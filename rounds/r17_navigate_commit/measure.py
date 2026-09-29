"""Round 17: every measurement on saved checkpoints. One run directory (a seed of a condition) in, measures.json out.

For each checkpoint, on the fixed evaluation bank (the same histories for every checkpoint and model):
  policy      expected local regret and mass on the optimal set at bank decisions, by number of clues seen
  use         on twin pairs whose exact optimal sets are disjoint: P(own optimal set) - P(the twin's optimal set)
  probes      held-out affine (ridge) decoders of the belief at the 9 sites; Adam-trained affine and one-hidden-layer
              decoders with the same budget; validity of the decoded probabilities
  writes      what each block's attention and MLP add, read through the block's shared decoder, between twins
  causal      patches from twin / same-belief / cross-position donors and norm-matched random moves, on
              discovery layouts and on held-out evaluation layouts
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from goalgeo import navbank as B, navcausal as C, navcommit as NC, navmodel as NM, navppo as P, navprobe as PR

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as T                                                    # noqa: E402

BANK_SEED = 170017


class Context:
    def __init__(self, q_mode, device, quick=False):
        self.dev = device
        self.q_mode = q_mode
        _, qs = T.q_pairs(q_mode, device)
        self.tab = tab = P.Tables(T.ENV["n"], T.ENV["H"], T.ENV["c"], qs, device)
        qp = T.q_pairs(q_mode, device, "all")[0] if q_mode == "grid" else T.q_pairs(q_mode, device)[0]
        k = dict(n_broad=3000, dense_layouts=10, dense_per=200) if quick else {}
        self.bank = bank = B.make_bank(tab, qp, BANK_SEED, **k)
        self.d = d = B.decisions(tab, bank["broad"])
        self.dd = B.decisions(tab, bank["dense"])
        self.fit = torch.isin(d["layout"], bank["split"]["fit_layouts"])
        self.test = ~self.fit
        if q_mode == "grid":                                         # probes are fitted on trained reliabilities only
            tr = T.q_pairs("grid", device, "train")[0]
            qi = d["cfg"] % (tab.nq * tab.nq)
            trained = torch.isin(qi, tr[:, 0] * tab.nq + tr[:, 1])
            self.fit = self.fit & trained
            self.test_sets = dict(trained=self.test & trained, held=self.test & ~trained)
        else:
            self.test_sets = dict(trained=self.test)
        self.Y = PR.targets(d)
        n = 600 if quick else 12000
        self.pairs = dict(discovery=C.build_pairs(tab, bank["dense"], self.dd, bank["split"]["dense_discovery"], n, 1),
                          eval=C.build_pairs(tab, bank["dense"], self.dd, bank["split"]["dense_eval"], n, 2))
        g = torch.Generator(device=device); g.manual_seed(5)
        idx = torch.nonzero(self.fit).squeeze(1)
        self.sub = idx[torch.randperm(len(idx), device=device, generator=g)[:60000]]
        X0 = PR.raw_features(bank["broad"], d, tab)
        self.raw = {k: PR.scores(PR.Affine(X0[self.fit], self.Y[self.fit])(X0[t]), self.Y[t]) for k, t in self.test_sets.items()}


def use_score(net, pairs):
    """On decisive pairs: P(own optimal set) - P(twin's optimal set), averaged over both members of the pair."""
    n = torch.arange(len(pairs["dec"]), device=pairs["tok"].device)
    sm = lambda tok: net(tok, pairs["q"])[0][n, pairs["pos"]].masked_fill(~pairs["legal"], -1e9).softmax(-1)
    pr, pt = sm(pairs["tok"]), sm(pairs["tok_twin"])
    dis = ~(pairs["opt"] & pairs["opt_cf"]).any(1)
    own, cf = pairs["opt"].float(), pairs["opt_cf"].float()
    u = 0.5 * (((pr * own).sum(1) - (pr * cf).sum(1)) + ((pt * cf).sum(1) - (pt * own).sum(1)))
    return dict(use=u[dis].mean().item(), tv=(0.5 * (pr - pt).abs().sum(1)).mean().item(),
                tv_decisive=(0.5 * (pr - pt).abs().sum(1))[dis].mean().item(), n_decisive=int(dis.sum()))


def block_writes(net, ctx, acts, pairs):
    """Shared decoder per block, fitted on the block's input, middle and output together. Between twins, the
    decoded displacement of each sublayer's contribution along the exact displacement, as a share of it; the
    displacement of the other marginal; and the relative size of the twin difference in each contribution."""
    fit, Y = ctx.sub, ctx.Y
    out = {}
    rec_r, _ = C.record_at(net, pairs["tok"], pairs["q"], pairs["pos"])
    rec_t, _ = C.record_at(net, pairs["tok_twin"], pairs["q"], pairs["pos"])
    target = (pairs["marg_cf"] - pairs["marg"]).double()
    on = target.abs() > 1e-9
    decs = []
    for l in range(net.nl):
        X = torch.cat([acts[2 * l][fit], acts[2 * l + 1][fit], acts[2 * l + 2][fit]])
        dec = PR.Affine(X, torch.cat([Y[fit]] * 3)); decs.append(dec)
        held = torch.cat([acts[2 * l + i][ctx.test] for i in range(3)])
        sc = PR.scores(dec(held), torch.cat([Y[ctx.test]] * 3))
        out[f"L{l}.shared_r2_marginals"] = (sc["r2"][4] + sc["r2"][5]) / 2
        for c in [("attn", l), ("mlp", l)] + [("head", l, h) for h in range(net.nh)]:
            diff = rec_t[c] - rec_r[c]
            dm = dec.delta(diff)[:, 4:6]
            out[C.name(c)] = dict(write=((dm * target).sum() / (target ** 2).sum()).item(),
                                  off=(dm[~on].abs().mean() / target[on].abs().mean()).item(),
                                  rel_norm=(diff.norm(dim=1).mean() / rec_r[c].norm(dim=1).mean()).item())
    return out, decs


def source_table(net, decs, pairs):
    """Per head: what the report token of a station sends to the decision token, in decoded probability of that
    station's coordinate, signed so that + is towards the reported side; the same for the other coordinate; and
    the attention paid to report tokens, configuration tokens and the decision token itself."""
    tok, q, pos = pairs["tok"], pairs["q"], pairs["pos"]
    contrib, rec = C.source_contributions(net, decs, tok, q, pos)            # [layers, N, h, L, 2]
    rep = tok[:, :, NM.F_REP]
    L = tok.shape[1]
    before = torch.arange(L, device=tok.device)[None] < pos[:, None]
    n = torch.arange(len(tok), device=tok.device)
    out = {}
    for l in range(net.nl):
        A = rec["pattern"][l][n, :, pos]                                     # [N, h, L]
        for h in range(net.nh):
            on = off = cnt = 0.0
            for lo, col in ((NM.REP_L, 0), (NM.REP_T, 1)):
                is_rep = ((rep == lo) | (rep == lo + 1)) & before
                sign = torch.where(rep == lo + 1, 1.0, -1.0) * is_rep
                on = on + (contrib[l, :, h, :, col] * sign).sum()
                off = off + (contrib[l, :, h, :, 1 - col] * sign).sum()
                cnt = cnt + is_rep.sum()
            any_rep = (rep > 0) & before
            out[f"L{l}.H{h}"] = dict(on=(on / cnt).item(), off=(off / cnt).item(),
                                     attn_report=((A[:, h] * any_rep).sum(1) / any_rep.sum(1).clamp(min=1)).mean().item(),
                                     attn_report_total=(A[:, h] * any_rep).sum(1).mean().item(),
                                     attn_config=A[:, h, :2].sum(1).mean().item(), attn_self=A[n, h, pos].mean().item())
    return out


@torch.no_grad()
def measure(net, ctx, causal=True, adam=True):
    bank, d = ctx.bank, ctx.d
    acts, lg = PR.collect(net, bank["broad"], d)
    names = PR.site_names(net.nl)
    m = dict(policy=PR.policy_on_bank(lg, d), use={k: use_score(net, p) for k, p in ctx.pairs.items()}, probes={})
    probes = [PR.Affine(acts[i][ctx.fit], ctx.Y[ctx.fit]) for i in range(len(names))]
    for k, t in ctx.test_sets.items():
        m["probes"][k] = {n: PR.scores(probes[i](acts[i][t]), ctx.Y[t]) for i, n in enumerate(names)}
        two = t & (d["sh"] > 0) & (d["sv"] > 0)
        m["probes"][k + "_two_clues"] = {n: PR.scores(probes[i](acts[i][two]), ctx.Y[two]) for i, n in enumerate(names)}
    if adam:
        with torch.enable_grad():
            for hid, key in ((0, "adam_affine"), (64, "adam_mlp")):
                pr = PR.fit_adam(acts[:, ctx.sub], ctx.Y[ctx.sub], acts[:, ctx.test], ctx.Y[ctx.test], hidden=hid)
                m["probes"][key] = {n: PR.scores(pr[i], ctx.Y[ctx.test]) for i, n in enumerate(names)}
    m["writes"], decs = block_writes(net, ctx, acts, ctx.pairs["eval"])
    m["sources"] = source_table(net, decs, ctx.pairs["eval"])
    if causal:
        m["causal"] = {k: run_causal(net, probes[-1], ctx, p) for k, p in ctx.pairs.items()}
    return m


def run_causal(net, probe, ctx, pairs):
    out = C.run_pairs(net, probe, ctx.bank["dense"], ctx.dd, pairs)
    out["cross_strict"] = cross_strict(net, ctx, pairs)
    return out


def cross_strict(net, ctx, pairs):
    """Strict cross-position pairs; 'counterfactual actions' are those optimal only at the recipient's position
    with the donor's belief, 'donor actions' those optimal only at the donor's position. For each component: the gain in probability of
    the counterfactual actions as a share of the twin's gain ('belief'), and the gain in probability of the donor
    actions as a share of the donor's own ('imported'). On the pairs where the model's own three distributions
    (recipient, twin, donor) are pairwise more than 0.5 apart in total variation: how far the patch moves the
    recipient's distribution to the twin's ('to_twin') and to the donor's ('to_donor'), 1 = all the way."""
    keep = torch.nonzero(pairs["cross_strict"]).squeeze(1)
    if len(keep) < 20:
        return dict(n=int(len(keep)))
    tok, q, pos, legal = pairs["tok"][keep], pairs["q"][keep], pairs["pos"][keep], pairs["legal"][keep]
    base, lg_b = C.record_at(net, tok, q, pos)
    don, lg_d = C.record_at(net, pairs["cross_tok"][keep], q, pos)
    _, lg_t = C.record_at(net, pairs["tok_twin"][keep], q, pos)
    sm = lambda lg: lg.masked_fill(~legal, -1e9).softmax(-1)
    pb, pt = sm(lg_b), sm(lg_t)
    pd = sm(lg_d)
    o, ocf, od = pairs["opt"][keep], pairs["opt_cf"][keep], pairs["cross_opt"][keep]
    cf, do = (ocf & ~o & ~od).float(), (od & ~o & ~ocf).float()
    tv = lambda a, b_: 0.5 * (a - b_).abs().sum(1)
    sep = (tv(pb, pt) > 0.5) & (tv(pb, pd) > 0.5) & (tv(pt, pd) > 0.5)        # the model itself acts differently in all three
    out = dict(n=int(len(keep)), twin_gain=((pt - pb) * cf).sum(1).mean().item(), donor_gain=((pd - pb) * do).sum(1).mean().item(),
               n_separated=int(sep.sum()))
    for comp in C.components(net):
        lg_p, _ = C.patched(net, tok, q, pos, comp, don[comp])
        pp = sm(lg_p)
        out[C.name(comp)] = dict(to_twin=(1 - tv(pp, pt)[sep].mean() / tv(pb, pt)[sep].mean()).item() if sep.sum() >= 20 else None,
                                 to_donor=(1 - tv(pp, pd)[sep].mean() / tv(pb, pd)[sep].mean()).item() if sep.sum() >= 20 else None,
                                 belief=(((pp - pb) * cf).sum(1).mean() / ((pt - pb) * cf).sum(1).mean()).item(),
                                 imported=(((pp - pb) * do).sum(1).mean() / ((pd - pb) * do).sum(1).mean()).item())
    return out


def on_policy_by_q(net, ctx, n=8192):
    """Grid models: on-policy regret for trained reliability pairs, held-out values and held-out combinations."""
    out = {}
    for which in ("train", "held_value", "held_combo"):
        qp, _ = T.q_pairs("grid", ctx.dev, which)
        g = torch.Generator(device=ctx.dev); g.manual_seed(99)
        cfg, start = P.sample_configs(ctx.tab, n, qp, g)
        for greedy in (False, True):
            s = P.summarize(P.rollout(net, ctx.tab, cfg, start, gen=g, greedy=greedy))
            s["v_star"] = ctx.tab.V[cfg, ctx.tab.H, start, 0, 0].mean().item()
            out[which + ("_greedy" if greedy else "_sampled")] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="seed directories")
    ap.add_argument("--q", default="fixed", choices=("fixed", "grid"))
    ap.add_argument("--only-final", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ctx = Context(a.q, a.device, quick=a.quick)
    for run in a.runs:
        run = Path(run)
        layers = json.loads((run / "log.json").read_text())["args"]["layers"]
        net = NM.Net(T.ENV["n"] ** 2, T.ENV["H"], layers=layers).to(a.device).eval()
        cks = sorted((run / "ckpt").glob("u*.pt"))
        if a.only_final:
            cks = [cks[0], cks[-1]]
        res = dict(raw_baseline=ctx.raw, sites=PR.site_names(layers), targets=PR.TARGETS, checkpoints={})
        for ck in cks:
            net.load_state_dict(torch.load(ck))
            m = measure(net, ctx, adam=not a.quick)
            if a.q == "grid":
                m["on_policy_by_q"] = on_policy_by_q(net, ctx, 1024 if a.quick else 8192)
            res["checkpoints"][int(ck.stem[1:])] = m
            print(run, ck.stem, f"bank regret {m['policy']['regret']:.4f} use {m['use']['eval']['use']:.3f}", flush=True)
        (run / "measures.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
