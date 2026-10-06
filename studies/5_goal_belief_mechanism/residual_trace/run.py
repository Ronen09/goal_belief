"""The residual-trace experiment: where does the residual r beyond the posterior change the decision? The
observation-prediction experiment's reward models, frozen. Strata, measures and decision rule: PLAN.md.

    .venv/bin/python studies/5_goal_belief_mechanism/residual_trace/run.py                         # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/residual_trace/run.py --untrained --seeds 0   # smoke test (initial checkpoint)

As in the history-mediator experiment: at the goal token's four attention outputs d(h) = x_b(h) + r(h), x_b the
fit-side table over (posterior, L). On informative pairs (h, h') under g, the real arm adds r(h') - r(h); the rot arm
adds the same vectors rotated, same norm per component. Per pair: flip (action differs from natural) and shift (the
logit gap toward the donor's natural action, against natural). Per-pair data are aggregated by stratum here; tables.py
does the medians and the rule.
"""

from __future__ import annotations

import argparse, importlib.util, json, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


HM = load("hmrun", ROUNDS / "5_goal_belief_mechanism" / "history_mediator" / "run.py")
R42 = load("r42run", ROUNDS / "5_goal_belief_mechanism" / "additive_code" / "run.py")
R36, R30, M26, outputs, table = HM.R36, HM.R30, HM.M26, HM.outputs, HM.table
CHUNK, SEED, MIN_FIT = 4096, 46, HM.MIN_FIT
TABLES = HM.TABLES


def quart(x):
    """Quartile index 0..3 of each value within x."""
    q = torch.quantile(x.float(), torch.tensor([0.25, 0.5, 0.75], device=x.device))
    return torch.bucketize(x.float(), q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    nh, nL = len(ctx.L), t.max_prefix + 1
    allh = torch.arange(nh, device=dev)
    fit = torch.nonzero(ctx.fit).squeeze(1)
    Lh = ctx.L
    bel = t.belief[ctx.node].double()
    pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)[1]
    ent = -(bel * torch.log(bel.clamp(min=1e-300))).sum(-1)
    G42 = json.load(open(ROUNDS / "5_goal_belief_mechanism" / "additive_code" / "results.json"))["runs"]
    cells0 = R36.Cells(ctx)
    res = dict(runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        A = [("attn", l) for l in range(net.nl)]
        d = net.d
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        xs, lgs = [], []
        for g in range(3):
            xa, la = [], []
            for i in range(0, nh, 8192):
                h = allh[i:i + 8192]
                o, lg = outputs(net, ctx, h, torch.full_like(h, g))
                xa.append(torch.cat([o[c] for c in A], 1)); la.append(lg)
            xs.append(torch.cat(xa)); lgs.append(torch.cat(la))
        xbar = torch.stack(xs, 1).mean(1)
        lg_all = torch.stack(lgs, 1)                                                          # [nh, 3, 4]
        del xs, lgs
        cntL = torch.bincount(Lh[fit], minlength=nL).clamp(min=1).float()[:, None]
        Xb = torch.zeros(nL, xbar.shape[1], device=dev).index_add_(0, Lh[fit], xbar[fit]) / cntL
        D = xbar - Xb[Lh]
        lc = lg_all.double() - lg_all.double().mean(-1, keepdim=True)
        H = lc.mean(1)
        order = H.argsort(-1, descending=True)
        # the history-mediator experiment's shared cell set (same coverage rule)
        mixF = torch.cat([HM.R43.candidates(t, ctx)[:, HM.R43.CANDS.index(k)] for k in ("popt", "meanMDP")], 1)
        feats = dict(top=order[:, 0], rank2=order[:, 0] * 4 + order[:, 1], mix=HM.keys_of(mixF), b=pid)
        cnt = {k: table(feats[k], Lh, nL, D, fit)[1] for k in TABLES}
        xb, _ = table(pid, Lh, nL, D, fit)
        Hb, _ = table(pid, Lh, nL, H.float(), fit)
        R = D - xb
        ok = torch.stack([cnt[k][cells0.h] >= MIN_FIT for k in TABLES]).all(0)
        ch, cg = cells0.h[ok], cells0.g[ok]
        N = len(ch)
        rows = torch.arange(N, device=dev)
        cL = Lh[ch]
        nat = lg_all[ch, cg].argmax(-1)
        donor = rows.clone()
        for l in range(nL):
            idx = torch.nonzero(cL == l).squeeze(1)
            if len(idx) > 1:
                donor[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
        dh = ch[donor]
        dnat = lg_all[dh, cg].argmax(-1)
        inf = (dh != ch) & (dnat != nat)
        P = torch.nonzero(inf).squeeze(1)                                                    # informative pairs
        h, g, hd = ch[P], cg[P], dh[P]
        an, ad = nat[P], dnat[P]
        delta = R[hd] - R[h]
        rot = torch.empty_like(delta)
        for j in range(len(A)):
            v = delta[:, j * d:(j + 1) * d]
            r_ = torch.randn(v.shape, device=dev, generator=gen)
            rot[:, j * d:(j + 1) * d] = r_ / r_.norm(dim=-1, keepdim=True).clamp(min=1e-12) * v.norm(dim=-1, keepdim=True)

        def logits(vec):
            out_ = []
            for i in range(0, len(h), CHUNK):
                hh, gg, vv = h[i:i + CHUNK], g[i:i + CHUNK], vec[i:i + CHUNK]
                n = torch.arange(len(hh), device=dev)
                p = 1 + Lh[hh]

                def adder(dv):
                    def f(z):
                        z = z.clone(); z[n, p] = z[n, p] + dv
                        return z
                    return f
                out_.append(net(R30.seq(ctx, hh, gg), patch={c: adder(vv[:, j * d:(j + 1) * d]) for j, c in enumerate(A)})[0][n, p])
            return torch.cat(out_)

        n_ = torch.arange(len(h), device=dev)
        l0 = lg_all[h, g]
        gap = lambda l: l[n_, ad] - l[n_, an]
        lr, lo = logits(delta), logits(rot)
        flip_r, flip_o = (lr.argmax(-1) != an).float(), (lo.argmax(-1) != an).float()
        sh_r, sh_o = gap(lr) - gap(l0), gap(lo) - gap(l0)
        # strata
        Gs = torch.tensor(G42[f"seed{s}"]["G"], device=dev, dtype=torch.float64) if not a.untrained else torch.zeros(3, nL, 4, device=dev, dtype=torch.float64)
        Gt = Gs[:, Lh[h]].permute(1, 0, 2)
        unsolv = R42.margin(ctx.Q[h].argmax(-1), Gt) <= 0
        add_act = (H[h][:, None] + Gt).argmax(-1)[n_, g]
        inter = add_act != an
        margin = l0[n_, an] - l0[n_, ad]
        enti = torch.bucketize(ent[h].float(), torch.quantile(ent[ch].float(), torch.tensor([1 / 3, 2 / 3], device=dev)))
        herr = (H[h] - Hb[h].double()).norm(dim=-1)
        strata = dict(margin=quart(margin), unsolvable=unsolv.long(), interaction=inter.long(), goal=g, entropy=enti,
                      H_error=quart(herr), r_norm=quart(delta.norm(dim=-1)))
        row = dict(checkpoint=ck, pairs=len(P), overall=dict(flip_real=float(flip_r.mean()), flip_rot=float(flip_o.mean()),
                                                              shift_real=float(sh_r.mean()), shift_rot=float(sh_o.mean())), strata={})
        tot_ex = float((flip_r - flip_o).sum())
        for name, b in strata.items():
            bins = {}
            for v in range(int(b.max()) + 1 if len(b) else 0):
                m = b == v
                if not m.any():
                    continue
                ex = float((flip_r - flip_o)[m].sum())
                bins[str(v)] = dict(n=int(m.sum()), share=float(m.float().mean()), flip_real=float(flip_r[m].mean()), flip_rot=float(flip_o[m].mean()),
                                    excess_flip=float((flip_r - flip_o)[m].mean()), excess_shift=float((sh_r - sh_o)[m].mean()),
                                    share_of_excess=ex / tot_ex if tot_ex != 0 else None)
            row["strata"][name] = bins
        # joint least-squares fit of the excess shift on all strata (standardized)
        z = lambda x: (x.double() - x.double().mean()) / x.double().std().clamp(min=1e-12)
        X = torch.stack([torch.ones(len(h), device=dev, dtype=torch.float64), z(margin), z(unsolv), z(inter), z(g == 1), z(g == 2), z(ent[h]), z(herr), z(delta.norm(dim=-1))], 1)
        y = (sh_r - sh_o).double()
        lsq = lambda Y: torch.linalg.lstsq(X.cpu(), Y.cpu()[:, None], driver="gelsd").solution.squeeze(1)    # rank-tolerant
        beta = lsq(y)
        row["joint"] = dict(zip(("intercept", "margin", "unsolvable", "interaction", "goal_G2", "goal_G3", "entropy", "H_error", "r_norm"), beta.tolist()))
        row["joint_flip"] = dict(zip(("intercept", "margin", "unsolvable", "interaction", "goal_G2", "goal_G3", "entropy", "H_error", "r_norm"),
                                     lsq((flip_r - flip_o).double()).tolist()))
        res["runs"][f"seed{s}"] = row
        o = row["overall"]
        print(f"seed{s} {ck} pairs {len(P)} flip real {o['flip_real']:.3f} rot {o['flip_rot']:.3f} shift real {o['shift_real']:.3f} rot {o['shift_rot']:.3f} | "
              + " ".join(f"{nm}:" + ",".join(f"{v['excess_flip']:.3f}/{v['share']:.2f}" for v in bb.values()) for nm, bb in row["strata"].items())
              + f" | joint {', '.join(f'{k} {v:.3f}' for k, v in row['joint'].items())} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
