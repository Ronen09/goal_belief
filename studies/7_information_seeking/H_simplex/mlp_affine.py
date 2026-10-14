"""What does the MLP do to the goal-history mixture? Its goal response is far from its local Jacobian but a single
linear map across decisions (section 13), so: replace an MLP by its best affine approximation on natural inputs,
attention intact, and measure ordinary behaviour (return in the environment; agreement with the natural model's
decisions and their goal dependence on the natural model's test decisions) and cross-goal transport (the ridge map
M(A->B) between goals' difference vectors at the input of block 2, refitted on the variant's states: held-out R², and
on equiv.py's A-equivalent, B-different pairs the switch of B's decision by the direct B-written difference and by the
mapped A-written one). If a shared affine map fails, compare per-goal affine maps (the goal selecting a linear response
to the history). Variants at block 1 (the MLP of section 13), block 0, and both; replacement at every token, or at
the decision token only (history tokens keep their MLP). The affine maps are fitted on LN(mid), keeping the layer
norm; one variant is affine in the residual itself (the layer norm removed too). Exploratory.

    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/H_simplex/mlp_affine.py        # writes mlp_affine.md / .json
"""

from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C, equiv as E          # noqa: E402

N_TRAIN_PAIRS, CHUNK = 12288, 1024


class Replaced:
    """The network with the MLP of some blocks replaced: spec[l] = dict(kind='shared'|'pergoal', W, c, on_ln);
    decision_only: only at each row's last non-PAD token (callers pad the sequence beyond the read position)."""

    def __init__(self, net, spec, decision_only=False):
        self.net, self.spec, self.decision_only = net, spec, decision_only

    def __getattr__(self, k):
        return getattr(self.net, k)

    def __call__(self, tok, q=None, record=False, patch=None):
        patch = dict(patch or {}); store = {}
        goal = tok[:, 1, MM.F_GOAL] - 1
        live = tok[..., MM.F_TYPE] != MM.PAD
        last = live.long().cumsum(1).argmax(1)
        for l, sp in self.spec.items():
            b = self.net.blocks[l]
            def cap(x, l=l):
                store[l] = x; return x
            def rep(m, l=l, sp=sp, b=b):
                mid = store[l]; z = b.ln2(mid) if sp["on_ln"] else mid
                if sp["kind"] == "shared":
                    out = z @ sp["W"] + sp["c"]
                else:
                    out = torch.einsum("nld,ndo->nlo", z, sp["W"][goal]) + sp["c"][goal][:, None]
                out = out.to(m.dtype)
                if self.decision_only:
                    mask = torch.arange(tok.shape[1], device=tok.device)[None] == last[:, None]
                    out = torch.where(mask[..., None], out, m)
                return out
            patch[("resid_mid", l)] = cap; patch[("mlp", l)] = rep
        return self.net(tok, q, record=record, patch=patch)


@torch.no_grad()
def fit_data(net, tok, K, layers):
    """LN(mid), mid and the MLP output at every live token under every goal: {l: (Z, X, Y, goal, ep)} (float32)."""
    N, dev = len(tok), tok.device
    out = {l: ([], [], [], [], [], []) for l in layers}
    for i in range(0, N, CHUNK):
        tk = tok[i:i + CHUNK].repeat(K, 1, 1); n_ = len(tok[i:i + CHUNK])
        gl = torch.arange(K, device=dev).repeat_interleave(n_); tk[:, 1, MM.F_GOAL] = gl + 1
        _, _, rec = net(tk, record=True)
        live = tk[..., MM.F_TYPE] != MM.PAD
        ep = (torch.arange(n_, device=dev) + i).repeat(K)[:, None].expand(-1, tk.shape[1])
        is_last = torch.arange(tk.shape[1], device=dev)[None] == live.long().cumsum(1).argmax(1)[:, None]
        for l in layers:
            mid = rec["mid"][l]
            out[l][0].append(net.blocks[l].ln2(mid)[live]); out[l][1].append(mid[live]); out[l][2].append(rec["mlp"][l][live])
            out[l][3].append(gl[:, None].expand(-1, tk.shape[1])[live]); out[l][4].append(ep[live]); out[l][5].append(is_last[live])
    return {l: tuple(torch.cat(v) for v in vs) for l, vs in out.items()}


def affine(X, Y, va):
    rg = R.Ridge(X, Y, va); return dict(W=rg.W.float(), c=(rg.my - rg.mu @ rg.W).float())


def pad_after(tok, p):
    tk = tok.clone(); tk[torch.arange(tok.shape[1], device=tok.device)[None] > p[:, None]] = 0; return tk


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K = len(t.goal_cell)
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        t0 = time.time()
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        nat, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        nat, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        gb = MS.episodes(t, lambda env, s_: MS.logits_goals(net, env, s_).mean(1).argmax(-1), goal, cell, 79)[0]
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        # ---- affine fits on the fit episodes, every live token under every goal
        data = fit_data(net, rf["tok"], K, (0, 1, 2, 3))
        maps = {}
        for l, (Z, X, Y, gl, ep, _) in data.items():
            va = ep % 10 == 0
            maps[("shared", l)] = dict(kind="shared", on_ln=True, **affine(Z, Y, va))
            maps[("resid", l)] = dict(kind="shared", on_ln=False, **affine(X, Y, va))
            per = [affine(Z[gl == g], Y[gl == g], va[gl == g]) for g in range(K)]
            maps[("pergoal", l)] = dict(kind="pergoal", on_ln=True, W=torch.stack([m["W"] for m in per]), c=torch.stack([m["c"] for m in per]))
            maps[("const", l)] = dict(kind="shared", on_ln=True, W=torch.zeros_like(maps[("shared", l)]["W"]), c=Y.mean(0))
        # fit quality at the decision token of the test decisions (under every goal; held-out episodes)
        gq = torch.Generator(device=dev); gq.manual_seed(1)
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gq)[:E.N_TEST]
        tokp, p = pad_after(r["tok"][dt["ep"][ti]], dt["st"][ti] + 1), dt["st"][ti] + 1
        tdata = fit_data(net, tokp, K, (0, 1, 2, 3))
        fitq = {}
        for l, (Z, X, Y, gl, ep, isl) in tdata.items():
            Zd, Xd, Yd, gd = Z[isl], X[isl], Y[isl], gl[isl]                                  # the decision token = the last live token of each row
            r2 = lambda P: float(1 - ((P - Yd) ** 2).sum() / ((Yd - Yd.mean(0)) ** 2).sum())
            m = maps[("shared", l)]; fitq[f"b{l}_shared"] = r2(Zd @ m["W"] + m["c"])
            m = maps[("resid", l)]; fitq[f"b{l}_resid"] = r2(Xd @ m["W"] + m["c"])
            m = maps[("pergoal", l)]; fitq[f"b{l}_pergoal"] = r2(torch.einsum("nd,ndo->no", Zd, m["W"][gd]) + m["c"][gd])
            fitq[f"b{l}_const"] = r2(maps[("const", l)]["c"][None].expand_as(Yd))
        variants = [("natural", {}, False),
                    ("b1 shared affine", {1: maps[("shared", 1)]}, False), ("b1 shared affine, decision token only", {1: maps[("shared", 1)]}, True),
                    ("b1 per-goal affine", {1: maps[("pergoal", 1)]}, False), ("b1 per-goal affine, decision token only", {1: maps[("pergoal", 1)]}, True),
                    ("b1 constant (MLP removed)", {1: maps[("const", 1)]}, False), ("b1 shared affine in the residual (LN removed)", {1: maps[("resid", 1)]}, False),
                    ("b0 shared affine", {0: maps[("shared", 0)]}, False), ("b0 per-goal affine", {0: maps[("pergoal", 0)]}, False),
                    ("b0+b1 shared affine", {0: maps[("shared", 0)], 1: maps[("shared", 1)]}, False), ("b0+b1 per-goal affine", {0: maps[("pergoal", 0)], 1: maps[("pergoal", 1)]}, False),
                    ("b2 shared affine", {2: maps[("shared", 2)]}, False), ("b3 shared affine", {3: maps[("shared", 3)]}, False),
                    ("b2+b3 shared affine", {2: maps[("shared", 2)], 3: maps[("shared", 3)]}, False),
                    ("all four MLPs shared affine", {l: maps[("shared", l)] for l in range(4)}, False), ("all four MLPs per-goal affine", {l: maps[("pergoal", l)] for l in range(4)}, False),
                    ("all four MLPs shared affine in the residual (LN removed)", {l: maps[("resid", l)] for l in range(4)}, False),
                    ("all four MLPs shared affine, decision token only", {l: maps[("shared", l)] for l in range(4)}, True),
                    ("all four MLPs constant (removed)", {l: maps[("const", l)] for l in range(4)}, False)]
        # ---- shared machinery for the transport test (equiv2's recipe on the natural model's episodes)
        gp = torch.Generator(device=dev); gp.manual_seed(21)
        mf = len(df["st"]); a_ = torch.randint(mf, (N_TRAIN_PAIRS * 4,), device=dev, generator=gp); by_step = {}
        for i in torch.randint(mf, (N_TRAIN_PAIRS * 4,), device=dev, generator=gp).tolist(): by_step.setdefault(int(df["st"][i]), []).append(i)
        tp = []
        for i in a_.tolist():
            cands = by_step.get(int(df["st"][i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
            if j != i and float((df["b"][i] - df["b"][j]).abs().sum()) >= 0.5: tp.append((i, j))
            if len(tp) >= N_TRAIN_PAIRS: break
        tp = torch.tensor(tp, device=dev); f1, f2 = tp[:, 0], tp[:, 1]; va = df["ep"][f1] % 10 == 0
        tokf1, pf1 = pad_after(rf["tok"][df["ep"][f1]], df["st"][f1] + 1), df["st"][f1] + 1; tokf2, pf2 = pad_after(rf["tok"][df["ep"][f2]], df["st"][f2] + 1), df["st"][f2] + 1
        Qt = E.qvals(t, dt, ti); cQ = (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Qt.clamp(min=1e-9)) / np.log(t.gamma))
        sett = Qt >= Qt.max(-1, keepdim=True).values - 1e-6
        gq.manual_seed(7); m = len(ti); cand = torch.randint(m, (m, E.CANDS), device=dev, generator=gq)
        same_step = dt["st"][ti][:, None] == dt["st"][ti][cand]
        dq = (cQ[:, None] - cQ[cand]).abs().amax(-1); same_set = (sett[:, None] == sett[cand]).all(-1); disjoint = ~(sett[:, None] & sett[cand]).any(-1)
        reach = Qt.amax(-1) >= E.MINQ
        eq = same_set & (dq < E.EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > E.DIFF)
        pairs = {}
        for A in range(K):
            for B in range(K):
                if A == B: continue
                ok = same_step & eq[..., A] & dif[..., B] & (cand != torch.arange(m, device=dev)[:, None])
                ii, cc = torch.nonzero(ok, as_tuple=True)
                if len(ii) < 30: continue
                _, first = np.unique(ii.cpu().numpy(), return_index=True); ii, cc = ii[torch.tensor(first, device=dev)], cc[torch.tensor(first, device=dev)]
                jj = cand[ii, cc]
                if len(ii) > 1500:
                    sel = torch.randperm(len(ii), device=dev, generator=gq)[:1500]; ii, jj = ii[sel], jj[sel]
                pairs[(A, B)] = (ii, jj)
        own = dt["goal"][ti]; nat_dec = dt["lc"][ti][torch.arange(m, device=dev), own].argmax(-1)
        nat_gd = (dt["lc"][ti].argmax(-1) != dt["lc"][ti].argmax(-1)[:, :1]).any(1)
        rows = {}
        for name, spec, donly in variants:
            v = Replaced(net, spec, donly) if spec else net
            on = MS.episodes(t, MS.natural(v), goal, cell, 79)[0]
            dv = R.decisions(t, v, r, K)                                                      # the natural model's test histories, the variant's logits
            lv = dv["lc"][ti]; dec = lv[torch.arange(m, device=dev), own].argmax(-1)
            row = dict(ret=on["ret"], success=on["success"], recovery=(on["ret"] - gb["ret"]) / (nat["ret"] - gb["ret"]),
                       agree=float((dec == nat_dec).double().mean()), goal_dep=float((lv.argmax(-1) != lv.argmax(-1)[:, :1]).any(1).double().mean()),
                       agree_gd=float((dec == nat_dec)[nat_gd].double().mean()))
            # transport at the input of block 2
            XF1, _ = C.states(v, tokf1, pf1, K, (2,)); XF2, _ = C.states(v, tokf2, pf2, K, (2,)); XF1, XF2 = XF1[2].double(), XF2[2].double()
            Xt, Lt = C.states(v, tokp, p, K, (2,))
            X2 = Xt[2].double()
            hist = X2.mean(1); Mg = (X2 - hist[:, None]).mean(0); I = X2 - hist[:, None] - Mg[None]
            row["inter_share_site2"] = float((I ** 2).sum() / ((X2 - X2.mean((0, 1))) ** 2).sum())
            tr = {k: [] for k in ("r2_linear", "direct_B", "linear_diff", "untransformed_A", "random")}
            for (A, B), (ii, jj) in pairs.items():
                lin = R.Ridge(XF2[:, A] - XF1[:, A], XF2[:, B] - XF1[:, B], va)
                X1, X2p = X2[ii], X2[jj]; dA, dB = X2p[:, A] - X1[:, A], X2p[:, B] - X1[:, B]
                tr["r2_linear"].append(float(1 - ((lin(dA) - dB) ** 2).sum() / ((dB - dB.mean(0)) ** 2).sum()))
                dec1B, dec2B = Lt[ii, B].argmax(-1), Lt[jj, B].argmax(-1); differ = dec1B != dec2B
                gr = torch.Generator(device=dev); gr.manual_seed(3)
                rnd = torch.randn(dB.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * dB.norm(dim=1, keepdim=True)
                for k, vec in (("direct_B", dB), ("linear_diff", lin(dA)), ("untransformed_A", dA), ("random", rnd)):
                    ns = X1.clone(); ns[:, B] = ns[:, B] + vec
                    Le = C.run_edit(v, tokp[ii], p[ii], K, 2, ns)
                    tr[k].append(float((Le[:, B].argmax(-1) == dec2B)[differ].double().mean()) if int(differ.sum()) else float("nan"))
            row.update({k: float(np.nanmean(vv)) for k, vv in tr.items()})
            rows[name] = row
            print(f"  {name}: " + " ".join(f"{k} {vv:.3f}" for k, vv in row.items()), flush=True)
        res[f"seed{s}"] = dict(rows=rows, fit=fitq, goal_blind_ret=gb["ret"])
        print(f"seed{s} done in {time.time() - t0:.0f}s; fit quality {fitq}", flush=True)
        (HERE / "mlp_affine.json").write_text(json.dumps(res))
    report()


def report():
    res = json.loads((HERE / "mlp_affine.json").read_text()); runs = list(res.values())
    names = list(runs[0]["rows"].keys())
    mm = lambda n, k, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([r_["rows"][n][k] for r_ in runs])
    fq = lambda k: (lambda xs: f"{np.median(xs):.2f} ({min(xs):.2f}–{max(xs):.2f})")([r_["fit"][k] for r_ in runs])
    L = ["# The MLP replaced by its best affine approximation", "",
         "Generated by `mlp_affine.py` (exploratory). Affine maps fitted by ridge on LN(mid) → MLP output at every live token of the fit episodes under every goal (shared: one map; per-goal: one map per goal; constant: the mean output; 'in the residual': fitted on mid, the layer norm removed too). "
         "Behaviour: greedy return on the test spawns (the natural model 0.548, QMDP 0.551, goal-blind H policy as the floor of the recovery); agreement with the natural model's own-goal decisions on its test decisions (all, and those where its move depends on the goal); the share of decisions whose move depends on the goal. Transport at the input of block 2 (equiv2's recipe, refitted on the variant's states): held-out R² of the ridge map M(A→B) between goals' difference vectors; on the A-equivalent, B-different pairs, B's decision switched to the donor's by the direct B-written difference, by M(A→B) Δ_A, by Δ_A and by random. Medians (min–max) over six models.", "",
         "## Fit quality of the replacements at the decision token (held-out R² of the MLP output)", "",
         "| block | shared affine on LN(mid) | per-goal affine | affine in the residual (no LN) | constant |", "|---|---|---|---|---|",
         f"| 0 | {fq('b0_shared')} | {fq('b0_pergoal')} | {fq('b0_resid')} | {fq('b0_const')} |",
         f"| 1 | {fq('b1_shared')} | {fq('b1_pergoal')} | {fq('b1_resid')} | {fq('b1_const')} |",
         f"| 2 | {fq('b2_shared')} | {fq('b2_pergoal')} | {fq('b2_resid')} | {fq('b2_const')} |",
         f"| 3 | {fq('b3_shared')} | {fq('b3_pergoal')} | {fq('b3_resid')} | {fq('b3_const')} |", "",
         "## Behaviour and transport", "",
         "| variant | return (success) | recovery of the goal-directed return | agreement with the natural decisions: all / goal-dependent | goal-dependent decisions | interaction share at site 2 | map R² | switch: direct Δ_B | M(A→B) Δ_A | Δ_A | random |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | {mm(n, 'ret', 3)} ({mm(n, 'success')}) | {mm(n, 'recovery')} | {mm(n, 'agree')} / {mm(n, 'agree_gd')} | {mm(n, 'goal_dep')} | {mm(n, 'inter_share_site2')} | {mm(n, 'r2_linear')} | {mm(n, 'direct_B')} | {mm(n, 'linear_diff')} | {mm(n, 'untransformed_A')} | {mm(n, 'random')} |")
    (HERE / "mlp_affine.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
