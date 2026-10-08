"""Does the change of representation across goals generalise? For each ordered goal pair (A, B), maps from the
A-written difference Delta_A = x2(A) - x1(A) to the B-written difference Delta_B at the input of block 2, fitted on
random same-step pairs of fit decisions (posteriors >= 0.5 apart): linear (ridge) on differences, an MLP on
differences, and an MLP from the A-written state to the B-written state (difference taken after). Applied to the
held-out A-equivalent, B-different pairs of equiv.py: x1(B) + T(Delta_A), logits read under B; against the
untransformed Delta_A, the direct Delta_B and random. Exploratory.

    .venv/bin/python studies/7_information_seeking/H_simplex/equiv2.py        # writes equiv2.md / equiv2.json
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C, equiv as E          # noqa: E402

N_TRAIN_PAIRS = 12288


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len, n = len(t.goal_cell), t.H, t.n
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        # ---- training pairs from the fit episodes: random same-step pairs, posteriors >= 0.5 apart
        gp = torch.Generator(device=dev); gp.manual_seed(21)
        mf = len(df["st"]); a_ = torch.randint(mf, (N_TRAIN_PAIRS * 4,), device=dev, generator=gp)
        by_step = {}
        for i in torch.randint(mf, (N_TRAIN_PAIRS * 4,), device=dev, generator=gp).tolist():
            by_step.setdefault(int(df["st"][i]), []).append(i)
        tp = []
        for i in a_.tolist():
            cands = by_step.get(int(df["st"][i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
            if j != i and float((df["b"][i] - df["b"][j]).abs().sum()) >= 0.5: tp.append((i, j))
            if len(tp) >= N_TRAIN_PAIRS: break
        tp = torch.tensor(tp, device=dev); f1, f2 = tp[:, 0], tp[:, 1]
        XF1, _ = C.states(net, rf["tok"][df["ep"][f1]], df["st"][f1] + 1, K, (2,)); XF2, _ = C.states(net, rf["tok"][df["ep"][f2]], df["st"][f2] + 1, K, (2,))
        XF1, XF2 = XF1[2].double(), XF2[2].double()                                                       # [P, K, d]
        va = df["ep"][f1] % 10 == 0
        # ---- held-out pairs: equiv.py's selection on the test decisions
        gsub = torch.Generator(device=dev); gsub.manual_seed(1)
        _ = torch.randperm(mf, device=dev, generator=gsub)[:E.N_FIT]
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:E.N_TEST]
        Xt, Lt = C.states(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K, (2,))
        Qt = E.qvals(t, dt, ti); cQ = (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Qt.clamp(min=1e-9)) / np.log(t.gamma))
        sett = Qt >= Qt.max(-1, keepdim=True).values - 1e-6
        gq = torch.Generator(device=dev); gq.manual_seed(7); m = len(ti); cand = torch.randint(m, (m, E.CANDS), device=dev, generator=gq)
        same_step = dt["st"][ti][:, None] == dt["st"][ti][cand]
        dq = (cQ[:, None] - cQ[cand]).abs().amax(-1); same_set = (sett[:, None] == sett[cand]).all(-1); disjoint = ~(sett[:, None] & sett[cand]).any(-1)
        reach = Qt.amax(-1) >= E.MINQ
        eq = same_set & (dq < E.EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > E.DIFF)
        row = {"by_goal_pair": {}}
        agg = {k: [] for k in ("linear_diff", "mlp_diff", "mlp_state", "untransformed_A", "direct_B", "random", "r2_linear", "r2_mlp_diff", "r2_mlp_state", "r2_identity", "pairs", "A_changed_linear")}
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
                # maps fitted on the training pairs for this (A, B)
                DA, DB = XF2[:, A] - XF1[:, A], XF2[:, B] - XF1[:, B]
                lin = R.Ridge(DA, DB, va)
                mlp_d = R.fit_mlp(DA, DB, va, steps=2500, d=256, seed=A * 4 + B)
                mlp_s = R.fit_mlp(torch.cat([XF1[:, A], XF2[:, A]]), torch.cat([XF1[:, B], XF2[:, B]]), torch.cat([va, va]), steps=2500, d=256, seed=40 + A * 4 + B)
                # held-out differences
                X1, X2 = Xt[2][ii].double(), Xt[2][jj].double()
                dA, dB = X2[:, A] - X1[:, A], X2[:, B] - X1[:, B]
                preds = {"linear_diff": lin(dA), "mlp_diff": mlp_d(dA), "mlp_state": mlp_s(X2[:, A]) - mlp_s(X1[:, A]), "untransformed_A": dA, "direct_B": dB}
                gr = torch.Generator(device=dev); gr.manual_seed(3)
                rnd = torch.randn(dB.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * dB.norm(dim=1, keepdim=True); preds["random"] = rnd
                r2 = lambda P: float(1 - ((P - dB) ** 2).sum() / ((dB - dB.mean(0)) ** 2).sum())
                agg["r2_linear"].append(r2(preds["linear_diff"])); agg["r2_mlp_diff"].append(r2(preds["mlp_diff"])); agg["r2_mlp_state"].append(r2(preds["mlp_state"])); agg["r2_identity"].append(r2(dA))
                dec1B, dec2B = Lt[ii, B].argmax(-1), Lt[jj, B].argmax(-1); differ = dec1B != dec2B
                dec1A = Lt[ii, A].argmax(-1)
                tok1, p1 = r["tok"][dt["ep"][ti][ii]], dt["st"][ti][ii] + 1
                ar = torch.arange(len(ii), device=dev)
                out = {}
                for k, vec in preds.items():
                    ns = X1.clone(); ns[:, B] = ns[:, B] + vec
                    Le = C.run_edit(net, tok1, p1, K, 2, ns)
                    out[k] = float((Le[:, B].argmax(-1) == dec2B)[differ].double().mean()); agg[k].append(out[k])
                nsA = X1.clone(); nsA[:, A] = nsA[:, A] + preds["linear_diff"]
                agg["A_changed_linear"].append(float((C.run_edit(net, tok1, p1, K, 2, nsA)[:, A].argmax(-1) != dec1A).double().mean()))
                agg["pairs"].append(len(ii))
                row["by_goal_pair"][f"{A}->{B}"] = dict(pairs=len(ii), **out, r2_linear=agg["r2_linear"][-1], r2_mlp_diff=agg["r2_mlp_diff"][-1], r2_mlp_state=agg["r2_mlp_state"][-1])
        row["mean"] = {k: float(np.mean(v)) for k, v in agg.items() if v}
        res[f"seed{s}"] = row
        print(f"seed{s}: " + " ".join(f"{k} {v:.2f}" for k, v in row["mean"].items()), flush=True)
        (HERE / "equiv2.json").write_text(json.dumps(res))
    res = json.loads((HERE / "equiv2.json").read_text()); runs = list(res.values())
    mm = lambda k, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([r_["mean"][k] for r_ in runs])
    L = ["# Does the change of representation across goals generalise?", "",
         f"Generated by `equiv2.py` (exploratory). For each ordered goal pair (A, B): maps from the A-written difference x2(A) − x1(A) to the B-written difference at the input of block 2, fitted on {N_TRAIN_PAIRS} random same-step pairs of fit decisions (posteriors ≥ 0.5 apart); "
         "applied to the held-out A-equivalent, B-different pairs of `equiv.py` (test episodes; " + mm("pairs", 0) + " pairs per goal pair): x1(B) + T(Δ_A), logits read under B. Means over goal pairs, medians (min–max) over six models.", "",
         "| vector added under B | B's decision to b2's | held-out R² for Δ_B |", "|---|---|---|",
         f"| direct B-written difference Δ_B | {mm('direct_B')} | 1 |",
         f"| **linear map of Δ_A (ridge on differences)** | **{mm('linear_diff')}** | {mm('r2_linear')} |",
         f"| MLP on differences | {mm('mlp_diff')} | {mm('r2_mlp_diff')} |",
         f"| MLP from the A-written state to the B-written state | {mm('mlp_state')} | {mm('r2_mlp_state')} |",
         f"| untransformed Δ_A | {mm('untransformed_A')} | {mm('r2_identity')} |",
         f"| random, Δ_B's norm | {mm('random')} | |", "",
         f"The linear map's vector added under A changes A's decision in {mm('A_changed_linear')} of pairs."]
    (HERE / "equiv2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
