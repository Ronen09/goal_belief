"""Do the pairwise maps between goals form one consistent system? The linear maps M_{A->B} of equiv2.py (ridge from the
A-written difference x2(A) - x1(A) to the B-written one at the input of block 2, fitted per ordered goal pair on
random same-step pairs of fit decisions) are composed and inverted, and evaluated on actual held-out state
differences, not as matrices:

  composition  M_{B->C} M_{A->B} Delta_A  against  M_{A->C} Delta_A  and the direct Delta_C  (24 ordered triples)
  round trip   M_{B->A} M_{A->B} Delta_A  against  Delta_A                                  (12 ordered pairs)
  three-cycle  M_{C->A} M_{B->C} M_{A->B} Delta_A  against  Delta_A                         (24 ordered triples)

Fidelity: held-out R^2 and cosine against the target difference and the norm ratio, on random same-step test pairs
(posteriors >= 0.5 apart). Effect: the vector added to x1 under the target goal, the share of pairs whose decision
under that goal switches to b2's; composition on equiv.py's A-equivalent, C-different pairs (where the direct map's
number is equiv2's), round trips on test pairs whose A-decisions differ. Exploratory.

    .venv/bin/python studies/7_information_seeking/H_simplex/equiv3.py        # writes equiv3.md / equiv3.json
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
N_EVAL_PAIRS = 4096


def same_step_pairs(d, n, gen, min_l1=0.5):
    """Random same-step pairs of decisions (i, j), posteriors >= min_l1 apart in L1."""
    m = len(d["st"]); dev = d["st"].device
    by_step = {}
    for i in torch.randint(m, (n * 4,), device=dev, generator=gen).tolist():
        by_step.setdefault(int(d["st"][i]), []).append(i)
    out = []
    for i in torch.randint(m, (n * 4,), device=dev, generator=gen).tolist():
        cands = by_step.get(int(d["st"][i]), [])
        if not cands: continue
        j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gen))]
        if j != i and float((d["b"][i] - d["b"][j]).abs().sum()) >= min_l1: out.append((i, j))
        if len(out) >= n: break
    return torch.tensor(out, device=dev)


def fid(P, T):
    """Fidelity of predicted differences P against target differences T: R^2, mean cosine, mean norm ratio."""
    r2 = float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())
    cos = float(((P * T).sum(1) / (P.norm(dim=1) * T.norm(dim=1)).clamp(min=1e-12)).mean())
    nr = float((P.norm(dim=1) / T.norm(dim=1).clamp(min=1e-12)).mean())
    return r2, cos, nr


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K = len(t.goal_cell)
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        # ---- the 12 maps, fitted exactly as in equiv2.py (same training pairs)
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
        XF1, XF2 = XF1[2].double(), XF2[2].double()
        va = df["ep"][f1] % 10 == 0
        M = {(A, B): R.Ridge(XF2[:, A] - XF1[:, A], XF2[:, B] - XF1[:, B], va) for A in range(K) for B in range(K) if A != B}
        # ---- held-out: equiv.py's selection on the test decisions (as in equiv2.py) and random same-step test pairs
        gsub = torch.Generator(device=dev); gsub.manual_seed(1)
        _ = torch.randperm(mf, device=dev, generator=gsub)[:E.N_FIT]
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:E.N_TEST]
        Xt, Lt = C.states(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K, (2,)); X2t = Xt[2].double()
        Qt = E.qvals(t, dt, ti); cQ = (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Qt.clamp(min=1e-9)) / np.log(t.gamma))
        sett = Qt >= Qt.max(-1, keepdim=True).values - 1e-6
        gq = torch.Generator(device=dev); gq.manual_seed(7); m = len(ti); cand = torch.randint(m, (m, E.CANDS), device=dev, generator=gq)
        same_step = dt["st"][ti][:, None] == dt["st"][ti][cand]
        dq = (cQ[:, None] - cQ[cand]).abs().amax(-1); same_set = (sett[:, None] == sett[cand]).all(-1); disjoint = ~(sett[:, None] & sett[cand]).any(-1)
        reach = Qt.amax(-1) >= E.MINQ
        eq = same_set & (dq < E.EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > E.DIFF)
        sub = {k: dt[k][ti] for k in ("st", "b", "ep")}
        ge = torch.Generator(device=dev); ge.manual_seed(33)
        rp = same_step_pairs(sub, N_EVAL_PAIRS, ge); r1, r2_ = rp[:, 0], rp[:, 1]
        D = X2t[r2_] - X2t[r1]                                                                      # [P, K, d] held-out differences, all goals
        tokr, pr = r["tok"][sub["ep"][r1]], sub["st"][r1] + 1
        gr = torch.Generator(device=dev); gr.manual_seed(3)
        rand_like = lambda T: (lambda z: z / z.norm(dim=1, keepdim=True) * T.norm(dim=1, keepdim=True))(torch.randn(T.shape, device=dev, generator=gr))

        def switch(X1, tok, p, G, vec, target, mask):
            ns = X1.clone(); ns[:, G] = ns[:, G] + vec
            Le = C.run_edit(net, tok, p, K, 2, ns)
            return float((Le[:, G].argmax(-1) == target)[mask].double().mean())

        row = {"triples": {}, "pairs": {}}
        agg = {}
        add = lambda k, v: agg.setdefault(k, []).append(v)
        # ---- composition over ordered triples
        for A in range(K):
            for B in range(K):
                for Cc in range(K):
                    if len({A, B, Cc}) < 3: continue
                    dA, dC = D[:, A], D[:, Cc]
                    comp, direct = M[(B, Cc)](M[(A, B)](dA)), M[(A, Cc)](dA)
                    tr = {}
                    tr["r2_comp_vs_C"], tr["cos_comp_vs_C"], tr["norm_comp_vs_C"] = fid(comp, dC)
                    tr["r2_direct_vs_C"], tr["cos_direct_vs_C"], _ = fid(direct, dC)
                    tr["r2_comp_vs_direct"], tr["cos_comp_vs_direct"], tr["norm_comp_vs_direct"] = fid(comp, direct)
                    tr["r2_identity_vs_C"], _, _ = fid(dA, dC)
                    # effect on equiv.py's A-equivalent, C-different pairs, under C
                    ok = same_step & eq[..., A] & dif[..., Cc] & (cand != torch.arange(m, device=dev)[:, None])
                    ii, cc = torch.nonzero(ok, as_tuple=True)
                    if len(ii) >= 30:
                        _, first = np.unique(ii.cpu().numpy(), return_index=True); ii, cc = ii[torch.tensor(first, device=dev)], cc[torch.tensor(first, device=dev)]
                        jj = cand[ii, cc]
                        if len(ii) > 1500:
                            sel = torch.randperm(len(ii), device=dev, generator=gq)[:1500]; ii, jj = ii[sel], jj[sel]
                        X1, X2 = X2t[ii], X2t[jj]; eA, eC = X2[:, A] - X1[:, A], X2[:, Cc] - X1[:, Cc]
                        dec2 = Lt[jj, Cc].argmax(-1); differ = Lt[ii, Cc].argmax(-1) != dec2
                        tok1, p1 = r["tok"][sub["ep"][ii]], sub["st"][ii] + 1
                        vecs = {"direct_C": eC, "map_A_to_C": M[(A, Cc)](eA), "composed_A_B_C": M[(B, Cc)](M[(A, B)](eA)), "untransformed_A": eA, "random": rand_like(eC)}
                        for k, v in vecs.items(): tr["sw_" + k] = switch(X1, tok1, p1, Cc, v, dec2, differ)
                        tr["pairs"] = int(len(ii))
                    for k, v in tr.items(): add("c_" + k, v)
                    row["triples"][f"{A}->{B}->{Cc}"] = tr
        # ---- round trips over ordered pairs, and three-cycles over ordered triples
        for A in range(K):
            dA = D[:, A]
            decA1, decA2 = Lt[r1, A].argmax(-1), Lt[r2_, A].argmax(-1); differA = decA1 != decA2
            base = {}
            base["sw_direct_A"] = switch(X2t[r1], tokr, pr, A, dA, decA2, differA)
            base["sw_random"] = switch(X2t[r1], tokr, pr, A, rand_like(dA), decA2, differA)
            base["sw_none"] = float((decA1 == decA2)[differA].double().mean())
            for B in range(K):
                if B == A: continue
                rt = M[(B, A)](M[(A, B)](dA)); pr_ = {}
                pr_["r2_rt"], pr_["cos_rt"], pr_["norm_rt"] = fid(rt, dA)
                pr_["sw_rt"] = switch(X2t[r1], tokr, pr, A, rt, decA2, differA)
                pr_["sw_one_hop_B_to_A"] = switch(X2t[r1], tokr, pr, A, M[(B, A)](D[:, B]), decA2, differA)
                pr_.update(base); pr_["differ"] = int(differA.sum())
                for k, v in pr_.items(): add("p_" + k, v)
                row["pairs"][f"{A}->{B}->{A}"] = pr_
                for Cc in range(K):
                    if Cc in (A, B): continue
                    cyc = M[(Cc, A)](M[(B, Cc)](M[(A, B)](dA)))
                    r2c, cosc, nc = fid(cyc, dA); add("y_r2", r2c); add("y_cos", cosc); add("y_norm", nc)
                    add("y_sw", switch(X2t[r1], tokr, pr, A, cyc, decA2, differA))
                    row["triples"][f"{A}->{B}->{Cc}"]["cycle_r2"] = r2c; row["triples"][f"{A}->{B}->{Cc}"]["cycle_sw"] = agg["y_sw"][-1]
        row["mean"] = {k: float(np.mean(v)) for k, v in agg.items() if v}
        res[f"seed{s}"] = row
        print(f"seed{s}: " + " ".join(f"{k} {v:.3f}" for k, v in row["mean"].items()), flush=True)
        (HERE / "equiv3.json").write_text(json.dumps(res))
    report()


def report():
    res = json.loads((HERE / "equiv3.json").read_text()); runs = list(res.values())
    mm = lambda k, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([r_["mean"][k] for r_ in runs])
    # per-triple spread of the composition's agreement with the direct map, pooled over models
    r2s = [v["r2_comp_vs_direct"] for r_ in runs for v in r_["triples"].values()]
    sws = [v["sw_composed_A_B_C"] - v["sw_map_A_to_C"] for r_ in runs for v in r_["triples"].values() if "sw_composed_A_B_C" in v]
    rts = [v["r2_rt"] for r_ in runs for v in r_["pairs"].values()]
    L = ["# Do the pairwise maps between goals form one consistent system?", "",
         f"Generated by `equiv3.py` (exploratory). The 12 linear maps M(A→B) of `equiv2.py` (ridge on differences at the input of block 2, {N_TRAIN_PAIRS} training pairs each), composed and inverted, evaluated on {N_EVAL_PAIRS} random same-step pairs of test decisions (posteriors ≥ 0.5 apart) and, for the effect of a composition, on `equiv.py`'s A-equivalent, C-different test pairs ({mm('c_pairs', 0)} per triple). "
         "Means over the 24 ordered triples / 12 ordered pairs, medians (min–max) over six models.", "",
         "## Composition: M(B→C) M(A→B) Δ_A against M(A→C) Δ_A", "",
         "| prediction of Δ_C from Δ_A | held-out R² for Δ_C | cosine with Δ_C | R² against the direct map's prediction | cosine with it | norm / direct's |", "|---|---|---|---|---|---|",
         f"| direct map M(A→C) | {mm('c_r2_direct_vs_C')} | {mm('c_cos_direct_vs_C')} | 1 | 1 | 1 |",
         f"| **composed M(B→C) M(A→B)** | **{mm('c_r2_comp_vs_C')}** | {mm('c_cos_comp_vs_C')} | **{mm('c_r2_comp_vs_direct')}** | {mm('c_cos_comp_vs_direct')} | {mm('c_norm_comp_vs_direct')} |",
         f"| untransformed Δ_A | {mm('c_r2_identity_vs_C')} | | | | |", "",
         f"Over all {len(r2s)} (model, triple) cases the composed prediction's R² against the direct map's ranges {min(r2s):.2f}–{max(r2s):.2f} (lowest decile {np.quantile(r2s, 0.1):.2f}).", "",
         "**Effect under C** on the A-equivalent, C-different pairs (the vector added to x1 under C; share of pairs whose C-decisions differ that switch to b2's):", "",
         "| vector added under C | C's decision to b2's |", "|---|---|",
         f"| direct C-written difference Δ_C | {mm('c_sw_direct_C')} |",
         f"| direct map M(A→C) Δ_A | {mm('c_sw_map_A_to_C')} |",
         f"| **composed M(B→C) M(A→B) Δ_A** | **{mm('c_sw_composed_A_B_C')}** |",
         f"| untransformed Δ_A | {mm('c_sw_untransformed_A')} |",
         f"| random, Δ_C's norm | {mm('c_sw_random')} |", "",
         f"Composed minus direct map, per (model, triple): mean {np.mean(sws):+.3f}, range {min(sws):+.2f} to {max(sws):+.2f}.", "",
         "## Round trip: M(B→A) M(A→B) Δ_A against Δ_A", "",
         "| vector | held-out R² for Δ_A | cosine with Δ_A | norm / Δ_A's | A's decision to b2's (pairs whose A-decisions differ) |", "|---|---|---|---|---|",
         f"| Δ_A itself | 1 | 1 | 1 | {mm('p_sw_direct_A')} |",
         f"| **round trip M(B→A) M(A→B) Δ_A** | **{mm('p_r2_rt')}** | {mm('p_cos_rt')} | {mm('p_norm_rt')} | **{mm('p_sw_rt')}** |",
         f"| one hop from the B-written difference, M(B→A) Δ_B | | | | {mm('p_sw_one_hop_B_to_A')} |",
         f"| three-cycle M(C→A) M(B→C) M(A→B) Δ_A | {mm('y_r2')} | {mm('y_cos')} | {mm('y_norm')} | {mm('y_sw')} |",
         f"| random, Δ_A's norm | | | | {mm('p_sw_random')} |",
         f"| no edit | | | | {mm('p_sw_none')} |", "",
         f"Round-trip R² over all {len(rts)} (model, pair) cases: {min(rts):.2f}–{max(rts):.2f}. Pairs whose A-decisions differ: {mm('p_differ', 0)} of {N_EVAL_PAIRS}."]
    (HERE / "equiv3.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
