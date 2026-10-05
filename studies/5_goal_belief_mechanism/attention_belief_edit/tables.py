"""The attention-belief-edit experiment: tables and the registered decision rule from results.json, with the direct-belief-edit experiment's for the paired comparison.

    .venv/bin/python studies/5_goal_belief_mechanism/attention_belief_edit/tables.py            # writes tables.md
"""

from __future__ import annotations

import importlib.util, json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
R32DIR = OUT.parent.parent / "5_goal_belief_mechanism" / "direct_belief_edit"
spec = importlib.util.spec_from_file_location("r32tables", R32DIR / "tables.py")
T32 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T32)
f, row, head, mr, p_gt, fp = T32.f, T32.row, T32.head, T32.mr, T32.p_gt, T32.fp


def main():
    L = T32.main(OUT, "# The attention-belief-edit experiment tables", "the goal token's state after block 0's attention, before its MLP (the direct-belief-edit experiment's tables and rule C1–C7 at this site)")
    runs = json.load(open(OUT / "results.json"))["runs"]
    r32 = json.load(open(R32DIR / "results.json"))["runs"]
    seeds = sorted(runs, key=lambda s: int(s[4:]))
    v = lambda R, *p: np.array([T32._get(R[s], p) for s in seeds], dtype=float)
    a, b = (lambda *p: v(runs, *p)), (lambda *p: v(r32, *p))
    L += ["", "## Against the direct-belief-edit experiment's site (goal token entering block 1), the same models", ""]
    L += head("", "the direct-belief-edit experiment's site", "this site (before block 0's MLP)", "difference (this − direct-belief-edit)")
    pairs = [("shared R² within (goal, length)", ("fit", "shared", "r2_within_goal_length")), ("goal-specific R² within", ("fit", "goal_specific", "r2_within_goal_length"))]
    for lab, p in pairs:
        L.append(row(lab, mr(b(*p)), mr(a(*p)), mr(a(*p) - b(*p))))
    gap = lambda x: x("fit", "goal_specific", "r2_within_goal_length") - x("fit", "shared", "r2_within_goal_length")
    L.append(row("goal-specific − shared R² (within)", mr(gap(b)), mr(gap(a)), mr(gap(a) - gap(b))))
    L.append(row("goal × history interaction share", mr(a("interaction", "r32_site")), mr(a("interaction", "site")), mr(a("interaction", "site") - a("interaction", "r32_site"))))
    D = lambda x, k: x("main", "change", k, "donor_optimal")
    for k, lab in (("none", "donor-optimal: none"), ("whole", "donor-optimal: whole"), ("hybrid", "donor-optimal: hybrid"), ("encoding", "donor-optimal: **shared encoding**"),
                   ("enc_goal", "donor-optimal: goal-specific encoding"), ("pca", "donor-optimal: PCA-13"), ("rotated", "donor-optimal: rotated")):
        L.append(row(lab, mr(D(b, k)), mr(D(a, k)), mr(D(a, k) - D(b, k))))
    S = lambda x, k: x("main", "change", k, "share_of_whole")
    L.append(row("S(shared encoding)", mr(S(b, "encoding"), 2), mr(S(a, "encoding"), 2), mr(S(a, "encoding") - S(b, "encoding"), 2)))
    gd = lambda x, k: x("main", "goal_dependent", k, "right_every_change_goal")
    rat = lambda x: gd(x, "encoding") / gd(x, "whole")
    L.append(row("goal-dependent: shared encoding / whole", mr(rat(b), 2), mr(rat(a), 2), mr(rat(a) - rat(b), 2)))
    cg = lambda x: D(x, "enc_goal") - D(x, "encoding")
    L.append(row("D(enc_goal) − D(shared encoding)", mr(cg(b)), mr(cg(a)), mr(cg(a) - cg(b))))
    L.append(row("preservation harm, shared encoding", mr(b("main", "preserve", "encoding", "harm")), mr(a("main", "preserve", "encoding", "harm")), ""))
    eqr = lambda x: x("equivalent", "change", "encoding", "all_donor_optimal") / x("equivalent", "change", "whole", "all_donor_optimal")
    L.append(row("equivalent: all four, shared / whole", mr(eqr(b), 2), mr(eqr(a), 2), mr(eqr(a) - eqr(b), 2)))
    chk = max(float(np.max(np.abs(D(a, k) - D(b, k)))) for k in ("none", "whole", "hybrid"))
    L += ["", f"Check (none, whole, hybrid are the same runs in both experiments): largest |Δ donor-optimal| {chk:.1e}.", ""]
    pr = lambda *p: a("propagation", *p)
    L += ["## Propagation: the shared edit here, carried through block 0's MLP to the direct-belief-edit experiment's site", ""]
    L += head("induced change against", "cosine", "norm ratio")
    L.append(row("the actual difference z(B,g) − z(A,g)", mr(pr("cos_actual")), mr(pr("norm_over_actual"))))
    L.append(row("the direct-belief-edit experiment's shared edit vector", mr(pr("cos_r32_shared")), "—"))
    L.append(row("the direct-belief-edit experiment's goal-specific edit vector", mr(pr("cos_r32_goal")), mr(pr("norm_over_r32_goal"))))
    L.append(row("(the direct-belief-edit experiment's own shared edit against the actual difference)", mr(b("edit_stats", "encoding", "cosine")), mr(b("edit_stats", "encoding", "norm_ratio"))))
    # registered additions to the rule
    G = np.median(gap(a))
    pU1, pU2 = p_gt(rat(a), rat(b)), p_gt(cg(b), cg(a))
    L += ["", "## Decision rule: additions (C1–C7 above)", ""]
    L += head("", "criterion", "value", "held")
    L.append(row("G shared map fits", "goal-specific − shared R² (within) ≤ 0.03", f(G), "**yes**" if G <= 0.03 else "no"))
    L.append(row("U upstream gain", "goal-dependent ratio higher here; enc_goal − encoding gap smaller here; both p < 0.05",
                 f"{f(np.median(rat(a)), 2)} against {f(np.median(rat(b)), 2)}, p {fp(pU1)}; {f(np.median(cg(a)))} against {f(np.median(cg(b)))}, p {fp(pU2)}",
                 "**yes**" if pU1 < 0.05 and pU2 < 0.05 else "no"))
    pI = p_gt(a("interaction", "r32_site"), a("interaction", "site"))
    L.append(row("(descriptive) interaction smaller here", "", f"{f(np.median(a('interaction', 'site')))} against {f(np.median(a('interaction', 'r32_site')))}; p {fp(pI)}; every model: {bool(np.all(a('interaction', 'site') < a('interaction', 'r32_site')))}", ""))
    (OUT / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
