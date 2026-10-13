"""tables.md from results.json (factorial patching). Pooled numbers: for one model, a statistic of a (quantity, patch
type) is the mean over the directions of that type (four per type) of its mean over goal pairs; then the median
(min–max) over models.

    .venv/bin/python studies/7_information_seeking/factorial_patch/tables.py [--results results.json] [--out DIR]
"""

from __future__ import annotations

import argparse, json
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROLES = ("h1A", "h1B", "h2A", "h2B")
RESID = ["dec@resid1", "dec@resid2", "dec@resid3", "dec@resid4", "hist@resid1", "hist@resid2", "hist@resid3", "goaltok@resid1", "goaltok@resid2", "goaltok@resid3"]
SUB = [f"attn {l}" for l in range(4)] + [f"mlp {l}" for l in range(4)]
HEADS = [f"head {l}.{k}" for l in range(4) for k in range(4)]
RANDOM = ["random dec@resid2"] + [f"random attn {l}" for l in range(4)] + [f"random mlp {l}" for l in range(4)]
TYPES = ("same_goal", "same_hist", "both")


def mm(xs, nd=2):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return "—" if not xs else f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})"


def wilcoxon_greater(x, y):
    """Exact one-sided Wilcoxon signed-rank p for median(x - y) > 0 (ties dropped)."""
    d = np.array([a - b for a, b in zip(x, y)]); d = d[d != 0]
    if len(d) == 0:
        return 1.0
    r = np.argsort(np.argsort(np.abs(d))) + 1
    W = float(r[d > 0].sum()); n = len(d)
    return sum(1 for signs in product((0, 1), repeat=n) if float(sum(ri for ri, s_ in zip(r, signs) if s_)) >= W) / 2 ** n


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


nm = lambda v: float(np.nanmean(v))


def pooled(run, st, q, ty, stat):
    vals = [nm(v[stat]) for v in run["sets"][st]["quantities"][q]["patches"].values() if v["type"] == ty and stat in v]
    return nm(vals) if vals else float("nan")


def direction(run, st, q, r, ty):
    for k, v in run["sets"][st]["quantities"][q]["patches"].items():
        if k.startswith(r + "<-") and v["type"] == ty:
            return v
    raise KeyError((q, r, ty))


def iota(run, st, q):
    return nm([nm(v["iota"]) for v in run["sets"][st]["quantities"][q]["additivity"].values()])


def iota_nat(run, st):
    return nm([nm(v["iota"]) for v in run["sets"][st]["natural"]["iota"].values()])


def oi(run, st, q):
    v = run["sets"][st]["natural"]["output_interaction"].get(q)
    return float("nan") if v is None else nm(v)


def where_table(runs, st, names, title):
    L = [f"### {title}", "",
         "| quantity | same goal, other history: to source | same history, other goal: to source | both: to source | both: to q_h (source's history, recipient's goal) | both: to q_g (recipient's history, source's goal) | both: kept | both: logit transfer to source |",
         "|---|---|---|---|---|---|---|---|"]
    for q in names:
        g = lambda ty, stat: mm([pooled(r, st, q, ty, stat) for r in runs])
        L.append(f"| {q} | {g('same_goal', 'to_source')} | {g('same_hist', 'to_source')} | {g('both', 'to_source')} | {g('both', 'to_qh')} | {g('both', 'to_qg')} | {g('both', 'kept')} | {g('both', 'transfer')} |")
    return L + [""]


def unique_table(runs, st, names):
    L = ["### Unique attribution: the both-type patch on quartets whose four natural decisions all differ", "",
         "| quantity | to source | to q_h | to q_g | kept |", "|---|---|---|---|---|"]
    for q in names:
        g = lambda stat: mm([pooled(r, st, q, 'both', stat) for r in runs])
        L.append(f"| {q} | {g('u_source')} | {g('u_qh')} | {g('u_qg')} | {g('u_kept')} |")
    return L + [""]


def additivity_table(runs, st, names):
    L = ["### Additivity: the two-factor patch's logit change against the sum of the one-factor changes", "",
         "ι = Σ‖Δ_both − Δ_h − Δ_g‖² / Σ‖Δ_both‖² over quartets (mean over the four recipients and the goal pairs); cos = mean cosine between the residual and Δ_both; output share = the variance share of the goal × history interaction in the quantity's own natural value at the decision token across the four cells.", "",
         "| quantity | ι | cos | ‖residual‖ / ‖Δ_both‖ | output share |", "|---|---|---|---|---|"]
    L.append(f"| **natural logits (ι_nat)** | {mm([iota_nat(r, st) for r in runs])} | {mm([nm([nm(v['cos']) for v in r['sets'][st]['natural']['iota'].values()]) for r in runs])} | {mm([nm([nm(v['rel_norm']) for v in r['sets'][st]['natural']['iota'].values()]) for r in runs])} | {mm([oi(r, st, 'logits') for r in runs])} |")
    for q in names:
        A = lambda stat: mm([nm([nm(v[stat]) for v in r["sets"][st]["quantities"][q]["additivity"].values()]) for r in runs])
        L.append(f"| {q} | {A('iota')} | {A('cos')} | {A('rel_norm')} | {mm([oi(r, st, q) for r in runs])} |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results.json")); ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    res = json.load(open(a.results)); runs = list(res["runs"].values()); n_models = len(runs)
    L = ["# Factorial patching: tables", "", f"Generated by `tables.py` from `{Path(a.results).name}`; {n_models} models, medians (min–max). "
         "Quartets (h1, h2) × (A, B) of same-step decisions after the first step; roles h1A, h1B, h2A, h2B. A patch writes one quantity of the source cell into the recipient cell's run, all 12 directions. "
         "'To source' is the share of the recipient's decisions switched to the source's among quartets where the two differ; q_h is the cell with the source's history and the recipient's goal, q_g the recipient's history and the source's goal. "
         "History tokens are positions 0 and 2 … p − 1 (position 0 is the first observation); the goal token (position 1) attends to position 0, so a same-goal goal-token patch is not an identity.", ""]
    L += ["## The quartet sets", "", "| set | quartets per model | decisions depend on the history (goal A / B) | on the goal (h1 / h2) | all four decisions distinct | ι_nat (the natural logits' non-additivity) | interaction share of the natural logits |", "|---|---|---|---|---|---|---|"]
    for st, lab in (("random", "random quartets (4 096 pairs × 6 goal pairs)"), ("equiv", "A-equivalent, B-different pairs")):
        S = lambda k: mm([nm(r["sets"][st][k]) for r in runs])
        L.append(f"| {lab} | {mm([r['sets'][st]['n'] for r in runs], 0)} | {S('hist_dependent_A')} / {S('hist_dependent_B')} | {S('goal_dependent_h1')} / {S('goal_dependent_h2')} | {S('four_distinct')} | {mm([iota_nat(r, st) for r in runs])} | {mm([oi(r, st, 'logits') for r in runs])} |")
    lc = [r["logit_check"] for r in runs if "logit_check" in r]
    if lc: L.append(f"\nLogit check (full-length against truncated sequences), largest absolute difference: {mm([c['max_abs_diff'] for c in lc], 3)} on logits up to {mm([c['logit_scale'] for c in lc], 1)}.\n")
    for st, lab in (("random", "Random quartets (primary)"), ("equiv", "A-equivalent quartets (secondary)")):
        L += [f"## {lab}", ""]
        L += where_table(runs, st, RESID + SUB, "Where the decision goes: residual sets and sublayer outputs")
        L += where_table(runs, st, HEADS, "Where the decision goes: single heads")
        L += where_table(runs, st, RANDOM, "Random floors (a random vector of the natural change's norm added at the recipient)")
        L += [f"Informative quartets per direction (source's decision ≠ recipient's), both type, dec@resid1: {mm([nm([sum(v['n_inf']) for v in r['sets'][st]['quantities']['dec@resid1']['patches'].values() if v['type'] == 'both']) for r in runs], 0)}; four-distinct quartets: {mm([nm([sum(v['n4']) for v in r['sets'][st]['quantities']['dec@resid1']['patches'].values() if v['type'] == 'both']) for r in runs], 0)}.", ""]
        L += unique_table(runs, st, RESID + SUB)
        L += additivity_table(runs, st, RESID + SUB + HEADS)
        # symmetry across recipients
        L += ["### Symmetry across recipients", "", "To source per recipient role; the spread is max − min over the four roles within a model (median and largest over models).", "",
              "| quantity, type | h1A | h1B | h2A | h2B | spread |", "|---|---|---|---|---|---|"]
        for q, ty in (("dec@resid1", "same_goal"), ("dec@resid2", "same_goal"), ("hist@resid1", "same_goal"), ("dec@resid1", "same_hist"), ("dec@resid1", "both"), ("attn 0", "same_hist")):
            per = [[nm(direction(r, st, q, role, ty)["to_source"]) for role in ROLES] for r in runs]
            spread = [max(p_) - min(p_) for p_ in per]
            L.append(f"| {q}, {ty} | " + " | ".join(mm([p_[i] for p_ in per]) for i in range(4)) + f" | {np.median(spread):.3f} (max {max(spread):.3f}) |")
        if st == "equiv":
            L += ["", "Informative quartets per recipient (same-goal type, dec@resid1), A recipients against B recipients: " +
                  ", ".join(f"{role} {mm([sum(direction(r, st, 'dec@resid1', role, 'same_goal')['n_inf']) for r in runs], 0)}" for role in ROLES) + "."]
        L.append("")
    # scorecard on the random quartets
    st = "random"
    whole = [pooled(r, st, "dec@resid1", "both", "to_source") for r in runs]
    bl1 = [pooled(r, st, "hist@resid1", "same_goal", "to_source") for r in runs]
    gt = [pooled(r, st, "goaltok@resid1", "same_hist", "to_source") for r in runs]
    single = {q: sum(pooled(r, st, q, "same_hist", "to_source") >= 0.5 for r in runs) for q in SUB}
    add_p = {q: wilcoxon_greater([iota_nat(r, st) for r in runs], [iota(r, st, q) for r in runs]) for q in SUB}
    rho = [spearman([iota(r, st, q) for q in SUB], [oi(r, st, q) for q in SUB]) for r in runs]
    sym = [max(p_) - min(p_) for p_ in [[nm(direction(r, st, "dec@resid2", role, "same_goal")["to_source"]) for role in ROLES] for r in runs]]
    L += ["## Scorecard (random quartets)", "", "| | criterion | numbers | held |", "|---|---|---|---|",
          f"| WHOLE | dec@resid1, both: to source ≥ 0.8 | {mm(whole)} | {'yes' if np.median(whole) >= 0.8 else 'no'} |",
          f"| BELIEF-L1 | hist@resid1, same goal: to source ≥ 0.5 | {mm(bl1)} | {'yes' if np.median(bl1) >= 0.5 else 'no'} |",
          f"| GOAL-TOK | goaltok@resid1, same history: to source ≥ 0.5 | {mm(gt)} | {'yes' if np.median(gt) >= 0.5 else 'no'} |",
          f"| NO-SINGLE | no sublayer output, same history: to source ≥ 0.5 in ≥ 5 of {n_models} models | models ≥ 0.5: " + ", ".join(f"{q} {c}" for q, c in single.items()) + f" | {'yes' if max(single.values()) < 5 else 'no'} |",
          f"| ADD | every sublayer: ι < ι_nat (one-sided Wilcoxon, p < 0.05) | p: " + ", ".join(f"{q} {p:.3f}" for q, p in add_p.items()) + f" | {'yes' if max(add_p.values()) < 0.05 else 'no'} |",
          f"| RANK | Spearman(ι, output share) over the 8 sublayers ≥ 0.5 in ≥ 5 of {n_models} models | ρ per model: " + ", ".join(f"{x:.2f}" for x in rho) + f" | {'yes' if sum(x >= 0.5 for x in rho) >= 5 else 'no'} |",
          f"| SYM | dec@resid2, same goal: the four recipients' to-source within 0.05 in every model | spread per model: " + ", ".join(f"{x:.3f}" for x in sym) + f" | {'yes' if max(sym) <= 0.05 else 'no'} |", ""]
    # expectation helpers
    L += ["## Numbers for the expectations", ""]
    for q in ("dec@resid2", "dec@resid3", "dec@resid4"):
        L.append(f"* E1: {q}, both, to source {mm([pooled(r, st, q, 'both', 'to_source') for r in runs])}.")
    L.append("* E2: hist@resid 1 / 2 / 3, same goal, to source " + " / ".join(mm([pooled(r, st, q, 'same_goal', 'to_source') for r in runs]) for q in ("hist@resid1", "hist@resid2", "hist@resid3")) + ".")
    L.append("* E3: goaltok@resid 1 / 2 / 3, same history, to source " + " / ".join(mm([pooled(r, st, q, 'same_hist', 'to_source') for r in runs]) for q in ("goaltok@resid1", "goaltok@resid2", "goaltok@resid3")) + ".")
    L.append(f"* E4: dec@resid1, both: to q_h {mm([pooled(r, st, 'dec@resid1', 'both', 'to_qh') for r in runs])}, to q_g {mm([pooled(r, st, 'dec@resid1', 'both', 'to_qg') for r in runs])}; hist@resid1 from the both source: to q_h {mm([pooled(r, st, 'hist@resid1', 'both', 'to_qh') for r in runs])}, to q_g {mm([pooled(r, st, 'hist@resid1', 'both', 'to_qg') for r in runs])}, to source {mm([pooled(r, st, 'hist@resid1', 'both', 'to_source') for r in runs])}.")
    best = [max(SUB, key=lambda q: pooled(r, st, q, 'same_hist', 'to_source')) for r in runs]
    L.append(f"* E5: the largest single carrier of the goal (same history, to source) per model: {', '.join(best)}; attn 0 {mm([pooled(r, st, 'attn 0', 'same_hist', 'to_source') for r in runs])}.")
    order = [sorted(SUB, key=lambda q: -iota(r, st, q))[:3] for r in runs]
    L.append("* E6: the three least additive sublayers per model: " + "; ".join(", ".join(o) for o in order) + ". ι: " + ", ".join(f"{q} {mm([iota(r, st, q) for r in runs])}" for q in SUB) + ".")
    eqA = [nm([nm(direction(r, 'equiv', 'dec@resid1', role, 'same_goal')['to_source']) for role in ('h1A', 'h2A')]) for r in runs]
    eqB = [nm([nm(direction(r, 'equiv', 'dec@resid1', role, 'same_goal')['to_source']) for role in ('h1B', 'h2B')]) for r in runs]
    L.append(f"* E7: A-equivalent quartets, same goal, dec@resid1, to source into the A recipients {mm(eqA)} against the B recipients {mm(eqB)}; the share of A-equivalent quartets whose decisions under A differ {mm([1 - nm(r['sets']['equiv']['hist_dependent_A']) for r in runs])} (same), under B {mm([nm(r['sets']['equiv']['hist_dependent_B']) for r in runs])} (differ).")
    L.append(f"* E8: four-distinct share {mm([nm(r['sets'][st]['four_distinct']) for r in runs])}; dec@resid1, both: unique to source {mm([pooled(r, st, 'dec@resid1', 'both', 'u_source') for r in runs])} against pooled {mm(whole)}.")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
