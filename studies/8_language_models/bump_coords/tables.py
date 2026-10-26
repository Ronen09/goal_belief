"""tables.md from compose.json, rank.json and preserve.json: parts A, B, C by site, the top-k curve at the best site,
and the scorecard of PLAN.md's criteria (a criterion holds if its bootstrap interval excludes the threshold).

    .venv/bin/python studies/8_language_models/bump_coords/tables.py [--out DIR]
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 896)


def f(x, d=3):
    if x is None: return "–"
    if isinstance(x, (list, tuple)): return f"{f(x[0], d)} ({f(x[1], d)}–{f(x[2], d)})" if len(x) == 3 else " / ".join(f(v, d) for v in x)
    if isinstance(x, float) and x != x: return "nan"
    if isinstance(x, (int,)) and not isinstance(x, bool): return str(x)
    return f"{x:.{d}f}"


def scorecard(compose, rank, preserve, site):
    """[dict(criterion, value, interval, threshold, holds)] at `site`; `holds` needs the interval to exclude the threshold."""
    s = str(site); cs, rs, ps = compose["by_site"][s]["summary"], rank["by_site"][s]["summary"], preserve["by_site"][s]["summary"]
    def ge(name, v, thr):
        return dict(criterion=name, value=None if v is None else v[0], interval=None if v is None else v[1:], threshold=f"≥ {thr}", holds=bool(v is not None and v[1] > thr))
    def le(name, v, thr):
        return dict(criterion=name, value=None if v is None else v[0], interval=None if v is None else v[1:], threshold=f"≤ {thr}", holds=bool(v is not None and v[2] < thr))
    rows = [ge("COMP-R", cs.get("r2_composed_vs_direct"), 0.95), ge("COMP-S", cs.get("switch_ratio"), 0.90), ge("ROUND", cs.get("keep_round"), 0.90),
            ge("RANK-C", rs.get("rank32_ratio"), 0.90), ge("RANK-SH", rs.get("pooled_at_kc_ratio"), 0.90), ge("PRES-D", ps.get("drop"), 0.05)]
    r, o = ge("PRES-R", ps.get("restoration"), 0.80), ge("over offset", ps.get("over_offset"), 0.10)
    r["threshold"] = "≥ 0.80 and ≥ 0.10 over the offset control"; r["holds"] = bool(r["holds"] and o["holds"]); r["over_offset"] = o["value"]; rows.append(r)
    rows.append(le("BUMP-SH", ps.get("bump_drop"), 0.05))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); a = ap.parse_args(); out = Path(a.out) if a.out else HERE
    J = lambda n: json.loads((out / n).read_text()); comp, rank, pres = J("compose.json"), J("rank.json"), J("preserve.json")
    sites = [str(s) for s in comp["sites"]]; best = str(comp["best_site"]); L = []
    L += ["# Bump coordinates: tables", "", f"Sites: residual entering layers {', '.join(sites)}; the best site (largest natural goal swap) is **{best}**. "
          f"{comp['n_pairs']} test pairs (A, B); probe pool {pres['n_fit']} fit / {pres['n_test']} test decisions held out by set. Intervals are bootstrap 95 %.", ""]
    # A
    L += ["## A1. Composition: the composed map M(B→C) M(A→B) against the direct map M(A→C), mean over the 24 ordered triples", "",
          "| site | R² composed vs direct | R² direct vs x_C | R² composed vs x_C | switch: direct | composed | composed / direct | natural x_C | random |", "|---|---|---|---|---|---|---|---|---|"]
    for s in sites:
        c = comp["by_site"][s]["summary"]; L.append(f"| {s} | {f(c['r2_composed_vs_direct'])} | {f(c['r2_direct_vs_xC'])} | {f(c['r2_composed_vs_xC'])} | {f(c['switch_direct'])} | {f(c['switch_composed'])} | **{f(c['switch_ratio'])}** | {f(c['switch_natural'])} | {f(c['switch_random'])} |")
    L += ["", "## A2. Round trips M(B→A) M(A→B) x_A and three-cycles back to A (installed under A; share of goal-dependent decisions keeping A's)", "",
          "| site | round trip: R² vs x_A | keeps A's decision | x_A re-installed keeps | x_B keeps | three-cycle: R² vs x_A | keeps |", "|---|---|---|---|---|---|---|"]
    for s in sites:
        c = comp["by_site"][s]["summary"]; L.append(f"| {s} | {f(c['r2_round'])} | **{f(c['keep_round'])}** | {f(c['keep_xA'])} | {f(c['keep_xB'])} | {f(c['r2_cycle'])} | {f(c['keep_cycle'])} |")
    L += ["", f"Per triple at site {best}:", "", "| A → B → C | R² composed vs direct | norm ratio | switch direct | composed | natural | n goal-dependent |", "|---|---|---|---|---|---|---|"]
    for t in comp["by_site"][best]["triples"]: L.append(f"| {t['A']} → {t['B']} → {t['C']} | {f(t['r2_composed_vs_direct'])} | {f(t['norm_ratio'], 2)} | {f(t['switch_direct'])} | {f(t['switch_composed'])} | {f(t['switch_natural'])} | {t['n_dep']} |")
    # B
    L += ["", "## B1. Reconstruction rank: singular directions of the per-step-centred goal difference for 50 / 90 / 99 % of its variance (median over the 12 ordered pairs; pooled over pairs)", "",
          "| site | k50 / k90 / k99 (median pair) | pooled k50 / k90 / k99 | goal × step offset: share of the difference's norm² | switch: full map | natural | k_c median (range; defined) | pooled subspace at k_c / full | rank-32 / full | overlap of pairs' top-k_c subspaces (random) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sites:
        r = rank["by_site"][s]; sm = r["summary"]; med = lambda k: f(sorted(p[k] for p in r["pairs"])[len(r["pairs"]) // 2], 0)
        L.append(f"| {s} | {med('k50')} / {med('k90')} / {med('k99')} | {r['pooled']['k50']} / {r['pooled']['k90']} / {r['pooled']['k99']} | {f(sm['offset_share_mean'])} | {f(sm['switch_full'])} | {f(sum(p['switch_natural'] for p in r['pairs']) / len(r['pairs']))} | {f(sm['k_c_median'], 0)} ({sm['k_c_range']}; {sm['k_c_defined']} of 12) | **{f(sm['pooled_at_kc_ratio'])}** | **{f(sm['rank32_ratio'])}** | {f(sm['overlap_mean'])} ({f(sm['overlap_random_mean'])}) |")
    rb = rank["by_site"][best]; ks = [str(k) for k in rank["ks"]]
    L += ["", f"## B2. Causal rank at site {best}: switch of the rank-k map, mean over the 12 ordered pairs (full map {f(rb['summary']['switch_full'])})", "", "| directions | " + " | ".join(f"k = {k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for v in rb["variants"]:
        vals = [sum(p["switch_by_k"][v][k] for p in rb["pairs"]) / len(rb["pairs"]) for k in ks]; L.append(f"| {v} | " + " | ".join(f(x, 2) for x in vals) + " |")
    L += ["", "| A → B | switch full | natural | k_c (top) | k_c (pooled) | k_c (map SVD) | k50 / k90 / k99 | n goal-dependent |", "|---|---|---|---|---|---|---|---|"]
    for p in rb["pairs"]: L.append(f"| {p['A']} → {p['B']} | {f(p['switch_full'])} | {f(p['switch_natural'])} | {p['k_c']} | {p['k_c_pooled']} | {p['k_c_mapsvd']} | {p['k50']} / {p['k90']} / {p['k99']} | {p['n_dep']} |")
    # C
    L += ["", "## C1. Probes by goal at each site (held-out sets): the soft set probe and the bump probe, with shuffled-label floors", "",
          "| site | goal | set: exact | Jaccard (floor) | bump: R² (floor) |", "|---|---|---|---|---|"]
    for s in sites:
        for g in pres["by_site"][s]["probes_by_goal"]: L.append(f"| {s} | {g['g']} | {f(g['set_exact'])} | {f(g['set_jaccard'])} ({f(g['set_jaccard_floor'])}) | {f(g['bump_r2'])} ({f(g['bump_r2_floor'])}) |")
    L += ["", "## C2. The A-probe under the other goal, by site (mean over the 12 ordered pairs)", "",
          "| site | set: drop j_A − j_B | restoration by the inverse map | over the offset control | restoration by the rank-k_c map | bump: R²_A − R²_B unmapped | causal check: pull-back installed under B switches to A's (natural x_A) | probe rows in the causal subspace (random) |", "|---|---|---|---|---|---|---|---|"]
    for s in sites:
        c = pres["by_site"][s]["summary"]; L.append(f"| {s} | **{f(c['drop'])}** | **{f(c['restoration'])}** | {f(c['over_offset'])} | {f(c['restoration_kc'])} | **{f(c['bump_drop'])}** | {f(c['check_switch_full'])} ({f(c['check_switch_natural'])}) | {f(c['overlap'])} ({f(c['overlap_random'])}) |")
    L += ["", f"Per pair at site {best}:", "", "| A → B | j_A | j_B | j_back | j_off | j_back (rank k_c) | j_BB | drop | restoration | bump R²: A / B / back / off / BB | check full / k_c / natural | overlap |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pres["by_site"][best]["pairs"]:
        L.append(f"| {p['A']} → {p['B']} | {f(p['j_A'])} | {f(p['j_B'])} | {f(p['j_back'])} | {f(p['j_off'])} | {f(p['j_back_kc'])} | {f(p['j_BB'])} | {f(p['drop'])} | {f(p['restoration'])} | {f(p['b_A'], 2)} / {f(p['b_B'], 2)} / {f(p['b_back'], 2)} / {f(p['b_off'], 2)} / {f(p['b_BB'], 2)} | {f(p['check_switch_full'], 2)} / {f(p['check_switch_kc'], 2)} / {f(p['check_switch_natural'], 2)} | {f(p['probe_overlap_kc'], 2)} ({f(p['probe_overlap_random'], 2)}) |")
    # scorecard
    L += ["", f"## Scorecard at site {best}", "", "| criterion | value (95 %) | threshold | holds |", "|---|---|---|---|"]
    for r in scorecard(comp, rank, pres, best):
        extra = f"; over the offset control {f(r['over_offset'])}" if "over_offset" in r else ""
        L.append(f"| {r['criterion']} | {f(r['value'])} ({f(r['interval'][0])}–{f(r['interval'][1])}){extra} | {r['threshold']} | {'**yes**' if r['holds'] else '**no**'} |" if r["value"] is not None else f"| {r['criterion']} | undefined | {r['threshold']} | **no** |")
    L += ["", "Scorecards at the other sites:", "", "| site | " + " | ".join(r["criterion"] for r in scorecard(comp, rank, pres, best)) + " |", "|---|" + "---|" * 8]
    for s in sites:
        if s != best: L.append(f"| {s} | " + " | ".join(("yes" if r["holds"] else "no") + f" ({f(r['value'], 2)})" for r in scorecard(comp, rank, pres, s)) + " |")
    # post hoc: maps refitted on the probe pool
    if (out / "posthoc_pool.json").exists():
        ph = J("posthoc_pool.json")
        L += ["", "## Post hoc C3. The maps refitted on the probe pool's fit half (`posthoc_pool.py`): the A-probe's cells on its test half", "",
              "| site | map R² on the probe pool: refitted | registered (swap-pool) map | identity | set: drop | restoration by the refitted inverse map | over the offset control | bump: R²_A − R²_B unmapped | after the pull-back | causal check: pull-back installed under B switches to A's (natural x_A) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in sites:
            c = ph["by_site"][s]["summary"]
            L.append(f"| {s} | {f(c['r2_map_probe_pool'])} | {f(c['r2_registered_map_probe_pool'])} | {f(c['r2_identity'])} | {f(c['drop'])} | **{f(c['restoration'])}** | {f(c['over_offset'])} | {f(c['bump_drop'])} | **{f(c['bump_drop_back'])}** | {f(c['check_switch_back'])} ({f(c['check_switch_natural'])}) |")
        L += ["", f"Per pair at site {best}:", "", "| A → B | λ | map R² refitted / registered / identity | j_A | j_B | j_back | j_off | j_BB | restoration | bump R²: A / B / back / off / BB | check back / natural |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in ph["by_site"][best]["pairs"]:
            L.append(f"| {p['A']} → {p['B']} | {p['lam']} | {f(p['r2_map_probe_pool'], 2)} / {f(p['r2_registered_map_probe_pool'], 2)} / {f(p['r2_identity'], 2)} | {f(p['j_A'])} | {f(p['j_B'])} | {f(p['j_back'])} | {f(p['j_off'])} | {f(p['j_BB'])} | {f(p['restoration'])} | {f(p['b_A'], 2)} / {f(p['b_B'], 2)} / {f(p['b_back'], 2)} / {f(p['b_off'], 2)} / {f(p['b_BB'], 2)} | {f(p['check_switch_back'], 2)} / {f(p['check_switch_natural'], 2)} |")
    (out / "tables.md").write_text("\n".join(L) + "\n"); print(f"wrote {out / 'tables.md'}")


if __name__ == "__main__":
    main()
