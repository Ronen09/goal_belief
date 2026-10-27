"""tables.md from subspace_<kind>.json (part 0) and channel_<kind>.json (parts 1–3): the from-scratch models' goal
subspaces, the upstream swap measured downstream, the natural history difference by site with the sweep over the
rank, the known / uncertain split, the overlaps between the subspaces, and the scorecard.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_channel .venv/bin/python studies/8_language_models/bump_channel/tables.py [--quick]
"""

from __future__ import annotations

import argparse, json
from pathlib import Path

import common as CH

KINDS = ("llm", "tfm", "gru"); NAMES = dict(llm="fine-tuned 0.5B", tfm="transformer", gru="GRU")
VARS = (("goal", "goal subspace"), ("comp", "complement"), ("bottom", "bottom-k"), ("random", "random-k (3 draws)"), ("readout", "readout"), ("history", "history"))
SITE_NAME = dict(llm=lambda s: f"layer {s}", tfm=lambda s: "final residual" if s == 4 else f"block {s} entrance", gru=lambda s: {0: "h_t", -1: "h_t−1", -2: "h_t−2"}[s])


def f(x, d=2):
    return "–" if x is None or (isinstance(x, float) and x != x) else f"{x:.{d}f}"


def ci(c, d=2):
    return "" if not c or c[0] != c[0] else f" ({f(c[0], d)}–{f(c[1], d)})"


def ratio(r):
    """through-dict -> 'ratio (lo–hi)' or the undefined marker."""
    if r is None: return "–"
    return "undef." if r.get("ratio") is None else f"{f(r['ratio'])}{ci(r.get('ratio_ci'))}"


def rand3(r):
    rs = [r[f"random{i}"]["ratio"] for i in range(3) if f"random{i}" in r and r[f"random{i}"]["ratio"] is not None]
    return f"{f(min(rs))}–{f(max(rs))}" if rs else "undef."


def part0(sub, kind):
    L = [f"### {NAMES[kind]} (`subspace_{kind}.json`; registered site: {SITE_NAME[kind](sub['registered_site']) if sub['registered_site'] is not None else 'none'})", "",
         "| site | natural goal swap | full map | k_c per pair (pooled subspace) | registered k | pooled rank-k / full | k90 (median pair / pooled) | goal × step offset share |", "|---|---|---|---|---|---|---|---|"]
    for s in sub["sites"]:
        b = sub["by_site"][str(s)]; m = b["summary"]; kc = " ".join("–" if k is None else str(k) for k in m["k_c_pooled"])
        L.append(f"| {SITE_NAME[kind](s)}{' (linear)' if m['linear'] else ''} | {f(m['natural'][0], 3)}{ci(m['natural'][1:], 3)} | {f(m['switch_full'][0], 3)} | {kc} | {m['k_registered'] if m['k_registered'] is not None else '–'} | {f(m['rank_at_k_ratio'][0]) + ci(m['rank_at_k_ratio'][1:]) if m['rank_at_k_ratio'] else '–'} | {f(m['k90_median'], 0)} / {b['pooled']['k90']} | {f(m['offset_share_mean'])} |")
    return L + [""]


def part2(ch, kind):
    L = [f"### {NAMES[kind]} (`channel_{kind}.json`; {ch['n_pairs']} pairs × 4 goals, {ch['n_cells']} history-dependent cells, {ch['n_known']} at known states; history-dependent share by goal {' / '.join(f(x) for x in ch['dep_share_by_goal'])})", "",
         "| site | k | natural swap | " + " | ".join(n for _, n in VARS) + " | floor | ‖δ‖² share: goal / complement |", "|---|---|---|" + "---|" * len(VARS) + "---|---|"]
    for s in ch["sites"]:
        r = ch["part2"][str(s)]; k = ch["k"][str(s)]; lin = ch["linear_site"] == s
        cells = [ratio(r.get("goal")), ratio(r.get("comp")), ratio(r.get("bottom")), rand3(r), ratio(r.get("readout")), ratio(r.get("history"))]
        L.append(f"| {SITE_NAME[kind](s)}{' (linear)' if lin else ''} | {k if k is not None else '–'} | {f(r['natural']['switch'], 3)}{ci(r['natural']['switch_ci'], 3)} | " + " | ".join(cells) + f" | {ratio(r.get('floor'))} | {f(r['goal']['norm_share'], 3) if 'goal' in r else '–'} / {f(r['comp']['norm_share'], 3) if 'comp' in r else '–'} |")
    L += ["", "The sweep over the rank (through ratio of the top-k goal directions):", "", "| site | " + " | ".join(f"k = {k}" for k in ch["ks"]) + " |", "|---|" + "---|" * len(ch["ks"])]
    for s in ch["sites"]:
        r = ch["part2"][str(s)]; L.append(f"| {SITE_NAME[kind](s)} | " + " | ".join(f(r[f'k{k}']['ratio']) if r[f'k{k}']['ratio'] is not None else "undef." for k in ch["ks"]) + " |")
    return L + [""]


def part1(ch, kind):
    L = [f"### {NAMES[kind]}", "", "| upstream → downstream | upstream swap | full change reproduces | " + " | ".join(n for _, n in VARS) + " | ‖δ‖² share: goal / complement |", "|---|---|---|" + "---|" * len(VARS) + "---|"]
    for key, r in ch["part1"].items():
        s_in, s_out = key.split("->"); s_in, s_out = int(s_in), int(s_out)
        cells = [ratio(r.get("goal")), ratio(r.get("comp")), ratio(r.get("bottom")), rand3(r), ratio(r.get("readout")), ratio(r.get("history"))]
        L.append(f"| {SITE_NAME[kind](s_in)} → {SITE_NAME[kind](s_out)} | {f(r['switch_in']['switch'], 3)}{ci(r['switch_in']['switch_ci'], 3)} | {ratio(r['full_vs_upstream'])} | " + " | ".join(cells) + f" | {f(r['goal']['norm_share'], 3) if 'goal' in r else '–'} / {f(r['comp']['norm_share'], 3) if 'comp' in r else '–'} |")
    return L + [""]


def part3(ch, kind):
    L = [f"### {NAMES[kind]}", "", "| part | site | goal subspace: known | uncertain | difference | complement: known | uncertain | cells known / uncertain |", "|---|---|---|---|---|---|---|---|"]
    for part, key_name in (("2", "part2"), ("1", "part1")):
        for key, r in ch[key_name].items():
            if "goal" not in r: continue
            g, c = r["goal"]["known"], r["comp"]["known"]; site = SITE_NAME[kind](int(key)) if part == "2" else " → ".join(SITE_NAME[kind](int(x)) for x in key.split("->"))
            L.append(f"| {part} | {site} | {ratio(g['known'])} | {ratio(g['uncertain'])} | {ci(g['diff_ci']).strip() or '–'} | {ratio(c['known'])} | {ratio(c['uncertain'])} | {g['known']['n']} / {g['uncertain']['n']} |")
    return L + [""]


def overlaps(ch, kind):
    L = [f"### {NAMES[kind]}", "", "| site | k | goal / readout | goal / history | history / readout | goal / random |", "|---|---|---|---|---|---|"]
    for s in ch["sites"]:
        o = ch["overlap"][str(s)]
        if o is None: L.append(f"| {SITE_NAME[kind](s)} | – | – | – | – | – |"); continue
        L.append(f"| {SITE_NAME[kind](s)} | {o['k']} | {f(o['goal_readout'], 3)} | {f(o['goal_history'], 3)} | {f(o['history_readout'], 3)} | {f(o['random'], 3)} |")
    return L + [""]


def scorecard(chs):
    L = ["| | criterion | " + " | ".join(NAMES[k] for k in chs) + " |", "|---|---|" + "---|" * len(chs)]
    def cell(reg, c):
        if reg is None: return "no registered site"
        v = reg.get(c)
        if v is None: return "–"
        tag = " (vacuous: k ≥ D/2)" if reg.get("degenerate") else ""
        if c in ("THRU_D", "THRU_U"): return f"{f(v['value'])}{ci(v['ci'])}: {'yes' if v['holds'] else 'no'}{tag}"
        if c in ("COMP", "CTRL"): return f"{', '.join(f(x) for x in v['values'])}: {'yes' if v['holds'] else 'no'}{tag}"
        if c == "DISTINCT": return f"{f(v['value'])}: {'yes' if v['holds'] else 'no'}"
        if c == "KNOWN": return f"{f(v['known'])} vs {f(v['uncertain'])}, diff{ci(v['diff_ci']) or ' –'}: {'yes' if v['holds'] else 'no'}"
    rows = (("THRU-D", "THRU_D", "the natural history difference through the goal subspace ≥ 0.65 × the natural swap"), ("THRU-U", "THRU_U", "the upstream swap's induced change through the goal subspace ≥ 0.65 × the upstream swap (fine-tuned model only)"),
            ("COMP", "COMP", "the complement ≤ 0.35 in both parts"), ("CTRL", "CTRL", "bottom-k and every random-k ≤ 0.20 in both parts"), ("DISTINCT", "DISTINCT", "goal / readout overlap ≤ 0.50"), ("KNOWN", "KNOWN", "the part 2 through ratio higher at known states"))
    for name, key, desc in rows: L.append(f"| **{name}** | {desc} | " + " | ".join(cell(chs[k]["registered"], key) for k in chs) + " |")
    L.append("| registered site, k, upstream | | " + " | ".join(f"{SITE_NAME[k](r['site'])}, k = {r['k']}, upstream {SITE_NAME[k](r['s_in']) if r['s_in'] is not None else '–'}" if r else "–" for k, r in ((k, chs[k]["registered"]) for k in chs)) + " |")
    L.append("| rejected (THRU-D interval below 0.40) | | " + " | ".join(("yes" if r["reject"] else "no") if r else "–" for r in (chs[k]["registered"] for k in chs)) + " |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
    here = CH.HERE; out = Path(a.out) if a.out else (here / "_smoke" if a.quick else here)
    load = lambda name, kind: json.loads((here / "_smoke" / f"{name}_{kind}" / f"{name}_{kind}.json").read_text()) if a.quick else json.loads((here / f"{name}_{kind}.json").read_text())
    subs = {k: load("subspace", k) for k in ("tfm", "gru") if (here / "_smoke" / f"subspace_{k}" / f"subspace_{k}.json" if a.quick else here / f"subspace_{k}.json").exists()}
    chs = {k: load("channel", k) for k in KINDS if (here / "_smoke" / f"channel_{k}" / f"channel_{k}.json" if a.quick else here / f"channel_{k}.json").exists()}
    L = ["# Bump channel: tables", "", "Generated by `tables.py`" + (" (quick)" if a.quick else "") + ". Through ratio = the candidate's switch over the reference's (the natural history swap in part 2, the upstream swap in part 1), on history-dependent cells, 95 % cluster-bootstrap intervals by the receiving history; `undef.` where the reference switches below 0.10.", ""]
    L += ["## 0. The from-scratch models' goal subspaces (part 0)", ""]
    for k, s in subs.items(): L += part0(s, k)
    L += ["## 1. The upstream belief swap measured downstream (part 1)", ""]
    for k, c in chs.items(): L += part1(c, k)
    L += ["## 2. The natural history difference by site (part 2)", ""]
    for k, c in chs.items(): L += part2(c, k)
    L += ["## 3. Known and uncertain states (part 3)", ""]
    for k, c in chs.items(): L += part3(c, k)
    L += ["## 4. Overlaps between the subspaces (mean squared cosine of the principal angles at the registered k)", ""]
    for k, c in chs.items(): L += overlaps(c, k)
    L += ["## Scorecard (each model at its registered site)", ""] + scorecard(chs)
    out.mkdir(parents=True, exist_ok=True); (out / "tables.md").write_text("\n".join(L)); print(f"wrote {out / 'tables.md'} ({len(L)} lines; models {list(chs)}; part 0 {list(subs)})")


if __name__ == "__main__":
    main()
