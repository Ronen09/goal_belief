"""TASK12 (round 13): tables, prediction checks and figures from results12/results12.json.

    .venv/bin/python scripts/task12_tables.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("results12")
OBJ = ("goal", "act_soft", "next_obs", "act_hard")
HIDDEN = (2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64)
R12_GOAL_ONLY = {6: 0.950, 12: None}          # round 12 part 2, GRU channel goal PROBE-E toward genuine-B (t = 12 filled below)


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


class D:
    def __init__(self, path):
        d = json.load(open(path)); self.runs = d["runs"]; self.base = d["baselines"]

    def sel(self, arch="gru", obj="goal", hidden=64, kind="channel", conv=True):
        return [r for r in self.runs if r["arch"] == arch and r["objective"] == obj and r["hidden"] == hidden
                and r["kind"] == kind and (not conv or obj == "init" or r.get("converged"))]

    def rec(self, obj, split, block, arch="gru", site=None, hidden=64, kind="channel", conv=True):
        rs = self.sel(arch, obj, hidden, kind, conv)
        if not rs:
            return float("nan")
        return float(np.mean([(r["recover"][site] if site else r["recover"])[split][block] for r in rs]))

    def I(self, obj, t, key, part="future", conv=True):
        rs = self.sel("gru", obj, conv=conv)
        if not rs:
            return float("nan")
        S = np.array([r["I"][f"t{t}"][key] for r in rs])
        return float((S[:, 0] if part == "k0" else S[:, 1:].mean(1)).mean())

    def E(self, obj, m, hidden=64, conv=True):
        rs = self.sel("gru", obj, hidden, conv=conv)
        if not rs or m not in rs[0]["E"]:
            return None
        ref = "Xr" if m in ("Ex", "Xr") else "R"
        mod = np.array([np.mean(r["E"][m]["model"][1:]) for r in rs]); bay = np.array([np.mean(r["E"][m]["bayes"][1:]) for r in rs])
        rnd = np.array([np.mean(r["E"][ref]["model"][1:]) for r in rs])
        return {"model": float(mod.mean()), "bayes": float(bay.mean()), "over_random": float((mod / rnd).mean()),
                "over_bayes": float((mod / np.clip(bay, 1e-12, None)).mean()),
                "spearman": float(np.nanmean([r["E"][m]["spearman_pairs"] for r in rs]))}

    def kl(self, kind, hidden):
        rs = self.sel("gru", "goal", hidden, kind, conv=False)
        return float(np.mean([r["behaviour"]["kl"] for r in rs])) if rs else float("nan")


def goal_only_reference():
    """Round 12 part 2: GRU channel goal, 3-coordinate PROBE-E, gap closed toward genuine-B (future)."""
    p = Path("results11/causal/causal.json")
    if not p.exists():
        return {}
    runs = [r for r in json.load(open(p))["runs"] if r["arch"] == "gru" and r["kind"] == "channel" and r["objective"] == "goal"]
    return {t: float(np.mean([np.mean(r["A"][f"t{t}_h"]["probe_e:F_model_S"][1:]) for r in runs])) for t in (6, 12)}


def tables(d: D) -> str:
    L = ["# TASK12 tables (round 13: the full filter state)", "",
         "Channel environment, K = 4, GRU unless stated, seed means over converged models (`act_hard` models did not converge "
         "in round 12 and are shown in brackets). Full state = goal block y (3 log-odds) + channel block l (logit P(on | g), 4).", ""]
    L += ["## 1. Recoverability of the full state", "",
          "| model | goal IID | channel IID | channel TIME | channel EXT | goal TIME | goal EXT |", "|---|---|---|---|---|---|---|"]
    for o in OBJ + ("init",):
        g = lambda s, b: d.rec(o, s, b, conv=o != "act_hard")        # noqa: E731
        row = [g("IID", "goal"), g("IID", "channel"), g("TIME", "channel"), g("EXT", "channel"), g("TIME", "goal"), g("EXT", "goal")]
        cells = [f(x) for x in row]
        if o == "act_hard":
            rs = d.sel("gru", o, conv=False)
            cells = [f"[{f(np.mean([r['recover'][s][b] for r in rs]))}]" for s, b in
                     (("IID", "goal"), ("IID", "channel"), ("TIME", "channel"), ("EXT", "channel"), ("TIME", "goal"), ("EXT", "goal"))]
        L.append(f"| GRU {o} | " + " | ".join(cells) + " |")
    b = d.base
    L += ["", f"Baselines: channel block from the true goal block {f(b['channel_from_true_y'])}; from the true marginals (y, logit P(on)) "
          f"{f(b['channel_from_true_marginals'])}; full state from the token counts: goal {f(b['full_from_counts']['goal'])}, "
          f"channel {f(b['full_from_counts']['channel'])}.", "",
          "| transformer | site | goal IID | channel IID | channel TIME | channel EXT |", "|---|---|---|---|---|---|"]
    for o in ("goal", "act_soft", "next_obs"):
        for s in ("res1", "res2", "u"):
            g = lambda sp, bl: d.rec(o, sp, bl, arch="tfm", site=s)   # noqa: E731
            L.append(f"| {o} | {s} | {f(g('IID', 'goal'))} | {f(g('IID', 'channel'))} | {f(g('TIME', 'channel'))} | {f(g('EXT', 'channel'))} |")
    ref = goal_only_reference()
    L += ["", "## 2. Matched interventions in both coordinates (PROBE-E on the 7 full-state coordinates)", "",
          "F = fraction of the gap closed toward the intervention's own Bayes future (k ≥ 1). 'vs genuine-B' = toward the model's own "
          f"run on B's history. Round 12's goal-only probe closed {f(ref.get(6))} (t = 6) and {f(ref.get(12))} (t = 12) toward genuine-B.", "",
          "| objective | t | G-only | R-only | both | swap | random | both vs genuine-B | swap vs genuine-B | gap G-only | gap R-only | gap both |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in OBJ:
        conv = o != "act_hard"
        for t in (6, 12):
            g = lambda k: d.I(o, t, k, conv=conv)                     # noqa: E731
            cells = [g("g_only:F_bayes"), g("r_only:F_bayes"), g("both:F_bayes"), g("swap:F_bayes"), g("rand:F_bayes"),
                     g("both:F_model_S"), g("swap:F_model_S")]
            gaps = [g("g_only:gap_size"), g("r_only:gap_size"), g("both:gap_size")]
            L.append(f"| {o if conv else '[' + o + ']'} | {t} | " + " | ".join(f(x) for x in cells) + " | " + " | ".join(f(x, 4) for x in gaps) + " |")
    L += ["", "## 3. The equivalence hierarchy", "",
          "Model divergence = JS between the model's outputs after the two histories (k ≥ 1, 8 continuations). Random = same-length random "
          "pairs (cross-length random pairs for Ex).", "",
          "| objective | pair set | model | Bayes | model ÷ Bayes | model ÷ random | Spearman(model, Bayes) |", "|---|---|---|---|---|---|---|"]
    names = {"Eb": "equal b", "Em": "equal marginals (b, P(on))", "Ej": "equal full state", "Ej3": "equal full state, ≥ 3 positions differ beyond neutral relabel",
             "Ex": "equal full state, lengths 8 vs 16"}
    for o in OBJ:
        for m in ("Eb", "Em", "Ej", "Ej3", "Ex"):
            e = d.E(o, m, conv=o != "act_hard")
            if e is None:
                continue
            L.append(f"| {o} | {names[m]} | {f(e['model'], 6)} | {f(e['bayes'], 6)} | {f(e['over_bayes'], 2)} | {f(e['over_random'], 5)} | {f(e['spearman'], 2)} |")
    L += ["", "## 4. Bottleneck (goal objective)", "",
          "| hidden | KL iid | KL channel | goal R² iid | goal R² channel | channel R² channel | Ej model ÷ Bayes | Ex model ÷ Bayes | Ex model ÷ random |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in HIDDEN:
        ej, ex = d.E("goal", "Ej", n, conv=False), d.E("goal", "Ex", n, conv=False)
        L.append(f"| {n} | {f(d.kl('iid', n), 4)} | {f(d.kl('channel', n), 4)} | {f(d.rec('goal', 'IID', 'goal', hidden=n, kind='iid', conv=False))} "
                 f"| {f(d.rec('goal', 'IID', 'goal', hidden=n, conv=False))} | {f(d.rec('goal', 'IID', 'channel', hidden=n, conv=False))} "
                 f"| {f(ej['over_bayes'], 1) if ej else '—'} | {f(ex['over_bayes'], 1) if ex else '—'} | {f(ex['over_random'], 5) if ex else '—'} |")
    return "\n".join(L) + "\n"


def predictions(d: D):
    P = []
    add = lambda n, ok, x: P.append((n, bool(ok), x))                  # noqa: E731
    b = d.base
    v = {o: (d.rec(o, "IID", "channel"), d.rec(o, "IID", "goal")) for o in ("goal", "next_obs")}
    ini = d.rec("init", "IID", "channel")
    add("F1 channel IID ≥ 0.85, goal ≥ 0.98 (goal, next_obs); > marginals baseline; untrained channel ≤ 0.5",
        all(c >= 0.85 and g >= 0.98 and c > b["channel_from_true_marginals"] for c, g in v.values()) and ini <= 0.5,
        "; ".join(f"{o}: channel {f(c)}, goal {f(g)}" for o, (c, g) in v.items()) + f"; marginals baseline {f(b['channel_from_true_marginals'])}; untrained channel {f(ini)}")
    t_, e_ = d.rec("goal", "TIME", "channel"), d.rec("goal", "EXT", "channel")
    add("F2 goal: channel TIME ≥ 0.8, EXT ≥ 0.6", t_ >= 0.8 and e_ >= 0.6, f"TIME {f(t_)}, EXT {f(e_)}")
    v = {(k, t): d.I("goal", t, f"{k}:F_bayes") for k in ("g_only", "r_only", "both", "rand") for t in (6, 12)}
    add("F3 goal: G-only, R-only, both ≥ 0.9 of own Bayes gap; random ≤ 0.1",
        all(v[(k, t)] >= 0.9 for k in ("g_only", "r_only", "both") for t in (6, 12)) and all(v[("rand", t)] <= 0.1 for t in (6, 12)),
        "; ".join(f"{k} t={t} {f(x)}" for (k, t), x in v.items()))
    ref = goal_only_reference()
    v = {t: d.I("goal", t, "both:F_model_S") for t in (6, 12)}
    add("F4 goal: both ≥ 0.97 toward genuine-B and above round 12's goal-only probe",
        all(v[t] >= 0.97 and v[t] > ref.get(t, 1) for t in (6, 12)), "; ".join(f"t={t}: {f(v[t])} vs goal-only {f(ref.get(t))}" for t in (6, 12)))
    eb, em, ej = d.E("goal", "Eb"), d.E("goal", "Em"), d.E("goal", "Ej")
    ok = all(0.5 <= e["over_bayes"] <= 2 and e["spearman"] >= 0.5 for e in (eb, em)) and ej["over_random"] <= 0.002 and ej["over_bayes"] <= 3
    add("F5 goal: Eb, Em model ÷ Bayes in [0.5, 2] (Spearman ≥ 0.5); Ej ≤ 0.002 of random and ≤ 3× Bayes", ok,
        f"Eb {f(eb['over_bayes'], 2)} (ρ {f(eb['spearman'], 2)}); Em {f(em['over_bayes'], 2)} (ρ {f(em['spearman'], 2)}); "
        f"Ej ÷ random {f(ej['over_random'], 5)}, ÷ Bayes {f(ej['over_bayes'], 2)}")
    ex = d.E("goal", "Ex")
    add("F6 goal cross-length equal state: ≤ 0.01 of random, ≤ 3× Bayes (tolerance 0.05, see report)",
        ex["over_random"] <= 0.01 and ex["over_bayes"] <= 3, f"÷ random {f(ex['over_random'], 5)}, ÷ Bayes {f(ex['over_bayes'], 2)}")
    v = {(o, m): d.E(o, m) for o in ("next_obs", "act_soft") for m in ("Ej", "Ex")}
    ok = all((e["over_random"] <= (0.002 if m == "Ej" else 0.01)) and e["over_bayes"] <= 3 for (o, m), e in v.items())
    add("F7 Ej and Ex also for next_obs, act_soft", ok,
        "; ".join(f"{o} {m}: ÷ random {f(e['over_random'], 5)}, ÷ Bayes {f(e['over_bayes'], 2)}" for (o, m), e in v.items()))
    first = {k: next((n for n in HIDDEN if d.kl(k, n) < 0.01), None) for k in ("iid", "channel")}
    add("F8 KL < 0.01 at n ≤ 4 (iid), needs n ≥ 6 (channel)",
        first["iid"] is not None and first["iid"] <= 4 and first["channel"] is not None and first["channel"] >= 6,
        f"smallest n with KL < 0.01: iid {first['iid']}, channel {first['channel']}")
    v = {n: (d.rec("goal", "IID", "goal", hidden=n, conv=False), d.rec("goal", "IID", "channel", hidden=n, conv=False)) for n in (3, 4, 5)}
    add("F9 channel n ∈ {3, 4, 5}: goal R² ≥ 0.9, channel R² ≤ 0.6", all(g >= 0.9 and c <= 0.6 for g, c in v.values()),
        "; ".join(f"n={n}: goal {f(g)}, channel {f(c)}" for n, (g, c) in v.items()))
    r2, u = d.rec("goal", "IID", "channel", arch="tfm", site="res2"), d.rec("goal", "IID", "channel", arch="tfm", site="u")
    add("F10 transformer goal: channel R² at res2 ≥ at u", r2 >= u, f"res2 {f(r2)}, u {f(u)}")
    return P


def main():
    d = D(OUT / "results12.json")
    txt = tables(d); P = predictions(d)
    txt += "\n## 5. Pre-registered predictions (docs/task12_filter_theory.md §5)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {x} |\n" for n, ok, x in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    (OUT / "tables12.md").write_text(txt)
    print(txt[txt.index("## 2."):])
    from goalgeo import plotting12
    plotting12.all_figures(d, OUT)


if __name__ == "__main__":
    main()
