"""TASK11 part 2: tables, prediction checks and figures from rounds/r12_hidden_goal/causal/causal.json.

    .venv/bin/python rounds/r12_hidden_goal/causal_tables.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import plots as P11  # noqa: E402

OUT = Path("rounds/r12_hidden_goal/causal")
OBJ = ("goal", "act_soft", "act_hard", "next_obs")
KINDS = ("iid", "channel")
INTS = ("swap", "probe_e", "probe_d", "probe_e_half", "rand")


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


class C:
    def __init__(self, path):
        self.runs = json.load(open(path))["runs"]

    def sel(self, arch, kind, obj, conv=True):
        return [r for r in self.runs if r["arch"] == arch and r["kind"] == kind and r["objective"] == obj
                and (r["converged"] or not conv)]

    def A(self, arch, kind, obj, key, t, site, part="future", conv=True):
        """Seed mean of an A-series: part 'k0' (offset 0), 'future' (mean over k >= 1) or 'curve'."""
        rs = self.sel(arch, kind, obj, conv)
        if not rs:
            return float("nan") if part != "curve" else None
        S = np.array([r["A"][f"t{t}_{site}"][key] for r in rs], float)
        if part == "curve":
            return S.mean(0)
        return float((S[:, 0] if part == "k0" else S[:, 1:].mean(1)).mean())

    def B(self, arch, kind, obj, mode, conv=True):
        rs = self.sel(arch, kind, obj, conv)
        if not rs:
            return None
        m = np.array([r["B"][mode]["model"] for r in rs]); b = np.array([r["B"][mode]["bayes"] for r in rs])
        rm = np.array([r["B"]["random"]["model"] for r in rs])
        return {"model": m.mean(0), "bayes": b.mean(0), "random": rm.mean(0),
                "ratio_k": (m / rm).mean(0), "ratio_mean": float((m.mean(1) / rm.mean(1)).mean()),
                "model_over_bayes": float((m[:, 1:].mean(1) / np.clip(b[:, 1:].mean(1), 1e-12, None)).mean()),
                "spearman": float(np.mean([r["B"][mode]["spearman_pairs"] for r in rs]))}

    def flip(self, arch, kind, obj, conv=True):
        rs = self.sel(arch, kind, obj, conv)
        return float(np.mean([r["C"]["flip"]["flip_rate"] for r in rs])) if rs else float("nan")

    def lad(self, arch, kind, obj, site, key, conv=True):
        rs = self.sel(arch, kind, obj, conv)
        return float(np.mean([r["C"]["ladder"][site][key] for r in rs])) if rs else float("nan")


def site_of(arch):
    return ("h",) if arch == "gru" else ("res1", "res2")


def tables(D: C) -> str:
    L = ["# TASK11 part 2 tables: causal tests of the decoded posterior", "",
         "Seed means over converged models (channel `act_hard` models, which did not converge, are shown in "
         "brackets from all seeds). 'fut' = mean over future offsets k ≥ 1; 'k0' = the output at the "
         "intervention position. F = fraction of the gap closed (1 − E[div(intervened, ref)] / E[div(unintervened, ref)]).", ""]
    L += ["## A. Posterior transplant", "",
          "Reference 'model B' = the model's own run on B's history followed by A's continuation. "
          "Bayes S = genuine-B Bayes target; Bayes T = transplant target (goal belief moved, channel belief given the goal kept).", ""]
    for t in (6, 12):
        L += [f"### t = {t}", "",
              "| arch | site | env | objective | intervention | F model B, k0 | F model B, fut | F Bayes T, fut | KL to Bayes S, fut | KL to Bayes T, fut | F Bayes T½, fut |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arch in ("gru", "tfm"):
            for site in site_of(arch):
                for kind in KINDS:
                    for o in OBJ:
                        conv = not (kind == "channel" and o == "act_hard")
                        br = (lambda s: s) if conv else (lambda s: f"[{s}]")
                        for it in INTS:
                            g = lambda key, part="future": D.A(arch, kind, o, f"{it}:{key}", t, site, part, conv)   # noqa: E731
                            half = f(g("F_bayesT_half")) if it == "probe_e_half" else ""
                            L.append(f"| {arch} | {site} | {kind} | {o} | {it} | {br(f(g('F_model_S', 'k0')))} | {br(f(g('F_model_S')))} "
                                     f"| {br(f(g('F_bayesT')))} | {br(f(g('kl_bayesS'), 4))} | {br(f(g('kl_bayesT'), 4))} | {half} |")
            L.append("")
    L += ["## B. Equal-belief equivalence (t = 12, 8 continuations per pair)", "",
          "Effect = JS between the model's outputs after H₁⊕c and H₂⊕c; at the GRU's complete cut this is the patch effect. "
          "'/random' = the same effect for random pairs.", "",
          "| arch | env | objective | pairs | effect k0 | effect fut | /random (mean) | /random (max over k) | Bayes divergence fut | model / Bayes | Spearman(model, Bayes) over pairs |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arch in ("gru", "tfm"):
        for kind in KINDS:
            for o in OBJ:
                conv = not (kind == "channel" and o == "act_hard")
                for mode in {"iid": ("iid_exact",), "channel": ("chan_equal_b", "chan_equal_joint")}[kind]:
                    b = D.B(arch, kind, o, mode, conv)
                    if b is None:
                        continue
                    L.append(f"| {arch} | {kind} | {o} | {mode} | {f(b['model'][0], 5)} | {f(b['model'][1:].mean(), 5)} | {f(b['ratio_mean'], 4)} "
                             f"| {f(b['ratio_k'].max(), 4)} | {f(b['bayes'][1:].mean(), 5)} | {f(b['model_over_bayes'], 2) if b['bayes'][1:].mean() > 1e-8 else '— (Bayes = 0)'} "
                             f"| {f(b['spearman'], 2) if b['bayes'][1:].mean() > 1e-8 else '—'} |")
    L += ["", "## C. Same action, different belief", "",
          "Flip rate: on future steps where the Bayes-optimal actions of the two runs differ, how often the model's actions differ "
          "(`goal` models act on their output posterior). Ladder on the evaluation set: R²_y overall and within the optimal-action "
          "classes, action decodability, and decodability of the neutral-token count (centred on its expectation given t).", "",
          "| arch | env | objective | flip rate |", "|---|---|---|---|"]
    for arch in ("gru", "tfm"):
        for kind in KINDS:
            for o in ("goal", "act_soft", "act_hard"):
                conv = not (kind == "channel" and o == "act_hard")
                v = f(D.flip(arch, kind, o, conv))
                L.append(f"| {arch} | {kind} | {o} | {v if conv else '[' + v + ']'} |")
    L += ["", "| arch | env | objective | site | R²_y | R²_y within action | action acc | neutral-count R² |", "|---|---|---|---|---|---|---|---|"]
    for arch in ("tfm", "gru"):
        for kind in KINDS:
            for o in OBJ:
                conv = not (kind == "channel" and o == "act_hard")
                for s in (("res0", "res1", "res2", "u") if arch == "tfm" else ("h",)):
                    g = lambda k: f(D.lad(arch, kind, o, s, k, conv))       # noqa: E731
                    row = [g("r2y"), g("r2y_within_action"), g("action_acc"), g("neutral_r2")]
                    if not conv:
                        row = [f"[{x}]" for x in row]
                    L.append(f"| {arch} | {kind} | {o} | {s} | " + " | ".join(row) + " |")
    return "\n".join(L) + "\n"


def predictions(D: C):
    P = []
    add = lambda n, ok, d: P.append((n, bool(ok), d))                      # noqa: E731
    Ts = (6, 12)
    v = {t: (D.A("gru", "iid", "goal", "probe_e:F_model_S", t, "h"), D.A("gru", "iid", "goal", "probe_e:F_model_S", t, "h", "k0"),
             D.A("gru", "iid", "goal", "probe_d:F_model_S", t, "h")) for t in Ts}
    add("A1 GRU iid goal: PROBE-E ≥ 0.9 fut, ≥ 0.95 k0; PROBE-D ≥ 0.8 fut",
        all(a >= 0.9 and b >= 0.95 and c >= 0.8 for a, b, c in v.values()),
        "; ".join(f"t={t}: E fut {f(a)}, k0 {f(b)}, D fut {f(c)}" for t, (a, b, c) in v.items()))
    v = {t: (D.A("gru", "channel", "goal", "probe_e:kl_bayesT", t, "h"), D.A("gru", "channel", "goal", "probe_e:kl_bayesS", t, "h"),
             D.A("gru", "channel", "goal", "probe_e:F_bayesT", t, "h")) for t in Ts}
    add("A2 GRU channel goal: PROBE-E closer to transplant than genuine-B; F_T ≥ 0.8",
        all(a < b and c >= 0.8 for a, b, c in v.values()),
        "; ".join(f"t={t}: KL to T {f(a, 4)} vs to S {f(b, 4)}, F_T {f(c)}" for t, (a, b, c) in v.items()))
    v = {t: D.A("gru", "iid", "goal", "probe_e_half:F_bayesT_half", t, "h") for t in Ts}
    add("A3 GRU iid goal: halfway PROBE-E ≥ 0.8 of its own transplant gap", all(x >= 0.8 for x in v.values()),
        "; ".join(f"t={t}: {f(x)}" for t, x in v.items()))
    cells = [(k, o) for k in KINDS for o in OBJ if not (k == "channel" and o == "act_hard")]
    v = {(k, o, t): D.A("gru", k, o, "rand:F_model_S", t, "h") for k, o in cells for t in Ts}
    add("A4 RAND ≤ 0.2 in every GRU cell", all(x <= 0.2 for x in v.values()), f"max {f(max(v.values()))}")
    v = {(o, t): D.A("gru", "iid", o, "probe_e:F_model_S", t, "h") for o in ("act_soft", "act_hard", "next_obs") for t in Ts}
    add("A5 GRU iid act_soft / act_hard / next_obs: PROBE-E ≥ 0.7 fut", all(x >= 0.7 for x in v.values()),
        "; ".join(f"{o} t={t}: {f(x)}" for (o, t), x in v.items()))
    v1 = {(k, it, t): D.A("tfm", k, "goal", f"{it}:F_model_S", t, "res1") for k in KINDS for it in ("probe_e", "swap") for t in Ts}
    v2 = {(k, t): D.A("tfm", k, "goal", "swap:F_model_S", t, "res2", "k0") for k in KINDS for t in Ts}
    add("A6 tfm goal: res1 single-position PROBE-E / SWAP ≤ 0.3 fut; res2 SWAP ≥ 0.95 at k0",
        all(x <= 0.3 for x in v1.values()) and all(x >= 0.95 for x in v2.values()),
        f"res1 fut max {f(max(v1.values()))}; res2 swap k0 min {f(min(v2.values()))}")
    b = D.B("gru", "iid", "goal", "iid_exact")
    add("B1 GRU iid goal: equal-belief effect ≤ 0.05 of random at every k", b["ratio_k"].max() <= 0.05,
        f"max over k {f(b['ratio_k'].max(), 4)}, mean {f(b['ratio_mean'], 4)}")
    b = D.B("gru", "channel", "goal", "chan_equal_b")
    add("B2 GRU channel goal equal-b: model/Bayes in [0.5, 2], Spearman ≥ 0.5",
        0.5 <= b["model_over_bayes"] <= 2 and b["spearman"] >= 0.5, f"model/Bayes {f(b['model_over_bayes'], 2)}, Spearman {f(b['spearman'], 2)}")
    b = D.B("gru", "channel", "goal", "chan_equal_joint")
    add("B3 GRU channel goal equal-joint: ≤ 0.1 of random", b["ratio_mean"] <= 0.1, f"{f(b['ratio_mean'], 4)}")
    v = {o: D.B("gru", "iid", o, "iid_exact")["ratio_mean"] for o in ("act_hard", "act_soft", "next_obs")}
    add("B4 GRU iid other objectives: equal-belief ≤ 0.1 of random", all(x <= 0.1 for x in v.values()),
        "; ".join(f"{o} {f(x, 4)}" for o, x in v.items()))
    x = D.flip("gru", "iid", "act_hard")
    add("C1 GRU iid act_hard flip rate ≥ 0.9", x >= 0.9, f(x))
    g_u, h_u = D.lad("tfm", "iid", "goal", "u", "r2y_within_action"), D.lad("tfm", "iid", "act_hard", "u", "r2y_within_action")
    g_1, h_1 = D.lad("tfm", "iid", "goal", "res1", "r2y_within_action"), D.lad("tfm", "iid", "act_hard", "res1", "r2y_within_action")
    add("C2 tfm iid: within-action R²_y at u, goal − act_hard ≥ 0.2; res1 ≥ 0.8 both",
        g_u - h_u >= 0.2 and min(g_1, h_1) >= 0.8, f"u: goal {f(g_u)} − act_hard {f(h_u)} = {f(g_u - h_u)}; res1 goal {f(g_1)}, act_hard {f(h_1)}")
    v = [D.lad("tfm", "iid", "goal", s, "r2y_within_action") for s in ("res2", "u")]
    add("C3 tfm iid goal: within-action R²_y ≥ 0.95 at res2 and u", min(v) >= 0.95, f"res2 {f(v[0])}, u {f(v[1])}")
    v = {o: (D.lad("tfm", "iid", o, "res1", "neutral_r2"), D.lad("tfm", "iid", o, "u", "neutral_r2")) for o in OBJ}
    add("C4 tfm iid: neutral-count R² lower at u than res1, every objective", all(u < r for r, u in v.values()),
        "; ".join(f"{o} res1 {f(r)} → u {f(u)}" for o, (r, u) in v.items()))
    return P


def main():
    D = C(OUT / "causal.json")
    txt = tables(D)
    P = predictions(D)
    txt += "\n## Pre-registered predictions (rounds/r12_hidden_goal/causal/THEORY.md §6)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {d} |\n" for n, ok, d in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    (OUT / "tables.md").write_text(txt)
    print(txt[txt.index("## Pre-registered"):])
    P11.causal_figures(D, OUT)


if __name__ == "__main__":
    main()
