"""TASK14 (round 15): tables, prediction checks and figure from results14/results14.json.

    .venv/bin/python scripts/task14_tables.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from goalgeo.plotting11 import INK, MUTED, _style  # noqa: E402

OUT = Path("results14")
CLASSES = [(2, 24), (2, 64), (4, 24), (4, 64)]
TS = {24: (2, 4, 8, 16, 23), 64: (2, 4, 8, 16, 32, 48, 63)}
CELLS = ("R1_baseline", "R2_swap", "R2_probe", "R3_corrupt", "R3_mask", "R4_consistent")


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


class D:
    def __init__(self, path):
        self.runs = json.load(open(path))["runs"]

    def sel(self, L, T, conv=True):
        return [r for r in self.runs if r["L"] == L and r["T"] == T and (not conv or r["kl"] < 0.01)]

    def v(self, L, T, t, cell, key, site="u"):
        rs = self.sel(L, T)
        if not rs:
            return float("nan")
        g = lambda r: r["t"][str(t)][cell][site][key] if site else r["t"][str(t)][cell][key]      # noqa: E731
        return float(np.mean([g(r) for r in rs]))

    def cal(self, L, T, t, cell, site="output"):
        """λ rescaled so that the baseline (R1) is 0 and the consistent counterfactual (R4) is 1, per seed."""
        rs = self.sel(L, T)
        g = lambda r, c: r["t"][str(t)][c][site]["lambda"]                    # noqa: E731
        return float(np.mean([(g(r, cell) - g(r, "R1_baseline")) / (g(r, "R4_consistent") - g(r, "R1_baseline")) for r in rs]))

    def attn(self, L, T, t):
        """Mean attention of query t+1 on key t in the blocks that read belief-carrying residuals (2..L)."""
        rs = self.sel(L, T)
        return float(np.mean([np.mean(r["t"][str(t)]["R1_baseline"]["attn_on_t"][1:]) for r in rs])) if rs else float("nan")


def tables(d: D) -> str:
    L_ = ["# TASK14 tables (round 15: prior vs recomputation through the K/V cache)", "",
          "Next-token transformers trained on sampled tokens in the channel environment. λ = position of the decoded full "
          "filter state at t+1 along A⁺ → B⁺ (0: the update of A's belief, 1: the update of B's), pooled over 1000 pairs; "
          "perp = distance off that line as a fraction of |B⁺ − A⁺|. Decoded at u(t+1) unless stated. Seed means over "
          "converged models (predictive KL < 0.01).", ""]
    L_ += ["## 1. Models", "", "| L | T | converged | predictive KL | probe R² of the full state at u |", "|---|---|---|---|---|"]
    for L, T in CLASSES:
        rs = d.sel(L, T, conv=False)
        L_.append(f"| {L} | {T} | {sum(r['kl'] < 0.01 for r in rs)}/{len(rs)} | {f(np.mean([r['kl'] for r in rs]), 4)} "
                  f"| {f(np.mean([r['probe_r2']['u'] for r in rs]))} |")
    L_ += ["", "## 2. The 2×2 at every t", "",
           "| L | T | t | R2 swap λ | R2 probe λ | R3 corrupt λ | R4 λ | R2 swap, output λ | R3 corrupt, output λ | perp R2 / R3 / R1 | R3 mask retention | attention t+1 → t (blocks ≥ 2) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for L, T in CLASSES:
        for t in TS[T]:
            g = lambda c, k="lambda", s="u": d.v(L, T, t, c, k, s)            # noqa: E731
            L_.append(f"| {L} | {T} | {t} | {f(g('R2_swap'))} | {f(g('R2_probe'))} | {f(g('R3_corrupt'))} | {f(g('R4_consistent'))} "
                      f"| {f(g('R2_swap', s='output'))} | {f(g('R3_corrupt', s='output'))} "
                      f"| {f(g('R2_swap', 'perp'), 2)} / {f(g('R3_corrupt', 'perp'), 2)} / {f(g('R1_baseline', 'perp'), 2)} "
                      f"| {f(d.v(L, T, t, 'R3_mask', 'retention', None))} | {f(d.attn(L, T, t))} |")
    L_ += ["", "## 3. Where the transplanted prior enters: R2 swap λ by site at t+1", "",
           "| L | T | t | " + " | ".join(f"res{i}" for i in range(1, 5)) + " | u |", "|---|---|---|---|---|---|---|---|"]
    for L, T in CLASSES:
        for t in (4, 8, 16):
            cells = [f(d.v(L, T, t, "R2_swap", "lambda", f"res{i}")) if i <= L else "" for i in range(1, 5)]
            L_.append(f"| {L} | {T} | {t} | " + " | ".join(cells) + f" | {f(d.v(L, T, t, 'R2_swap', 'lambda'))} |")
    rp = json.load(open(OUT / "recompute_prediction.json")) if (OUT / "recompute_prediction.json").exists() else {}
    L_ += ["", "## 5. Post hoc (not pre-registered): calibrated λ and the pure-recomputation prediction", "",
           "Output λ (the next-token predictive; needs no probe) and decoded λ at u, each rescaled so that R1 = 0 and R4 = 1. "
           "'Recompute' = the exact Bayes λ of the token sequence the R3 cell shows position t+1 (B's tokens before t, A's x_t and "
           "x_{t+1}), i.e. what pure recomputation from tokens predicts (scripts/task14_recompute.py).", "",
           "| L | T | t | prior weight: R2 swap (output) | R2 probe (output) | R2 swap (decoded u) | R3 corrupt (output) | recompute prediction for R3 | R3 − prediction |",
           "|---|---|---|---|---|---|---|---|---|"]
    for L, T in CLASSES:
        for t in TS[T]:
            r3 = d.cal(L, T, t, "R3_corrupt"); pr = rp.get(f"{T}_{t}", float("nan"))
            L_.append(f"| {L} | {T} | {t} | {f(d.cal(L, T, t, 'R2_swap'))} | {f(d.cal(L, T, t, 'R2_probe'))} | {f(d.cal(L, T, t, 'R2_swap', 'u'))} "
                      f"| {f(r3)} | {f(pr)} | {f(r3 - pr)} |")
    return "\n".join(L_) + "\n"


def predictions(d: D):
    P = []
    add = lambda n, ok, x: P.append((n, bool(ok), x))                   # noqa: E731
    v = [d.v(L, T, t, "R4_consistent", "lambda") for L, T in CLASSES for t in TS[T]]
    add("Q1 R4 λ ≥ 0.9 everywhere", min(v) >= 0.9, f"min {f(min(v))}")
    v = {(T, t): d.v(2, T, t, "R2_swap", "lambda") for T in (24, 64) for t in TS[T] if t >= 8}
    add("Q2 2-layer R2 swap λ ≤ 0.3 at t ≥ 8", max(v.values()) <= 0.3, "; ".join(f"T={T} t={t} {f(x)}" for (T, t), x in v.items()))
    ok, det = True, []
    for L in (2, 4):
        a, b = d.v(L, 24, 4, "R2_swap", "lambda"), d.v(L, 24, 16, "R2_swap", "lambda")
        c, e = d.v(L, 64, 8, "R2_swap", "lambda"), d.v(L, 64, 48, "R2_swap", "lambda")
        ok &= a > b and c > e; det.append(f"L={L}: T24 t4 {f(a)} vs t16 {f(b)}; T64 t8 {f(c)} vs t48 {f(e)}")
    add("Q3 prior weight falls with context", ok, "; ".join(det))
    v = {(T, t): (d.v(4, T, t, "R2_swap", "lambda"), d.v(2, T, t, "R2_swap", "lambda")) for T in (24, 64) for t in (8, 16)}
    add("Q4 4-layer R2 swap λ > 2-layer", all(a > b for a, b in v.values()), "; ".join(f"T={T} t={t}: {f(a)} vs {f(b)}" for (T, t), (a, b) in v.items()))
    ok, det = True, []
    for L, T in CLASSES:
        pr = np.mean([d.v(L, T, t, "R2_probe", "lambda") for t in TS[T] if t >= 4]); sw = np.mean([d.v(L, T, t, "R2_swap", "lambda") for t in TS[T] if t >= 4])
        ok &= pr >= 0.7 * sw; det.append(f"L={L} T={T}: probe {f(pr)} vs swap {f(sw)}")
    add("Q5 R2 probe λ ≥ 0.7 × R2 swap λ", ok, "; ".join(det))
    v = {(L, T, t): d.v(L, T, t, "R2_swap", "lambda") + 1 - d.v(L, T, t, "R3_corrupt", "lambda") for L, T in CLASSES for t in (8, 16)}
    add("Q6 λ(R2) + 1 − λ(R3) in [0.8, 1.2]", all(0.8 <= x <= 1.2 for x in v.values()), "; ".join(f"L={L} T={T} t={t} {f(x)}" for (L, T, t), x in v.items()))
    v = {(L, T, t, c): d.v(L, T, t, c, "perp") for L, T in CLASSES for t in TS[T] if t >= 8 for c in ("R2_swap", "R3_corrupt")}
    add("Q7 perp ≤ 0.3", max(v.values()) <= 0.3, f"max {f(max(v.values()))}; baseline R1 perp (probe floor) mean "
        f"{f(np.mean([d.v(L, T, t, 'R1_baseline', 'perp') for L, T in CLASSES for t in TS[T] if t >= 8]))}")
    v = {(T, t): d.v(2, T, t, "R3_mask", "retention", None) for T in (24, 64) for t in TS[T] if t >= 8}
    add("Q8 2-layer R3 mask retention ≤ 0.5 at t ≥ 8", max(v.values()) <= 0.5, "; ".join(f"T={T} t={t} {f(x)}" for (T, t), x in v.items()))
    xs = [(d.v(L, T, t, "R2_swap", "lambda"), d.attn(L, T, t)) for L, T in CLASSES for t in TS[T]]
    rho = spearmanr([a for a, _ in xs], [b for _, b in xs]).correlation
    add("Q9 Spearman(R2 swap λ, attention on t) ≥ 0.7", rho >= 0.7, f"ρ = {f(rho, 2)} over {len(xs)} (class, t) cells")
    return P


def figure(d: D):
    rp = json.load(open(OUT / "recompute_prediction.json"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    cols = {(2, 24): "#2a78d6", (2, 64): "#1baf7a", (4, 24): "#eb6834", (4, 64): "#e87ba4"}
    ax = axes[0]; _style(ax)
    for L, T in CLASSES:
        ax.plot(TS[T], [d.cal(L, T, t, "R2_swap") for t in TS[T]], color=cols[(L, T)], lw=2, marker="o", ms=4, label=f"{L} layers, T = {T}")
    ax.set_ylim(-0.05, 1.05); ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_title("prior edited (B's export at t, A's tokens):\nweight on the transplanted belief", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax = axes[1]; _style(ax)
    for L, T in CLASSES:
        ax.plot(TS[T], [d.cal(L, T, t, "R3_corrupt") for t in TS[T]], color=cols[(L, T)], lw=2, marker="o", ms=4)
    t64 = TS[64]
    ax.plot(t64, [rp[f"64_{t}"] for t in t64], color=INK, lw=1, ls="--", label="pure recomputation from tokens (exact)")
    ax.set_ylim(-0.05, 1.05); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("evidence edited (B's tokens before t, A's export at t):\nposition along A⁺ → B⁺", fontsize=9, color=INK)
    for ax in axes[:2]:
        ax.set_xscale("log", base=2); ax.set_xlabel("position t of the intervention", fontsize=8, color=INK)
    ax = axes[2]; _style(ax)
    for L, T in CLASSES:
        ax.scatter([d.attn(L, T, t) for t in TS[T]], [d.cal(L, T, t, "R2_swap") for t in TS[T]], color=cols[(L, T)], s=24)
    ax.set_xlabel("attention of t+1 on t (blocks ≥ 2)", fontsize=8, color=INK); ax.set_ylabel("weight on the transplanted belief", fontsize=8, color=INK)
    ax.set_title("the prior's weight vs the attention it receives", fontsize=9, color=INK)
    fig.suptitle("Next-token transformer, channel env: output-level λ, rescaled so baseline = 0 and consistent counterfactual = 1", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "fig1_prior_vs_recompute.png", dpi=150); plt.close(fig)


def main():
    d = D(OUT / "results14.json")
    txt = tables(d); P = predictions(d)
    txt += "\n## 4. Pre-registered predictions (docs/task14_prior_theory.md §5)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {x} |\n" for n, ok, x in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    (OUT / "tables14.md").write_text(txt); print(txt)
    figure(d)


if __name__ == "__main__":
    main()
