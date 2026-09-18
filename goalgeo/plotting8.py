"""TASK8 figures: steering effect curves and matched cross-model divergence; on/off-manifold
linearisation error and cross-model CV; the propagation-depth hierarchy."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import SERIES, MUTED, _save

ALPHAS = np.linspace(0, 1.5, 16); SCALES = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
FAM = {"lr0.1": SERIES[0], "base": SERIES[6], "lr10": SERIES[1], "k4_lr0.1": SERIES[0], "k4_lr10": SERIES[1]}


def _fam(name):
    return name.rsplit("_s", 1)[0]


def fig_steering(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for a in R["A"][:-1]:
        for tag, col in (("1", SERIES[0]), ("2", SERIES[1])):
            axes[0].plot(ALPHAS, a["curves"][f"{tag}_meandiff"], "-", color=col, lw=1.4, alpha=0.8, label={"1": "lr ×0.1 (large ‖Δh‖)", "2": "lr ×10 (small ‖Δh‖)"}[tag] if a is R["A"][0] else None)
            axes[0].plot(ALPHAS, a["curves"][f"{tag}_gradient"], "--", color=col, lw=1.0, alpha=0.6, label={"1": "gradient direction", "2": None}[tag] if a is R["A"][0] else None)
    axes[0].set_xlabel("α (steering in units of the model's own Δh)"); axes[0].set_ylabel("JS(F(h+αv), F(h))"); axes[0].set_title("Effect curves, same α"); axes[0].grid(True); axes[0].legend(fontsize=7)
    x = np.arange(4); w = 0.2
    for j, (kind, col) in enumerate((("meandiff", SERIES[2]), ("gradient", SERIES[3]))):
        same = np.mean([[r["cross_js_same_alpha"] for r in a["matched"][kind]] for a in R["A"][:-1]], 0)
        matched = np.mean([[r["cross_js_matched"] for r in a["matched"][kind]] for a in R["A"][:-1]], 0)
        axes[1].bar(x + (2 * j - 1.5) * w, same, w, color=col, alpha=0.45, label=f"{kind}: same α", edgecolor="white")
        axes[1].bar(x + (2 * j - 0.5) * w, matched, w, color=col, label=f"{kind}: matched effect", edgecolor="white")
    axes[1].axhline(np.mean([a["cross_js_alpha0"] for a in R["A"][:-1]]), color=MUTED, ls="--", lw=0.8, label="α = 0")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["α₁=0.25", "0.5", "0.75", "1.0"]); axes[1].set_ylabel("cross-model JS"); axes[1].set_title("lr ×0.1 vs ×10: outputs after steering"); axes[1].set_yscale("log"); axes[1].legend(fontsize=6.5); axes[1].grid(True, axis="y")
    labels = [a["pair"].replace(" | ", "\nvs ") for a in R["A"]]
    axes[2].bar(np.arange(len(labels)) - 0.2, [a["norm_v"][0] / a["norm_v"][1] for a in R["A"]], 0.4, color=SERIES[1], label="‖v₁‖ / ‖v₂‖", edgecolor="white")
    axes[2].bar(np.arange(len(labels)) + 0.2, [a["cos_v1_mapped_v2"] for a in R["A"]], 0.4, color=SERIES[4], label="cos(v₁ mapped, v₂)", edgecolor="white")
    axes[2].set_xticks(range(len(labels))); axes[2].set_xticklabels(labels, fontsize=6.5); axes[2].set_title("Steering-vector geometry"); axes[2].legend(fontsize=7); axes[2].grid(True, axis="y")
    _save(fig, out / "fig1_intervention_equivalence.png")


def fig_manifold(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    cols = {"meandiff": SERIES[2], "top_pc": SERIES[0], "random": SERIES[6], "bottom_pc": SERIES[1]}
    for kind, d in R["B_summary"].items():
        axes[0].plot(SCALES, d["lin_err"], "o-", color=cols[kind], label=kind); axes[1].plot(SCALES, d["cv_finite"], "o-", color=cols[kind], label=kind); axes[2].plot(SCALES, d["cv_linear"], "o-", color=cols[kind], label=kind)
    for ax, t, yl in zip(axes, ("Linearisation error of the effect", "Cross-model CV of the finite effect", "Cross-model CV of the linear prediction"), ("|JS_finite − JS_lin| / JS_finite", "CV", "CV")):
        ax.set_xscale("log"); ax.set_xlabel("perturbation scale s (× own ‖Δh‖)"); ax.set_ylabel(yl); ax.set_title(t); ax.grid(True); ax.legend(fontsize=7)
    _save(fig, out / "fig2_on_off_manifold.png")


def fig_depth(R, out):
    fig, axes = plt.subplots(4, 2, figsize=(10, 10), layout="constrained")
    for col, (rows, K, title) in enumerate(((R["C_k1"], 4, "delay-1 HMM: relevant token at step 1"), (R["C_k4"], 8, "delay-4 HMM: relevant token at step 4"))):
        for row, (q, yl) in enumerate((("raw", "‖h_A^(k) − h_B^(k)‖ (raw)"), ("lin", "‖J^(k) Δh‖ (linearised)"), ("finite_logit", "‖Δ logits‖ (finite)"), ("js", "JS of predictions (behaviour)"))):
            ax = axes[row, col]; seen = set()
            for r in rows:
                f = _fam(r["name"]); ax.plot(range(K + 1), r[q][:K + 1], "o-", color=FAM.get(f, MUTED), ms=3, lw=1.2, alpha=0.8, label=f if f not in seen else None); seen.add(f)
            ax.set_ylabel(yl, fontsize=8.5); ax.grid(True)
            if row == 0:
                ax.set_title(title); ax.legend(fontsize=7)
            if row == 3:
                ax.set_xlabel("propagation depth k (tokens consumed after the cue)")
    _save(fig, out / "fig3_propagation_depth.png")


def make_figures(R, out):
    fig_steering(R, out); fig_manifold(R, out); fig_depth(R, out)
