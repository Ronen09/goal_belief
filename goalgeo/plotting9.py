"""TASK9 figures: transformer reproduction of the core dissociation, the allocation under fixed
readout gain / lr ratio with and without final LayerNorm, and the invariance CVs."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import SERIES, MUTED, _save


def _rows(R, labels, ln):
    suf = "" if ln else "_noln"; return [R["agg"][l + suf] for l in labels if l + suf in R["agg"]]


def fig_core(R, out):
    labels = ["r0", "lam0", "d0.1", "d0.2", "base"]; names = ["r=0", "λ=0", "δ=0.1", "δ=0.2", "δ=0.4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for ax, ln in zip(axes, (True, False)):
        rows = _rows(R, labels, ln); x = np.arange(len(rows)); w = 0.2
        for j, (k, lab, col) in enumerate((("P_metric", "P_metric (layer-1 cue)", SERIES[1]), ("branch_acc_pre_l1", "branch acc at t*−1, layer 1", SERIES[2]),
                                            ("branch_acc_pre_postnorm", "branch acc at t*−1, post-norm", SERIES[0]), ("JS_future", "JS after patching", SERIES[3]))):
            ax.bar(x + (j - 1.5) * w, [a[k] for a in rows], w, yerr=[a[k + "_std"] for a in rows], color=col, edgecolor="white", capsize=2, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([n for n, a in zip(names, rows)]); ax.set_title(("with" if ln else "without") + " final LayerNorm"); ax.grid(True, axis="y")
    axes[0].legend(fontsize=7); _save(fig, out / "fig1_core_dissociation.png")


def fig_alloc(R, out):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), layout="constrained")
    for row, ln in zip(axes, (True, False)):
        gains = _rows(R, ["gain0.5", "base", "gain2", "gain8"], ln); lrs = _rows(R, ["lr0.1", "base", "lr10"], ln)
        for ax, (k, lab) in zip(row, (("g_eff", "‖w_x − w_y‖"), ("sep_pre_postnorm", "‖Δh̄‖ at the readout interface"), ("sep_pre_final", "‖Δh̄‖ in the final residual (pre-norm)"), ("ln_gain_norm", "‖γ‖ of the final LayerNorm"))):
            xg = [a["g_eff"] for a in gains]; xl = [a["g_eff"] for a in lrs]
            if k == "ln_gain_norm" and not ln:
                ax.text(0.5, 0.5, "no final LayerNorm", ha="center", va="center", transform=ax.transAxes, color=MUTED); ax.set_axis_off(); continue
            ax.errorbar(xg, [a[k] for a in gains], [a[k + "_std"] for a in gains], fmt="s-", color=SERIES[1], capsize=2, ms=4, label="fixed gain 0.5/free/2/8")
            ax.errorbar(xl, [a[k] for a in lrs], [a[k + "_std"] for a in lrs], fmt="o-", color=SERIES[0], capsize=2, ms=4, label="readout lr ×0.1/×1/×10")
            ax.set_xscale("log"); ax.set_xlabel("effective readout gain"); ax.set_title(lab + (" (with LN)" if ln else " (no LN)"), fontsize=9); ax.grid(True)
            if k != "g_eff":
                ax.set_yscale("log")
        row[0].legend(fontsize=7)
    _save(fig, out / "fig2_allocation.png")


def fig_cv(R, out):
    keys = list(R["cv_ln"]); x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    ax.bar(x - w / 2, [R["cv_ln"][k] for k in keys], w, color=SERIES[0], edgecolor="white", label=f"with final LN (n={R['n_eq'][0]})")
    ax.bar(x + w / 2, [R["cv_noln"][k] for k in keys], w, color=SERIES[1], edgecolor="white", label=f"without final LN (n={R['n_eq'][1]})")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=50, ha="right", fontsize=8); ax.set_ylabel("CV across equivalent models"); ax.set_title("Which measures are stable across functionally equivalent transformers?"); ax.legend(fontsize=8); ax.grid(True, axis="y")
    _save(fig, out / "fig3_invariance_cv.png")


def make_figures(R, out):
    fig_core(R, out); fig_alloc(R, out); fig_cv(R, out)
