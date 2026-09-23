"""TASK7 figures: same function / different geometry, coordinate stress test, horizon drift, δ sensitivity."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from goalgeo.plotting import SERIES, MUTED, _save
from goalgeo import invariants as Iv

CLS = {"raw": (Iv.RAW, SERIES[1]), "information": (Iv.INFO, SERIES[0]), "subspace": (Iv.SUBSPACE, SERIES[4]), "functional": (Iv.FUNCTIONAL, SERIES[2]), "allocation": (Iv.ALLOC, SERIES[3])}


def _color(k):
    for keys, col in CLS.values():
        if k in keys:
            return col
    return MUTED


def fig_same_function(R, out):
    rows = [r for r in R["runs"] if r["cond"]["delta"] == 0.4 and r["family"] in ("lr", "base") and r["seed"] < 3 and r["metrics"]["kl"] < 0.003]
    keys = [("sep_cue", "‖Δh‖ at the cue (raw)"), ("g_eff", "readout gain"), ("D_future", "‖J Δh‖ (functional)"), ("D_F", "D_F (Fisher)"), ("JS_future", "JS after propagation"), ("r2_belief", "belief R² (OLS)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.2), layout="constrained")
    for ax, (k, title) in zip(axes, keys):
        x = [r["cond"]["lr_ratio"] for r in rows]; y = [r["metrics"][k] for r in rows]
        ax.plot(x, y, "o", color=_color(k), ms=5, alpha=0.8); ax.set_xscale("log"); ax.set_title(title, fontsize=9); ax.set_xlabel("readout lr ratio"); ax.grid(True)
        ax.set_ylim(0, 1.15 * max(y))
    _save(fig, out / "fig1_same_function_different_geometry.png")


def fig_transforms(R, out):
    tr = R["transform"]["transforms"]; names = list(tr); keys = list(Iv.ALL)
    fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
    marks = ["o", "o", "s", "^", "D", "D", "D", "D"]
    for j, (name, mk) in enumerate(zip(names, marks)):
        ax.plot(np.arange(len(keys)) + (j - 3.5) * 0.08, [abs(tr[name]["ratio"][k]) for k in keys], mk, ms=4, alpha=0.8, label=name, color=SERIES[j % 7])
    ax.axhline(1, color=MUTED, lw=0.8); ax.set_yscale("log"); ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=7.5)
    for i, k in enumerate(keys):
        ax.get_xticklabels()[i].set_color(_color(k))
    ax.set_ylabel("|M(AH) / M(H)|"); ax.set_title("Exact coordinate transforms at the interface (base model, seed 0)"); ax.legend(fontsize=7, ncol=4); ax.grid(True, axis="y")
    _save(fig, out / "fig2_coordinate_stress_test.png")


def fig_horizon(R, out):
    H = R["horizon"]
    keys = [("kl", "KL to targets"), ("sep_cue", "‖Δh‖ at the cue"), ("g_eff", "readout gain"), ("cos_theta", "cos θ"), ("I_contrast_pre", "g·D·cos θ"), ("D_future", "‖J Δh‖"), ("D_F", "D_F"), ("r2_belief", "belief R²"), ("rsa_euclid", "Euclidean RSA")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), layout="constrained", sharex=True)
    for ax, (k, title) in zip(axes.ravel(), keys):
        for seed, d in H.items():
            steps = sorted(d, key=int); ax.plot([int(s) for s in steps], [d[s][k] for s in steps], "o-", color=_color(k), ms=2.5, lw=1.2, alpha=0.45 + 0.55 * (int(seed) == 0), label=f"seed {seed}")
        ax.set_xscale("symlog", linthresh=50); ax.set_title(title); ax.grid(True)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    axes[0, 0].legend(fontsize=7)
    _save(fig, out / "fig3_horizon_drift.png")


def fig_sensitivity(R, out):
    rows = R["sensitivity_rows"]
    keys = [("sep_cue", "‖Δh‖ at the cue"), ("D_delay", "mean pairwise distance"), ("rsa_euclid", "Euclidean RSA"), ("r2_belief", "belief R²"), ("I_contrast_pre", "readout contrast"), ("D_future", "‖J Δh‖"), ("D_F", "D_F"), ("JS_future", "JS after propagation")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), layout="constrained")
    mk = {0.1: "v", 1.0: "o", 10.0: "^"}
    for ax, (k, title) in zip(axes.ravel(), keys):
        for d, rr in rows.items():
            for r in rr:
                ax.plot(float(d), r[k], mk[r["cond"]["lr_ratio"]], color=_color(k), ms=6, alpha=0.6, mec="black", mew=0.4)
        ax.set_title(title); ax.set_xlabel("δ"); ax.set_xticks([0.1, 0.2, 0.4]); ax.grid(True)
    axes[0, 0].plot([], [], "v", color=MUTED, label="readout lr ×0.1"); axes[0, 0].plot([], [], "o", color=MUTED, label="×1"); axes[0, 0].plot([], [], "^", color=MUTED, label="×10"); axes[0, 0].legend(fontsize=7)
    _save(fig, out / "fig4_functional_sensitivity.png")


def make_figures(R, out):
    fig_same_function(R, out); fig_transforms(R, out); fig_horizon(R, out); fig_sensitivity(R, out)
