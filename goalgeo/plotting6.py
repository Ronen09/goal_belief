"""TASK6 figures: readout gain and hidden separation under the three allocation manipulations, and their dynamics."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import SERIES, MUTED, _save

SWEEPS = (("lr_ratio", "readout lr / recurrent lr", "log"), ("init_scale", "readout init scale", "log"), ("control", "control contrast P(x|A′)", "linear"))


def _errbar(ax, x, rows, key, color, label, marker="o"):
    ax.errorbar(x, [a[key] for a in rows], [a[f"{key}_std"] for a in rows], fmt=marker + "-", color=color, ecolor=MUTED, capsize=2, ms=4, lw=1.5, label=label)


def fig_allocation(R, out):
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.4), layout="constrained")
    for row, (sweep, xl, scale) in zip(axes, SWEEPS):
        rows = sorted([a for a in R["agg"].values() if sweep in a["sweeps"]], key=lambda a: a["cond"][sweep])
        x = [a["cond"][sweep] for a in rows]
        _errbar(row[0], x, rows, "g_eff", SERIES[3], "‖w_x − w_y‖ final")
        row[0].plot(x, [a["init_g_eff"] for a in rows], "--", color=MUTED, lw=0.9, label="at init")
        _errbar(row[1], x, rows, "mean_sep_pre", SERIES[1], "delayed pair at t*−1")
        _errbar(row[1], x, rows, "mean_sep_control", SERIES[0], "control pair", "s")
        _errbar(row[2], x, rows, "cos_readout_pre", SERIES[2], "cos θ, delayed")
        _errbar(row[2], x, rows, "cos_readout_control", SERIES[4], "cos θ, control", "s")
        for ax, t in zip(row, ("readout gain", "hidden separation", "alignment with readout")):
            ax.set_xlabel(xl); ax.set_xscale(scale); ax.grid(True); ax.legend(fontsize=7); ax.set_title(t)
    _save(fig, out / "fig1_allocation.png")


def fig_dynamics(R, out):
    rows = sorted([a for a in R["agg"].values() if "lr_ratio" in a["sweeps"]], key=lambda a: a["cond"]["lr_ratio"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained")
    cols = [SERIES[6], SERIES[0], SERIES[2], SERIES[1], SERIES[3]]
    for a, col in zip(rows, cols):
        rr = [r for r in R["runs"] if r["cond"] == a["cond"] and r["seed"] == 0]
        if not rr:
            continue
        d = rr[0]["dynamics"]; steps = sorted(d, key=int)
        for ax, key in zip(axes, ("g_eff", "mean_sep_pre", "achieved_gap_pre")):
            ax.plot(steps, [d[s][key] for s in steps], "o-", color=col, ms=2.5, lw=1.2, label=f"readout lr ×{a['cond']['lr_ratio']:g}")
    for ax, t in zip(axes, ("‖w_x − w_y‖", "‖h̄_C − h̄_D‖ at t*−1", "achieved logit gap")):
        ax.set_xscale("symlog", linthresh=50); ax.set_xlabel("training step"); ax.set_title(t + " (seed 0)"); ax.grid(True); ax.legend(fontsize=7)
    _save(fig, out / "fig2_lr_ratio_dynamics.png")


def make_figures(R, out):
    fig_allocation(R, out); fig_dynamics(R, out)
