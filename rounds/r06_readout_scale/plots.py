"""TASK5 figures: hidden separation against readout gain, gap accounting, free-readout allocation dynamics."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from goalgeo.plotting import SERIES, MUTED, _save

COL = {0.1: SERIES[0], 0.4: SERIES[1]}


def fig_separation_vs_gain(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for delta, col in COL.items():
        con = [r for r in R["runs"] if r["delta"] == delta and r["gain"] is not None]
        free = [r for r in R["runs"] if r["delta"] == delta and r["gain"] is None]
        for ax, key, lab in zip(axes, ("mean_sep_pre", "mean_sep_control", "mean_sep_cue"), ("‖h̄_C − h̄_D‖ at t*−1", "‖h̄_U − h̄_V‖ (control)", "‖h̄_A − h̄_B‖ at the cue")):
            ok = [r for r in con if r["final"]["kl"] < 0.01]; bad = [r for r in con if r["final"]["kl"] >= 0.01]
            ax.loglog([r["final"]["g_eff"] for r in ok], [r["final"][key] for r in ok], "o", color=col, ms=5, label=f"δ={delta}, fixed gain")
            if bad:
                ax.loglog([r["final"]["g_eff"] for r in bad], [r["final"][key] for r in bad], "o", mfc="none", color=col, ms=5, label=f"δ={delta}, not converged")
            ax.loglog([r["final"]["g_eff"] for r in free], [r["final"][key] for r in free], "*", color=col, ms=11, mec="black", mew=0.5, label=f"δ={delta}, free readout")
            if key == "mean_sep_pre":
                req = free[0]["final"]["required_gap"] if free else 2 * np.log((0.5 + delta) / (0.5 - delta))
                g = np.logspace(-1, 1.3, 20); ax.loglog(g, req / g, "--", color=col, lw=0.9, label=f"Δℓ/‖w‖, Δℓ={req:.2f}")
            ax.set_title(lab); ax.set_xlabel("effective readout gain ‖w_x − w_y‖"); ax.grid(True, which="both", lw=0.4)
    axes[0].set_ylabel("mean-state separation"); axes[0].legend(fontsize=6.5)
    _save(fig, out / "fig1_separation_vs_gain.png")


def fig_gap_accounting(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for delta, col in COL.items():
        rows = sorted([a for a in R["agg"].values() if a["delta"] == delta and a["gain"] != "free"], key=lambda a: float(a["gain"]))
        x = [float(a["gain"]) for a in rows]
        axes[0].errorbar(x, [a["achieved_gap_pre"] for a in rows], [a["achieved_gap_pre_std"] for a in rows], fmt="o-", color=col, ecolor=MUTED, capsize=2, ms=4, label=f"δ={delta} achieved")
        axes[0].axhline(rows[0]["required_gap"], color=col, ls="--", lw=0.9, label=f"δ={delta} required")
        axes[1].errorbar(x, [a["cos_readout_pre"] for a in rows], [a["cos_readout_pre_std"] for a in rows], fmt="o-", color=col, ecolor=MUTED, capsize=2, ms=4, label=f"δ={delta}")
    axes[0].set_xscale("log", base=2); axes[1].set_xscale("log", base=2)
    axes[0].set_xlabel("readout gain c"); axes[0].set_ylabel("(w_x − w_y)·(h̄_C − h̄_D)"); axes[0].set_title("Achieved vs required logit gap"); axes[0].grid(True); axes[0].legend(fontsize=7)
    axes[1].set_xlabel("readout gain c"); axes[1].set_ylabel("cosine"); axes[1].set_title("Alignment of the separation with the readout"); axes[1].grid(True); axes[1].legend(fontsize=7)
    _save(fig, out / "fig2_gap_accounting.png")


def fig_free_dynamics(R, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained")
    for delta, col in COL.items():
        for r in [r for r in R["runs"] if r["delta"] == delta and r["gain"] is None]:
            steps = sorted(r["dynamics"], key=int); d = r["dynamics"]
            a = 0.45 + 0.55 * (r["seed"] == 0)
            axes[0].plot(steps, [d[s]["g_eff"] for s in steps], "o-", color=col, ms=2.5, lw=1.2, alpha=a, label=f"δ={delta}" if r["seed"] == 0 else None)
            axes[1].plot(steps, [d[s]["mean_sep_pre"] for s in steps], "o-", color=col, ms=2.5, lw=1.2, alpha=a, label=f"δ={delta}" if r["seed"] == 0 else None)
            axes[2].plot(steps, [d[s]["achieved_gap_pre"] for s in steps], "o-", color=col, ms=2.5, lw=1.2, alpha=a, label=f"δ={delta}" if r["seed"] == 0 else None)
    for ax, t in zip(axes, ("‖w_x − w_y‖ (free readout)", "‖h̄_C − h̄_D‖ at t*−1 (free readout)", "achieved logit gap (free readout)")):
        ax.set_xscale("symlog", linthresh=50); ax.set_xlabel("training step"); ax.set_title(t); ax.grid(True); ax.legend(fontsize=7)
    _save(fig, out / "fig3_free_readout_dynamics.png")


def make_figures(R, out):
    fig_separation_vs_gain(R, out); fig_gap_accounting(R, out); fig_free_dynamics(R, out)
