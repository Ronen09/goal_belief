"""TASK4 figures: information vs prominence, prominence vs forgetting cost, gradient
pressure vs prominence, training dynamics, delay sweep. One colour and one marker per
sweep, fixed across figures; separate panels instead of dual axes."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from goalgeo.plotting import SERIES, MUTED, _save

PARAM = {"frequency": "r", "strength": "delta", "weight": "lam", "delay": "k", "delay_matched": "k"}
MARK = {"frequency": ("o", SERIES[0]), "strength": ("s", SERIES[1]), "weight": ("^", SERIES[2]), "delay": ("D", SERIES[3]),
        "delay_matched": ("d", SERIES[4]), "grid": ("x", MUTED)}


def _sweep(agg, sweep):
    rows = [a for a in agg.values() if sweep in a["sweeps"]]
    return sorted(rows, key=lambda a: a["cond"][PARAM[sweep]])


def _errbar(ax, x, rows, key, color, label, marker="o"):
    ax.errorbar(x, [a[key] for a in rows], [a[f"{key}_std"] for a in rows], fmt=marker + "-", color=color, ecolor=MUTED,
                capsize=2, ms=4, lw=1.6, label=label)


def fig_info_vs_prominence(R, out):
    rows = _sweep(R["agg"], "frequency")
    if not rows:
        return
    x = [a["cond"]["r"] for a in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    _errbar(axes[0], x, rows, "belief_r2", SERIES[0], "belief R² (all positions)")
    _errbar(axes[0], x, rows, "branch_acc_pre", SERIES[2], "branch accuracy at t*−1", "s")
    axes[0].set_xlabel("relevance frequency r"); axes[0].set_ylabel("decodability"); axes[0].set_title("Information: is the cue decodable?")
    axes[0].grid(True); axes[0].legend(); axes[0].set_ylim(0, 1.05)
    _errbar(axes[1], x, rows, "P_metric", SERIES[1], "P_metric at the cue")
    _errbar(axes[1], x, rows, "P_metric_pre", SERIES[3], "P_metric at t*−1", "s")
    axes[1].axhline(rows[0]["init_P_metric"], color=MUTED, ls="--", lw=0.8, label="initialisation")
    axes[1].set_xlabel("relevance frequency r"); axes[1].set_ylabel("delayed-pair / control-pair distance"); axes[1].set_title("Prominence: is the cue amplified?")
    axes[1].grid(True); axes[1].legend()
    _save(fig, out / "fig1_info_vs_prominence.png")


def _scatter_by_sweep(R, xkey, xlabel, title, path, ykey="P_metric", logx=False):
    agg = R["agg"]
    fig, ax = plt.subplots(figsize=(5.4, 3.9), layout="constrained")
    for sweep, (mk, col) in MARK.items():
        rows = [a for a in agg.values() if sweep in a["sweeps"]]
        if not rows:
            continue
        ax.errorbar([a[xkey] for a in rows], [a[ykey] for a in rows], [a[f"{ykey}_std"] for a in rows], fmt=mk, color=col, ecolor=MUTED,
                    capsize=2, ms=5, ls="none", label=sweep.replace("_", " "))
    if logx:
        ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(xlabel); ax.set_ylabel("P_metric = d(delayed pair) / d(control pair)"); ax.set_title(title); ax.grid(True); ax.legend(fontsize=7.5)
    _save(fig, path)


def fig_prominence_vs_cost(R, out):
    _scatter_by_sweep(R, "cost", "expected forgetting cost λ·ΔL_forget (nats per position)", "Prominence vs cost of forgetting",
                      out / "fig2_prominence_vs_cost.png", logx=True)


def fig_gradient_vs_prominence(R, out):
    _scatter_by_sweep(R, "Gpar_rel_int", "∫ |∇_h ℓ_t* · d̂| over training (mean over checkpoints)", "Prominence vs gradient pressure",
                      out / "fig3_gradient_vs_prominence.png", logx=True)


def fig_dynamics(R, out):
    sets = R.get("dynamics_dense") or {"base": R["dynamics"]["base"]}
    if not sets:
        return
    panels = [("branch_acc_pre", "branch accuracy at t*−1", SERIES[2]), ("D_delay", "delayed-pair distance (raw)", SERIES[1]),
              ("P_metric", "P_metric", SERIES[4]), ("Gpar_rel", "|∇_h ℓ_t* · d̂| at the cue", SERIES[3]), ("kl_relevant", "KL to target at t*", SERIES[0])]
    fig, axes = plt.subplots(len(sets), len(panels), figsize=(3.0 * len(panels), 2.6 * len(sets)), layout="constrained", sharex=True)
    axes = axes.reshape(len(sets), len(panels))
    for row, (label, D) in zip(axes, sets.items()):
        for ax, (key, title, col) in zip(row, panels):
            for seed, d in D.items():
                ax.plot(d["steps"], d["curves"][key], "o-", color=col, ms=2.5, lw=1.2, alpha=0.45 + 0.55 * (int(seed) == 0), label=f"seed {seed}")
            ax.grid(True); ax.set_xscale("symlog", linthresh=10)
            if key == "D_delay":
                for seed, d in D.items():
                    ax.plot(d["steps"], d["curves"]["D_control"], "-", color=MUTED, lw=1.0, alpha=0.6, label="control pair" if seed == list(D)[0] else None)
        row[0].set_ylabel(label, fontsize=8.5)
    for ax, (_, title, _) in zip(axes[0], panels):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    axes[0, 1].legend(fontsize=7)
    _save(fig, out / "fig4_dynamics.png")


def fig_target_gap(R, out):
    _scatter_by_sweep(R, "target_gap", "required log-odds gap between branches at t* (0 when λ = 0)", "Prominence vs required output separation",
                      out / "fig6_prominence_vs_target_gap.png")


def fig_delay(R, out):
    plain, matched = _sweep(R["agg"], "delay"), _sweep(R["agg"], "delay_matched")
    if not plain:
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), layout="constrained")
    for rows, col, lab, mk in ((plain, SERIES[3], "λ = 1", "D"), (matched, SERIES[4], "λ matched to k = 1", "d")):
        if rows:
            x = [a["cond"]["k"] for a in rows]
            _errbar(axes[0], x, rows, "P_metric", col, lab, mk); _errbar(axes[1], x, rows, "D_delay_over_median", col, lab, mk)
            _errbar(axes[2], x, rows, "branch_acc_pre", col, lab, mk); _errbar(axes[3], x, rows, "Gpar_rel_int", col, lab, mk)
    axes[0].set_ylabel("P_metric"); axes[1].set_ylabel("delayed-pair distance / median"); axes[2].set_ylabel("branch accuracy at t*−1"); axes[3].set_ylabel("∫ |∇_h ℓ_t* · d̂|")
    for ax, title in zip(axes, ("Prominence vs delay", "Median-normalised", "Decodability vs delay", "Gradient pressure vs delay")):
        ax.set_xscale("log", base=2); ax.set_xlabel("delay k (filler steps)"); ax.grid(True); ax.legend(); ax.set_title(title)
    _save(fig, out / "fig5_delay.png")


def make_figures(R, out):
    fig_info_vs_prominence(R, out); fig_prominence_vs_cost(R, out); fig_gradient_vs_prominence(R, out); fig_dynamics(R, out); fig_delay(R, out)
    fig_target_gap(R, out)
