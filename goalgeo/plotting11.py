"""TASK11 figures. Categorical colours follow one fixed order: goal, act_soft, act_hard, next_obs;
untrained networks are neutral grey."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"goal": "#2a78d6", "act_soft": "#eb6834", "act_hard": "#1baf7a", "next_obs": "#eda100", "init": "#9a9a94"}
OBJ = ("goal", "act_soft", "act_hard", "next_obs", "init")
INK, MUTED, GRID = "#2b2b28", "#6b6a63", "#e4e3dc"
SPLITS = ("IID", "EXT", "CONF", "TIME")
FLOOR = -0.5


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def fig_interface(Rs, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey=True)
    w = 0.16
    for i, a in enumerate(("tfm", "gru")):
        for j, k in enumerate(("iid", "channel")):
            ax = axes[i, j]; _style(ax)
            for n, o in enumerate(OBJ):
                v = [Rs.m(a, k, o, f"{s}_r2y") for s in SPLITS]
                x = np.arange(len(SPLITS)) + (n - 2) * w
                ax.bar(x, np.maximum(v, 0), w * 0.88, color=COL[o], label=o if (i, j) == (0, 0) else None)
                for xx, vv in zip(x, v):
                    if np.isnan(vv):
                        ax.text(xx, 0.02, "not converged", ha="center", va="bottom", fontsize=6, color=MUTED, rotation=90)
            base = Rs.base[f"{k}_K4"]["counts"]
            for s_i, s in enumerate(SPLITS):
                ax.plot([s_i - 2.5 * w, s_i + 2.5 * w], [base[f"{s}_r2y"]] * 2, color=INK, lw=1, ls="--",
                        label="affine in counts" if (i, j, s_i) == (0, 0, 0) else None)
            ax.set_xticks(range(len(SPLITS)), ["IID", "EXT\n(confident)", "CONF\n(conflict)", "TIME\n(t > 12)"])
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{'transformer (post-LN u)' if a == 'tfm' else 'GRU (h)'} — {k}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel("held-out R² of affine probe → log-odds", fontsize=8, color=INK)
    fig.suptitle("Readout interface: affine recoverability of the posterior log-odds by split", y=0.99, fontsize=10, color=INK)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(out / "fig1_interface.png", dpi=150); plt.close(fig)


def fig_sites(Rs, out: Path):
    sites = ("res0", "res1", "res2", "u")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for j, k in enumerate(("iid", "channel")):
        ax = axes[j]; _style(ax)
        for o in OBJ:
            v = [Rs.m("tfm", k, o, "IID_r2y", site=s) for s in sites]
            ax.plot(range(4), v, color=COL[o], lw=2, marker="o", ms=5, label=o)
        ax.axhline(Rs.base[f"{k}_K4"]["counts"]["IID_r2y"], color=INK, ls="--", lw=1, label="affine in counts")
        ax.axhline(Rs.base[f"{k}_K4"]["mean_t"]["IID_r2y"], color=MUTED, ls=":", lw=1, label="affine in (counts/t, t)")
        ax.set_xticks(range(4), ["res0\n(embed)", "res1", "res2", "u\n(interface)"]); ax.set_ylim(0, 1.02)
        ax.set_title(f"transformer, {k}: IID R²_y by site", fontsize=9, color=INK)
    ax = axes[2]; _style(ax)
    for o in OBJ:
        v = [Rs.m("tfm", "channel", o, "G_count", site=s) for s in sites]
        ax.plot(range(4), np.maximum(v, -1), color=COL[o], lw=2, marker="o", ms=5)
    ax.axhline(0, color=INK, ls="--", lw=1)
    ax.set_xticks(range(4), ["res0\n(embed)", "res1", "res2", "u\n(interface)"]); ax.set_ylim(-1, 1.02)
    ax.set_title("transformer, channel: gain over counts G_count\n(res0 is −5.6, clipped)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig2_sites.png", dpi=150); plt.close(fig)


def fig_scatter(out: Path):
    """Probe prediction against the true first log-odds coordinate on EXT test states
    (probe fit on |y| < 2), one transformer seed per objective, channel env."""
    import torch
    from . import belief_train as BT, beliefprobe as BP
    E = BP.EvalSet("channel", 4, n=3000)
    fit, test = E.splits()["EXT"]
    rng = np.random.default_rng(0); pick = rng.choice(np.flatnonzero(test), 3000, replace=False)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
    for ax, o in zip(axes, ("goal", "act_soft", "act_hard", "next_obs")):
        _style(ax)
        job = BT.Job("tfm", "channel", 4, o, 0)
        p = out / "models" / f"{job.name}.pt"
        if not p.exists():
            continue
        net = BT.load_net(job, p)
        S = BP.tfm_sites(net, E.X, device="cuda" if torch.cuda.is_available() else "cpu")
        P = BP._pred(S["u"][pick], BP._fit(S["u"][fit], E.y[fit]))
        ax.scatter(E.y[pick, 0], P[:, 0], s=4, color=COL[o], alpha=0.35, lw=0)
        lim = (-12, 12); ax.plot(lim, lim, color=INK, lw=1, ls="--")
        ax.axvspan(-EXT, EXT, color=GRID, alpha=0.6, lw=0)
        ax.set_xlim(lim); ax.set_ylim(lim)
        note = "" if o != "act_hard" else " (not converged)"
        ax.set_title(f"{o}: R² {BP.r2(E.y[pick], P):.2f}{note}", fontsize=9, color=INK)
        ax.set_xlabel("true log b(g₁)/b(g₄)", fontsize=8, color=INK)
    axes[0].set_ylabel("affine probe on u", fontsize=8, color=INK)
    fig.suptitle("Channel env, transformer seed 0: probe fit on |y| < 2 (shaded), scored on max|y| ≥ 3", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "fig3_extrapolation.png", dpi=150); plt.close(fig)


EXT = 2.0


def all_figures(Rs, out: Path):
    fig_interface(Rs, out); fig_sites(Rs, out); fig_scatter(out)
