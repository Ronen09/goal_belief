"""TASK12 figures: matched interventions, the equivalence hierarchy, the hidden-size bottleneck."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from goalgeo.style import COL, GRID, INK, MUTED, _style  # noqa: E402

HIDDEN = (2, 3, 4, 5, 6, 7, 8, 12, 16, 32, 64)


def fig_interventions(d, out: Path):
    ints = [("g_only", "goal block only"), ("r_only", "channel block only"), ("both", "both blocks"), ("swap", "state swap"), ("rand", "random")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    w = 0.25
    for ax, t in zip(axes, (6, 12)):
        _style(ax)
        for n, o in enumerate(("goal", "act_soft", "next_obs")):
            v = [d.I(o, t, f"{k}:F_bayes") for k, _ in ints]
            x = np.arange(len(ints)) + (n - 1) * w
            ax.bar(x, np.maximum(v, -0.2), w * 0.88, color=COL[o], label=o)
            for xx, vv in zip(x, v):
                if vv < -0.2:
                    ax.text(xx, -0.19, f"{vv:.2f}", ha="center", va="bottom", fontsize=6, color=INK, rotation=90)
        ax.axhline(0, color=MUTED, lw=0.6); ax.set_ylim(-0.2, 1.05)
        ax.set_xticks(range(len(ints)), [lab for _, lab in ints], fontsize=8)
        ax.set_title(f"GRU, channel, t = {t}: gap closed to each edit's own Bayes future (k ≥ 1)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig1_interventions.png", dpi=150); plt.close(fig)


def fig_hierarchy(d, out: Path):
    sets = [("Eb", "equal b"), ("Em", "equal\nmarginals"), ("Ej", "equal full\nstate"), ("Ej3", "equal full,\n≥3 positions"), ("Ex", "equal full,\nlength 8 vs 16")]
    fig, ax = plt.subplots(figsize=(9, 3.8)); _style(ax)
    w = 0.25
    for n, o in enumerate(("goal", "act_soft", "next_obs")):
        es = [d.E(o, m) for m, _ in sets]
        x = np.arange(len(sets)) + (n - 1) * w
        ax.bar(x, [max(e["model"], 1e-8) for e in es], w * 0.88, color=COL[o], label=f"{o}: model")
        ax.scatter(x, [max(e["bayes"], 1e-8) for e in es], marker="_", s=160, color=INK, zorder=3,
                   label="Bayes divergence of the same pairs" if n == 0 else None)
    r = d.E("goal", "R")
    ax.axhline(r["model"], color=MUTED, ls="--", lw=1, label="random pairs")
    ax.set_yscale("log"); ax.set_ylim(1e-8, 1)
    ax.set_xticks(range(len(sets)), [lab for _, lab in sets], fontsize=8)
    ax.set_ylabel("JS between the two histories' futures", fontsize=8, color=INK)
    ax.set_title("GRU, channel: the future divergence of histories matched on more and more of the filter state", fontsize=9, color=INK)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "fig2_hierarchy.png", dpi=150); plt.close(fig)


def fig_bottleneck(d, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(HIDDEN))
    ax = axes[0]; _style(ax)
    for kind, col in (("iid", "#2a78d6"), ("channel", "#eb6834")):
        ax.plot(x, [d.kl(kind, n) for n in HIDDEN], color=col, lw=2, marker="o", ms=4, label=kind)
    ax.axhline(0.01, color=MUTED, ls=":", lw=1); ax.set_yscale("log")
    ax.set_title("KL to the exact posterior", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]; _style(ax)
    ax.plot(x, [d.rec("goal", "IID", "goal", hidden=n, conv=False) for n in HIDDEN], color="#2a78d6", lw=2, marker="o", ms=4, label="goal block (supervised projection)")
    ax.plot(x, [d.rec("goal", "IID", "channel", hidden=n, conv=False) for n in HIDDEN], color="#eb6834", lw=2, marker="o", ms=4, label="channel block (unsupervised)")
    ax.set_ylim(0, 1.02); ax.set_title("channel env: affine R² of each block", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[2]; _style(ax)
    for m, col, lab in (("Ej", "#1baf7a", "equal full state, same length"), ("Ex", "#e87ba4", "equal full state, lengths 8 vs 16")):
        ax.plot(x, [(d.E("goal", m, n, conv=False) or {}).get("over_bayes", np.nan) for n in HIDDEN], color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(1, color=MUTED, ls=":", lw=1); ax.set_yscale("log")
    ax.set_title("channel env: model ÷ Bayes future divergence", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xticks(x, [str(n) for n in HIDDEN], fontsize=7); ax.set_xlabel("GRU hidden size", fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(out / "fig3_bottleneck.png", dpi=150); plt.close(fig)


def all_figures(d, out: Path):
    fig_interventions(d, out); fig_hierarchy(d, out); fig_bottleneck(d, out)
