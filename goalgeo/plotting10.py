"""TASK10 figures: (1) single-node vs complete-cut effects under forced routing,
(2) which term of C = g·D·cos θ absorbs a change of readout budget, and the LayerNorm
ceiling. Colour carries identity, marker and line style repeat it (CVD secondary encoding)."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plotting import SERIES, MUTED, TEXT, _save

COND = ["l1_diag", "pen_l1", "free", "pen_l2", "l2_diag"]
COND_LABEL = {"l1_diag": "block 1 diagonal", "pen_l1": "penalty on block 1", "free": "free",
              "pen_l2": "penalty on block 2", "l2_diag": "block 2 diagonal"}
NORM_STYLE = {"none": (SERIES[0], "o", "-"), "frozen": (SERIES[1], "s", "--"), "learn": (SERIES[2], "^", "-.")}
NORM_LABEL = {"none": "no final norm", "frozen": "LayerNorm, γ frozen", "learn": "LayerNorm, γ learned"}


def fig_routing(R, out):
    A = R["routing"]; pc = A["per_condition"]
    conds = [c for c in COND if c in pc]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout="constrained")
    ax = axes[0]; x = np.arange(len(conds)); w = 0.26
    bars = (("phi_L1{t}", "single node: residual at t", SERIES[0]),
            ("phi_L1{t+1}", "single node: residual at t+1", SERIES[1]),
            ("phi_L1{t,t+1}", "complete cut: {t, t+1}", SERIES[2]))
    for j, (k, lab, col) in enumerate(bars):
        ax.bar(x + (j - 1) * w, [pc[c][k] for c in conds], w, color=col, edgecolor="white", label=lab)
    ax.plot(x, [pc[c]["phi_behav"] for c in conds], "k_", ms=22, mew=1.6, label="behavioural divergence")
    ax.set_xticks(x); ax.set_xticklabels([f"{COND_LABEL[c]}\n({pc[c]['n']}/{pc[c]['n_all']} seeds)" for c in conds],
                                         rotation=15, ha="right", fontsize=7)
    ax.set_ylabel("interchange effect (JS at t+1)"); ax.grid(True, axis="y")
    ax.set_title("forcing the route moves the node, not the cut", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    runs = R["routing_runs"]
    marks = {"l1_diag": "s", "pen_l1": "D", "free": "o", "pen_l2": "^", "l2_diag": "v"}
    for c in conds:
        rr = [r for r in runs if r["label"] == c and r["kl"] < 0.003]
        ax.scatter([r["attn2_t1_to_t"] for r in rr], [r["phi_L1{t}"] for r in rr], s=44, marker=marks[c],
                   color=SERIES[COND.index(c) % len(SERIES)], edgecolor="white", lw=0.8, alpha=0.9,
                   label=COND_LABEL[c], zorder=3)
    ax.set_xlabel("block-2 attention, t+1 → t (the fetch route)")
    ax.set_ylabel("single-node effect at t")
    ax.set_title(f"r = {A.get('corr_attn2_vs_phi_t', float('nan')):.2f}: the node effect is the route share", fontsize=10)
    ax.grid(True); ax.legend(fontsize=7)
    _save(fig, out / "fig1_routing.png")


def _cells(A, norm, delta):
    rows = [(float(k.split("|")[1][1:]), v) for k, v in A["cells"].items()
            if k.split("|")[0] == norm and abs(float(k.split("|")[2][1:]) - delta) < 1e-9]
    return sorted(rows)


def fig_factorization(R, out):
    A = R["factor"]; d = 64; delta = 0.4
    req = float(2 * np.log((0.5 + delta) / (0.5 - delta)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), layout="constrained")
    for norm in ("none", "frozen", "learn"):
        rows = _cells(A, norm, delta)
        if not rows:
            continue
        c = np.array([g for g, _ in rows]); col, mk, ls = NORM_STYLE[norm]
        D = np.array([v["D"] for _, v in rows]); cos = np.array([abs(v["cos_theta"]) for _, v in rows])
        C = np.array([v["C"] for _, v in rows]); conv = np.array([v["converged"] for _, v in rows])
        for ax, y, lab in ((axes[0], D, "D = ‖Δh̃‖"), (axes[1], cos, "cos θ"), (axes[2], C / req, "C / C*")):
            ax.plot(c, y, ls, color=col, marker=mk, ms=5, lw=1.6, label=NORM_LABEL[norm])
            ax.scatter(c[conv < 1], y[conv < 1], s=70, facecolors="none", edgecolors="k", lw=1.0, zorder=4)
    cc = np.logspace(np.log10(0.09), np.log10(2.6), 50)
    axes[0].plot(cc, 2 * np.sqrt(d) * np.ones_like(cc), ":", color=MUTED, lw=1.4)
    axes[0].text(0.5, 2 * np.sqrt(d) * 1.06, "ceiling 2√d (frozen LN)", color=MUTED, fontsize=7)
    axes[0].plot(cc, 4.4 / (cc * 1.4), ":", color=TEXT, lw=1.0)
    axes[0].text(0.35, 4.4 / (0.35 * 1.4) * 1.1, "slope −1", color=TEXT, fontsize=7)
    axes[0].set_yscale("log"); axes[0].set_ylabel("D = ‖Δh̃‖ at the interface")
    axes[1].set_ylabel("|cos θ|"); axes[1].set_ylim(0, 1.05)
    axes[2].axhline(1.0, color=MUTED, lw=1.0, ls=":")
    axes[2].plot(cc, np.minimum(4 * cc * np.sqrt(d) / (2 * np.log(0.9 / 0.1)), 1.3), "-", color=MUTED, lw=1.2)
    axes[2].text(0.105, 0.35, "ceiling 4c√d / C*", color=MUTED, fontsize=7, rotation=52)
    axes[2].set_ylabel("achieved contrast C / C*")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("fixed readout row gain c")
        ax.axvline(A["c_crit"]["0.4"] if "0.4" in A["c_crit"] else A["c_crit"][0.4], color=TEXT, lw=1.0, ls="--")
        ax.grid(True)
    axes[0].legend(fontsize=7); axes[0].set_title("δ = 0.4; rings: not every seed reached the floor", fontsize=9)
    axes[1].set_title("dashed line: predicted boundary c_crit", fontsize=9)
    axes[2].set_title("the contrast the function requires", fontsize=9)
    _save(fig, out / "fig2_factorization.png")


def fig_boundary(R, out):
    """Where the frozen-LayerNorm models actually stop reaching the floor, against the formal
    ceiling 4c√d and the ceiling implied by the g and D that training actually attains."""
    A = R["factor"]; d = 64
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), layout="constrained")
    for ax, delta in zip(axes, (0.4, 0.2)):
        req = float(2 * np.log((0.5 + delta) / (0.5 - delta)))
        att = A.get("attained", {}).get(f"d{delta:g}", {})
        for norm in ("none", "frozen", "learn"):
            rows = _cells(A, norm, delta)
            if not rows:
                continue
            c = np.array([g for g, _ in rows]); col, mk, ls = NORM_STYLE[norm]
            C = np.maximum(np.array([v["C"] for _, v in rows]), 1e-3)   # degenerate runs reach ~0
            conv = np.array([v["converged"] for _, v in rows])
            ax.plot(c, C, ls, color=col, marker=mk, ms=5, lw=1.6, label=NORM_LABEL[norm])
            ax.scatter(c[conv >= 1], C[conv >= 1], s=80, facecolors="none", edgecolors=TEXT, lw=1.2, zorder=4)
        cc = np.logspace(np.log10(0.028 if delta < 0.4 else 0.09), np.log10(2.6), 60)
        ax.plot(cc, 4 * cc * np.sqrt(d), "-", color=MUTED, lw=1.3, label="formal ceiling 4c√d")
        if att:
            ax.plot(cc, att["kappa"] * cc, ":", color=MUTED, lw=1.6,
                    label=f"attained ceiling = {att['kappa']:.1f}c")
        ax.axhline(req, color=TEXT, lw=1.0, ls=":"); ax.text(cc[0], req * 1.08, "C*(δ)", fontsize=7, color=TEXT)
        ax.axvline(req / (4 * np.sqrt(d)), color=TEXT, lw=1.0, ls="--")
        if att and att.get("c_attained"):
            ax.axvline(att["c_attained"], color=MUTED, lw=1.0, ls="-.")
        if A.get("boundary_observed", {}).get(f"frozen_d{delta:g}"):
            ax.axvline(A["boundary_observed"][f"frozen_d{delta:g}"], color=SERIES[1], lw=1.4, ls="-", alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("fixed readout row gain c")
        ax.set_ylabel("achieved contrast C"); ax.grid(True); ax.set_ylim(max(1e-3, 0.02 * req), None)
        ax.set_title(f"δ = {delta}: c_crit = {req / (4 * np.sqrt(d)):.3f} (dashed), attained (dash-dot),\n"
                     f"observed frozen boundary (orange); rings = reached the floor", fontsize=8)
    axes[0].legend(fontsize=7, loc="lower right")
    _save(fig, out / "fig3_boundary.png")


def make_figures(R, out):
    if "routing" in R and "routing_runs" in R:
        fig_routing(R, out)
    if "factor" in R:
        fig_factorization(R, out); fig_boundary(R, out)
