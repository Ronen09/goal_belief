"""Figures for TASK2 (results2/)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .analysis import HIDDEN_LAYERS
from .plotting import SERIES, MUTED, HYP_COLOR, _save

REG_COLOR = {"occupancy": SERIES[0], "SR": SERIES[3], "policy": SERIES[4], "advantage": SERIES[2],
             "margin": SERIES[5], "spatial": SERIES[1], "value": SERIES[6]}
L3 = ["h1", "h2", "h3"]


def plot_sweep_ground_truth(gt: dict, path: str, n_curves: int = 40):
    lams = np.array(gt["lams"]); mid = (lams[:-1] + lams[1:]) / 2
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.0))
    sm = gt["signed_margin"]                       # [L, K, S]
    K, S = sm.shape[1:]
    rng = np.random.default_rng(0)
    idx = [(g, s) for g in range(K) for s in range(S) if len(gt["boundaries"][(g, s)]) > 0]
    for g, s in (idx if len(idx) <= n_curves else [idx[i] for i in rng.choice(len(idx), n_curves, replace=False)]):
        axes[0].plot(lams, sm[:, g, s], color=SERIES[0], lw=0.8, alpha=0.45)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_xlabel("λ (hazard failure probability)"); axes[0].set_ylabel("signed margin (λ=0 action − best other)")
    axes[0].set_title(f"(s,g) pairs that switch policy ({len(idx)} of {K*S})")
    occ_ch = np.linalg.norm(gt["Z_sa"][1:] - gt["Z_sa"][:-1], axis=(1, 2))
    sr_ch = np.linalg.norm(gt["SR"][1:] - gt["SR"][:-1], axis=(1, 2))
    axes[1].plot(mid, occ_ch / occ_ch.max(), color=SERIES[0], lw=2, marker="o", ms=4, label="‖ΔZ_sa‖ (normalised)")
    axes[1].plot(mid, sr_ch / sr_ch.max(), color=SERIES[3], lw=2, marker="o", ms=4, label="‖ΔSR‖ (normalised)")
    axes[1].set_xlabel("λ"); axes[1].set_title("Ground-truth change per λ step"); axes[1].legend(fontsize=7.5); axes[1].grid(True, axis="y")
    axes[2].bar(mid, gt["n_flips"], width=0.04, color=SERIES[4], edgecolor="white")
    axes[2].set_xlabel("λ"); axes[2].set_title("(s,g) pairs whose optimal action set changes"); axes[2].grid(True, axis="y")
    _save(fig, path)


def plot_change_across_lambda(agg: dict, path: str, layer: str = "h2"):
    """agg[measure] = dict(mid=…, mean=…, floor=…); plus agg['occ'], agg['n_flips']."""
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(13, 3.2))
    mid = np.array(agg["mid"])
    for key, lab, col in [("chain", "warm-start chain ‖Δh‖", SERIES[0]), ("procrustes", "Procrustes-aligned ‖Δh‖ (÷ between-seed distance)", SERIES[2]),
                          ("rdm", "RDM change (1 − ρ)", SERIES[4])]:
        m = np.array(agg[layer][key]["mean"]); f = np.array(agg[layer][key]["floor"])
        axes[0].plot(mid, m / np.maximum(f, 1e-12), color=col, lw=2, marker="o", ms=4, label=lab)
    axes[0].axhline(1, color=MUTED, lw=0.8, ls="--")
    axes[0].set_xlabel("λ"); axes[0].set_ylabel("change / reference (same-λ retrain or between-seed)"); axes[0].set_title(f"Hidden change per λ step ({layer})"); axes[0].legend(fontsize=7.5); axes[0].grid(True, axis="y")
    ax2 = axes[1]
    occ = np.array(agg["occ_change"]); ax2.plot(mid, occ / occ.max(), color=SERIES[0], lw=2, marker="o", ms=4, label="‖ΔZ_sa‖ (norm.)")
    nf = np.array(agg["n_flips"]); ax2.bar(mid, nf / max(nf.max(), 1), width=0.04, color=SERIES[4], alpha=0.5, label="action flips (norm.)")
    ch = np.array(agg[layer]["chain"]["mean"]); ax2.plot(mid, ch / ch.max(), color=SERIES[1], lw=2, marker="s", ms=4, label=f"chain ‖Δh‖ (norm., {layer})")
    ax2.set_xlabel("λ"); ax2.set_title("Hidden change vs ground-truth change"); ax2.legend(fontsize=7.5); ax2.grid(True, axis="y")
    ax3 = axes[2]
    cats = ["flipped (s,g)", "same state, other goal", "all other (s,g)"]
    x = np.arange(len(cats)); w = 0.26
    for k, (key, lab, col) in enumerate([("chain", "chain", SERIES[0]), ("procrustes", "Procrustes", SERIES[2])]):
        vals = [agg[layer][key][c] for c in ("flipped", "same_state", "other")]
        ax3.bar(x + (k - 0.5) * w, vals, w, color=col, edgecolor="white", label=lab)
    ax3.set_xticks(x); ax3.set_xticklabels(cats, fontsize=8); ax3.set_ylabel("mean per-(s,g) change / noise floor")
    ax3.set_title(f"Where does the change land? ({layer})"); ax3.legend(fontsize=7.5); ax3.grid(True, axis="y")
    _save(fig, path)


def plot_change_scatter(tab: dict, path: str, layer: str = "h2"):
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.5, 3.2), sharey=True)
    flip = tab["flip"].astype(bool)
    for ax, key, xl in [(axes[0], "d_occ", "‖Δρ(g|s,·)‖ per (s,g)"), (axes[1], "d_margin", "|Δ margin| per (s,g)")]:
        ax.scatter(tab[key][~flip], tab["d_h"][~flip], s=6, color=SERIES[0], alpha=0.35, ec="none", label="action unchanged")
        ax.scatter(tab[key][flip], tab["d_h"][flip], s=10, color=SERIES[1], alpha=0.8, ec="none", label="action changed")
        ax.set_xlabel(xl); ax.grid(True)
    axes[0].set_ylabel(f"chain ‖Δh‖ ({layer}) / noise floor"); axes[0].legend(fontsize=7.5)
    fig.suptitle("Per-(s,g) hidden change across successive λ", fontsize=10, fontweight="bold")
    _save(fig, path)


def plot_quotient_pairs(reports: dict[str, list[dict]], path: str, layers=("h2", "h3")):
    envs = list(reports)
    fig, axes = plt.subplots(1, len(layers), layout="constrained", figsize=(5.5 * len(layers), 3.2), sharey=True)
    x = np.arange(len(envs)); w = 0.2
    for ax, layer in zip(np.atleast_1d(axes), layers):
        for k, (key, lab, col, alpha) in enumerate([("d_h_A", "case A: same actions, far occupancy", SERIES[0], 1.0),
                                                     ("d_h_B", "case B: near occupancy, different actions", SERIES[1], 1.0),
                                                     ("d_h_A_random", "case A, random init", SERIES[0], 0.4),
                                                     ("d_h_B_random", "case B, random init", SERIES[1], 0.4)]):
            vals = [np.mean([r[layer][key] for r in reports[e]]) for e in envs]
            ax.bar(x + (k - 1.5) * w, vals, w, color=col, alpha=alpha, edgecolor="white", label=lab)
        ax.set_xticks(x); ax.set_xticklabels(envs); ax.set_title(f"Hidden distance / median ({layer})"); ax.grid(True, axis="y")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, path)


def plot_extended_regression(reports: dict[str, list[dict]], path: str, regressors=("occupancy", "SR", "policy", "advantage", "margin", "spatial")):
    envs = list(reports)
    ncol = min(5, len(envs)); nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, layout="constrained", figsize=(3.0 * ncol, 3.0 * nrow), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(envs):]:
        ax.axis("off")
    x = np.arange(len(L3)); w = 0.8 / len(regressors)
    for ax, e in zip(axes, envs):
        for k, reg in enumerate(regressors):
            vals = [np.mean([r[l]["regression"][f"beta_{reg}"] for r in reports[e]]) for l in L3]
            ax.bar(x + (k - len(regressors) / 2 + 0.5) * w, vals, w, color=REG_COLOR[reg], edgecolor="white", label=reg)
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(L3); ax.set_title(e); ax.grid(True, axis="y")
    axes[0].set_ylabel("standardised β"); axes[len(envs) - 1].legend(fontsize=7)
    fig.suptitle("Six-way RDM regression of hidden-state distances", fontsize=10, fontweight="bold")
    _save(fig, path)


def plot_ablation(reports: dict[str, list[dict]], path: str):
    conds = [("baseline", "baseline", SERIES[6]), ("remove_component", "− h_sg (exact)", SERIES[0]), ("random_component", "− random, same norm", SERIES[0]),
             ("remove_subspace", "− top-k subspace of h_sg", SERIES[4]), ("random_subspace", "− random k-subspace", SERIES[4]),
             ("only_interaction_plus_state", "keep h_s + h_sg only (drop h_g)", SERIES[2])]
    metrics = [("accuracy", "action accuracy"), ("goal_sensitivity", "goal sensitivity"), ("steer_success", "steering success (α=1)")]
    envs = list(reports)
    fig, axes = plt.subplots(len(envs), 3, layout="constrained", figsize=(12.5, 3.0 * len(envs)), squeeze=False)
    x = np.arange(len(L3)); w = 0.8 / len(conds)
    for i, e in enumerate(envs):
        for j, (mk, ml) in enumerate(metrics):
            ax = axes[i, j]
            for k, (ck, cl, col) in enumerate(conds):
                vals = [np.mean([r[l][ck][mk] for r in reports[e]]) for l in L3]
                ax.bar(x + (k - len(conds) / 2 + 0.5) * w, vals, w, color=col, alpha=1.0 if "random" not in ck else 0.4, edgecolor="white", label=cl)
            ax.set_xticks(x); ax.set_xticklabels(L3); ax.set_title(f"{e}: {ml}"); ax.grid(True, axis="y"); ax.set_ylim(0, 1.02)
    axes[0, -1].legend(fontsize=6.5)
    _save(fig, path)


def plot_steering_predictors(summ: dict[str, list[dict]], path: str):
    preds = [("alpha_lin", "linearised boundary crossing"), ("logit_margin", "network logit margin"), ("m1", "occupancy margin g1"),
             ("m2", "occupancy margin g2"), ("d_occ", "‖ρ(g2|s,·) − ρ(g1|s,·)‖"), ("occ_model", "occupancy model m1/(m1+m2)")]
    envs = list(summ)
    fig, axes = plt.subplots(1, len(envs), layout="constrained", figsize=(5.5 * len(envs), 3.2), sharey=True)
    x = np.arange(len(L3)); w = 0.8 / len(preds)
    for ax, e in zip(np.atleast_1d(axes), envs):
        for k, (pk, pl) in enumerate(preds):
            vals = [np.nanmean([r[l][f"spearman_{pk}"] for r in summ[e]]) for l in L3]
            ax.bar(x + (k - len(preds) / 2 + 0.5) * w, vals, w, color=SERIES[k], edgecolor="white", label=pl)
        ax.axhline(0, color=MUTED, lw=0.8); ax.set_xticks(x); ax.set_xticklabels(L3); ax.set_title(f"{e}: Spearman(α*, predictor)"); ax.grid(True, axis="y")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, path)


def plot_stochastic_tracking(track: dict, path: str):
    lams = np.array(track["lams"])
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.0), sharey=True)
    for ax, layer in zip(axes, L3):
        t = track[layer]
        ax.plot(lams, t["rsa_policy_lam"], color=SERIES[4], lw=2, marker="o", ms=4, label="policy table at λ")
        ax.plot(lams, t["rsa_policy_det"], color=SERIES[3], lw=2, marker="o", ms=4, label="policy table at λ=0 (deterministic)")
        ax.plot(lams, t["rsa_occ_lam"], color=SERIES[0], lw=1.5, marker="s", ms=3, label="occupancy at λ")
        ax.plot(lams, t["rsa_occ_det"], color=SERIES[2], lw=1.5, marker="s", ms=3, label="occupancy at λ=0")
        ax.plot(lams, t["partial_policy_lam_given_det"], color=SERIES[4], lw=1.2, ls="--", label="partial: policy λ | policy 0")
        ax.set_title(f"{layer}"); ax.set_xlabel("λ of the trained network"); ax.grid(True, axis="y")
    axes[0].set_ylabel("RSA with hidden geometry"); axes[-1].legend(fontsize=6.5)
    fig.suptitle("Does hidden geometry track π*_λ (stochastic-optimal) or π*_0 (deterministic)?", fontsize=10, fontweight="bold")
    _save(fig, path)


def plot_dimensionality(table: dict[str, dict], path: str):
    envs = list(table)
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.3))
    for ax, (xk, xl) in zip(axes[:2], [("occupancy", "PR of Z_sa"), ("policy", "PR of policy table")]):
        xs = [table[e][xk]["participation_ratio"] for e in envs]
        ys = [table[e]["h3"]["participation_ratio"] for e in envs]
        ax.scatter(xs, ys, color=SERIES[0], s=36, ec="white", label="h3 trained")
        for e, xv, yv in zip(envs, xs, ys):
            ax.annotate(e, (xv, yv), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        from scipy.stats import spearmanr
        r = spearmanr(xs, ys).correlation
        ax.set_xlabel(xl); ax.set_ylabel("PR of hidden h3 (trained)"); ax.set_title(f"ρ = {r:+.2f} across environments"); ax.grid(True)
    ax = axes[2]
    x = np.arange(len(envs)); w = 0.15
    for k, (key, lab, col) in enumerate([("occupancy", "Z_sa", SERIES[0]), ("policy", "policy table", SERIES[4]), ("SR", "SR", SERIES[3]),
                                         ("h3", "h3 trained", SERIES[1]), ("h3_random", "h3 random", SERIES[6])]):
        ax.bar(x + (k - 2) * w, [table[e][key]["entropy_rank"] for e in envs], w, color=col, edgecolor="white", label=lab)
    ax.set_xticks(x); ax.set_xticklabels(envs, fontsize=7, rotation=20); ax.set_ylabel("entropy effective rank"); ax.legend(fontsize=6.5); ax.grid(True, axis="y")
    ax.set_title("Effective rank per representation")
    _save(fig, path)
