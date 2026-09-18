"""Matplotlib figures for every phase. Colours follow one fixed categorical
order per hypothesis so the same hypothesis has the same colour in every figure."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from .gridworld import GridWorld, DELTAS
from .analysis import HIDDEN_LAYERS

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6b6b6b"]
HYP_COLOR = {
    "occupancy": SERIES[0], "spatial": SERIES[1], "geodesic": SERIES[2], "SR": SERIES[3],
    "occupancy_state": SERIES[4], "occupancy_log": SERIES[5], "identity": SERIES[6], "policy": SERIES[4],
    "trained": SERIES[0], "random": SERIES[6],
}
BLUE_RAMP = LinearSegmentedColormap.from_list("blue", ["#f4f8fe", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f6c7b3", "#e6e6e6", "#b7d3f6", "#2a78d6"])
TEXT, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e2e2e2"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_env(env: GridWorld, sol, goal_idx: int, path: str, title: str | None = None):
    fig, ax = plt.subplots(layout="constrained", figsize=(4.2, 4.2 * env.height / env.width))
    ax.imshow(np.where(env.grid, 1.0, 0.0), cmap=ListedColormap(["#3a3a3a", "#ffffff"]), vmin=0, vmax=1)
    for (r, c) in env.goal_coords:
        ax.add_patch(plt.Circle((c, r), 0.18, color="#cde2fb", ec=SERIES[0], lw=1.2))
    g = env.goal_states[goal_idx]
    gr, gc = env.coords_of(g)
    ax.add_patch(plt.Circle((gc, gr), 0.3, color=SERIES[1]))
    for s in range(env.n_states):
        if s == g:
            continue
        r, c = env.coords_of(s)
        for a in range(4):
            p = sol.pi[goal_idx, s, a]
            if p > 0:
                dr, dc = DELTAS[a]
                ax.annotate("", xy=(c + 0.32 * dc, r + 0.32 * dr), xytext=(c, r),
                            arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=0.8 * p + 0.4, alpha=0.5 + 0.5 * p))
    for (r, c), (dr_, dc_) in env.portals.items():
        ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, fill=False, ec=SERIES[4], lw=2))
        ax.annotate("", xy=(dc_, dr_), xytext=(c, r), arrowprops=dict(arrowstyle="->", color=SERIES[4], lw=1.2, ls="--"))
    for e in env.blocked_edges:
        (r1, c1), (r2, c2) = sorted(e)
        if r1 == r2:
            ax.plot([c1 + 0.5, c1 + 0.5], [r1 - 0.5, r1 + 0.5], color=SERIES[1], lw=3)
        else:
            ax.plot([c1 - 0.5, c1 + 0.5], [r1 + 0.5, r1 + 0.5], color=SERIES[1], lw=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f"{env.name}: goals (light) and π* for goal at {(gr, gc)} (orange)")
    _save(fig, path)


def plot_occupancy_geometry(env: GridWorld, occ, summary: dict, path: str):
    from scipy.cluster.hierarchy import leaves_list
    from . import geometry as G
    Z = occ.Z_sa
    order = leaves_list(summary["linkage"])
    R = G.rdm(Z)[np.ix_(order, order)]
    fig, axes = plt.subplots(1, 4, layout="constrained", figsize=(14, 3.4))
    im = axes[0].imshow(R, cmap=BLUE_RAMP)
    axes[0].set_title("Z_sa distances (cluster order)"); axes[0].set_xlabel("state"); axes[0].set_ylabel("state")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    ev = np.array(summary["explained_variance"])
    axes[1].plot(np.arange(1, len(ev) + 1), np.cumsum(ev), color=SERIES[0], lw=2, marker="o", ms=3)
    axes[1].axhline(0.9, color=MUTED, lw=0.8, ls="--")
    axes[1].set_xlim(0.5, min(20, len(ev)) + 0.5); axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("principal component"); axes[1].set_ylabel("cumulative variance")
    axes[1].set_title(f"PCA of Z_sa (PR {summary['participation_ratio']:.1f}, 90% at {summary['n_pcs_90pct']} PCs)")
    axes[1].grid(True, axis="y")
    Zc = Z - Z.mean(0); U, s, _ = np.linalg.svd(Zc, full_matrices=False)
    pcs = U[:, :2] * s[:2]
    sc = axes[2].scatter(pcs[:, 0], pcs[:, 1], c=env.coords[:, 0] * env.width + env.coords[:, 1], cmap=BLUE_RAMP, s=28, ec="white", lw=0.5)
    axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2"); axes[2].set_title("States in PC1–PC2 (shade = row-major position)")
    cmap = ListedColormap(SERIES[:6])
    axes[3].imshow(env.render(summary["cluster_labels"] - 1), cmap=cmap, vmin=-0.5, vmax=5.5)
    axes[3].set_title("Ward clusters of Z_sa on the grid"); axes[3].set_xticks([]); axes[3].set_yticks([])
    for (r, c) in env.goal_coords:
        axes[3].add_patch(plt.Circle((c, r), 0.15, color="white", ec=TEXT, lw=0.8))
    _save(fig, path)


def plot_training(histories: list, path: str):
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8, 2.8))
    for i, h in enumerate(histories):
        axes[0].plot(h.loss, color=SERIES[0], alpha=0.35 + 0.65 * (i == 0), lw=1.2)
        axes[1].plot(h.acc, color=SERIES[0], alpha=0.35 + 0.65 * (i == 0), lw=1.2)
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_title("BC loss (soft cross-entropy)"); axes[1].set_title("Argmax-in-optimal-set accuracy")
    axes[0].set_xlabel("step"); axes[1].set_xlabel("step"); axes[1].set_ylim(0, 1.02)
    for ax in axes: ax.grid(True, axis="y")
    _save(fig, path)


def _agg(reports: list[dict], layer: str, key: str):
    vals = np.array([r[layer][key] for r in reports], dtype=float)
    return vals.mean(), vals.std()


def plot_layer_rsa(trained: list[dict], random: list[dict], path: str, hyps=("occupancy", "occupancy_log", "policy", "spatial", "geodesic", "SR", "identity")):
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.2))
    x = np.arange(len(HIDDEN_LAYERS))
    for h in hyps:
        m = [ _agg(trained, l, f"rsa_{h}") for l in HIDDEN_LAYERS ]
        axes[0].errorbar(x, [a for a, _ in m], [b for _, b in m], color=HYP_COLOR[h], lw=2, marker="o", ms=5, label=h, capsize=2)
        m = [ _agg(random, l, f"rsa_{h}") for l in HIDDEN_LAYERS ]
        axes[0].plot(x, [a for a, _ in m], color=HYP_COLOR[h], lw=1, ls="--", marker="o", ms=3, alpha=0.6)
    axes[0].set_title("RSA (Spearman) with each hypothesis\nsolid = trained, dashed = random init")
    axes[0].set_ylabel("Spearman ρ"); axes[0].legend(ncol=2, fontsize=7.5)
    for key, lab, col in [("partial_occ_given_spatial", "occupancy | spatial", HYP_COLOR["occupancy"]),
                          ("partial_spatial_given_occ", "spatial | occupancy", HYP_COLOR["spatial"]),
                          ("partial_occ_given_geodesic", "occupancy | geodesic", SERIES[5]),
                          ("partial_geodesic_given_occ", "geodesic | occupancy", HYP_COLOR["geodesic"])]:
        m = [ _agg(trained, l, key) for l in HIDDEN_LAYERS ]
        axes[1].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].set_title("Partial RSA (trained)"); axes[1].legend(fontsize=7.5)
    for key, lab, col, ls in [("decode_r2_occupancy", "→ Z_sa (trained)", HYP_COLOR["occupancy"], "-"),
                              ("decode_r2_spatial", "→ (x,y) (trained)", HYP_COLOR["spatial"], "-")]:
        m = [ _agg(trained, l, key) for l in HIDDEN_LAYERS ]
        axes[2].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2, ls=ls)
        m = [ _agg(random, l, key) for l in HIDDEN_LAYERS ]
        axes[2].plot(x, [a for a, _ in m], color=col, lw=1, ls="--", marker="o", ms=3, alpha=0.6)
    axes[2].axhline(0, color=MUTED, lw=0.8); axes[2].set_ylim(-0.6, 1.05)
    axes[2].set_title("Ridge decoding, held-out states (CV R²)\nsolid = trained, dashed = random init"); axes[2].legend(fontsize=7.5)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(HIDDEN_LAYERS); ax.grid(True, axis="y")
    _save(fig, path)


def plot_dimensionality(trained: list[dict], random: list[dict], occ_pr: float, path: str):
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.0))
    x = np.arange(len(HIDDEN_LAYERS))
    for reps, lab, ls in [(trained, "trained", "-"), (random, "random init", "--")]:
        m = [_agg(reps, l, "participation_ratio") for l in HIDDEN_LAYERS]
        axes[0].errorbar(x, [a for a, _ in m], [b for _, b in m], color=HYP_COLOR["trained" if ls == "-" else "random"], lw=2, ls=ls, marker="o", ms=5, label=lab, capsize=2)
        m = [_agg(reps, l, "cka_occupancy") for l in HIDDEN_LAYERS]
        axes[1].errorbar(x, [a for a, _ in m], [b for _, b in m], color=HYP_COLOR["occupancy"], lw=2, ls=ls, marker="o", ms=5, label=f"occupancy ({lab})", capsize=2)
        m = [_agg(reps, l, "cka_spatial") for l in HIDDEN_LAYERS]
        axes[1].errorbar(x, [a for a, _ in m], [b for _, b in m], color=HYP_COLOR["spatial"], lw=2, ls=ls, marker="o", ms=5, label=f"spatial ({lab})", capsize=2)
        m = [_agg(reps, l, "subspace_overlap_k3") for l in HIDDEN_LAYERS]
        axes[2].errorbar(x, [a for a, _ in m], [b for _, b in m], color=HYP_COLOR["trained" if ls == "-" else "random"], lw=2, ls=ls, marker="o", ms=5, label=f"top-3 ({lab})", capsize=2)
    axes[0].axhline(occ_pr, color=HYP_COLOR["occupancy"], lw=1.2, ls=":", label="Z_sa participation ratio")
    axes[0].set_title("Participation ratio of h̄(s)"); axes[0].legend(fontsize=7.5)
    axes[1].set_title("Linear CKA with Z_sa / (x,y)"); axes[1].legend(fontsize=7); axes[1].set_ylim(0, 1.02)
    axes[2].set_title("Overlap of top-k principal subspaces with Z_sa"); axes[2].legend(fontsize=7.5); axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(HIDDEN_LAYERS); ax.grid(True, axis="y")
    _save(fig, path)


def plot_emergence(steps: list[int], curves: dict[str, dict[str, list[float]]], acc: list[float], path: str, layer: str):
    """curves[metric][hyp] -> list over steps (already averaged over seeds)."""
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.0))
    xs = np.array(steps, dtype=float); xs[xs == 0] = 0.5
    for h in ("occupancy", "spatial", "geodesic", "SR"):
        axes[0].plot(xs, curves["rsa"][h], color=HYP_COLOR[h], lw=2, marker="o", ms=4, label=h)
    axes[0].set_xscale("log"); axes[0].set_title(f"RSA with hypotheses during training ({layer})"); axes[0].legend(fontsize=7.5)
    axes[1].plot(xs, curves["decode"]["occupancy"], color=HYP_COLOR["occupancy"], lw=2, marker="o", ms=4, label="→ Z_sa")
    axes[1].plot(xs, curves["decode"]["spatial"], color=HYP_COLOR["spatial"], lw=2, marker="o", ms=4, label="→ (x,y)")
    axes[1].set_xscale("log"); axes[1].set_title(f"Held-out-state decoding R² ({layer})"); axes[1].legend(fontsize=7.5); axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[2].plot(xs, acc, color=SERIES[5], lw=2, marker="o", ms=4)
    axes[2].set_xscale("log"); axes[2].set_title("BC accuracy at checkpoint"); axes[2].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xlabel("training step (0 = init)"); ax.grid(True, axis="y")
    _save(fig, path)


def plot_discriminating(results: dict, path: str, envs: list[str], encodings: list[str], layer: str = "h2"):
    """results[(env, enc)] = list over seeds of discriminating_report dicts."""
    fig, axes = plt.subplots(layout="constrained", nrows=1, ncols=3, figsize=(13, 4.0))
    w = 0.38
    xs = np.arange(len(envs))
    for j, enc in enumerate(encodings):
        for k, hyp in enumerate(("occupancy", "spatial", "policy")):
            vals = []
            for e in envs:
                reps = results[(e, enc)]
                vals.append(np.mean([r[layer]["policy_regression"][f"beta_{hyp}"] for r in reps]))
            off = (j - 0.5) * w + (k - 1) * (w / 3)
            axes[0].bar(xs + off, vals, w / 3, color=HYP_COLOR[hyp], alpha=1.0 if enc == "onehot" else 0.45,
                        label=f"{hyp}, {enc} input", edgecolor="white")
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(envs); axes[0].set_title(f"RDM regression β at {layer} (occupancy + spatial + policy)")
    axes[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for j, enc in enumerate(encodings):
        for k, hyp in enumerate(("occupancy", "spatial")):
            vals = [np.mean([r[layer][f"disagree_rsa_{hyp}"] for r in results[(e, enc)]]) for e in envs]
            off = (j - 0.5) * w + (k - 0.5) * (w / 2)
            axes[1].bar(xs + off, vals, w / 2, color=HYP_COLOR[hyp], alpha=1.0 if enc == "onehot" else 0.45, edgecolor="white")
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(envs); axes[1].set_title(f"Spearman on hypothesis-disagreement pairs ({layer})")
    # designed pairs
    ex = [e for e in envs if e != "base"]
    xs2 = np.arange(len(ex))
    for j, enc in enumerate(encodings):
        sp = [np.mean([r[layer]["special_pairs_median"] for r in results[(e, enc)]]) for e in ex]
        ctl = [np.mean([r[layer]["special_pairs_spatial_matched_control"] for r in results[(e, enc)]]) for e in ex]
        ctl2 = [np.nanmean([r[layer]["special_pairs_occ_matched_control"] for r in results[(e, enc)]]) for e in ex]
        off = (j - 0.5) * 0.4
        axes[2].bar(xs2 + off - 0.13, sp, 0.13, color=SERIES[0], alpha=1.0 if enc == "onehot" else 0.45, edgecolor="white", label=f"designed pairs ({enc})")
        axes[2].bar(xs2 + off, ctl, 0.13, color=SERIES[1], alpha=1.0 if enc == "onehot" else 0.45, edgecolor="white", label=f"same spatial dist. ({enc})")
        axes[2].bar(xs2 + off + 0.13, ctl2, 0.13, color=SERIES[2], alpha=1.0 if enc == "onehot" else 0.45, edgecolor="white", label=f"same occupancy dist. ({enc})")
    axes[2].set_xticks(xs2); axes[2].set_xticklabels(ex); axes[2].set_title(f"Designed pairs: activation distance / median ({layer})")
    axes[2].legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    for ax in axes: ax.grid(True, axis="y")
    _save(fig, path)


def plot_goal_dependence(reports: list[dict], path: str):
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.0))
    x = np.arange(len(HIDDEN_LAYERS))
    bottom = np.zeros(len(x))
    for comp, col in [("state", SERIES[0]), ("goal", SERIES[1]), ("interaction", SERIES[2])]:
        m = np.array([_agg(reports, l, comp)[0] for l in HIDDEN_LAYERS])
        axes[0].bar(x, m, 0.6, bottom=bottom, color=col, label=comp, edgecolor="white")
        bottom += m
    axes[0].set_title("Variance of h(s,g): state + goal + interaction"); axes[0].legend(fontsize=7.5); axes[0].set_ylim(0, 1)
    for key, lab, col in [("state_component_rsa_occupancy", "state comp. ~ Z_sa", HYP_COLOR["occupancy"]),
                          ("state_component_rsa_spatial", "state comp. ~ (x,y)", HYP_COLOR["spatial"]),
                          ("goal_component_rsa_goal_occupancy", "goal comp. ~ Z_sa(g)", SERIES[4]),
                          ("goal_component_rsa_goal_spatial", "goal comp. ~ (x,y)(g)", SERIES[3])]:
        m = [_agg(reports, l, key) for l in HIDDEN_LAYERS]
        axes[1].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    axes[1].set_title("RSA of additive components"); axes[1].legend(fontsize=7); axes[1].axhline(0, color=MUTED, lw=0.8)
    for key, lab, col in [("interaction_rsa_Q", "interaction ~ ρ(g|s,·)", HYP_COLOR["occupancy"]),
                          ("interaction_rsa_action", "interaction ~ π*(·|s,g)", SERIES[5]),
                          ("interaction_decode_r2_Q", "decode ρ(g|s,·) from interaction (R²)", SERIES[4])]:
        m = [_agg(reports, l, key) for l in HIDDEN_LAYERS]
        axes[2].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    axes[2].set_title("Interaction term vs goal-specific action relevance"); axes[2].legend(fontsize=7); axes[2].axhline(0, color=MUTED, lw=0.8)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(HIDDEN_LAYERS); ax.grid(True, axis="y")
    _save(fig, path)


def plot_steering(reports: list[dict], path: str):
    layers = list(reports[0]["layers"].keys())
    fig, axes = plt.subplots(1, len(layers), layout="constrained", figsize=(3.2 * len(layers), 3.0), sharey=True)
    alphas = reports[0]["alphas"]
    for ax, layer in zip(axes, layers):
        def m(key):
            return np.mean([r["layers"][layer][key] for r in reports], 0)
        ax.plot(alphas, m("agree_g2"), color=SERIES[0], lw=2, label="action optimal for g2")
        ax.plot(alphas, m("agree_g2_excl"), color=SERIES[0], lw=1.2, ls="-.", label="optimal for g2 only")
        ax.plot(alphas, m("agree_g1"), color=SERIES[1], lw=2, label="action optimal for g1")
        ax.plot(alphas, m("agree_g2_random_dir"), color=SERIES[6], lw=1.5, ls="--", label="random direction")
        ax.plot(alphas, m("agree_g2_other_goal_dir"), color=SERIES[3], lw=1.5, ls=":", label="other goal's direction")
        ax.set_title(f"patch at {layer}"); ax.set_xlabel("α"); ax.grid(True, axis="y"); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("fraction of differing states"); axes[-1].legend(fontsize=6.5, loc="center right")
    fig.suptitle("Adding α·E_s[h(s,g2) − h(s,g1)] while the input goal is g1", fontsize=10, fontweight="bold")
    _save(fig, path)


def plot_flip_vs_margin(reports: list[dict], path: str, layer: str = "h2"):
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8.5, 3.0), sharey=True)
    fa = np.concatenate([r["layers"][layer]["flip_alpha"] for r in reports])
    mg = np.concatenate([r["layers"][layer]["margin"] for r in reports])
    ap = np.concatenate([r["layers"][layer]["alpha_pred"] for r in reports])
    fin = np.isfinite(fa)
    jit = np.random.default_rng(0).normal(0, 0.01, fin.sum())
    for ax, x, xl in [(axes[0], mg, "g2 occupancy margin  max_a ρ(g2|s,a) − ρ(g2|s,a*_g1)"),
                      (axes[1], ap, "occupancy-model prediction  α* = m1 / (m1 + m2)")]:
        ax.scatter(x[fin], fa[fin] + jit, s=10, color=SERIES[0], alpha=0.5, ec="none")
        ax.scatter(x[~fin], np.full((~fin).sum(), 2.1), s=10, color=SERIES[1], alpha=0.5, ec="none", label="never flipped (α ≤ 2)")
        ax.set_xlabel(xl); ax.grid(True)
    axes[1].plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", label="identity")
    axes[0].set_ylabel("α at first g2-optimal action"); axes[0].set_title(f"Flip threshold vs g2 margin ({layer})")
    axes[1].set_title(f"Flip threshold vs occupancy-model prediction ({layer})"); axes[1].legend(fontsize=7)
    _save(fig, path)


def plot_stochastic(reports: list[dict], path: str, variant: str = ""):
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(12.5, 3.0))
    x = np.arange(len(HIDDEN_LAYERS))
    for key, lab, col in [("rsa_occupancy", "stochastic occupancy (exact)", SERIES[0]),
                          ("rsa_shortest_path", "shortest-path γ^T", SERIES[1]),
                          ("rsa_spatial", "spatial", SERIES[2]),
                          ("rsa_policy_stochastic", "policy table (stochastic-optimal)", SERIES[4]),
                          ("rsa_policy_shortest", "policy table (deterministic)", SERIES[3])]:
        m = [_agg(reports, l, key) for l in HIDDEN_LAYERS]
        axes[0].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    axes[0].set_title(f"RSA, network trained on '{variant}'"); axes[0].legend(fontsize=7)
    for key, lab, col in [("partial_polstoch_given_poldet", "stochastic policy | deterministic policy", SERIES[4]),
                          ("partial_poldet_given_polstoch", "deterministic policy | stochastic policy", SERIES[3])]:
        m = [_agg(reports, l, key) for l in HIDDEN_LAYERS]
        axes[2].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    axes[2].axhline(0, color=MUTED, lw=0.8); axes[2].set_title("Partial RSA: which policy table?"); axes[2].legend(fontsize=7)
    for key, lab, col in [("partial_stoch_given_sp", "stochastic | shortest-path", SERIES[0]),
                          ("partial_sp_given_stoch", "shortest-path | stochastic", SERIES[1]),
                          ("decode_r2_stochastic", "decode R² stochastic", SERIES[4]),
                          ("decode_r2_shortest_path", "decode R² shortest-path", SERIES[3])]:
        m = [_agg(reports, l, key) for l in HIDDEN_LAYERS]
        axes[1].errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].set_title("Partial RSA and decoding"); axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(HIDDEN_LAYERS); ax.grid(True, axis="y")
    _save(fig, path)


def plot_pair_level(reports: list[dict], path: str):
    fig, ax = plt.subplots(layout="constrained", figsize=(6, 3.0))
    x = np.arange(len(HIDDEN_LAYERS))
    for key, lab, col in [("rsa_pair_occupancy", "[Z_sa(s), ρ(g|s,·)]", SERIES[0]),
                          ("rsa_state_occupancy_only", "Z_sa(s) only", SERIES[4]),
                          ("rsa_pair_Q", "ρ(g|s,·) only", SERIES[2]),
                          ("rsa_pair_spatial", "[(x,y)(s), (x,y)(g)]", SERIES[1]),
                          ("rsa_pair_action", "π*(·|s,g)", SERIES[5]),
                          ("rsa_goal_identity_only", "goal identity", SERIES[6])]:
        m = [_agg(reports, l, key) for l in HIDDEN_LAYERS]
        ax.errorbar(x, [a for a, _ in m], [b for _, b in m], color=col, lw=2, marker="o", ms=5, label=lab, capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(HIDDEN_LAYERS); ax.grid(True, axis="y"); ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("RSA over (s, g) pairs with pair-level models"); ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_gamma_sweep(gammas, curves: dict[str, list[float]], path: str, trained_gamma: float):
    """curves[layer] -> RSA with occupancy RDM computed at each gamma."""
    fig, ax = plt.subplots(layout="constrained", figsize=(4.6, 3.0))
    for i, (layer, vals) in enumerate(curves.items()):
        ax.plot(gammas, vals, color=SERIES[i], lw=2, marker="o", ms=4, label=layer)
    ax.axvline(trained_gamma, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("γ used to build the occupancy RDM"); ax.set_ylabel("RSA with h̄(s)")
    ax.set_title("Which discount best matches the network?"); ax.legend(fontsize=7.5); ax.grid(True, axis="y")
    _save(fig, path)
