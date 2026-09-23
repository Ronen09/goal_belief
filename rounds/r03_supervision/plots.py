"""Figures for TASK3 (rounds/r03_supervision/)."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from goalgeo.plotting import SERIES, MUTED, _save

MODEL_COLOR = {"target": SERIES[4], "policy": SERIES[5], "occupancy": SERIES[0], "advantage": SERIES[2],
               "SR": SERIES[3], "spatial": SERIES[1], "Q": SERIES[0], "occupancy_state": SERIES[0], "policy_row": SERIES[5], "advantage_row": SERIES[2]}


def _xt(conds):
    return [("hard" if c[0] == "hard" else f"τ={c[1]:g}") for c in conds]


def plot_temperature_sweep(conds, rep, path, layers=("h2", "h3")):
    """rep[(kind,tau)] = list over seeds of geometry_report dicts."""
    fig, axes = plt.subplots(1, len(layers) + 1, layout="constrained", figsize=(4.3 * (len(layers) + 1), 3.2))
    x = np.arange(len(conds))
    for ax, l in zip(axes, layers):
        for m in ("target", "policy", "occupancy", "advantage", "SR", "spatial"):
            ys = [np.mean([r[l][f"rsa_{m}"] for r in rep[c]]) for c in conds]
            es = [np.std([r[l][f"rsa_{m}"] for r in rep[c]]) for c in conds]
            ax.errorbar(x, ys, es, color=MODEL_COLOR[m], lw=2, marker="o", ms=4, label=m, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(_xt(conds), rotation=35, fontsize=7.5); ax.set_title(f"state-level RSA at {l}"); ax.grid(True, axis="y"); ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].legend(fontsize=7)
    ax = axes[-1]
    for l, col in zip(("h1", "h2", "h3"), SERIES[:3]):
        ax.plot(x, [np.mean([r[l]["participation_ratio"] for r in rep[c]]) for c in conds], color=col, lw=2, marker="o", ms=4, label=l)
    ax.set_xticks(x); ax.set_xticklabels(_xt(conds), rotation=35, fontsize=7.5); ax.set_title("participation ratio of h̄(s)"); ax.legend(fontsize=7.5); ax.grid(True, axis="y")
    _save(fig, path)


def plot_target_geometry(conds, rep, path, layer="h3"):
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(13, 3.2))
    x = np.arange(len(conds))
    for key, lab, col in [("rsa_target", "target (state level)", MODEL_COLOR["target"]), ("rsa_occupancy", "occupancy", MODEL_COLOR["occupancy"]),
                          ("pair_rsa_target", "target (pair level)", SERIES[6]), ("pair_rsa_Q", "Q (pair level)", SERIES[1])]:
        axes[0].plot(x, [np.mean([r[layer][key] for r in rep[c]]) for c in conds], color=col, lw=2, marker="o", ms=4, label=lab)
    axes[0].set_title(f"RSA with hidden geometry ({layer})"); axes[0].legend(fontsize=7)
    for key, lab, col in [("partial_target_given_env", "target | occupancy, SR, spatial, policy, advantage", MODEL_COLOR["target"]),
                          ("partial_occupancy_given_target", "occupancy | target", MODEL_COLOR["occupancy"])]:
        axes[1].plot(x, [np.mean([r[layer][key] for r in rep[c]]) for c in conds], color=col, lw=2, marker="o", ms=4, label=lab)
    axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].set_title(f"partial RSA ({layer})"); axes[1].legend(fontsize=7)
    for key, lab, col in [("r2", "with target", MODEL_COLOR["target"])]:
        axes[2].plot(x, [np.mean([r[layer]["regression"]["r2"] for r in rep[c]]) for c in conds], color=col, lw=2, marker="o", ms=4, label="R² with target regressor")
        axes[2].plot(x, [np.mean([r[layer]["regression_no_target"]["r2"] for r in rep[c]]) for c in conds], color=SERIES[6], lw=2, marker="o", ms=4, label="R² environment regressors only")
        axes[2].plot(x, [np.mean([r[layer]["regression"]["beta_target"] for r in rep[c]]) for c in conds], color=MODEL_COLOR["target"], lw=1.5, ls="--", marker="s", ms=3, label="β target")
    axes[2].set_title(f"RDM regression ({layer})"); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(_xt(conds), rotation=35, fontsize=7.5); ax.grid(True, axis="y")
    _save(fig, path)


def plot_counterfactual(synth, natural, conds, margins, path, layer="h3"):
    """synth[(kind,tau)][layer] = list over seeds of arrays d_h(probe m)/median; natural[(kind,tau)][layer] = list of counterfactual_report dicts."""
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(13, 3.3))
    cmap = plt.get_cmap("Blues")
    for i, c in enumerate(conds):
        ys = np.mean(synth[c][layer], 0)
        col = SERIES[1] if c[0] == "hard" else cmap(0.3 + 0.7 * i / max(len(conds) - 1, 1))
        axes[0].plot(np.abs(np.array(margins) - 0.5), ys, color=col, lw=2, marker="o", ms=4, label=_xt([c])[0])
    axes[0].set_xlabel("|margin(probe) − margin(reference)| (same argmax)"); axes[0].set_ylabel("hidden distance / median (background pairs)")
    axes[0].set_title(f"Synthetic fixed-argmax sweep ({layer})"); axes[0].legend(fontsize=6.5, ncol=2); axes[0].grid(True)
    x = np.arange(len(conds))
    for key, lab, col in [("low_dmargin_dist", "same argmax, small |Δmargin|", SERIES[0]), ("high_dmargin_dist", "same argmax, large |Δmargin|", SERIES[1])]:
        axes[1].plot(x, [np.mean([r[key] for r in natural[c][layer]]) for c in conds], color=col, lw=2, marker="o", ms=4, label=lab)
    axes[1].axhline(1, color=MUTED, lw=0.8, ls="--"); axes[1].set_title(f"Base grid: same-argmax row pairs ({layer})"); axes[1].legend(fontsize=7); axes[1].set_ylabel("hidden distance / median")
    for key, lab, col in [("spearman_dist_dmargin", "ρ(d_h, |Δ margin|)", SERIES[0]), ("spearman_dist_dtarget", "ρ(d_h, ‖Δ target‖)", MODEL_COLOR["target"])]:
        axes[2].plot(x, [np.mean([r[key] for r in natural[c][layer]]) for c in conds], color=col, lw=2, marker="o", ms=4, label=lab)
    axes[2].axhline(0, color=MUTED, lw=0.8); axes[2].set_title(f"Within same-argmax pairs ({layer})"); axes[2].legend(fontsize=7)
    for ax in axes[1:]:
        ax.set_xticks(x); ax.set_xticklabels(_xt(conds), rotation=35, fontsize=7.5); ax.grid(True, axis="y")
    _save(fig, path)


def plot_fixed_env(names, cross, rep, path, layer="h3"):
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 3.8))
    M = np.array([[cross[(a, b)] for b in names] for a in names])
    im = axes[0].imshow(M, vmin=0, vmax=1, cmap=plt.get_cmap("Blues"))
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=35, fontsize=7.5); axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(names, fontsize=7.5)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.6 else "#1f1f1f")
    axes[0].set_title(f"RSA between hidden geometries of conditions ({layer})\n(diagonal = between seeds)")
    x = np.arange(len(names)); w = 0.13
    for k, m in enumerate(("target", "policy", "occupancy", "advantage", "SR", "spatial")):
        axes[1].bar(x + (k - 2.5) * w, [np.mean([r[layer][f"rsa_{m}"] for r in rep[n]]) for n in names], w, color=MODEL_COLOR[m], edgecolor="white", label=m)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=35, fontsize=7.5); axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, axis="y")
    axes[1].set_title(f"RSA with each model ({layer}); same inputs, only T(Q) differs")
    _save(fig, path)


def plot_bottleneck(names, stats, hidden_pr, path):
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(13, 3.2))
    x = np.arange(len(names))
    axes[0].bar(x, [stats[n]["n_classes"] for n in names], color=SERIES[4], edgecolor="white"); axes[0].set_yscale("log"); axes[0].set_title("distinct target rows (equivalence classes)")
    w = 0.27
    for k, (key, lab, col) in enumerate([("recover_Q_r2", "Q", SERIES[0]), ("recover_adv_r2", "advantage", SERIES[2]), ("recover_occ_r2", "Z_sa (held-out states)", SERIES[3])]):
        axes[1].bar(x + (k - 1) * w, [stats[n][key] for n in names], w, color=col, edgecolor="white", label=lab)
    axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].set_title("ridge-CV recoverability from the target (R²)"); axes[1].legend(fontsize=7); axes[1].set_ylim(-0.3, 1.02)
    axes[2].plot(x, [stats[n]["participation_ratio"] for n in names], color=SERIES[4], lw=2, marker="o", ms=4, label="PR of target rows")
    axes[2].plot(x, [hidden_pr[n] for n in names], color=SERIES[1], lw=2, marker="s", ms=4, label="PR of hidden rows (h3)")
    axes[2].set_title("dimensionality of target versus hidden"); axes[2].legend(fontsize=7.5)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, fontsize=7.5); ax.grid(True, axis="y")
    _save(fig, path)


def plot_training(steps, traj, path, layer="h3"):
    """traj[cond][metric] = list over steps (seed-averaged)."""
    fig, axes = plt.subplots(1, 4, layout="constrained", figsize=(16, 3.2))
    xs = np.array(steps, float); xs[xs == 0] = 0.5
    cols = {c: (SERIES[1] if c[0] == "hard" else SERIES[0] if c[1] < 0.2 else SERIES[2]) for c in traj}
    for c, t in traj.items():
        lab = _xt([c])[0]
        axes[0].plot(xs, t["ratio"], color=cols[c], lw=2, marker="o", ms=3, label=lab)
        axes[1].plot(xs, np.array(t["within"]) / t["within"][0], color=cols[c], lw=2, marker="o", ms=3, label=f"{lab} within")
        axes[1].plot(xs, np.array(t["between"]) / t["between"][0], color=cols[c], lw=1.5, ls="--", marker="s", ms=3, label=f"{lab} between")
        axes[2].plot(xs, t["rsa_target"], color=cols[c], lw=2, marker="o", ms=3, label=f"{lab} target")
        axes[2].plot(xs, t["rsa_occupancy"], color=cols[c], lw=1.5, ls="--", marker="s", ms=3, label=f"{lab} occupancy")
        axes[3].plot(xs, t["participation_ratio"], color=cols[c], lw=2, marker="o", ms=3, label=lab)
    axes[0].set_title(f"within / between hard-class distance ({layer})"); axes[1].set_title("within and between, relative to init")
    axes[2].set_title("RSA with target and occupancy"); axes[3].set_title("participation ratio")
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("step"); ax.grid(True, axis="y")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=6); axes[2].legend(fontsize=6); axes[3].legend(fontsize=7)
    _save(fig, path)


def plot_generalisation(rep, path):
    names = list(rep)
    fig, axes = plt.subplots(1, 3, layout="constrained", figsize=(13, 3.2))
    x = np.arange(len(names)); w = 0.38
    for k, (split, col) in enumerate([("train", SERIES[0]), ("heldout", SERIES[1])]):
        axes[0].bar(x + (k - 0.5) * w, [np.mean([r[f"acc_{split}"] for r in rep[n]]) for n in names], w, color=col, edgecolor="white", label=split)
        axes[1].bar(x + (k - 0.5) * w, [np.mean([r[f"ratio_{split}"] for r in rep[n]]) for n in names], w, color=col, edgecolor="white", label=split)
    axes[2].bar(x - w / 2, [np.mean([r["centroid_acc_heldout"] for r in rep[n]]) for n in names], w, color=SERIES[1], edgecolor="white", label="held-out rows")
    axes[2].bar(x + w / 2, [np.mean([r["centroid_acc_chance"] for r in rep[n]]) for n in names], w, color=SERIES[6], edgecolor="white", label="chance")
    axes[0].set_title("argmax accuracy"); axes[1].set_title("within / between hard-class distance (h3)"); axes[2].set_title("nearest-class-centroid accuracy (h3)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, fontsize=7.5); ax.grid(True, axis="y"); ax.legend(fontsize=7)
    _save(fig, path)


def plot_architectures(archs, conds, rep, synth, margins, path):
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(11, 3.4))
    x = np.arange(len(archs)); w = 0.8 / (3 * len(conds))
    k = 0
    for c in conds:
        for m, col in (("target", MODEL_COLOR["target"]), ("policy", MODEL_COLOR["policy"]), ("occupancy", MODEL_COLOR["occupancy"])):
            vals = [np.mean([r["last"][f"rsa_{m}"] for r in rep[(a, c)]]) for a in archs]
            axes[0].bar(x + (k - 1.5 * len(conds) + 0.5) * w, vals, w, color=col, alpha=0.4 + 0.6 * (conds.index(c) / max(len(conds) - 1, 1)), edgecolor="white",
                        label=f"{m}, {_xt([c])[0]}")
            k += 1
    axes[0].set_xticks(x); axes[0].set_xticklabels(archs); axes[0].set_title("RSA at the last hidden layer (darker = softer)"); axes[0].legend(fontsize=6, ncol=3); axes[0].grid(True, axis="y")
    for a, col in zip(archs, SERIES):
        for c, ls in zip(conds[:2], ("-", "--")):
            axes[1].plot(np.abs(np.array(margins) - 0.5), np.mean(synth[(a, c)], 0), color=col, lw=2, ls=ls, marker="o", ms=3, label=f"{a}, {_xt([c])[0]}")
    axes[1].set_xlabel("|Δ margin| (same argmax)"); axes[1].set_ylabel("hidden distance / median"); axes[1].set_title("Synthetic fixed-argmax sweep, last layer"); axes[1].legend(fontsize=6, ncol=2); axes[1].grid(True)
    _save(fig, path)


def plot_unique_interaction(rep, path):
    layers = list(rep[0].keys())
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(10, 3.2))
    x = np.arange(len(layers)); w = 0.2
    for k, (key, lab, col) in enumerate([("interaction", "interaction (all)", SERIES[4]), ("unique_of_total", "interaction unique to complement of additive span", SERIES[0])]):
        axes[0].bar(x + (k - 0.5) * w, [np.mean([r[l][key] for r in rep]) for l in layers], w, color=col, edgecolor="white", label=lab)
    axes[0].set_xticks(x); axes[0].set_xticklabels(layers); axes[0].set_title("variance fractions"); axes[0].legend(fontsize=7); axes[0].grid(True, axis="y")
    conds = [("baseline", SERIES[6]), ("remove_unique", SERIES[0]), ("random_in_complement", SERIES[3]), ("remove_all_interaction", SERIES[4])]
    for k, (ck, col) in enumerate(conds):
        axes[1].bar(x + (k - 1.5) * w, [np.mean([r[l][ck]["accuracy"] for r in rep]) for l in layers], w, color=col, edgecolor="white", label=ck)
    axes[1].set_xticks(x); axes[1].set_xticklabels(layers); axes[1].set_title("action accuracy after ablation"); axes[1].legend(fontsize=7); axes[1].grid(True, axis="y"); axes[1].set_ylim(0, 1.02)
    _save(fig, path)


def plot_steering_tau(conds, rep, path, layer="h3"):
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(11, 3.3))
    x = np.arange(len(conds))
    axes[0].plot(x, [np.mean([r[layer]["steer_success_alpha1"] for r in rep[c]]) for c in conds], color=SERIES[0], lw=2, marker="o", ms=4)
    axes[0].set_title(f"steering success at α = 1 ({layer})"); axes[0].set_ylim(0, 1.02)
    for key, lab, col in [("spearman_alpha_lin", "linearised boundary", SERIES[0]), ("spearman_logit_margin", "logit margin", SERIES[1]), ("spearman_target_margin", "target-probability margin", MODEL_COLOR["target"]),
                          ("spearman_m1", "Q margin g1", SERIES[2]), ("spearman_m2", "Q margin g2", SERIES[3]), ("spearman_d_occ", "‖Δρ‖", SERIES[6])]:
        axes[1].plot(x, [np.nanmean([r[layer][key] for r in rep[c]]) for c in conds], color=col, lw=2, marker="o", ms=4, label=lab)
    axes[1].axhline(0, color=MUTED, lw=0.8); axes[1].set_title(f"Spearman(α*, predictor) ({layer})"); axes[1].legend(fontsize=7, ncol=2)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(_xt(conds), rotation=35, fontsize=7.5); ax.grid(True, axis="y")
    _save(fig, path)


def plot_losses(names, cross, rep, path, layer="h3"):
    plot_fixed_env(names, cross, rep, path, layer)


def plot_abc(rep, path):
    conds = list(rep)
    layers = list(rep[conds[0]][0].keys())
    fig, axes = plt.subplots(1, len(layers), layout="constrained", figsize=(4 * len(layers), 3.2), sharey=True)
    x = np.arange(3); w = 0.8 / len(conds)
    for ax, l in zip(np.atleast_1d(axes), layers):
        for k, c in enumerate(conds):
            vals = [np.mean([r[l][p] for r in rep[c]]) for p in "ABC"]
            ax.bar(x + (k - len(conds) / 2 + 0.5) * w, vals, w, color=SERIES[k], edgecolor="white", label=c)
        ax.axhline(1, color=MUTED, lw=0.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels(["A: same hard target,\nfar occupancy", "B: near occupancy,\ndifferent hard target", "C: same hard target,\ndifferent soft targets"], fontsize=7)
        ax.set_title(f"ring, {l}"); ax.grid(True, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("hidden distance / median"); np.atleast_1d(axes)[0].legend(fontsize=7)
    _save(fig, path)
