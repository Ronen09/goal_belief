"""Round 12 figures. Categorical colours follow one fixed order: goal, act_soft, act_hard, next_obs;
untrained networks are neutral grey."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from goalgeo.style import COL, GRID, INK, MUTED, _style  # noqa: E402

OBJ = ("goal", "act_soft", "act_hard", "next_obs", "init")
SPLITS = ("IID", "EXT", "CONF", "TIME")
FLOOR = -0.5


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
    from goalgeo import belief_train as BT, beliefprobe as BP
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


# ---- part 2: causal tests -------------------------------------------------------------------
ICOL = {"swap": "#2a78d6", "probe_e": "#eb6834", "probe_d": "#1baf7a", "probe_e_half": "#e87ba4", "rand": "#9a9a94"}


def fig_transplant(D, out: Path, t: int = 6):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, kind in zip(axes[:2], ("iid", "channel")):
        _style(ax)
        for it, lab in (("swap", "GRU swap"), ("probe_e", "GRU PROBE-E"), ("probe_d", "GRU PROBE-D"), ("rand", "GRU random")):
            c = D.A("gru", kind, "goal", f"{it}:F_model_S", t, "h", "curve")
            ax.plot(range(len(c)), c, color=ICOL[it], lw=2, marker="o", ms=3, label=lab)
        for it, ls in (("swap", "--"), ("probe_e", ":")):
            c = D.A("tfm", kind, "goal", f"{it}:F_model_S", t, "res1", "curve")
            ax.plot(range(len(c)), c, color=ICOL[it], lw=1.5, ls=ls, label=f"transformer res1(t) {it}")
        ax.axhline(0, color=MUTED, lw=0.6); ax.set_ylim(-0.1, 1.05)
        ax.set_xlabel("offset k after the intervention", fontsize=8, color=INK)
        ax.set_title(f"{kind}, goal models, t = {t}: gap closed to genuine-B", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="center right")
    ax = axes[2]; _style(ax)
    for key, lab, col in (("probe_e:kl_bayesS", "to genuine-B Bayes", "#2a78d6"), ("probe_e:kl_bayesT", "to transplant Bayes", "#eb6834"),
                          ("swap:kl_bayesS", "swap, to genuine-B Bayes", "#1baf7a")):
        c = D.A("gru", "channel", "goal", key, t, "h", "curve")
        ax.plot(range(len(c)), c, color=col, lw=2, marker="o", ms=3, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("offset k", fontsize=8, color=INK)
    ax.set_title("channel GRU goal, PROBE-E: KL to each Bayes target", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig4_transplant.png", dpi=150); plt.close(fig)


def fig_equal(D, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, arch in zip(axes, ("gru", "tfm")):
        _style(ax)
        for kind, mode, col, lab in (("iid", "iid_exact", "#2a78d6", "iid, exactly equal b"),
                                     ("channel", "chan_equal_b", "#eb6834", "channel, equal b, different channel belief"),
                                     ("channel", "chan_equal_joint", "#1baf7a", "channel, equal joint state")):
            b = D.B(arch, kind, "goal", mode)
            ax.plot(range(len(b["model"])), np.maximum(b["model"], 1e-7), color=col, lw=2, marker="o", ms=3, label=lab)
            if mode == "chan_equal_b":
                ax.plot(range(len(b["bayes"])), np.maximum(b["bayes"], 1e-7), color=col, lw=1, ls="--", label="  its Bayes divergence")
        b = D.B(arch, "iid", "goal", "iid_exact")
        ax.plot(range(len(b["random"])), b["random"], color="#9a9a94", lw=2, label="random pairs (iid)")
        ax.set_yscale("log"); ax.set_xlabel("offset k after t = 12", fontsize=8, color=INK)
        ax.set_title(f"{'GRU (patch at the complete cut)' if arch == 'gru' else 'transformer (behaviour)'}, goal models", fontsize=9, color=INK)
    axes[0].set_ylabel("JS between the two runs' outputs", fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(out / "fig5_equal_belief.png", dpi=150); plt.close(fig)


def fig_ladder(D, out: Path):
    sites = ("res0", "res1", "res2", "u")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, title in zip(axes, ("r2y_within_action", "action_acc", "neutral_r2"),
                              ("belief within an action class (R²_y | a)", "optimal action (linear accuracy)",
                               "neutral-token count (no evidential value)")):
        _style(ax)
        for o in ("goal", "act_soft", "act_hard", "next_obs"):
            ax.plot(range(4), [D.lad("tfm", "iid", o, s, key) for s in sites], color=COL[o], lw=2, marker="o", ms=5, label=o)
        ax.set_xticks(range(4), ["res0\n(embed)", "res1\n(read by the future)", "res2", "u\n(interface)"], fontsize=7)
        ax.set_title(f"transformer, iid: {title}", fontsize=9, color=INK); ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig6_ladder.png", dpi=150); plt.close(fig)


def causal_figures(D, out: Path):
    fig_transplant(D, out); fig_equal(D, out); fig_ladder(D, out)
