"""Post hoc (not registered): the belief simplex. The exact beliefs of decision states in barycentric coordinates; the
beliefs decoded from held-out activations at each site, in the same triangle; and how curved the embedding of the
simplex is (polynomial fits in b of increasing degree against the node-table ceiling; PCA of the activations at a
fixed step coloured by the belief). The reward-bandit experiment's trained transformer, seed 0 (figures) and all
six seeds (the curvature table).

    .venv/bin/python studies/2_belief_state/belief_formation/simplex.py        # writes simplex_*.png, simplex.md, simplex.json
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from goalgeo import bandit as BD, beliefprobe as BP
from goalgeo.style import INK, MUTED, _style

HERE = Path(__file__).resolve().parent
RB = HERE.parent / "reward_bandit"
import importlib.util                                                # noqa: E402
_load = lambda name, path: (lambda sp: (sys.modules.__setitem__(name, m := importlib.util.module_from_spec(sp)), sp.loader.exec_module(m), m)[2])(importlib.util.spec_from_file_location(name, path))
ME = _load("measure", RB / "measure.py")                             # the reward-bandit experiment's measure.py, registered as `measure`
BF = _load("bf_measure", HERE / "measure.py")                        # this experiment's, which imports `measure` (the one above) as ME
TR, record, SITES = ME.TR, BF.record, BF.SITES

V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])        # the triangle's vertices: goal 1, 2, 3


def bary(b):
    """Probabilities [n, 3] -> plane coordinates [n, 2]."""
    return b @ V


def triangle(ax, title):
    ax.plot(*np.c_[V, V[:1]].T if False else np.r_[V, V[:1]].T, color=MUTED, lw=1)
    for i, (x, y) in enumerate(V):
        ax.text(x + (-0.05 if i == 0 else 0.02 if i == 1 else 0), y + (0.03 if i == 2 else -0.06), f"G{i + 1}", fontsize=8, color=INK, ha="center")
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.0)


def rgb(b):
    return np.clip(b, 0, 1)


def poly_features(b, degree):
    """Monomials of b[:, :2] up to `degree` (b3 = 1 - b1 - b2 is redundant)."""
    x, y = b[:, 0], b[:, 1]
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            if i + j >= 1:
                cols.append(x ** i * y ** j)
    return np.stack(cols, 1)


def mlp_fit(X, H, fold, hidden=64, steps=1500):
    """R² of a small MLP (two hidden layers) from X to H, fitted on folds != 0 and scored on fold 0: a smooth
    function of the belief, as flexible as needed."""
    dev = "cuda"
    Xt, Ht = torch.tensor(X, dtype=torch.float32, device=dev), torch.tensor(H, dtype=torch.float32, device=dev)
    mu, sd = Ht.mean(0), Ht.std(0) + 1e-6; Ht = (Ht - mu) / sd
    fit, test = torch.tensor(fold != 0, device=dev), torch.tensor(fold == 0, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, H.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    with torch.enable_grad():
        for i in range(steps):
            idx = torch.nonzero(fit).squeeze(1)[torch.randint(int(fit.sum()), (4096,), device=dev)]
            loss = ((net(Xt[idx]) - Ht[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = net(Xt[test])
        return float(1 - ((P - Ht[test]) ** 2).sum() / ((Ht[test] - Ht[fit].mean(0)) ** 2).sum())


def softmax_probe(H, B, fit, test, steps=600, wd=1e-4):
    """A decoder z = W x + c, p = softmax(z), fitted by cross-entropy to the exact posterior on `fit`, applied to
    `test`: decoded probabilities [n_test, K] (inside the simplex by construction) and their R²."""
    dev = "cuda"
    X = torch.tensor(H, dtype=torch.float32, device=dev); Y = torch.tensor(B, dtype=torch.float32, device=dev)
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6; X = (X - mu) / sd
    f, t_ = torch.tensor(fit, device=dev), torch.tensor(test, device=dev)
    lin = torch.nn.Linear(X.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=wd)
    with torch.enable_grad():
        for i in range(steps):
            lp = torch.log_softmax(lin(X[f]), -1)
            loss = -(Y[f] * lp).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        P = torch.softmax(lin(X[t_]), -1).double().cpu().numpy()
    return P, BP.r2(B[test], P)


def curvature(H, D):
    """R² (5-fold) of the activations from polynomials in b of degree 1..3, affine in y and in [y, b], a small MLP
    from b (a smooth function of the belief), the node table, and PCA variance shares."""
    out = {}
    # fitted within each step (the node includes the step; a function of b alone cannot), R² pooled over steps
    def within_step(X):
        P = np.zeros_like(H)
        for st in range(D.T):
            m = D.step == st
            P[m] = BP.cv_pred(X[m], H[m], D.fold[m])
        return BP.r2(H, P)
    for deg in (1, 2, 3):
        out[f"poly{deg}"] = within_step(poly_features(D.b, deg))
    out["affine_y"] = within_step(D.y)
    out["affine_y_b"] = within_step(np.c_[D.y, D.b[:, :2]])
    out["mlp_from_b"] = mlp_fit(np.c_[D.b, np.eye(D.T)[D.step]], H, D.fold)
    _, inv = np.unique(D.node, return_inverse=True)
    means = np.zeros((inv.max() + 1, H.shape[1])); cnt = np.zeros(inv.max() + 1)
    np.add.at(means, inv, H); np.add.at(cnt, inv, 1); means /= cnt[:, None]
    out["node_table"] = float(1 - ((H - means[inv]) ** 2).sum() / ((H - H.mean(0)) ** 2).sum())
    # within a step (the step is a large part of the variance): PCA shares of the top components
    m = D.step == 3
    X = H[m] - H[m].mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2; s /= s.sum()
    out["pca_step3_top2"], out["pca_step3_top3"], out["pca_step3_top6"] = float(s[:2].sum()), float(s[:3].sum()), float(s[:6].sum())
    return out


def main():
    dev = "cuda"
    s = BD.Spec(**TR.TASK); t = BD.Sim(BD.Graph(s), dev)
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BD.Env(t, ME.N_EVAL, gen); goal, cues, u = e.goal, e.cues, e.u
    args = json.load(open(RB / "runs" / "tfm" / "task.json"))["args"]
    res = {}
    for seed in range(6):
        d = RB / "runs" / "tfm" / f"seed{seed}" / "ckpt"
        for which, ck in (("trained", sorted(d.glob("u*.pt"))[-1]), ("untrained", d / "u000000.pt")):
            net = BD.build("tfm", s, args.get("d", 128), args.get("layers", 2))
            net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
            r = BD.rollout(net, t, len(goal), None, greedy=True, goal=goal, cues=cues, u=u)
            D = ME.Decisions(t, r, seed)
            pos = torch.as_tensor(D.step[: D.T] + t.n_cue, device=dev)
            sites, _ = record(net, r.tok, pos)
            res[f"seed{seed}/{which}"] = {k: curvature(sites[k], D) for k in ("mid0", "mlp0", "res1", "res2")}
            for k in ("mid0", "mlp0", "res1", "res2"):
                res[f"seed{seed}/{which}"][k]["softmax_EXT_r2b"] = softmax_probe(sites[k], D.b, D.ext_fit, D.ext_test)[1]
                res[f"seed{seed}/{which}"][k]["softmax_IID_r2b"] = softmax_probe(sites[k], D.b, D.fold != 0, D.fold == 0)[1]
                res[f"seed{seed}/{which}"][k]["affine_EXT_r2b"] = BP.r2(D.b[D.ext_test], BP.probe_pred(sites[k], D.b, D.ext_fit, D.ext_test))
            if seed == 0:
                figures(sites, D, which)
            print(seed, which, {k: {m: round(v, 3) for m, v in c.items()} for k, c in res[f"seed{seed}/{which}"].items()}, flush=True)
    (HERE / "simplex.json").write_text(json.dumps(res, indent=1))
    f = lambda x: f"{x:.3f}"
    med = lambda w, k, m: f(np.median([res[f"seed{s_}/{w}"][k][m] for s_ in range(6)]))
    L = ["# The belief simplex (post hoc)", "", "Generated by `simplex.py`. Figures: `simplex_exact.png` (the exact beliefs of decision states), `simplex_decoded.png` (beliefs decoded from held-out activations by an affine probe, coloured by the exact belief as RGB), `simplex_pca.png` (the activations at step 3, top principal components, coloured the same way).", "",
         "How curved the embedding is: R² (5-fold) of the activations from functions of the belief fitted within each step (pooled), the MLP from (b, step), against the node table (any function of the belief state); and the variance share of the top principal components within one step. Median over six seeds.", "",
         "| site | affine in y | affine in b | affine in y and b | quadratic in b | cubic in b | **MLP from b** | node table | PCA within step 3: top 2 / 3 / 6 components |", "|---|---|---|---|---|---|---|---|---|"]
    for w in ("trained", "untrained"):
        for k in ("mid0", "mlp0", "res1", "res2"):
            L.append(f"| {w} {k} | {med(w, k, 'affine_y')} | {med(w, k, 'poly1')} | {med(w, k, 'affine_y_b')} | {med(w, k, 'poly2')} | {med(w, k, 'poly3')} | {med(w, k, 'mlp_from_b')} | {med(w, k, 'node_table')} | {med(w, k, 'pca_step3_top2')} / {med(w, k, 'pca_step3_top3')} / {med(w, k, 'pca_step3_top6')} |")
    L += ["", "A softmax decoder (z = W x + c, p = softmax(z), fitted by cross-entropy to the exact posterior; the decoded beliefs stay inside the simplex): R² of b, against the affine probe, on the extrapolation split and 5-fold. Median over six seeds.", "",
          "| site | affine probe, EXT | **softmax decoder, EXT** | softmax decoder, IID |", "|---|---|---|---|"]
    for w in ("trained", "untrained"):
        for k in ("mid0", "mlp0", "res1", "res2"):
            L.append(f"| {w} {k} | {med(w, k, 'affine_EXT_r2b')} | {med(w, k, 'softmax_EXT_r2b')} | {med(w, k, 'softmax_IID_r2b')} |")
    (HERE / "simplex.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def figures(sites, D, which):
    b = D.b
    if which == "trained":
        # 1. the exact beliefs
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        xy = bary(b)
        ent = -(b * np.log(np.clip(b, 1e-12, None))).sum(1)
        for ax, (c, title, cmap) in zip(axes, ((D.step, "coloured by decision (0–5)", "viridis"), (D.action, "by the optimal action (a1–a4)", "tab10"),
                                               (ent, "by posterior entropy", "magma"), (rgb(b), "by the belief itself (RGB = b)", None))):
            triangle(ax, title)
            ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=2, alpha=0.5, linewidths=0, vmax=3 if cmap == "tab10" else None)
        fig.suptitle("The exact posterior of every decision state (16 384 episodes × 6 decisions)", fontsize=10, color=INK)
        fig.tight_layout(); fig.savefig(HERE / "simplex_exact.png", dpi=150); plt.close(fig)
    # 2. decoded beliefs from held-out activations (affine probe fitted on even episodes, shown on odd)
    fit, test = D.ep % 2 == 0, D.ep % 2 == 1
    keys = ("res0", "attn0", "mid0", "mlp0", "res1", "res2")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.4))
    for ax, k in zip(axes, keys):
        P = BP.probe_pred(sites[k], b, fit, test)
        xy = bary(P)
        triangle(ax, f"{k}: decoded b (R² {BP.r2(b[test], P):.2f})")
        ax.scatter(xy[:, 0], xy[:, 1], c=rgb(b[test]), s=2, alpha=0.4, linewidths=0)
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.2)
    fig.suptitle(f"Beliefs decoded from held-out activations ({which}), coloured by the exact belief (RGB = b)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(HERE / f"simplex_decoded_{which}.png", dpi=150); plt.close(fig)
    # 2b. the extrapolation split: probe fitted on the states with every |y| < 2, applied to those with one >= 3
    keys = ("mid0", "mlp0", "res2")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6))
    for ax, k in zip(axes, keys):
        P = BP.probe_pred(sites[k], b, D.ext_fit, D.ext_test)
        xy = bary(P)
        triangle(ax, f"{k}: EXT-decoded b (R² {BP.r2(b[D.ext_test], P):.2f})")
        fx = bary(b[D.ext_fit]); ax.scatter(fx[:, 0], fx[:, 1], c="#bbbbbb", s=1, alpha=0.3, linewidths=0)
        ax.scatter(xy[:, 0], xy[:, 1], c=rgb(b[D.ext_test]), s=2, alpha=0.4, linewidths=0)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.3)
    fig.suptitle(f"Probe fitted on the uncertain states (grey: their exact beliefs, every |log-odds| < 2), applied to the confident ones ({which})", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(HERE / f"simplex_ext_{which}.png", dpi=150); plt.close(fig)
    # 2c. the same split, decoded through a softmax
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6))
    for ax, k in zip(axes, keys):
        P, r2 = softmax_probe(sites[k], b, D.ext_fit, D.ext_test)
        xy = bary(P)
        triangle(ax, f"{k}: softmax-decoded b (R² {r2:.2f})")
        fx = bary(b[D.ext_fit]); ax.scatter(fx[:, 0], fx[:, 1], c="#bbbbbb", s=1, alpha=0.3, linewidths=0)
        ax.scatter(xy[:, 0], xy[:, 1], c=rgb(b[D.ext_test]), s=2, alpha=0.4, linewidths=0)
    fig.suptitle(f"Softmax decoder fitted on the uncertain states (grey), applied to the confident ones ({which})", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(HERE / f"simplex_softmax_{which}.png", dpi=150); plt.close(fig)
    # 3. raw activations at one step: top principal components
    m = D.step == 3
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
    for row, k in enumerate(("mlp0", "res2")):
        X = sites[k][m] - sites[k][m].mean(0)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        pc = U[:, :3] * S[:3]
        share = S ** 2 / (S ** 2).sum()
        for col, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axes[row, col]; _style(ax)
            ax.scatter(pc[:, i], pc[:, j], c=rgb(b[m]), s=3, alpha=0.5, linewidths=0)
            ax.set_xlabel(f"PC{i + 1} ({share[i]:.2f})", fontsize=8); ax.set_ylabel(f"PC{j + 1} ({share[j]:.2f})", fontsize=8)
            ax.set_title(f"{k}, step 3", fontsize=9, color=INK)
        ax = axes[row, 3]; _style(ax)
        ax.plot(np.arange(1, 11), np.cumsum(share[:10]), "o-", color=INK, ms=3)
        ax.set_ylim(0, 1.02); ax.set_xlabel("components", fontsize=8); ax.set_ylabel("cumulative variance", fontsize=8); ax.set_title(f"{k}: PCA spectrum", fontsize=9, color=INK)
    fig.suptitle(f"Activations at decision 4 ({which}), coloured by the exact belief (RGB = b)", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(HERE / f"simplex_pca_{which}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
