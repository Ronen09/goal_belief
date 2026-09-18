"""TASK4 measurements on a trained GRU: pairwise metric prominence of the delayed
cue, decodability, RSA against belief/predictive geometries, gradient relevance
of the delayed prediction, and KL to the exact targets."""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as G
from . import hmm4 as H4


def make_eval(m: H4.HMM4, n: int = 2000, T: int = 48, seed: int = 123) -> dict:
    X, Z = H4.sample(m, n, T, seed)
    B = H4.beliefs_seq(m, X)
    return {"X": X, "Z": Z, "B": B, "Y": H4.next_token(m, B)}


def linear_cv_accuracy(X: np.ndarray, y: np.ndarray, n_folds: int = 5, alpha: float = 1.0,
                       rng: np.random.Generator | None = None) -> float:
    """Cross-validated accuracy of a ridge classifier (least squares onto +/-1, sign readout)."""
    rng = rng or np.random.default_rng(0)
    n = len(X); folds = np.array_split(rng.permutation(n), n_folds)
    s = np.where(y, 1.0, -1.0); correct = 0
    for held in folds:
        tr = np.ones(n, bool); tr[held] = False
        mx, my = X[tr].mean(0), s[tr].mean(); Xc = X[tr] - mx
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ (s[tr] - my))
        correct += (np.sign((X[held] - mx) @ w + my) == s[held]).sum()
    return float(correct / n)


def cross_distance(ha: np.ndarray, hb: np.ndarray, cap: int = 400, rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    ha = ha[rng.choice(len(ha), min(cap, len(ha)), replace=False)]
    hb = hb[rng.choice(len(hb), min(cap, len(hb)), replace=False)]
    return float(np.linalg.norm(ha[:, None] - hb[None], axis=-1).mean())


def positions(m: H4.HMM4, Z: np.ndarray):
    """Boolean masks [n, T] for the cue pair (after p / after q), the pre-relevant pair
    (last filler of each branch) and the control pair (after u / after v), plus gradient
    anchors (sequence, cue position) using the last cue whose relevant loss is inside the sequence."""
    S, k = m.S, m.k
    P = {"cueA": Z == S["P"], "cueB": Z == S["Q"], "preA": Z == S[f"FA{k}"], "preB": Z == S[f"FB{k}"],
         "ctrlU": Z == S["U"], "ctrlV": Z == S["V"]}
    n, T = Z.shape; anchors = []
    for i in range(n):
        ts = np.flatnonzero((Z[i] == S["P"]) | (Z[i] == S["Q"])); ts = ts[ts + k <= T - 2]
        if len(ts):
            anchors.append((i, int(ts[-1])))
    return P, np.array(anchors, dtype=int).reshape(-1, 2)


def gradients(net, X: np.ndarray, Y: np.ndarray, anchors: np.ndarray, k: int, dhat: np.ndarray,
              Hs: np.ndarray | None = None) -> dict[str, float]:
    """Gradient of the relevant loss l_{t+k} (prediction of the token at t+k+1) and of the
    immediate loss l_t with respect to the cue state h_t, per anchored sequence.

    The GRU output tensor's autograd node only carries the direct readout path at each
    position, so the cue state is re-injected as a leaf: the immediate loss reads it out
    directly, the relevant loss re-runs the GRU over the k following tokens from it.
    Returns mean norms and mean absolute projections onto the unit direction dhat."""
    dev = next(net.parameters()).device
    i, t = anchors[:, 0], anchors[:, 1]
    if Hs is None:
        with torch.no_grad():
            Hs = net.states(torch.as_tensor(X, device=dev)).cpu().numpy()
    h0 = torch.as_tensor(Hs[i, t], dtype=torch.float32, device=dev).requires_grad_(True)
    out = {}
    # immediate loss l_t: readout of h_t predicts token t+1
    y_imm = torch.as_tensor(Y[i, t], dtype=torch.float32, device=dev)
    loss = -(y_imm * torch.log_softmax(net.out(h0), -1)).sum()
    g = torch.autograd.grad(loss, h0)[0].cpu().numpy()
    out["G_imm"] = float(np.linalg.norm(g, axis=1).mean()); out["Gpar_imm"] = float(np.abs(g @ dhat).mean())
    # relevant loss l_{t+k}: run the GRU over tokens t+1..t+k from h_t, readout predicts token t+k+1
    win = np.stack([X[ii, tt + 1:tt + k + 1] for ii, tt in zip(i, t)])
    y_rel = torch.as_tensor(Y[i, t + k], dtype=torch.float32, device=dev)
    hk, _ = net.gru(net.emb(torch.as_tensor(win, device=dev)), h0[None].contiguous())
    loss = -(y_rel * torch.log_softmax(net.out(hk[:, -1]), -1)).sum()
    g = torch.autograd.grad(loss, h0)[0].cpu().numpy()
    out["G_rel"] = float(np.linalg.norm(g, axis=1).mean()); out["Gpar_rel"] = float(np.abs(g @ dhat).mean())
    return out


def measure(net, m: H4.HMM4, ev: dict, n_rsa: int = 800, n_dec: int = 3000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    X, Z, B, Y = ev["X"], ev["Z"], ev["B"], ev["Y"]
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(X)).cpu().numpy()          # [n, T, H]
    P, anchors = positions(m, Z)
    h = lambda key: Hs[P[key]]                                       # noqa: E731
    d_delay = cross_distance(h("cueA"), h("cueB"), rng=rng)
    d_ctrl = cross_distance(h("ctrlU"), h("ctrlV"), rng=rng)
    d_pre = cross_distance(h("preA"), h("preB"), rng=rng)
    flat = Hs.reshape(-1, Hs.shape[-1]); Bf = B.reshape(-1, B.shape[-1])
    sub = rng.choice(len(flat), min(n_rsa, len(flat)), replace=False)
    Rh = G.rdm(flat[sub]); med = float(np.median(G.upper(Rh)))
    dhat = h("cueA").mean(0) - h("cueB").mean(0); dhat = dhat / (np.linalg.norm(dhat) + 1e-12)
    dirs = rng.standard_normal((20, Hs.shape[-1])); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    var_d = float((flat @ dhat).var()); var_rand = float((flat @ dirs.T).var(0).mean())
    res = {"D_delay": d_delay, "D_control": d_ctrl, "D_pre": d_pre,
           "P_metric": d_delay / (d_ctrl + 1e-12), "P_metric_pre": d_pre / (d_ctrl + 1e-12),
           "D_delay_over_median": d_delay / (med + 1e-12), "var_ratio_delayed_vs_random": var_d / (var_rand + 1e-12)}
    # decodability
    dec = rng.choice(len(flat), min(n_dec, len(flat)), replace=False)
    res["belief_r2"] = G.ridge_cv_r2(flat[dec], Bf[dec], alpha=1.0)
    pre_mask = (P["preA"] | P["preB"]).ravel(); pre = np.flatnonzero(pre_mask)
    pre = rng.choice(pre, min(n_dec, len(pre)), replace=False)
    res["branch_acc_pre"] = linear_cv_accuracy(flat[pre], P["preA"].ravel()[pre], rng=rng)
    # RSA against belief and predictive geometries
    Bs = Bf[sub]
    refs = {"belief": Bs, "P1": H4.next_token(m, Bs), "P2": H4.joint_predictive(m, Bs, 2), "future": H4.marginals(m, Bs, m.k + 2)}
    for name, ref in refs.items():
        res[f"rsa_{name}"] = G.rsa(Rh, G.rdm(ref))
    # performance: KL to the exact targets
    with torch.no_grad():
        logp = torch.log_softmax(net.forward_all(torch.as_tensor(X)), -1)[:, :-1].cpu().numpy()
    Yn = Y[:, :-1]; kl = (Yn * (np.log(Yn + 1e-12) - logp)).sum(-1)
    rel = m.relevant[Z[:, 1:]]
    res["kl"] = float(kl.mean()); res["kl_relevant"] = float(kl[rel].mean()) if rel.any() else 0.0
    res.update(gradients(net, X, Y, anchors, m.k, dhat, Hs=Hs))
    res.update(readout_scale(net, m, Hs, P))
    return res


def readout_scale(net, m: H4.HMM4, Hs: np.ndarray, P: dict) -> dict[str, float]:
    """How the required x-vs-y logit gap between the two branches is split between the
    readout (effective gain ||w_x - w_y||) and the hidden separation at t*-1."""
    W = net.out.weight.detach().cpu().numpy()
    wd = W[H4.TOK["x"]] - W[H4.TOK["y"]]; g = float(np.linalg.norm(wd))
    d_pre = Hs[P["preA"]].mean(0) - Hs[P["preB"]].mean(0)
    d_ctrl = Hs[P["ctrlU"]].mean(0) - Hs[P["ctrlV"]].mean(0)
    d_cue = Hs[P["cueA"]].mean(0) - Hs[P["cueB"]].mean(0)
    req = 2 * np.log((0.5 + m.delta) / (0.5 - m.delta)) if m.delta < 0.5 else float("inf")
    return {"g_eff": g, "required_gap": float(req),
            "achieved_gap_pre": float(wd @ d_pre), "achieved_gap_control": float(wd @ d_ctrl),
            "cos_readout_pre": float(wd @ d_pre / (g * np.linalg.norm(d_pre) + 1e-12)),
            "cos_readout_control": float(wd @ d_ctrl / (g * np.linalg.norm(d_ctrl) + 1e-12)),
            "mean_sep_pre": float(np.linalg.norm(d_pre)), "mean_sep_control": float(np.linalg.norm(d_ctrl)), "mean_sep_cue": float(np.linalg.norm(d_cue)),
            "readout_frobenius": float(np.linalg.norm(W))}
