"""TASK9 measurements for the causal transformer: the TASK4 prominence / decodability set on
the layer-1 residual, the readout-scale identities at the post-norm interface, and the
functional measures (finite patching divergence, patch Jacobian, propagation depth)."""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as G
from . import hmm4 as H4
from . import invariants as Iv
from . import prominence as Pr


def _np(t):
    return t.detach().cpu().numpy().astype(np.float64)


def anchors_of(m, Z, K, n_anchor, rng):
    P, anchors = Pr.positions(m, Z); T = Z.shape[1]
    anchors = anchors[anchors[:, 1] + K <= T - 1]
    aA = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["P"]]; aB = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["Q"]]
    aA = aA[rng.choice(len(aA), min(n_anchor, len(aA)), replace=False)]; jB = aB[rng.integers(0, len(aB), len(aA))]
    return P, aA, jB


def patch_jacobian(net, X, R1, aA, t_read):
    """J = d logits[t_read] / d (cut residual at the cue) at each A anchor, via patching."""
    i, t = aA[:, 0], aA[:, 1]
    h = torch.as_tensor(R1[i, t], dtype=torch.float32).requires_grad_(True)
    z = net.patched(X, i, t, h)[torch.arange(len(i)), torch.as_tensor(t_read)]
    J = np.stack([_np(torch.autograd.grad(z[:, v].sum(), h, retain_graph=True)[0]) for v in range(z.shape[1])], axis=1)
    return J, _np(torch.softmax(z, -1))


def measure_tfm(net, m: H4.HMM4, ev: dict, n_dec: int = 3000, n_anchor: int = 200, K: int = 4, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed); X, Z, B, Y = ev["X"], ev["Z"], ev["B"], ev["Y"]
    Xt = torch.as_tensor(X)
    with torch.no_grad():
        res = net.residuals(Xt); R1 = _np(res[net.cut]); R2 = _np(res[-1]); Rf = _np(net.ln_f(res[-1]))
        logp = _np(torch.log_softmax(net.out(net.ln_f(res[-1])), -1))
    P, aA, jB = anchors_of(m, Z, K, n_anchor, rng)
    flat1 = R1.reshape(-1, R1.shape[-1]); Bf = B.reshape(-1, B.shape[-1])
    g = lambda R, key: R[P[key]]                                     # noqa: E731
    r = {}
    # --- prominence on the layer-1 residual (the analogue of the GRU state) ---
    dA, dB = g(R1, "cueA"), g(R1, "cueB"); dU, dV = g(R1, "ctrlU"), g(R1, "ctrlV")
    r["D_delay"] = Pr.cross_distance(dA, dB, rng=rng); r["D_control"] = Pr.cross_distance(dU, dV, rng=rng); r["P_metric"] = r["D_delay"] / (r["D_control"] + 1e-12)
    r["D_pre_l1"] = Pr.cross_distance(g(R1, "preA"), g(R1, "preB"), rng=rng); r["P_metric_pre_l1"] = r["D_pre_l1"] / (r["D_control"] + 1e-12)
    r["sep_cue"] = float(np.linalg.norm(dA.mean(0) - dB.mean(0))); r["hidden_norm_l1"] = float(np.linalg.norm(flat1, axis=1).mean())
    sub = rng.choice(len(flat1), 800, replace=False); r["rsa_euclid_l1"] = G.rsa(G.rdm(flat1[sub]), G.rdm(Bf[sub]))
    # --- decodability: layer-1 residual, final residual, post-norm, at the cue and at t*-1 ---
    dec = rng.choice(len(flat1), min(n_dec, len(flat1)), replace=False)
    r["belief_r2_l1"] = Iv.ols_cv_r2(flat1[dec], Bf[dec])
    pre = np.flatnonzero((P["preA"] | P["preB"]).ravel()); pre = rng.choice(pre, min(n_dec, len(pre)), replace=False); ypre = P["preA"].ravel()[pre]
    for name, R in (("l1", R1), ("final", R2), ("postnorm", Rf)):
        r[f"branch_acc_pre_{name}"] = Iv.ols_cv_acc(R.reshape(-1, R.shape[-1])[pre], ypre)
    cue = np.flatnonzero((P["cueA"] | P["cueB"]).ravel()); cue = rng.choice(cue, min(n_dec, len(cue)), replace=False)
    r["branch_acc_cue_l1"] = Iv.ols_cv_acc(flat1[cue], P["cueA"].ravel()[cue])
    # --- readout-scale identities at the post-norm interface (and the pre-norm final residual) ---
    W = _np(net.out.weight); x, y = H4.TOK["x"], H4.TOK["y"]; wd = W[x] - W[y]; r["g_eff"] = float(np.linalg.norm(wd))
    for name, R in (("postnorm", Rf), ("final", R2)):
        d = g(R, "preA").mean(0) - g(R, "preB").mean(0); dc = g(R, "ctrlU").mean(0) - g(R, "ctrlV").mean(0)
        r[f"sep_pre_{name}"] = float(np.linalg.norm(d)); r[f"sep_control_{name}"] = float(np.linalg.norm(dc))
        r[f"achieved_gap_{name}"] = float(wd @ d); r[f"cos_theta_{name}"] = float(wd @ d / (r["g_eff"] * np.linalg.norm(d) + 1e-12))
    r["ln_gain_norm"] = float(np.linalg.norm(_np(net.ln_f.weight))) if net.final_ln else float("nan")
    r["required_gap"] = float(2 * np.log((0.5 + m.delta) / (0.5 - m.delta))) if m.delta < 0.5 else float("inf")
    # --- performance ---
    Yn = Y[:, :-1]; kl = (Yn * (np.log(Yn + 1e-12) - logp[:, :-1])).sum(-1); rel = m.relevant[Z[:, 1:]]
    r["kl"] = float(kl.mean()); r["kl_relevant"] = float(kl[rel].mean()) if rel.any() else 0.0
    # --- functional: finite patching divergence and depth curve, patch Jacobian ---
    i, t = aA[:, 0], aA[:, 1]; ar = torch.arange(len(i)); tt = torch.as_tensor(t)
    with torch.no_grad():
        hB = torch.as_tensor(R1[jB[:, 0], jB[:, 1]], dtype=torch.float32)
        zp, resp = net.patched(X, i, t, hB, return_res=True); z0, res0 = net.patched(X, i, t, torch.as_tensor(R1[i, t], dtype=torch.float32), return_res=True)
        pp, p0 = torch.softmax(zp, -1), torch.softmax(z0, -1)
        js = [float(Iv.js_divergence(_np(pp[ar, tt + k]), _np(p0[ar, tt + k])).mean()) for k in range(K + 1)]
        raw = [float(np.linalg.norm(_np(resp[-1][ar, tt + k]) - _np(res0[-1][ar, tt + k]), axis=1).mean()) for k in range(K + 1)]
    r["JS_future"] = js[m.k]; r["depth_js"] = js; r["depth_raw_final"] = raw
    dh = dA.mean(0) - dB.mean(0)
    J, p_star = patch_jacobian(net, X, R1, aA, t + m.k)
    Jd = J @ dh; r["D_future"] = float(np.linalg.norm(Jd - Jd.mean(1, keepdims=True), axis=1).mean())
    fisher = (p_star * Jd ** 2).sum(1) - (p_star * Jd).sum(1) ** 2; r["D_F"] = float(np.sqrt(np.maximum(fisher, 0).mean()))
    r["rank_l1"] = Iv.exact_rank(flat1[sub])
    return r
