"""TASK8: intervention equivalence, on/off-manifold local metrics, and propagation depth,
all built on one primitive: push a (possibly perturbed) hidden state through the network's
actual downstream computation and read the output distribution at each step."""

from __future__ import annotations

import numpy as np
import torch

from . import invariants as Iv


def _t(a, dev=None):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)


def propagate(net, h0: np.ndarray, win: np.ndarray):
    """Run the GRU from states h0 [M, H] over tokens win [M, K]. Returns states at every step
    [M, K, H] and output probabilities at steps 0..K [M, K+1, V] (step 0 = readout of h0)."""
    with torch.no_grad():
        h0t = _t(h0); hk, _ = net.gru(net.emb(torch.as_tensor(win)), h0t[None].contiguous())
        z = net.out(torch.cat([h0t[:, None], hk], 1))
        return hk.numpy().astype(np.float64), torch.softmax(z, -1).numpy().astype(np.float64)


def directional_logit_derivative(net, h0: np.ndarray, win: np.ndarray, v: np.ndarray, eps: float = 1e-3):
    """Central finite difference of the logits at every step along v (v [H] or [M, H]):
    (z(h0 + eps u) - z(h0 - eps u)) / (2 eps) * ||v||, with u = v / ||v||. Returns [M, K+1, V]."""
    v = np.broadcast_to(v, h0.shape); n = np.linalg.norm(v, axis=1, keepdims=True); u = v / (n + 1e-12)
    with torch.no_grad():
        def z(h):
            ht = _t(h); hk, _ = net.gru(net.emb(torch.as_tensor(win)), ht[None].contiguous())
            return net.out(torch.cat([ht[:, None], hk], 1)).numpy().astype(np.float64)
        return (z(h0 + eps * u) - z(h0 - eps * u)) / (2 * eps) * n[:, :, None]


def js_linear_prediction(dz: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Second-order prediction of JS(p, p') for a logit perturbation dz at p: dz^T F(p) dz / 8."""
    return ((p * dz ** 2).sum(-1) - (p * dz).sum(-1) ** 2) / 8


def effect_curve(net, h0: np.ndarray, win: np.ndarray, v: np.ndarray, alphas, step: int = 1):
    """Mean JS between the prediction at `step` from h0 + alpha v and from h0, per alpha,
    plus the full probability arrays [len(alphas), M, V] for cross-model comparisons."""
    _, p0 = propagate(net, h0, win); P = []
    for a in alphas:
        _, p = propagate(net, h0 + a * v, win); P.append(p[:, step])
    P = np.stack(P); js = np.array([Iv.js_divergence(p, p0[:, step]).mean() for p in P])
    return js, P, p0[:, step]


def match_alpha(js_target: float, alphas, js_curve: np.ndarray) -> float:
    """Smallest alpha at which the (interpolated) effect curve reaches js_target."""
    alphas = np.asarray(alphas, float)
    for i in range(1, len(alphas)):
        if js_curve[i] >= js_target:
            lo, hi = js_curve[i - 1], js_curve[i]
            return float(alphas[i - 1] + (alphas[i] - alphas[i - 1]) * (js_target - lo) / max(hi - lo, 1e-12))
    return float(alphas[-1])


def align_states(H1: np.ndarray, H2: np.ndarray):
    """Least-squares linear map from model-1 coordinates to model-2 coordinates on shared states."""
    m1, m2 = H1.mean(0), H2.mean(0)
    B, *_ = np.linalg.lstsq(H1 - m1, H2 - m2, rcond=None)
    r2 = 1 - (((H1 - m1) @ B - (H2 - m2)) ** 2).sum() / ((H2 - m2) ** 2).sum()
    return B, float(r2)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def gradient_direction(net, h0: np.ndarray, win: np.ndarray, tok_up: int, tok_down: int, step: int = 1) -> np.ndarray:
    """Per-anchor gradient of the log-odds (tok_up vs tok_down) at `step` with respect to h0."""
    ht = _t(h0).requires_grad_(True)
    hk, _ = net.gru(net.emb(torch.as_tensor(win)), ht[None].contiguous())
    z = net.out(torch.cat([ht[:, None], hk], 1))[:, step]
    g = torch.autograd.grad((z[:, tok_up] - z[:, tok_down]).sum(), ht)[0]
    return g.detach().numpy().astype(np.float64)


def depth_curves(net, hA: np.ndarray, hB: np.ndarray, win: np.ndarray):
    """For k = 0..K: JS between the predictions from hA and hB after k shared tokens, the raw
    distance between the propagated states, and the linearised logit displacement ||J^(k) dh||
    (centred), all averaged over anchors."""
    K = win.shape[1]; dh = hA - hB
    sA, pA = propagate(net, hA, win); sB, pB = propagate(net, hB, win)
    dz = directional_logit_derivative(net, hA, win, dh); dz = dz - dz.mean(-1, keepdims=True)
    js = np.array([Iv.js_divergence(pA[:, k], pB[:, k]).mean() for k in range(K + 1)])
    raw = np.concatenate([[np.linalg.norm(dh, axis=1).mean()], np.linalg.norm(sA - sB, axis=2).mean(0)])
    lin = np.linalg.norm(dz, axis=2).mean(0)
    zA = np.log(pA + 1e-12); zB = np.log(pB + 1e-12); fin = np.linalg.norm((zA - zB) - (zA - zB).mean(-1, keepdims=True), axis=2).mean(0)
    return {"js": js, "raw": raw, "lin": lin, "finite_logit": fin}
