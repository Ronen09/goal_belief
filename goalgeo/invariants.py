"""TASK7: representation measures split into raw geometry, information, represented
subspace and functional effect. Everything is computed from extracted arrays so that an
exact change of coordinates at the hidden/readout interface, h -> A h with W -> W A^-1
and J -> J A^-1, can be applied to the arrays and each measure tested for invariance."""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as G
from . import hmm4 as H4
from . import prominence as Pr


# ---- unregularised decoding ----------------------------------------------------------
def _folds(n, n_folds, rng):
    return np.array_split(rng.permutation(n), n_folds)


def ols_cv_r2(X: np.ndarray, Y: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    """5-fold cross-validated R² of ordinary least squares X -> Y (no regulariser, so the
    value is exactly invariant to invertible linear maps of X up to numerical error)."""
    rng = np.random.default_rng(seed); n = len(X); Y = Y.reshape(n, -1); sse = sst = 0.0
    for held in _folds(n, n_folds, rng):
        tr = np.ones(n, bool); tr[held] = False
        mx, my = X[tr].mean(0), Y[tr].mean(0)
        Wm, *_ = np.linalg.lstsq(X[tr] - mx, Y[tr] - my, rcond=None)
        pred = (X[held] - mx) @ Wm + my
        sse += ((Y[held] - pred) ** 2).sum(); sst += ((Y[held] - my) ** 2).sum()
    return float(1 - sse / sst) if sst > 0 else 0.0


def ols_cv_acc(X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    rng = np.random.default_rng(seed); n = len(X); s = np.where(y, 1.0, -1.0); correct = 0
    for held in _folds(n, n_folds, rng):
        tr = np.ones(n, bool); tr[held] = False
        mx, my = X[tr].mean(0), s[tr].mean()
        w, *_ = np.linalg.lstsq(X[tr] - mx, s[tr] - my, rcond=None)
        correct += (np.sign((X[held] - mx) @ w + my) == s[held]).sum()
    return float(correct / n)


# ---- subspace ------------------------------------------------------------------------
def exact_rank(X: np.ndarray, rtol: float = 1e-6) -> int:
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return int((s > rtol * s[0]).sum()) if s[0] > 0 else 0


def subspace_basis(X: np.ndarray, rtol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis Q [N, r] of the column space of the centred representation."""
    U, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    r = int((s > rtol * s[0]).sum()) if s[0] > 0 else 0
    return U[:, :r]


def projection(Q: np.ndarray) -> np.ndarray:
    return Q @ Q.T


def whitened_rdm(Q: np.ndarray) -> np.ndarray:
    P = projection(Q); d = np.diag(P)
    return np.sqrt(np.maximum(d[:, None] + d[None] - 2 * P, 0))


def subspace_overlap(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """tr(P1 P2) / min(rank): 1 iff the two represented sample subspaces coincide."""
    return float(np.linalg.norm(Q1.T @ Q2, "fro") ** 2 / min(Q1.shape[1], Q2.shape[1]))


# ---- functional ------------------------------------------------------------------------
def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(-1)   # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def future_jacobian(net, X: np.ndarray, Hs: np.ndarray, anchors: np.ndarray, k: int):
    """Explicit J = d z_{t*} / d h_t at each anchor (z: logits at t*-1 after running the
    GRU over the k following tokens from h_t). Returns J [M, V, H], p_star [M, V]."""
    dev = next(net.parameters()).device
    i, t = anchors[:, 0], anchors[:, 1]
    h0 = torch.as_tensor(Hs[i, t], dtype=torch.float32, device=dev).requires_grad_(True)
    win = np.stack([X[ii, tt + 1:tt + k + 1] for ii, tt in zip(i, t)])
    hk, _ = net.gru(net.emb(torch.as_tensor(win, device=dev)), h0[None].contiguous())
    z = net.out(hk[:, -1])
    J = np.stack([torch.autograd.grad(z[:, v].sum(), h0, retain_graph=True)[0].cpu().numpy() for v in range(z.shape[1])], axis=1)
    return J, torch.softmax(z, -1).detach().cpu().numpy()


def finite_future_js(net, X: np.ndarray, Hs: np.ndarray, anchors_A: np.ndarray, anchors_B: np.ndarray, k: int, rng) -> float:
    """Each A anchor paired with a random B anchor; both states run through the A anchor's
    future tokens; JS divergence between the resulting predictions at t*-1."""
    dev = next(net.parameters()).device
    j = anchors_B[rng.integers(0, len(anchors_B), len(anchors_A))]
    hA = torch.as_tensor(Hs[anchors_A[:, 0], anchors_A[:, 1]], dtype=torch.float32, device=dev)
    hB = torch.as_tensor(Hs[j[:, 0], j[:, 1]], dtype=torch.float32, device=dev)
    win = torch.as_tensor(np.stack([X[ii, tt + 1:tt + k + 1] for ii, tt in anchors_A]), device=dev)
    with torch.no_grad():
        e = net.emb(win)
        pA = torch.softmax(net.out(net.gru(e, hA[None].contiguous())[0][:, -1]), -1).cpu().numpy()
        pB = torch.softmax(net.out(net.gru(e, hB[None].contiguous())[0][:, -1]), -1).cpu().numpy()
    return float(js_divergence(pA, pB).mean())


def _softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)


# ---- extraction ------------------------------------------------------------------------
def extract(net, m: H4.HMM4, ev: dict, sub: np.ndarray, dec: np.ndarray, n_group: int = 400, n_anchor: int = 200, seed: int = 0) -> dict:
    """All arrays the measures need, on the shared evaluation set. `sub` and `dec` are
    fixed flat-state indices so that rows correspond across models."""
    rng = np.random.default_rng(seed)
    X, Z, B, Y = ev["X"], ev["Z"], ev["B"], ev["Y"]
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(X)).cpu().numpy().astype(np.float64)
        logp = torch.log_softmax(net.forward_all(torch.as_tensor(X[:300])), -1).cpu().numpy()[:, :-1]
    P, anchors = Pr.positions(m, Z)
    flat = Hs.reshape(-1, Hs.shape[-1]); Bf = B.reshape(-1, B.shape[-1]); Yf = Y.reshape(-1, Y.shape[-1])
    pick = lambda mask: flat[rng.choice(np.flatnonzero(mask.ravel()), n_group, replace=False)]      # noqa: E731
    pre_idx = rng.choice(np.flatnonzero((P["preA"] | P["preB"]).ravel()), min(3000, int((P["preA"] | P["preB"]).sum())), replace=False)
    W = net.out.weight.detach().cpu().numpy().astype(np.float64); b = net.out.bias.detach().cpu().numpy().astype(np.float64)
    aA = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["P"]]; aB = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["Q"]]
    aA = aA[rng.choice(len(aA), min(n_anchor, len(aA)), replace=False)]
    J, p_star = future_jacobian(net, X, Hs, aA, m.k); J = J.astype(np.float64)
    H_U = pick(P["ctrlU"]); H_V = pick(P["ctrlV"])
    pU, pV = _softmax(H_U @ W.T + b), _softmax(H_V @ W.T + b)
    return {"H_sub": flat[sub], "B_sub": Bf[sub], "H_dec": flat[dec], "B_dec": Bf[dec], "Y_dec": Yf[dec],
            "H_pre": flat[pre_idx], "y_pre": P["preA"].ravel()[pre_idx],
            "H_cueA": pick(P["cueA"]), "H_cueB": pick(P["cueB"]), "H_preA": pick(P["preA"]), "H_preB": pick(P["preB"]), "H_U": H_U, "H_V": H_V,
            "W": W, "b": b, "J": J, "p_star": p_star, "p_U": pU.mean(0),
            "JS_future": finite_future_js(net, X, Hs, aA, aB, m.k, rng), "JS_control": float(js_divergence(pU, pV[rng.permutation(len(pV))]).mean()),
            "required_gap": float(2 * np.log((0.5 + m.delta) / (0.5 - m.delta))) if m.delta < 0.5 else float("inf"),
            "logp": logp.astype(np.float32)}


def transform(ex: dict, A: np.ndarray) -> dict:
    """Exact change of coordinates at the interface: h -> A h, W -> W A^-1, J -> J A^-1."""
    Ainv = np.linalg.inv(A); out = dict(ex)
    for key in ("H_sub", "H_dec", "H_pre", "H_cueA", "H_cueB", "H_preA", "H_preB", "H_U", "H_V"):
        out[key] = ex[key] @ A.T
    out["W"] = ex["W"] @ Ainv; out["J"] = ex["J"] @ Ainv
    return out


def random_transform(kind: str, d: int, rng, cond: float = 10.0, scale: float = 10.0) -> np.ndarray:
    if kind == "scalar":
        return scale * np.eye(d)
    if kind == "diagonal":
        return np.diag(np.exp(rng.uniform(np.log(0.1), np.log(10), d)))
    if kind == "orthogonal":
        return np.linalg.qr(rng.standard_normal((d, d)))[0]
    if kind == "full":
        U = np.linalg.qr(rng.standard_normal((d, d)))[0]; V = np.linalg.qr(rng.standard_normal((d, d)))[0]
        return U @ np.diag(np.logspace(0, np.log10(cond), d)) @ V.T
    raise ValueError(kind)


# ---- measures ------------------------------------------------------------------------
RAW = ("sep_cue", "sep_pre", "sep_control", "D_delay", "D_control", "P_metric", "cos_pair", "hidden_norm", "rsa_euclid", "rsa_cosine", "PR", "top5_frac", "eff_rank")
INFO = ("r2_belief", "acc_branch_pre", "r2_target", "rank")
SUBSPACE = ("rsa_whitened",)
FUNCTIONAL = ("D_logit_pre", "I_contrast_pre", "D_logit_control", "D_future", "D_F", "D_F_control", "JS_future", "JS_control")
ALLOC = ("g_eff", "cos_theta")
ALL = RAW + INFO + SUBSPACE + FUNCTIONAL + ALLOC


def metrics_from_arrays(ex: dict, seed: int = 0):
    """Returns (scalars, arrays) where arrays holds Q (subspace basis) and the RDM for
    cross-model comparisons."""
    rng = np.random.default_rng(seed)
    mA, mB = ex["H_cueA"].mean(0), ex["H_cueB"].mean(0); dh = mA - mB
    dpre = ex["H_preA"].mean(0) - ex["H_preB"].mean(0); dctrl = ex["H_U"].mean(0) - ex["H_V"].mean(0)
    Hs = ex["H_sub"]; Rh = G.rdm(Hs); Rb = G.rdm(ex["B_sub"])
    Hn = Hs / (np.linalg.norm(Hs, axis=1, keepdims=True) + 1e-12); Rc = 1 - Hn @ Hn.T
    lam = G.pca_spectrum(Hs)
    r = {"sep_cue": float(np.linalg.norm(dh)), "sep_pre": float(np.linalg.norm(dpre)), "sep_control": float(np.linalg.norm(dctrl)),
         "D_delay": Pr.cross_distance(ex["H_cueA"], ex["H_cueB"], rng=rng), "D_control": Pr.cross_distance(ex["H_U"], ex["H_V"], rng=rng)}
    r["P_metric"] = r["D_delay"] / (r["D_control"] + 1e-12)
    r["cos_pair"] = float(mA @ mB / (np.linalg.norm(mA) * np.linalg.norm(mB) + 1e-12))
    r["hidden_norm"] = float(np.linalg.norm(Hs, axis=1).mean())
    r["rsa_euclid"] = G.rsa(Rh, Rb); r["rsa_cosine"] = G.rsa(Rc, Rb)
    r["PR"] = G.participation_ratio(Hs); r["top5_frac"] = float(lam[:5].sum() / lam.sum()); r["eff_rank"] = G.entropy_rank(Hs)
    # information
    r["r2_belief"] = ols_cv_r2(ex["H_dec"], ex["B_dec"]); r["r2_target"] = ols_cv_r2(ex["H_dec"], ex["Y_dec"])
    r["acc_branch_pre"] = ols_cv_acc(ex["H_pre"], ex["y_pre"]); r["rank"] = exact_rank(Hs)
    # subspace
    Q = subspace_basis(Hs); r["rsa_whitened"] = G.rsa(whitened_rdm(Q), Rb)
    # functional
    W = ex["W"]; Wc = W - W.mean(0, keepdims=True); x, y = H4.TOK["x"], H4.TOK["y"]
    r["D_logit_pre"] = float(np.linalg.norm(Wc @ dpre)); r["I_contrast_pre"] = float((W[x] - W[y]) @ dpre)
    r["D_logit_control"] = float(np.linalg.norm(Wc @ dctrl))
    Jd = ex["J"] @ dh                                           # [M, V]
    r["D_future"] = float(np.linalg.norm(Jd - Jd.mean(1, keepdims=True), axis=1).mean())
    p = ex["p_star"]; fisher = (p * Jd ** 2).sum(1) - (p * Jd).sum(1) ** 2
    r["D_F"] = float(np.sqrt(np.maximum(fisher, 0).mean()))
    Wd = W @ dctrl; pu = ex["p_U"]
    r["D_F_control"] = float(np.sqrt(max((pu * Wd ** 2).sum() - (pu * Wd).sum() ** 2, 0)))
    r["JS_future"] = ex["JS_future"]; r["JS_control"] = ex["JS_control"]; r["required_gap"] = ex["required_gap"]
    wd = W[x] - W[y]; r["g_eff"] = float(np.linalg.norm(wd)); r["cos_theta"] = float(wd @ dpre / (r["g_eff"] * r["sep_pre"] + 1e-12))
    return r, {"Q": Q, "RDM": Rh}
