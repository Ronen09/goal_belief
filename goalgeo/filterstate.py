"""TASK12 (round 13): the full filter state of the channel environment
(docs/task12_filter_theory.md).

Coordinates of the joint filter α(g, c) = P(G = g, c_t = c | history), K goals x {off, on}:
    y_g = log b(g) / b(K)             goal block (K-1)
    l_g = logit P(c_t = on | G = g)   channel block (K)
an invertible reparametrisation of α; `joint_from` inverts it."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import beliefcausal as BC
from . import beliefprobe as BP
from . import latentgoal as LG

P_CLIP = 1e-6


def _logit(p):
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def coords(J):
    """J [..., K, 2] -> (y [..., K-1], l [..., K])."""
    b = J.sum(-1)
    return LG.log_odds(b), _logit(J[..., 1] / np.clip(b, 1e-300, None))


def full(J):
    y, l = coords(J)
    return np.concatenate([y, l], -1)


def joint_from(y, l):
    z = np.concatenate([y, np.zeros(y.shape[:-1] + (1,))], -1)
    b = np.exp(z - z.max(-1, keepdims=True)); b /= b.sum(-1, keepdims=True)
    on = 1 / (1 + np.exp(-l))
    return np.stack([b * (1 - on), b * on], -1)


# ---- recoverability -----------------------------------------------------------------------
class FullEval:
    """The round-12 channel evaluation set with the full-state coordinates of every state."""

    def __init__(self, K: int = 4, n: int = 8000, seed: int = 12345):
        self.E = E = BP.EvalSet("channel", K, n=n, seed=seed)
        J = LG.filter_joint(E.env, E.X)[:, 1:].reshape(-1, K, 2)
        self.J = J
        self.y, self.l = coords(J)
        self.Y = np.c_[self.y, self.l]
        on = J[..., 1].sum(-1)
        self.marg = np.c_[self.y, _logit(on)]
        lmax = np.abs(self.l).max(1)
        self.ext_fit, self.ext_test = lmax < 2, lmax >= 3


def block_r2(Y, P, K):
    return {"goal": BP.r2(Y[:, :K - 1], P[:, :K - 1]), "channel": BP.r2(Y[:, K - 1:], P[:, K - 1:]), "all": BP.r2(Y, P)}


def recover(H, F: FullEval):
    """Per-block R² of an affine probe h -> (y, l) on the IID, TIME and EXT-channel splits."""
    K = F.E.env.K; H = np.asarray(H, np.float64)
    out = {"IID": block_r2(F.Y, BP.cv_pred(H, F.Y, F.E.fold), K)}
    for name, (fit, test) in {"TIME": (F.E.t <= 12, F.E.t > 12), "EXT": (F.ext_fit, F.ext_test)}.items():
        out[name] = block_r2(F.Y[test], BP.probe_pred(H, F.Y, fit, test), K)
    return out


def baselines(F: FullEval):
    K = F.E.env.K
    ch = lambda X: BP.r2(F.l, BP.cv_pred(X, F.l, F.E.fold))       # noqa: E731
    return {"channel_from_true_y": ch(F.y), "channel_from_true_marginals": ch(F.marg),
            "full_from_counts": block_r2(F.Y, BP.cv_pred(F.E.counts, F.Y, F.E.fold), K)}


# ---- pairs --------------------------------------------------------------------------------
def _pool(env, t, n, seed):
    X, _, _ = LG.sample(env, n, T=t, seed=seed)
    J = LG.filter_joint(env, X)[:, -1]
    return X, J


def _disjoint(P, n, rng):
    P = P[rng.permutation(len(P))]; used, out = set(), []
    for i, j in P:
        if i not in used and j not in used:
            out.append((i, j)); used.update((i, j))
        if len(out) == n:
            break
    return np.array(out, int).reshape(-1, 2)


def pairs_equal_marginals(env, n, seed, t=12, pool=300000, tol=0.05, min_dl=0.5):
    X, J = _pool(env, t, pool, seed)
    y, l = coords(J); m = np.c_[y, _logit(J[..., 1].sum(-1))]
    P = cKDTree(m).query_pairs(tol, p=np.inf, output_type="ndarray")
    P = P[np.abs(l[P[:, 0]] - l[P[:, 1]]).max(1) >= min_dl]
    P = _disjoint(P, n, np.random.default_rng(seed))
    return X[P[:, 0]], X[P[:, 1]], J[P[:, 0]]


def pairs_equal_full(env, n, seed, t=12, pool=1000000, tol=0.02, min_positions=1):
    """Equal full filter state, different token sequences. min_positions > 1 counts differing
    positions after merging the two neutral tokens, which are exchangeable under every (g, c)
    and so give exactly equal states for a trivial reason."""
    X, J = _pool(env, t, pool, seed)
    P = cKDTree(full(J)).query_pairs(tol, p=np.inf, output_type="ndarray")
    if min_positions > 1:
        Xm = X.copy(); Xm[Xm == env.K + 2] = env.K + 1
        P = P[(Xm[P[:, 0]] != Xm[P[:, 1]]).sum(1) >= min_positions]
    else:
        P = P[(X[P[:, 0]] != X[P[:, 1]]).any(1)]                      # different token sequences
    P = _disjoint(P, n, np.random.default_rng(seed))
    return X[P[:, 0]], X[P[:, 1]], J[P[:, 0]]


def pairs_cross_length(env, n, seed, t1=8, t2=16, pool=600000, tol=0.05):
    """H1 of length t1 and H2 of length t2 with equal full filter states."""
    X1, J1 = _pool(env, t1, pool, seed); X2, J2 = _pool(env, t2, pool, seed + 1)
    tree = cKDTree(full(J2)); d, j = tree.query(full(J1), k=1, p=np.inf, distance_upper_bound=tol)
    i = np.flatnonzero(np.isfinite(d))
    i, j = i, j[i]
    _, first = np.unique(j, return_index=True)                        # each H2 used once
    i, j = i[first], j[first]
    sel = np.random.default_rng(seed).permutation(len(i))[:n]
    return X1[i[sel]], X2[j[sel]], J1[i[sel]], J2[j[sel]]


def pairs_cross_random(env, n, seed, t1=8, t2=16):
    X1, J1 = _pool(env, t1, n, seed); X2, J2 = _pool(env, t2, n, seed + 1)
    return X1, X2, J1, J2


# ---- tests --------------------------------------------------------------------------------
def test_matched(run_full, from_state, get_state, bmap, env, objective, XA, XB, t, rng):
    """Matched interventions in the goal and channel blocks at a complete cut."""
    K = env.K
    XS = np.concatenate([XB[:, :t + 1], XA[:, t + 1:]], 1)
    pA, pS = run_full(XA), run_full(XS)
    JA = LG.filter_joint(env, XA)[:, t]; JB = LG.filter_joint(env, XB)[:, t]
    yA, lA = coords(JA); yB, lB = coords(JB)
    Xc = XA[:, t + 1:]
    hA, hB = get_state(XA), get_state(XB)
    targets = {"g_only": (yB, lA), "r_only": (yA, lB), "both": (yB, lB)}
    out = {"none_vs_S_model": BC.js(pA, pS).mean(0)}
    shift_both = None
    for name, (y, l) in targets.items():
        tgt = BC.ideal(env, BC.filter_continue(env, joint_from(y, l), Xc), objective)
        h = bmap.probe_e(hA, np.c_[y, l])
        if name == "both":
            shift_both = h - hA
        p = from_state(h)
        out[f"{name}:F_bayes"] = BC.gap_closed(BC.kl(tgt, p), BC.kl(tgt, pA))
        out[f"{name}:F_model_S"] = BC.gap_closed(BC.js(p, pS), BC.js(pA, pS))
        out[f"{name}:gap_size"] = BC.kl(tgt, pA).mean(0)
    tgtS = BC.ideal(env, BC.filter_continue(env, JB, Xc), objective)
    for name, h in (("swap", hB), ("rand", bmap.rand_like(hA, shift_both, rng))):
        p = from_state(h)
        out[f"{name}:F_bayes"] = BC.gap_closed(BC.kl(tgtS, p), BC.kl(tgtS, pA))
        out[f"{name}:F_model_S"] = BC.gap_closed(BC.js(p, pS), BC.js(pA, pS))
    return out


def test_from_states(from_state_pair, env, objective, H1, H2, J1, J2, rng, n_cont=8, L=8):
    """Equal-state pairs of possibly different lengths: both states run the same continuation
    (sampled from J1's predictive) from the complete cut. Returns model / Bayes divergences."""
    from scipy.stats import spearmanr
    dm, db = [], []
    for _ in range(n_cont):
        c = BC.sample_continuation(env, J1, L, rng)
        p1, p2 = from_state_pair(H1, H2, c)
        i1 = BC.ideal(env, BC.filter_continue(env, J1, c), objective)
        i2 = BC.ideal(env, BC.filter_continue(env, J2, c), objective)
        dm.append(BC.js(p1, p2)); db.append(BC.js(i1, i2))
    dm, db = np.stack(dm, 1), np.stack(db, 1)
    pm, pb = dm[:, :, 1:].mean((1, 2)), db[:, :, 1:].mean((1, 2))
    rho = spearmanr(pm, pb).correlation if pb.std() > 0 else float("nan")
    return {"model": dm.mean((0, 1)), "bayes": db.mean((0, 1)), "spearman_pairs": float(rho)}
