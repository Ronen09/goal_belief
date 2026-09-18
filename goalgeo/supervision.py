"""TASK3 analyses: target geometry, information bottleneck of target maps,
equivalence classes, within/between-class compression, unique interaction
variance, counterfactual pairs and the A/B/C environment pairs."""

from __future__ import annotations

import numpy as np

from . import geometry as G
from .gridworld import GridWorld
from .occupancy import Occupancy, successor_representation
from .planning import Solution
from .quotient import policy_table, advantage_table, margins
from .targets import Targets, boltzmann


# ---------------------------------------------------------------------------
# target geometry (Task 2)
# ---------------------------------------------------------------------------
def state_level_target(tg: Targets, env: GridWorld) -> np.ndarray:
    """Concatenate a target's rows over goals -> [S, K*out_dim] (goal rows = 0 for s == g)."""
    S, K = env.n_states, env.n_goals
    M = np.zeros((S, K, tg.out_dim))
    M[tg.s, tg.g] = tg.Y
    return M.reshape(S, K * tg.out_dim)


def state_level_rdms(env: GridWorld, occ: Occupancy, sol: Solution, tg: Targets, gamma: float) -> dict[str, np.ndarray]:
    return {
        "target": G.rdm(state_level_target(tg, env)),
        "occupancy": G.rdm(occ.Z_sa),
        "SR": G.rdm(successor_representation(env, gamma)),
        "spatial": G.rdm(env.coords),
        "policy": G.rdm(policy_table(sol)),
        "advantage": G.rdm(advantage_table(sol)),
    }


def pair_level_rdms(env: GridWorld, occ: Occupancy, sol: Solution, tg: Targets, rows: np.ndarray) -> dict[str, np.ndarray]:
    s, g = tg.s[rows], tg.g[rows]
    return {
        "target": G.rdm(tg.Y[rows]),
        "Q": G.rdm(tg.Q[rows]),
        "occupancy_state": G.rdm(occ.Z_sa[s]),
        "policy_row": G.rdm(sol.pi[g, s]),
        "advantage_row": G.rdm(tg.Q[rows] - tg.Q[rows].max(-1, keepdims=True)),
        "spatial": G.rdm(np.hstack([env.coords[s], env.coords[env.goal_states][g]])),
    }


def geometry_report(H_layers: dict[str, np.ndarray], layers, env: GridWorld, occ: Occupancy, sol: Solution,
                    tg: Targets, gamma: float, rows: np.ndarray) -> dict[str, dict]:
    """Per layer: state-level RSA with each model, regression, pair-level RSA, PR."""
    Rs = state_level_rdms(env, occ, sol, tg, gamma)
    Rp = pair_level_rdms(env, occ, sol, tg, rows)
    S, K = env.n_states, env.n_goals
    out = {}
    for l in layers:
        H = H_layers[l]
        X = H.mean(1); R = G.rdm(X)
        rep = {f"rsa_{k}": G.rsa(R, v) for k, v in Rs.items()}
        rep["regression"] = G.rdm_regression(R, Rs)
        rep["regression_no_target"] = G.rdm_regression(R, {k: v for k, v in Rs.items() if k != "target"})
        rep["partial_target_given_env"] = G.partial_rsa(R, Rs["target"], [Rs[k] for k in ("occupancy", "SR", "spatial", "policy", "advantage")])
        rep["partial_occupancy_given_target"] = G.partial_rsa(R, Rs["occupancy"], [Rs["target"]])
        rep["participation_ratio"] = G.participation_ratio(X)
        Hf = H.reshape(S * K, -1)
        idx = tg.s[rows] * K + tg.g[rows]
        Rh = G.rdm(Hf[idx])
        rep.update({f"pair_rsa_{k}": G.rsa(Rh, v) for k, v in Rp.items()})
        rep["pair_regression"] = G.rdm_regression(Rh, {k: Rp[k] for k in ("target", "Q", "occupancy_state", "policy_row", "spatial")})
        rep["pair_participation_ratio"] = G.participation_ratio(Hf)
        out[l] = rep
    return out


# ---------------------------------------------------------------------------
# information bottleneck of T (Task 5)
# ---------------------------------------------------------------------------
def n_equivalence_classes(Y: np.ndarray, tol: float = 1e-6) -> int:
    return len(np.unique(np.round(Y / tol).astype(np.int64), axis=0))


def bottleneck_stats(tg: Targets, sol: Solution, occ: Occupancy, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    Y, Q = tg.Y, tg.Q
    adv = Q - Q.max(-1, keepdims=True)
    Z = occ.Z_sa[tg.s]
    sv = np.linalg.svd(Y - Y.mean(0), compute_uv=False)
    return {
        "rank": int((sv > 1e-8 * sv.max()).sum()) if sv.max() > 0 else 0,
        "participation_ratio": G.participation_ratio(Y),
        "n_classes": n_equivalence_classes(Y, 1e-6 if tg.kind != "boltzmann" else 1e-3),
        "n_rows": len(Y),
        "rsa_with_Q": G.rsa(G.rdm(Y[:800]), G.rdm(Q[:800])),
        "recover_Q_r2": G.ridge_cv_r2(Y, Q, alpha=1e-3, rng=rng),
        "recover_adv_r2": G.ridge_cv_r2(Y, adv, alpha=1e-3, rng=rng),
        "recover_occ_r2": G.ridge_cv_r2(Y, Z, alpha=1e-3, rng=rng, groups=tg.s),
    }


# ---------------------------------------------------------------------------
# classes, compression (Tasks 6, 7)
# ---------------------------------------------------------------------------
def hard_classes(tg_hard: Targets) -> np.ndarray:
    """Integer class id per row = argmax set of Q."""
    keys = np.round(tg_hard.Y * 1e6).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.ravel()


def within_between(X: np.ndarray, classes: np.ndarray, max_rows: int = 1500, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    if len(X) > max_rows:
        sel = rng.choice(len(X), max_rows, replace=False); X, classes = X[sel], classes[sel]
    R = G.rdm(X); iu = np.triu_indices(len(X), 1)
    same = classes[iu[0]] == classes[iu[1]]
    d = R[iu]
    w = float(d[same].mean()) if same.any() else float("nan"); b = float(d[~same].mean()) if (~same).any() else float("nan")
    return {"within": w, "between": b, "ratio": w / b if b > 0 else float("nan")}


def sufficient_statistic_pairs(tg: Targets, occ: Occupancy, q: float = 0.25) -> dict:
    """Row pairs with identical targets but different Z_sa(s), and different
    targets but similar Z_sa(s). Returns index pairs into tg rows."""
    n = len(tg.Y); rng = np.random.default_rng(0)
    sel = rng.choice(n, min(n, 900), replace=False)
    Y, Z = tg.Y[sel], occ.Z_sa[tg.s[sel]]
    iu = np.triu_indices(len(sel), 1)
    dT = np.linalg.norm(Y[iu[0]] - Y[iu[1]], axis=1); dZ = np.linalg.norm(Z[iu[0]] - Z[iu[1]], axis=1)
    same_state = tg.s[sel][iu[0]] == tg.s[sel][iu[1]]
    ident = (dT < 1e-6) & ~same_state
    A = np.where(ident & (dZ >= np.quantile(dZ, 1 - q)))[0]
    B = np.where((dT >= np.quantile(dT[dT > 1e-6], 0.5)) & (dZ <= np.quantile(dZ, q)) & ~same_state)[0]
    return {"rows": sel, "iu": iu, "same_target_far_env": A, "diff_target_near_env": B}


def pair_distances(Hf: np.ndarray, pairs: dict) -> dict:
    sel, iu = pairs["rows"], pairs["iu"]
    R = G.rdm(Hf[sel]); med = np.median(R[np.triu_indices(len(sel), 1)])
    out = {}
    for k in ("same_target_far_env", "diff_target_near_env"):
        idx = pairs[k]
        out[k] = float(np.median(R[iu[0][idx], iu[1][idx]]) / med) if len(idx) else float("nan")
        out[f"n_{k}"] = int(len(idx))
    return out


# ---------------------------------------------------------------------------
# fixed-argmax counterfactual in the real environment (Task 3a)
# ---------------------------------------------------------------------------
def same_argmax_pairs(tg_hard: Targets, sol: Solution, n_max: int = 20000, rng=None) -> dict:
    """Pairs of rows with identical argmax set; returns |Δmargin| and row indices."""
    rng = rng or np.random.default_rng(0)
    cls = hard_classes(tg_hard)
    m = margins(sol)[tg_hard.g, tg_hard.s]
    out_i, out_j = [], []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        if len(idx) < 2:
            continue
        ii, jj = np.triu_indices(len(idx), 1)
        out_i.append(idx[ii]); out_j.append(idx[jj])
    i, j = np.concatenate(out_i), np.concatenate(out_j)
    if len(i) > n_max:
        k = rng.choice(len(i), n_max, replace=False); i, j = i[k], j[k]
    return {"i": i, "j": j, "d_margin": np.abs(m[i] - m[j]), "margin_i": m[i], "margin_j": m[j]}


def counterfactual_report(Hf: np.ndarray, pairs: dict, tg: Targets) -> dict:
    from scipy.stats import spearmanr
    R_all = np.linalg.norm(Hf[pairs["i"]] - Hf[pairs["j"]], axis=1)
    rng = np.random.default_rng(0); n = len(Hf)
    a, b = rng.integers(0, n, 4000), rng.integers(0, n, 4000); keep = a != b
    med = np.median(np.linalg.norm(Hf[a[keep]] - Hf[b[keep]], axis=1))
    dT = np.linalg.norm(tg.Y[pairs["i"]] - tg.Y[pairs["j"]], axis=1)
    dm = pairs["d_margin"]
    lo, hi = dm <= np.quantile(dm, 0.25), dm >= np.quantile(dm, 0.75)
    return {"median_same_argmax_dist": float(np.median(R_all) / med),
            "low_dmargin_dist": float(np.median(R_all[lo]) / med), "high_dmargin_dist": float(np.median(R_all[hi]) / med),
            "spearman_dist_dmargin": float(spearmanr(R_all, dm).correlation) if np.std(dm) > 0 else float("nan"),
            "spearman_dist_dtarget": float(spearmanr(R_all, dT).correlation) if np.std(dT) > 0 else float("nan")}


# ---------------------------------------------------------------------------
# unique interaction variance (Task 10)
# ---------------------------------------------------------------------------
def additive_decomposition(H: np.ndarray) -> dict:
    """Centred two-way decomposition plus the part of the interaction that is
    orthogonal (in feature space) to the span of the additive effect vectors."""
    mu = H.mean((0, 1))
    a = H.mean(1) - mu                          # [S, d]
    b = H.mean(0) - mu                          # [K, d]
    r = H - mu - a[:, None, :] - b[None, :, :]  # [S, K, d]
    span = np.vstack([a, b])
    U, sv, Vt = np.linalg.svd(span, full_matrices=False)
    keep = sv > 1e-10 * max(sv.max(), 1e-30)
    B = Vt[keep]                                # orthonormal basis of the additive feature span
    S, K, d = H.shape
    rf = r.reshape(S * K, d)
    r_unique = (rf - (rf @ B.T) @ B).reshape(S, K, d)
    tot = ((H - mu) ** 2).sum()
    return {"mu": mu, "a_s": a, "b_g": b, "r": r, "r_unique": r_unique, "basis_additive": B,
            "state": float((a ** 2).sum() * K / tot), "goal": float((b ** 2).sum() * S / tot),
            "interaction": float((r ** 2).sum() / tot),
            "unique_fraction": float((r_unique ** 2).sum() / max((r ** 2).sum(), 1e-30)),
            "unique_of_total": float((r_unique ** 2).sum() / tot),
            "additive_span_dim": int(keep.sum())}


# ---------------------------------------------------------------------------
# A/B/C pairs in the ring (Task 14)
# ---------------------------------------------------------------------------
def abc_pairs(tg_hard: Targets, tau: float = 0.02, q: float = 0.25) -> dict:
    """Row-level pairs. Environment difference = ‖Q_i − Q_j‖ (the row's occupancy
    profile up to 1/γ); hard target = argmax set; soft target = softmax(Q/τ).
    A: same hard target, top-quartile dQ, soft difference no larger than dQ predicts.
    B: different hard target, bottom-quartile dQ.
    C: same hard target, dQ not in the top quartile, soft difference in the top
       quartile of the residual after regressing dS on dQ (soft targets differ
       more than the Q difference would suggest)."""
    cls = hard_classes(tg_hard); Q = tg_hard.Q; soft = boltzmann(Q, tau)
    n = len(Q); iu = np.triu_indices(n, 1)
    same = cls[iu[0]] == cls[iu[1]]
    dQ = np.linalg.norm(Q[iu[0]] - Q[iu[1]], axis=1); dS = np.linalg.norm(soft[iu[0]] - soft[iu[1]], axis=1)
    # soft-target difference beyond what the Q difference predicts (linear fit within same-class pairs)
    beta = float(dQ[same] @ dS[same]) / max(float(dQ[same] @ dQ[same]), 1e-12)
    resid = dS - beta * dQ
    A = np.where(same & (dQ >= np.quantile(dQ[same], 1 - q)) & (resid <= 0))[0]
    B = np.where(~same & (dQ <= np.quantile(dQ[~same], q)))[0]
    C = np.where(same & (dQ <= np.quantile(dQ[same], 0.75)) & (resid >= np.quantile(resid[same], 1 - q)))[0]
    return {"A": A, "B": B, "C": C, "iu": iu, "dQ": dQ, "dS": dS,
            "stats": {k: {"n": int(len(v)), "dQ": float(np.median(dQ[v])) if len(v) else float("nan"), "dS": float(np.median(dS[v])) if len(v) else float("nan")} for k, v in (("A", A), ("B", B), ("C", C))}}


def abc_report(Hf_rows: np.ndarray, pairs: dict) -> dict:
    """Hf_rows: hidden activations for the target rows, in row order."""
    R = G.rdm(Hf_rows); iu = pairs["iu"]; med = np.median(R[iu])
    return {k: float(np.median(R[iu[0][pairs[k]], iu[1][pairs[k]]]) / med) if len(pairs[k]) else float("nan") for k in "ABC"}
