"""Phase-level analysis functions. Each returns plain dicts/arrays; plotting is
separate (``plotting.py``) and orchestration lives in ``rounds/r01_occupancy/run.py``."""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as G
from .gridworld import GridWorld
from .model import PolicyNet, LAYERS
from .occupancy import Occupancy, successor_representation
from .planning import Solution

HIDDEN_LAYERS = ["emb", "h1", "h2", "h3"]


# ---------------------------------------------------------------------------
# Phase 3: theoretical geometries
# ---------------------------------------------------------------------------
def hypothesis_rdms(env: GridWorld, occ: Occupancy, gamma: float,
                    extra: dict[str, np.ndarray] | None = None, sol: Solution | None = None) -> dict[str, np.ndarray]:
    """State-level RDMs for every competing hypothesis. With ``sol`` given, adds
    the *policy table* geometry: each state represented by its optimal action
    distribution for every goal (what occupancy reduces to at the output)."""
    geo = env.geodesic()
    geo = np.where(np.isfinite(geo), geo, np.nanmax(geo[np.isfinite(geo)]) + 1)
    hyps = {
        "occupancy": G.rdm(occ.Z_sa),
        "occupancy_state": G.rdm(occ.Z_state),
        "spatial": G.rdm(env.coords),
        "geodesic": geo,
        "SR": G.rdm(successor_representation(env, gamma)),
        "occupancy_log": G.rdm(np.log(np.clip(occ.Z_sa, 1e-6, None))),
        "identity": 1.0 - np.eye(env.n_states),
    }
    if sol is not None:
        K, S, Aa = sol.pi.shape
        hyps["policy"] = G.rdm(np.transpose(sol.pi, (1, 0, 2)).reshape(S, K * Aa))
    if extra:
        hyps.update({k: G.rdm(v) for k, v in extra.items()})
    return hyps


def occupancy_geometry_summary(occ: Occupancy, n_clusters: int = 6) -> dict:
    Z = occ.Z_sa
    lam = G.pca_spectrum(Z)
    labels, link = G.hierarchical_clusters(Z, n_clusters)
    return {
        "n_states": int(Z.shape[0]),
        "dim": int(Z.shape[1]),
        "participation_ratio": G.participation_ratio(Z),
        "entropy_rank": G.entropy_rank(Z),
        "explained_variance": (lam / lam.sum()).tolist(),
        "n_pcs_90pct": int(np.searchsorted(np.cumsum(lam / lam.sum()), 0.9) + 1),
        "mean_pairwise_distance": float(G.upper(G.rdm(Z)).mean()),
        "mean_cosine": float(G.upper(G.cosine_similarity(Z)).mean()),
        "cluster_labels": labels,
        "linkage": link,
    }


# ---------------------------------------------------------------------------
# Phase 5: learned vs predicted geometry
# ---------------------------------------------------------------------------
def state_views(H: np.ndarray) -> dict[str, np.ndarray]:
    """H[s, g, :] -> state-level matrices."""
    S, K, d = H.shape
    return {"mean": H.mean(1), "concat": H.reshape(S, K * d)}


def layer_report(H_layers: dict[str, np.ndarray], hyps: dict[str, np.ndarray],
                 occ: Occupancy, env: GridWorld, view: str = "mean",
                 rng: np.random.Generator | None = None) -> dict[str, dict]:
    """Per-layer RSA / partial RSA / CKA / decoding / dimensionality."""
    rng = rng or np.random.default_rng(0)
    out = {}
    Z = occ.Z_sa
    for layer in HIDDEN_LAYERS:
        X = state_views(H_layers[layer])[view]
        R = G.rdm(X)
        rep = {f"rsa_{k}": G.rsa(R, Rh) for k, Rh in hyps.items()}
        rep["partial_occ_given_spatial"] = G.partial_rsa(R, hyps["occupancy"], [hyps["spatial"]])
        rep["partial_spatial_given_occ"] = G.partial_rsa(R, hyps["spatial"], [hyps["occupancy"]])
        rep["partial_occ_given_geodesic"] = G.partial_rsa(R, hyps["occupancy"], [hyps["geodesic"]])
        rep["partial_geodesic_given_occ"] = G.partial_rsa(R, hyps["geodesic"], [hyps["occupancy"]])
        rep["cka_occupancy"] = G.linear_cka(X, Z)
        rep["cka_spatial"] = G.linear_cka(X, env.coords)
        rep["cka_SR"] = G.linear_cka(X, successor_representation(env, occ.gamma))
        rep["decode_r2_occupancy"] = G.ridge_cv_r2(X, Z, n_folds=5, alpha=1.0, rng=rng)
        rep["decode_r2_spatial"] = G.ridge_cv_r2(X, env.coords, n_folds=5, alpha=1.0, rng=rng)
        # decoding from the full (s,g) activations with states held out
        Hf = H_layers[layer]
        S, K, d = Hf.shape
        groups = np.repeat(np.arange(S), K)
        rep["decode_r2_occupancy_pairs"] = G.ridge_cv_r2(
            Hf.reshape(S * K, d), np.repeat(Z, K, axis=0), n_folds=5, alpha=1.0, rng=rng, groups=groups)
        rep["participation_ratio"] = G.participation_ratio(X)
        rep["entropy_rank"] = G.entropy_rank(X)
        rep["subspace_overlap_k3"] = G.subspace_overlap(X, Z, 3)
        rep["subspace_overlap_k5"] = G.subspace_overlap(X, Z, 5)
        rep["rdm_regression"] = G.rdm_regression(
            R, {k: hyps[k] for k in ("occupancy", "spatial", "geodesic", "SR")})
        out[layer] = rep
    return out


def pair_level_report(H_layers: dict[str, np.ndarray], occ: Occupancy, env: GridWorld,
                      sol: Solution, max_pairs_rows: int = 1200,
                      rng: np.random.Generator | None = None) -> dict[str, dict]:
    """RSA over (s, g) pairs against pair-level theoretical representations."""
    rng = rng or np.random.default_rng(0)
    S, K = env.n_states, env.n_goals
    s_idx = np.repeat(np.arange(S), K); g_idx = np.tile(np.arange(K), S)
    if S * K > max_pairs_rows:
        sel = rng.choice(S * K, max_pairs_rows, replace=False)
    else:
        sel = np.arange(S * K)
    s_sel, g_sel = s_idx[sel], g_idx[sel]
    Q = occ.Q[g_sel, s_sel]                       # [n, A]  rho(g|s,·)
    pi = sol.pi[g_sel, s_sel]                     # [n, A]  optimal action distribution
    onehot_g = np.eye(K)[g_sel]
    models = {
        "pair_occupancy": G.rdm(np.hstack([occ.Z_sa[s_sel], Q])),
        "pair_occ_goalid": G.rdm(np.hstack([occ.Z_sa[s_sel], onehot_g * occ.Z_sa.std() * 3])),
        "pair_Q": G.rdm(Q),
        "pair_spatial": G.rdm(np.hstack([env.coords[s_sel], env.coords[env.goal_states][g_sel]])),
        "pair_action": G.rdm(pi),
        "state_occupancy_only": G.rdm(occ.Z_sa[s_sel]),
        "goal_identity_only": G.rdm(onehot_g),
    }
    out = {}
    for layer in HIDDEN_LAYERS:
        Hf = H_layers[layer].reshape(S * K, -1)[sel]
        R = G.rdm(Hf)
        rep = {f"rsa_{k}": G.rsa(R, Rm) for k, Rm in models.items()}
        rep["rdm_regression"] = G.rdm_regression(
            R, {k: models[k] for k in ("state_occupancy_only", "pair_Q", "pair_spatial", "pair_action", "goal_identity_only")})
        rep["decode_r2_Q"] = G.ridge_cv_r2(Hf, Q, n_folds=5, alpha=1.0, rng=rng)
        out[layer] = rep
    return out


def gamma_sweep(H_layers: dict[str, np.ndarray], env: GridWorld, sol_fn, gammas, view: str = "mean") -> dict[str, list[float]]:
    """RSA between h̄(s) and the occupancy RDM recomputed at each gamma.
    ``sol_fn(gamma)`` must return the Occupancy object for that discount."""
    out = {l: [] for l in HIDDEN_LAYERS}
    Rs = {l: G.rdm(state_views(H_layers[l])[view]) for l in HIDDEN_LAYERS}
    for g in gammas:
        Rg = G.rdm(sol_fn(g).Z_sa)
        for l in HIDDEN_LAYERS:
            out[l].append(G.rsa(Rs[l], Rg))
    return out


# ---------------------------------------------------------------------------
# Phase 6: discriminating pairs
# ---------------------------------------------------------------------------
def disagreement_pairs(R_occ: np.ndarray, R_sp: np.ndarray, quantile: float = 0.25):
    """Indices (into the upper triangle) of pairs where occupancy says far but
    spatial says near (occ_far) and vice versa (sp_far)."""
    a = G.upper(R_occ); b = G.upper(R_sp)
    za = (a - a.mean()) / a.std(); zb = (b - b.mean()) / b.std()
    diff = za - zb
    hi, lo = np.quantile(diff, 1 - quantile), np.quantile(diff, quantile)
    return np.where(diff >= hi)[0], np.where(diff <= lo)[0]


def discriminating_report(H_layers: dict[str, np.ndarray], hyps: dict[str, np.ndarray],
                          env: GridWorld, view: str = "mean") -> dict[str, dict]:
    out = {}
    R_occ, R_sp = hyps["occupancy"], hyps["spatial"]
    occ_far, sp_far = disagreement_pairs(R_occ, R_sp)
    for layer in HIDDEN_LAYERS:
        X = state_views(H_layers[layer])[view]
        R = G.rdm(X); u = G.upper(R)
        rep = G.rdm_regression(R, {k: hyps[k] for k in ("occupancy", "spatial", "geodesic", "SR")})
        rep["rsa_occupancy"] = G.rsa(R, R_occ); rep["rsa_spatial"] = G.rsa(R, R_sp)
        rep["rsa_geodesic"] = G.rsa(R, hyps["geodesic"])
        if "policy" in hyps:
            rep["rsa_policy"] = G.rsa(R, hyps["policy"])
            rep["policy_regression"] = G.rdm_regression(R, {k: hyps[k] for k in ("occupancy", "spatial", "policy")})
        # on pairs where the hypotheses disagree: which one does activation distance follow?
        idx = np.concatenate([occ_far, sp_far])
        from scipy.stats import spearmanr
        rep["disagree_rsa_occupancy"] = float(spearmanr(u[idx], G.upper(R_occ)[idx]).correlation)
        rep["disagree_rsa_spatial"] = float(spearmanr(u[idx], G.upper(R_sp)[idx]).correlation)
        # normalised activation distance on the two disagreement sets
        med = np.median(u)
        rep["dist_occ_far_spatial_near"] = float(np.median(u[occ_far]) / med)
        rep["dist_spatial_far_occ_near"] = float(np.median(u[sp_far]) / med)
        # designed pairs
        if env.special_pairs:
            d_special = np.array([R[a, b] for a, b in env.special_pairs]) / med
            rep["special_pairs_norm_dist"] = d_special.tolist()
            rep["special_pairs_median"] = float(np.median(d_special))
            # matched controls: pairs with the same spatial distance (±0.5) as each special pair
            ctrl = []
            for a, b in env.special_pairs:
                dsp = R_sp[a, b]
                mask = np.abs(G.upper(R_sp) - dsp) <= 0.5
                ctrl.append(np.median(u[mask]) / med)
            rep["special_pairs_spatial_matched_control"] = float(np.median(ctrl))
            ctrl2 = []
            for a, b in env.special_pairs:
                docc = R_occ[a, b]
                tol = 0.05 * np.median(G.upper(R_occ))
                mask = np.abs(G.upper(R_occ) - docc) <= tol
                if mask.any():
                    ctrl2.append(np.median(u[mask]) / med)
            rep["special_pairs_occ_matched_control"] = float(np.median(ctrl2)) if ctrl2 else float("nan")
        out[layer] = rep
    return out


# ---------------------------------------------------------------------------
# Phase 7: goal dependence
# ---------------------------------------------------------------------------
def goal_dependence_report(H_layers: dict[str, np.ndarray], hyps: dict[str, np.ndarray],
                           occ: Occupancy, env: GridWorld, sol: Solution) -> dict[str, dict]:
    out = {}
    K = env.n_goals
    goal_coords = env.coords[env.goal_states]
    R_goal_sp = G.rdm(goal_coords)
    R_goal_occ = G.rdm(occ.Z_sa[env.goal_states])
    for layer in HIDDEN_LAYERS:
        H = H_layers[layer]
        parts = G.two_way_decomposition(H)
        _, a_s, b_g, r = parts["components"]
        rep = {k: parts[k] for k in ("state", "goal", "interaction")}
        Ra = G.rdm(a_s)
        rep["state_component_rsa_occupancy"] = G.rsa(Ra, hyps["occupancy"])
        rep["state_component_rsa_spatial"] = G.rsa(Ra, hyps["spatial"])
        Rb = G.rdm(b_g)
        rep["goal_component_rsa_goal_spatial"] = G.rsa(Rb, R_goal_sp)
        rep["goal_component_rsa_goal_occupancy"] = G.rsa(Rb, R_goal_occ)
        # interaction vs goal-specific action relevance Q_g(s,·) and vs optimal action
        S = env.n_states
        r_flat = r.reshape(S * K, -1)
        Qf = np.transpose(occ.Q, (1, 0, 2)).reshape(S * K, -1)
        pif = np.transpose(sol.pi, (1, 0, 2)).reshape(S * K, -1)
        rng = np.random.default_rng(0)
        sel = rng.choice(S * K, min(S * K, 1000), replace=False)
        Rr = G.rdm(r_flat[sel])
        rep["interaction_rsa_Q"] = G.rsa(Rr, G.rdm(Qf[sel]))
        rep["interaction_rsa_action"] = G.rsa(Rr, G.rdm(pif[sel]))
        rep["interaction_decode_r2_Q"] = G.ridge_cv_r2(r_flat, Qf, n_folds=5, alpha=1.0, rng=rng)
        # goal-invariance of a state's representation: mean cosine across goals
        Hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-9)
        cos = np.einsum("sgd,shd->sgh", Hn, Hn)
        iu = np.triu_indices(K, 1)
        rep["mean_cos_across_goals"] = float(cos[:, iu[0], iu[1]].mean())
        out[layer] = rep
    return out


# ---------------------------------------------------------------------------
# Phase 8: causal intervention
# ---------------------------------------------------------------------------
@torch.no_grad()
def steer(net: PolicyNet, sol: Solution, layer: str, g1: int, g2: int, alphas,
          direction: np.ndarray | None = None, exclude_goals: bool = True) -> dict:
    """Add alpha * v_{g1->g2} at ``layer`` while the input goal is g1."""
    H = net.all_activations()[layer]              # [S, K, d]
    S = H.shape[0]
    v = (H[:, g2] - H[:, g1]).mean(0) if direction is None else direction
    base = H[:, g1]
    supp1 = sol.pi[g1] > 0; supp2 = sol.pi[g2] > 0   # [S, A]
    differ = (supp1 != supp2).any(1)                  # states where the optimal action sets differ
    if exclude_goals:
        differ &= ~np.isin(np.arange(S), [net.env.goal_states[g1], net.env.goal_states[g2]])
    only2 = supp2 & ~supp1
    actions, agree1, agree2, agree2x = [], [], [], []
    def _m(x):
        return float(x[differ].mean()) if differ.any() else float("nan")
    for a in alphas:
        x = torch.as_tensor(base + a * v, dtype=torch.float32)
        act = net.forward_from(layer, x).argmax(-1).numpy()
        actions.append(act)
        agree1.append(_m(supp1[np.arange(S), act]))
        agree2.append(_m(supp2[np.arange(S), act]))
        agree2x.append(_m(only2[np.arange(S), act]))
    return {"alphas": list(alphas), "actions": actions, "agree_g1": agree1, "agree_g2": agree2,
            "agree_g2_excl": agree2x,
            "n_differ": int(differ.sum()), "differ_mask": differ, "v_norm": float(np.linalg.norm(v)),
            "direction": v}


def steering_report(net: PolicyNet, sol: Solution, occ: Occupancy, layers=("emb", "h1", "h2", "h3"),
                    n_goal_pairs: int = 12, alphas=np.linspace(0, 2, 21), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    K = net.env.n_goals
    pairs = [tuple(rng.choice(K, 2, replace=False)) for _ in range(n_goal_pairs)]
    out = {"alphas": alphas.tolist(), "layers": {}}
    for layer in layers:
        curves2, curves1, rand2, other2, curves2x = [], [], [], [], []
        flip_alpha, margin, margin1 = [], [], []
        for g1, g2 in pairs:
            res = steer(net, sol, layer, g1, g2, alphas)
            curves2.append(res["agree_g2"]); curves1.append(res["agree_g1"]); curves2x.append(res["agree_g2_excl"])
            # control 1: random direction of the same norm
            d = rng.normal(size=res["direction"].shape); d *= res["v_norm"] / np.linalg.norm(d)
            rand2.append(steer(net, sol, layer, g1, g2, alphas, direction=d)["agree_g2"])
            # control 2: a third goal's direction (specificity)
            g3 = int(rng.choice([g for g in range(K) if g not in (g1, g2)]))
            H = net.all_activations()[layer]
            v3 = (H[:, g3] - H[:, g1]).mean(0)
            other2.append(steer(net, sol, layer, g1, g2, alphas, direction=v3)["agree_g2"])
            # per-state alpha at which the action first becomes g2-optimal, vs occupancy margin
            acts = np.stack(res["actions"])            # [n_alpha, S]
            supp2 = sol.pi[g2] > 0
            ok = supp2[np.arange(acts.shape[1])[None, :], acts]  # [n_alpha, S]
            for s in np.where(res["differ_mask"])[0]:
                a1 = int(np.argmax(sol.pi[g1, s])); a2 = int(np.argmax(sol.pi[g2, s]))
                m2 = occ.Q[g2, s].max() - occ.Q[g2, s, a1]
                if m2 <= 1e-12:      # g1's action is already g2-optimal: nothing to flip
                    continue
                hits = np.where(ok[:, s])[0]
                flip_alpha.append(alphas[hits[0]] if len(hits) else np.inf)
                margin.append(m2)
                margin1.append(occ.Q[g1, s, a1] - occ.Q[g1, s, a2])
        flip_alpha = np.array(flip_alpha); margin = np.array(margin); margin1 = np.array(margin1)
        # occupancy model with linear interpolation of rho(g|s,·): flip when
        # (1-a) Q1 + a Q2 changes argmax  ->  a* = m1 / (m1 + m2)
        alpha_pred = margin1 / np.maximum(margin1 + margin, 1e-12)
        fin = np.isfinite(flip_alpha)
        from scipy.stats import spearmanr
        def _corr(a, b):
            return float(spearmanr(a, b).correlation) if len(a) > 5 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
        corr = _corr(flip_alpha[fin], margin[fin])
        corr_pred = _corr(flip_alpha[fin], alpha_pred[fin])
        mae_pred = float(np.mean(np.abs(flip_alpha[fin] - alpha_pred[fin]))) if fin.any() else float("nan")
        out["layers"][layer] = {
            "agree_g2": np.mean(curves2, 0).tolist(),
            "agree_g1": np.mean(curves1, 0).tolist(),
            "agree_g2_excl": np.mean(curves2x, 0).tolist(),
            "agree_g2_random_dir": np.mean(rand2, 0).tolist(),
            "agree_g2_other_goal_dir": np.mean(other2, 0).tolist(),
            "frac_flipped_by_alpha1": float(np.mean(flip_alpha <= 1.0)),
            "frac_flipped_by_alpha2": float(np.mean(flip_alpha <= 2.0)),
            "flip_alpha_vs_margin_spearman": corr,
            "flip_alpha_vs_predicted_spearman": corr_pred,
            "flip_alpha_vs_predicted_mae": mae_pred,
            "flip_alpha": flip_alpha.tolist(), "margin": margin.tolist(), "alpha_pred": alpha_pred.tolist(),
        }
    out["goal_pairs"] = [(int(a), int(b)) for a, b in pairs]
    return out


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
def activations_from_state_dict(net: PolicyNet, state_dict: dict) -> dict[str, np.ndarray]:
    net.load_state_dict(state_dict)
    return net.all_activations()


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if k not in ("linkage", "cluster_labels", "components", "flip_alpha", "margin", "alpha_pred", "differ_mask", "direction", "actions")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
