"""TASK2 analyses: policy-switch sweep ground truth, quotient pairs, extended
hypothesis set, interaction ablation, steering predictors, dimensionality."""

from __future__ import annotations

import numpy as np
import torch

from . import geometry as G
from .analysis import HIDDEN_LAYERS, state_views, steer
from .envs import make_switch_env
from .gridworld import GridWorld
from .model import PolicyNet
from .occupancy import Occupancy, goal_occupancy, successor_representation
from .planning import Solution, solve_all_goals, optimal_dataset


# ---------------------------------------------------------------------------
# Task 2: ground truth across the sweep
# ---------------------------------------------------------------------------
def policy_table(sol: Solution) -> np.ndarray:
    K, S, A = sol.pi.shape
    return np.transpose(sol.pi, (1, 0, 2)).reshape(S, K * A)


def support_table(sol: Solution) -> np.ndarray:
    K, S, A = sol.pi.shape
    return np.transpose(sol.pi > 0, (1, 0, 2)).reshape(S, K * A)


def margins(sol: Solution) -> np.ndarray:
    """Best-minus-second-best Q per (goal, state): [K, S]."""
    q = np.sort(sol.Q, axis=-1)
    return q[..., -1] - q[..., -2]


def advantage_table(sol: Solution) -> np.ndarray:
    K, S, A = sol.Q.shape
    adv = sol.Q - sol.Q.max(-1, keepdims=True)
    return np.transpose(adv, (1, 0, 2)).reshape(S, K * A)


def sweep_ground_truth(lams, gamma: float = 0.9) -> dict:
    envs, sols, occs = [], [], []
    for lam in lams:
        e = make_switch_env(float(lam)); sol = solve_all_goals(e, gamma)
        envs.append(e); sols.append(sol); occs.append(goal_occupancy(e, sol.pi, gamma))
    L = len(lams)
    tables = np.stack([policy_table(s) for s in sols])          # [L, S, K*A]
    supports = np.stack([support_table(s) for s in sols])       # [L, S, K*A] bool
    marg = np.stack([margins(s) for s in sols])                  # [L, K, S]
    a0 = sols[0].Q.argmax(-1)                                    # [K, S] lambda=0 optimal action
    signed = np.zeros_like(marg)
    for i, s in enumerate(sols):
        Q = s.Q
        q0 = np.take_along_axis(Q, a0[..., None], -1)[..., 0]
        Qm = Q.copy(); np.put_along_axis(Qm, a0[..., None], -np.inf, -1)
        signed[i] = q0 - Qm.max(-1)
    K, S = a0.shape
    supp_ks = np.stack([s.pi > 0 for s in sols])                 # [L, K, S, A]
    boundaries = {}
    for g in range(K):
        for st in range(S):
            boundaries[(g, st)] = [i for i in range(L - 1) if (supp_ks[i, g, st] != supp_ks[i + 1, g, st]).any()]
    return {"lams": list(map(float, lams)), "envs": envs, "sols": sols, "occs": occs,
            "policy_tables": tables, "supports": supports, "margin": marg, "signed_margin": signed,
            "boundaries": boundaries,
            "Z_sa": np.stack([o.Z_sa for o in occs]),
            "SR": np.stack([successor_representation(e, gamma) for e in envs]),
            "advantage": np.stack([advantage_table(s) for s in sols]),
            "n_flips": [int((supp_ks[i] != supp_ks[i + 1]).any(-1).sum()) for i in range(L - 1)]}


# ---------------------------------------------------------------------------
# extended hypothesis set (Task 6)
# ---------------------------------------------------------------------------
def extended_rdms(env: GridWorld, occ: Occupancy, sol: Solution, gamma: float) -> dict[str, np.ndarray]:
    m = margins(sol).T                                            # [S, K]
    return {
        "occupancy": G.rdm(occ.Z_sa),                             # == Q geometry up to 1/gamma
        "SR": G.rdm(successor_representation(env, gamma)),
        "policy": G.rdm(policy_table(sol)),
        "advantage": G.rdm(advantage_table(sol)),
        "margin": G.rdm(m),
        "value": G.rdm(occ.Z_state),
        "spatial": G.rdm(env.coords),
    }


def extended_regression(H_layers: dict[str, np.ndarray], rdms: dict[str, np.ndarray],
                        regressors=("occupancy", "SR", "policy", "advantage", "margin", "spatial"),
                        view: str = "mean") -> dict[str, dict]:
    out = {}
    for layer in HIDDEN_LAYERS:
        R = G.rdm(state_views(H_layers[layer])[view])
        rep = {f"rsa_{k}": G.rsa(R, v) for k, v in rdms.items()}
        rep["regression"] = G.rdm_regression(R, {k: rdms[k] for k in regressors})
        rep["regression_no_policy"] = G.rdm_regression(R, {k: rdms[k] for k in regressors if k != "policy"})
        rep["partial_policy_given_rest"] = G.partial_rsa(R, rdms["policy"], [rdms[k] for k in regressors if k != "policy"])
        rep["partial_occupancy_given_rest"] = G.partial_rsa(R, rdms["occupancy"], [rdms[k] for k in regressors if k != "occupancy"])
        rep["partial_margin_given_rest"] = G.partial_rsa(R, rdms["margin"], [rdms[k] for k in regressors if k != "margin"])
        rep["partial_advantage_given_rest"] = G.partial_rsa(R, rdms["advantage"], [rdms[k] for k in regressors if k != "advantage"])
        out[layer] = rep
    return out


# ---------------------------------------------------------------------------
# Task 5: quotient pairs
# ---------------------------------------------------------------------------
def quotient_pairs(env: GridWorld, sol: Solution, occ: Occupancy, q: float = 0.25) -> dict:
    tables = support_table(sol)
    Docc = G.rdm(occ.Z_sa)
    iu = np.triu_indices(env.n_states, 1)
    same = np.array([np.array_equal(tables[i], tables[j]) for i, j in zip(*iu)])
    d = Docc[iu]
    goal_mask = np.isin(iu[0], env.goal_states) | np.isin(iu[1], env.goal_states)
    lo = np.quantile(d[~same & ~goal_mask], q)          # "very similar occupancy" among different-table pairs
    B = [(int(i), int(j)) for i, j, s, dd, gm in zip(*iu, same, d, goal_mask) if (not s) and dd <= lo and not gm]
    thr = max(np.median([Docc[i, j] for i, j in B]) if B else 0.0,
              np.quantile(d[same & ~goal_mask], 1 - q) if (same & ~goal_mask).any() else np.inf)
    A = [(int(i), int(j)) for i, j, s, dd, gm in zip(*iu, same, d, goal_mask) if s and dd >= thr and not gm]
    return {"case_A": A, "case_B": B, "Docc": Docc, "same_table": same, "n_same": int(same.sum()), "n_pairs": len(d)}


def quotient_pair_report(H_layers: dict[str, np.ndarray], H_random: dict[str, np.ndarray], pairs: dict, view: str = "mean") -> dict:
    out = {}
    for layer in HIDDEN_LAYERS:
        R = G.rdm(state_views(H_layers[layer])[view]); Rr = G.rdm(state_views(H_random[layer])[view])
        med, medr = np.median(G.upper(R)), np.median(G.upper(Rr))
        dA = np.array([R[i, j] for i, j in pairs["case_A"]]) / med
        dB = np.array([R[i, j] for i, j in pairs["case_B"]]) / med
        dAr = np.array([Rr[i, j] for i, j in pairs["case_A"]]) / medr
        dBr = np.array([Rr[i, j] for i, j in pairs["case_B"]]) / medr
        oA = np.array([pairs["Docc"][i, j] for i, j in pairs["case_A"]])
        oB = np.array([pairs["Docc"][i, j] for i, j in pairs["case_B"]])
        out[layer] = {"d_h_A": float(np.median(dA)), "d_h_B": float(np.median(dB)),
                      "d_h_A_random": float(np.median(dAr)), "d_h_B_random": float(np.median(dBr)),
                      "d_occ_A": float(np.median(oA)), "d_occ_B": float(np.median(oB)),
                      "frac_pairs_A_below_B": float(np.mean(dA[:, None] < dB[None, :])) if len(dA) and len(dB) else float("nan"),
                      "n_A": len(dA), "n_B": len(dB)}
    return out


# ---------------------------------------------------------------------------
# Task 7: interaction ablation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _actions_from(net: PolicyNet, layer: str, H: np.ndarray) -> np.ndarray:
    S, K, d = H.shape
    logits = net.forward_from(layer, torch.as_tensor(H.reshape(S * K, d), dtype=torch.float32))
    return logits.argmax(-1).numpy().reshape(S, K)


def _behaviour_metrics(net: PolicyNet, sol: Solution, layer: str, H: np.ndarray, rng: np.random.Generator, n_pairs: int = 10) -> dict:
    S, K, d = H.shape
    acts = _actions_from(net, layer, H)                          # [S, K]
    supp = np.transpose(sol.pi > 0, (1, 0, 2))                   # [S, K, A]
    ok = supp[np.arange(S)[:, None], np.arange(K)[None, :], acts]
    not_goal = np.ones((S, K), bool)
    for gi, g in enumerate(net.env.goal_states):
        not_goal[g, gi] = False
    acc = float(ok[not_goal].mean())
    iu = np.triu_indices(K, 1)
    sens = float((acts[:, iu[0]] != acts[:, iu[1]]).mean())
    # steering success with this activation tensor
    succ = []
    for _ in range(n_pairs):
        g1, g2 = rng.choice(K, 2, replace=False)
        v = (H[:, g2] - H[:, g1]).mean(0)
        patched = H[:, g1] + v
        a = net.forward_from(layer, torch.as_tensor(patched, dtype=torch.float32)).argmax(-1).numpy()
        differ = (supp[:, g1] != supp[:, g2]).any(-1) & not_goal[:, g1] & not_goal[:, g2]
        if differ.any():
            succ.append(float(supp[np.arange(S), g2, a][differ].mean()))
    return {"accuracy": acc, "goal_sensitivity": sens, "steer_success": float(np.mean(succ)) if succ else float("nan")}


def interaction_ablation(net: PolicyNet, sol: Solution, occ: Occupancy, layer: str, n_random: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    H = net.all_activations()[layer]
    parts = G.two_way_decomposition(H)
    mu, a_s, b_g, r = parts["components"]
    S, K, d = H.shape
    out = {"interaction_fraction": parts["interaction"], "baseline": _behaviour_metrics(net, sol, layer, H, rng)}
    # (i) remove the exact interaction component
    out["remove_component"] = _behaviour_metrics(net, sol, layer, H - r, rng)
    ctrl = []
    for _ in range(n_random):
        noise = rng.normal(size=r.shape); noise *= np.linalg.norm(r, axis=-1, keepdims=True) / (np.linalg.norm(noise, axis=-1, keepdims=True) + 1e-12)
        ctrl.append(_behaviour_metrics(net, sol, layer, H - noise, rng))
    out["random_component"] = {k: float(np.mean([c[k] for c in ctrl])) for k in ctrl[0]}
    # (ii) project out the top-k principal subspace of the interaction term
    rf = r.reshape(S * K, d)
    U, sv, Vt = np.linalg.svd(rf - rf.mean(0), full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum(); k = int(np.searchsorted(np.cumsum(var), 0.9) + 1)
    Pk = Vt[:k].T @ Vt[:k]
    out["k_subspace"] = k
    add = (a_s[:, None, :] + b_g[None, :, :]).reshape(S * K, d)
    out["additive_var_in_interaction_subspace"] = float(((add @ Pk) ** 2).sum() / ((add ** 2).sum() + 1e-12))
    out["remove_subspace"] = _behaviour_metrics(net, sol, layer, H - H @ Pk, rng)
    ctrl = []
    for _ in range(n_random):
        Qr, _ = np.linalg.qr(rng.normal(size=(d, k)))
        ctrl.append(_behaviour_metrics(net, sol, layer, H - H @ (Qr @ Qr.T), rng))
    out["random_subspace"] = {k_: float(np.mean([c[k_] for c in ctrl])) for k_ in ctrl[0]}
    # (iii) keep ONLY the interaction (plus mean): does h_sg alone carry the action?
    out["only_interaction_plus_state"] = _behaviour_metrics(net, sol, layer, mu + a_s[:, None, :] + r, rng)
    out["only_additive_plus_goal"] = out["remove_component"]
    return out


# ---------------------------------------------------------------------------
# Task 8: steering predictors
# ---------------------------------------------------------------------------
def _jacobian_from(net: PolicyNet, layer: str, h: np.ndarray) -> np.ndarray:
    """d logits / d h at a single activation h (numerical via autograd)."""
    x = torch.as_tensor(h, dtype=torch.float32).requires_grad_(True)
    return torch.autograd.functional.jacobian(lambda z: net.forward_from(layer, z[None])[0], x).detach().numpy()


def _linearised_alpha(logits0: np.ndarray, slope: np.ndarray, supp2: np.ndarray) -> float:
    """Smallest alpha >= 0 at which argmax of logits0 + alpha*slope lies in supp2."""
    best = np.inf
    for a in np.where(supp2)[0]:
        need = 0.0; feasible = True
        for b in range(len(logits0)):
            if b == a:
                continue
            gap = logits0[b] - logits0[a]; ds = slope[a] - slope[b]
            if gap > 0:
                if ds <= 0:
                    feasible = False; break
                need = max(need, gap / ds)
        if feasible:
            best = min(best, need)
    return float(best)


def steering_predictors(net: PolicyNet, sol: Solution, occ: Occupancy, layer: str, n_goal_pairs: int = 15,
                        alphas=np.linspace(0, 2, 41), seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    K = net.env.n_goals; S = net.env.n_states
    H = net.all_activations()[layer]
    cols = {k: [] for k in ("alpha_star", "alpha_lin", "m1", "m2", "d_occ", "logit_margin", "occ_model", "state", "g1", "g2")}
    for _ in range(n_goal_pairs):
        g1, g2 = map(int, rng.choice(K, 2, replace=False))
        res = steer(net, sol, layer, g1, g2, alphas)
        v = res["direction"]; acts = np.stack(res["actions"])
        supp1, supp2 = sol.pi[g1] > 0, sol.pi[g2] > 0
        with torch.no_grad():
            logits0 = net.forward_from(layer, torch.as_tensor(H[:, g1], dtype=torch.float32)).numpy()
        for s in np.where(res["differ_mask"])[0]:
            a1 = int(np.argmax(sol.pi[g1, s])); a2 = int(np.argmax(sol.pi[g2, s]))
            m2 = occ.Q[g2, s].max() - occ.Q[g2, s, a1]
            if m2 <= 1e-12:
                continue
            m1 = occ.Q[g1, s, a1] - occ.Q[g1, s, a2]
            hits = np.where(supp2[s][acts[:, s]])[0]
            cols["alpha_star"].append(alphas[hits[0]] if len(hits) else np.inf)
            J = _jacobian_from(net, layer, H[s, g1])
            cols["alpha_lin"].append(_linearised_alpha(logits0[s], J @ v, supp2[s]))
            cols["m1"].append(m1); cols["m2"].append(m2)
            cols["d_occ"].append(float(np.linalg.norm(occ.Q[g2, s] - occ.Q[g1, s])))
            l = logits0[s]; cur = int(np.argmax(l))
            cols["logit_margin"].append(float(l[cur] - l[supp2[s]].max()))
            cols["occ_model"].append(m1 / max(m1 + m2, 1e-12))
            cols["state"].append(s); cols["g1"].append(g1); cols["g2"].append(g2)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def steering_predictor_summary(tab: dict[str, np.ndarray]) -> dict:
    from scipy.stats import spearmanr
    y = tab["alpha_star"]; fin = np.isfinite(y)
    out = {"n": int(fin.sum()), "n_never_flipped": int((~fin).sum())}
    preds = ("alpha_lin", "logit_margin", "m1", "m2", "d_occ", "occ_model")
    for k in preds:
        x = tab[k]; ok = fin & np.isfinite(x)
        out[f"spearman_{k}"] = float(spearmanr(y[ok], x[ok]).correlation) if ok.sum() > 5 and np.std(x[ok]) > 0 else float("nan")
    ok = fin & np.isfinite(tab["alpha_lin"])
    out["mae_alpha_lin"] = float(np.mean(np.abs(y[ok] - tab["alpha_lin"][ok]))) if ok.any() else float("nan")
    out["mae_occ_model"] = float(np.mean(np.abs(y[fin] - tab["occ_model"][fin]))) if fin.any() else float("nan")
    # multiple regression (rank-transformed) of alpha* on the predictors
    from scipy.stats import rankdata
    X = np.column_stack([rankdata(np.nan_to_num(tab[k][fin], nan=0.0, posinf=1e6)) for k in preds])
    X = (X - X.mean(0)) / (X.std(0) + 1e-12); yr = rankdata(y[fin]); yr = (yr - yr.mean()) / yr.std()
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yr)), X]), yr, rcond=None)
    out["regression_beta"] = dict(zip(preds, map(float, beta[1:])))
    out["regression_r2"] = float(1 - np.var(yr - np.column_stack([np.ones(len(yr)), X]) @ beta) / np.var(yr))
    return out


# ---------------------------------------------------------------------------
# Task 10: dimensionality
# ---------------------------------------------------------------------------
def dimensionality(X: np.ndarray) -> dict:
    lam = G.pca_spectrum(X); ev = lam / lam.sum() if lam.sum() > 0 else lam
    return {"participation_ratio": G.participation_ratio(X), "entropy_rank": G.entropy_rank(X),
            "n_pcs_90": int(np.searchsorted(np.cumsum(ev), 0.9) + 1) if lam.sum() > 0 else 0,
            "two_nn": G.two_nn_dimension(X) if len(X) >= 10 else float("nan"),
            "spectrum": ev[:10].tolist()}
