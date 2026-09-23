"""TASK11 part 2: causal tests of the decoded posterior (rounds/r12_hidden_goal/causal/THEORY.md).

A. posterior transplant: move A's state at t to B's decoded belief (or swap in B's state) and
   compare the whole future with genuine-B runs and with Bayes targets.
B. equal-belief equivalence: histories with (near-)equal posteriors, patched at the GRU's
   complete cut (= run on the other history), compared with random pairs and with Bayes.
C. same action, different belief: whether the pair stays causally distinguishable (GRU flip
   rate) and where along depth the belief is quotiented by the action (transformer ladder)."""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from . import beliefprobe as BP
from . import latentgoal as LG

T_OBS = LG.T_OBS
EPS = 1e-12


# ---- exact filtering from an arbitrary joint state ------------------------------------------
def filter_continue(env, J0, Xc):
    """J0 [n, K, C] = P(G, c_t | history); Xc [n, L] continuation tokens (1..M).
    Returns the joint at offsets k = 0..L: [n, L+1, K, C]."""
    n, L = Xc.shape
    out = np.zeros((n, L + 1, env.K, env.C)); out[:, 0] = J0
    a = J0.copy()
    for k in range(L):
        a = a @ env.A
        lik = env.L.transpose(1, 0, 2)[:, :, Xc[:, k] - 1].transpose(2, 0, 1)
        a = a * lik; a /= a.sum((1, 2), keepdims=True)
        out[:, k + 1] = a
    return out


def ideal(env, J, objective):
    """Bayes-optimal output from joint states [..., K, C] (all positions t >= 1)."""
    b = J.sum(-1)
    if objective == "goal":
        return b
    if objective == "act_soft":
        return LG.act_soft(b)
    if objective == "act_hard":
        return LG.act_hard(b)
    if objective == "next_obs":
        return np.einsum("...kc,ckm->...m", J @ env.A, env.L)
    raise ValueError(objective)


def transplant(J, b_new):
    """Replace the goal marginal of joint states J [n, K, C] by b_new, keeping P(c | g)."""
    b = J.sum(-1, keepdims=True)
    return J / np.clip(b, EPS, None) * b_new[:, :, None]


def sample_continuation(env, J0, L, rng):
    """Continuation tokens drawn from the posterior predictive of the joint states J0 [n, K, C]."""
    n = len(J0); flat = J0.reshape(n, -1)
    idx = (rng.random((n, 1)) < flat.cumsum(1)).argmax(1)
    g, c = idx // env.C, idx % env.C
    X = np.zeros((n, L), int)
    for k in range(L):
        if env.C == 2:
            stay = rng.random(n) < env.A[c, c]; c = np.where(stay, c, 1 - c)
        P = env.L[c, g]
        X[:, k] = (rng.random((n, 1)) < P.cumsum(1)).argmax(1) + 1
    return X


# ---- divergences ---------------------------------------------------------------------------
def kl(p, q):
    return (p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(q, EPS, None)))).sum(-1)


def js(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def gap_closed(d_int, d_none):
    """1 - E[d_int] / E[d_none] per offset k (ratio of means over pairs). d: [n, L+1]."""
    return 1 - d_int.mean(0) / np.clip(d_none.mean(0), EPS, None)


# ---- pairs (environment only) ---------------------------------------------------------------
def pairs_transplant(env, t, n, seed):
    """(X_A, X_B) full sequences with ||y_A - y_B|| >= 2 at position t."""
    XA, _, _ = LG.sample(env, 6 * n, seed=seed); XB, _, _ = LG.sample(env, 6 * n, seed=seed + 1)
    yA = LG.log_odds(LG.posterior(env, XA)[:, t]); yB = LG.log_odds(LG.posterior(env, XB)[:, t])
    keep = np.flatnonzero(np.linalg.norm(yA - yB, axis=1) >= 2.0)[:n]
    return XA[keep], XB[keep]


def pairs_equal(env, n, seed, mode, t=12, pool=300000):
    """Prefix pairs (H1, H2) of length t with (near-)equal posteriors.
    mode 'iid_exact' | 'chan_equal_b' | 'chan_equal_joint'. Returns token arrays incl. BOS."""
    X, _, _ = LG.sample(env, pool, T=t, seed=seed)
    J = LG.filter_joint(env, X)[:, -1]; b = J.sum(-1); y = LG.log_odds(b)
    C = LG.counts(X, env.M)[:, -1]
    tol = 1e-7 if mode == "iid_exact" else 0.05
    P = cKDTree(y).query_pairs(tol, p=np.inf, output_type="ndarray")
    d = np.abs(C[P[:, 0]] - C[P[:, 1]]).sum(1)
    if mode == "iid_exact":
        P = P[d >= 8]
    else:
        pon = J[..., 1] / np.clip(b, EPS, None)                    # P(on | g, H)  [n, K]
        dmax = np.abs(pon[P[:, 0]] - pon[P[:, 1]]).max(1)
        don = np.abs(J[P[:, 0], :, 1].sum(1) - J[P[:, 1], :, 1].sum(1))
        P = P[(d > 0) & (don >= 0.15)] if mode == "chan_equal_b" else P[(d > 0) & (dmax < 0.02)]
    rng = np.random.default_rng(seed)
    P = P[rng.permutation(len(P))]
    used, out = set(), []                                           # disjoint pairs
    for i, j in P:
        if i not in used and j not in used:
            out.append((i, j)); used.update((i, j))
        if len(out) == n:
            break
    out = np.array(out)
    return X[out[:, 0]], X[out[:, 1]], J[out[:, 0]]


def pairs_random(env, n, seed, t=12):
    X, _, _ = LG.sample(env, 2 * n, T=t, seed=seed)
    J = LG.filter_joint(env, X)[:, -1]
    return X[:n], X[n:], J[:n]


def pairs_same_action(env, n, seed, t=12, tries=40):
    """Prefixes with the same untied optimal action, ||dy|| >= 1.5, and a continuation (from
    H1's predictive) under which the Bayes-optimal action later differs between them."""
    rng = np.random.default_rng(seed)
    X, _, _ = LG.sample(env, 200000, T=t, seed=seed)
    J = LG.filter_joint(env, X)[:, -1]; b = J.sum(-1); y = LG.log_odds(b)
    q = LG.q_values(b); best = q >= q.max(1, keepdims=True) - 1e-9
    a = np.where(best.sum(1) == 1, q.argmax(1), -1)
    H1, H2, Cs = [], [], []
    for _ in range(tries):
        i = rng.integers(0, len(X), 20000); j = rng.integers(0, len(X), 20000)
        ok = (a[i] >= 0) & (a[i] == a[j]) & (np.linalg.norm(y[i] - y[j], axis=1) >= 1.5)
        i, j = i[ok], j[ok]
        c = sample_continuation(env, J[i], T_OBS - t, rng)
        a1 = LG.act_hard(filter_continue(env, J[i], c).sum(-1)); a2 = LG.act_hard(filter_continue(env, J[j], c).sum(-1))
        u1, u2 = a1.max(-1) == 1, a2.max(-1) == 1
        differ = ((a1.argmax(-1) != a2.argmax(-1)) & u1 & u2)[:, 1:].any(1)
        H1.append(X[i[differ]]); H2.append(X[j[differ]]); Cs.append(c[differ])
        if sum(len(h) for h in H1) >= n:
            break
    return np.concatenate(H1)[:n], np.concatenate(H2)[:n], np.concatenate(Cs)[:n]


# ---- running models from a state ------------------------------------------------------------
@torch.no_grad()
def gru_states(net, X):
    return net.states(torch.as_tensor(X)).double().numpy()


@torch.no_grad()
def gru_outputs(net, X, t):
    """Output distributions at positions t..T of full sequences X."""
    h = net.states(torch.as_tensor(X))
    return torch.softmax(net.out(h[:, t:]), -1).double().numpy()


@torch.no_grad()
def gru_from(net, h, Xc):
    """Outputs at offsets k = 0..L starting from state h [n, d] at position t, continuation Xc."""
    h0 = torch.as_tensor(h, dtype=torch.float32)
    hs, _ = net.gru(net.emb(torch.as_tensor(Xc)), h0[None].contiguous())
    H = torch.cat([h0[:, None], hs], 1)
    return torch.softmax(net.out(H), -1).double().numpy()


def _tfm_probs(net, z):
    dim = int((net.out_mask == 0).sum())
    return torch.softmax(z + net.out_mask, -1)[..., :dim].double().numpy()


@torch.no_grad()
def tfm_residuals(net, X):
    return [r.double().numpy() for r in net.residuals(torch.as_tensor(X))]


@torch.no_grad()
def tfm_outputs(net, X, t):
    return _tfm_probs(net, net.forward_all(torch.as_tensor(X))[:, t:])


@torch.no_grad()
def tfm_patch(net, X, t, site, h_new):
    """Replace the residual after block `site` at position t by h_new; outputs at t..T."""
    R = net.residuals(torch.as_tensor(X))[site].clone()
    R[:, t] = torch.as_tensor(h_new, dtype=R.dtype)
    return _tfm_probs(net, net.run_from(R, site)[:, t:])


# ---- probe / encoder and the interventions ----------------------------------------------------
class BeliefMap:
    """Decoder z(h) = h Wd + cd (least squares) and encoder h ~ y We + ce (reverse regression)."""

    def __init__(self, H, Y):
        W = BP._fit(H, Y); self.Wd, self.cd = W[:-1], W[-1]
        E = BP._fit(Y, H); self.We, self.ce = E[:-1], E[-1]
        self.Wd_pinv = np.linalg.pinv(self.Wd)                      # [K-1, d]
        self.M = self.We @ self.Wd                                   # [K-1, K-1]: encoder moves, decoded

    def z(self, h):
        return h @ self.Wd + self.cd

    def probe_e(self, h, y_star):
        g = np.linalg.solve(self.M.T, (y_star - self.z(h)).T).T
        return h + g @ self.We

    def probe_d(self, h, y_star):
        return h + (y_star - self.z(h)) @ self.Wd_pinv

    def rand_like(self, h, shift, rng):
        v = rng.normal(size=h.shape); v /= np.linalg.norm(v, axis=1, keepdims=True)
        return h + v * np.linalg.norm(shift, axis=1, keepdims=True)


def fit_map(states, E: "BP.EvalSet"):
    """states [n_seq, T+1, d] on the eval set -> BeliefMap fit on positions 1..T."""
    return BeliefMap(states[:, 1:].reshape(-1, states.shape[-1]), E.y)


# ---- tests ------------------------------------------------------------------------------------
INTERVENTIONS = ("swap", "probe_e", "probe_d", "probe_e_half", "rand")


def test_transplant(run_full, run_from_state, get_state, bmap, env, objective, XA, XB, t, rng):
    """Generic over architectures. run_full(X) -> outputs t..T; run_from_state(h) -> outputs
    t..T of A's sequence with the state at t replaced; get_state(X) -> state at t."""
    XS = np.concatenate([XB[:, :t + 1], XA[:, t + 1:]], 1)
    pA, pS = run_full(XA), run_full(XS)
    JA = LG.filter_joint(env, XA)[:, t]; JB = LG.filter_joint(env, XB)[:, t]
    bA, bB = JA.sum(-1), JB.sum(-1); yA, yB = LG.log_odds(bA), LG.log_odds(bB)
    Xc = XA[:, t + 1:]
    tgt_S = ideal(env, filter_continue(env, JB, Xc), objective)
    tgt_T = ideal(env, filter_continue(env, transplant(JA, bB), Xc), objective)
    y_half = yA + 0.5 * (yB - yA)
    b_half = np.exp(np.c_[y_half, np.zeros(len(y_half))]); b_half /= b_half.sum(1, keepdims=True)
    tgt_T_half = ideal(env, filter_continue(env, transplant(JA, b_half), Xc), objective)
    tgt_A = ideal(env, filter_continue(env, JA, Xc), objective)
    hA, hB = get_state(XA), get_state(XB)
    he = bmap.probe_e(hA, yB)
    states = {"swap": hB, "probe_e": he, "probe_d": bmap.probe_d(hA, yB),
              "probe_e_half": bmap.probe_e(hA, y_half), "rand": bmap.rand_like(hA, he - hA, rng)}
    out = {"none_vs_S_model": js(pA, pS).mean(0), "none_vs_bayesS": kl(tgt_S, pA).mean(0),
           "none_vs_bayesT": kl(tgt_T, pA).mean(0), "none_vs_bayesT_half": kl(tgt_T_half, pA).mean(0),
           "bayes_S_vs_T": js(tgt_S, tgt_T).mean(0), "model_A_vs_bayesA": kl(tgt_A, pA).mean(0)}
    for name, h in states.items():
        p = run_from_state(h)
        out[f"{name}:F_model_S"] = gap_closed(js(p, pS), js(pA, pS))
        out[f"{name}:kl_bayesS"] = kl(tgt_S, p).mean(0)
        out[f"{name}:kl_bayesT"] = kl(tgt_T, p).mean(0)
        out[f"{name}:F_bayesT"] = gap_closed(kl(tgt_T, p), kl(tgt_T, pA))
        out[f"{name}:F_bayesS"] = gap_closed(kl(tgt_S, p), kl(tgt_S, pA))
        if name == "probe_e_half":
            out[f"{name}:F_bayesT_half"] = gap_closed(kl(tgt_T_half, p), kl(tgt_T_half, pA))
        if name == "swap":
            out["swap:max_abs_vs_S"] = float(np.abs(p - pS).max())
    return out


def test_equal(run_full, env, objective, H1, H2, J1, rng, n_cont=8):
    """Downstream divergence between H1+c and H2+c, per offset, with the Bayes divergence."""
    t = H1.shape[1] - 1
    dm, db = [], []
    for _ in range(n_cont):
        c = sample_continuation(env, J1, T_OBS - t, rng)
        X1, X2 = np.concatenate([H1, c], 1), np.concatenate([H2, c], 1)
        p1, p2 = run_full(X1), run_full(X2)
        i1 = ideal(env, LG.filter_joint(env, X1)[:, t:], objective); i2 = ideal(env, LG.filter_joint(env, X2)[:, t:], objective)
        dm.append(js(p1, p2)); db.append(js(i1, i2))
    dm, db = np.stack(dm, 1), np.stack(db, 1)                        # [pairs, cont, L+1]
    pair_m, pair_b = dm[:, :, 1:].mean((1, 2)), db[:, :, 1:].mean((1, 2))
    rho = spearmanr(pair_m, pair_b).correlation if pair_b.std() > 0 else float("nan")
    return {"model": dm.mean((0, 1)), "bayes": db.mean((0, 1)), "spearman_pairs": float(rho)}


def test_flip(run_full, env, objective, H1, H2, C):
    """Same-action pairs: on the future steps where the Bayes action differs, how often the
    model's action differs between the two runs."""
    t = H1.shape[1] - 1
    X1, X2 = np.concatenate([H1, C], 1), np.concatenate([H2, C], 1)
    p1, p2 = run_full(X1), run_full(X2)
    b1 = LG.filter_joint(env, X1)[:, t:].sum(-1); b2 = LG.filter_joint(env, X2)[:, t:].sum(-1)
    a1, a2 = LG.act_hard(b1), LG.act_hard(b2)
    diff = (a1.max(-1) == 1) & (a2.max(-1) == 1) & (a1.argmax(-1) != a2.argmax(-1))
    diff[:, 0] = False
    if objective == "goal":
        m1, m2 = LG.q_values(p1).argmax(-1), LG.q_values(p2).argmax(-1)
    elif objective.startswith("act"):
        m1, m2 = p1.argmax(-1), p2.argmax(-1)
    else:
        return {"flip_rate": float("nan"), "n_steps": int(diff.sum())}
    same_now = float((m1[:, 0] == m2[:, 0]).mean())
    return {"flip_rate": float((m1 != m2)[diff].mean()), "n_steps": int(diff.sum()), "same_action_at_t": same_now}


def ladder(sites: dict, E: "BP.EvalSet"):
    """Per site: overall R²_y, within-action R²_y, action decodability, and the decodability of
    the neutral-token count (centred on its expectation given t: zero evidential value)."""
    K = E.env.K
    neutral = E.counts[:, K:].sum(1)
    neutral = neutral - E.t * (neutral.sum() / E.t.sum())
    out = {}
    for name, H in sites.items():
        H = np.asarray(H, np.float64)
        sse = sst = 0.0
        for a in np.unique(E.action):
            m = E.action == a
            if m.sum() < 200:
                continue
            P = BP.cv_pred(H[m], E.y[m], E.fold[m])
            sse += ((E.y[m] - P) ** 2).sum(); sst += ((E.y[m] - E.y[m].mean(0)) ** 2).sum()
        onehot = np.eye(K + 1)[E.action]
        out[name] = {"r2y": BP.r2(E.y, BP.cv_pred(H, E.y, E.fold)), "r2y_within_action": float(1 - sse / sst),
                     "action_acc": float((BP.cv_pred(H, onehot, E.fold).argmax(1) == E.action).mean()),
                     "neutral_r2": BP.r2(neutral[:, None], BP.cv_pred(H, neutral[:, None], E.fold))}
    return out
