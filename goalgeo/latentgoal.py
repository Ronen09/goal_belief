"""TASK11 (round 12): a latent goal G, noisy evidence about it, and the exact Bayesian filter.

An episode samples G ~ prior over K goals and emits T observation tokens. Two evidence
processes:

  'iid'     o_t ~ L[G] independently. The posterior log-odds are then an affine function of
            the token counts (y_t = y_0 + sum_s lambda(o_s)), so any representation that holds
            the counts linearly also holds the log-odds linearly.
  'channel' a hidden sticky reliability channel c_t in {off, on} (Markov, P(stay) = STAY);
            o_t ~ L_on[G] when on and uniform over the alphabet when off. The exact filter runs
            over (G, c) jointly and the posterior over G is not an affine function of the
            counts: a run of consistent tokens is evidence that the channel is on, which makes
            each later token count more.

Token alphabet: 0 = BOS, 1..K = the goal-indicative tokens (token k favours goal k-1),
K+1, K+2 = two neutral tokens whose likelihood is the same under every goal.

Everything the models are trained on is computed exactly here: b_t over goals, the Bayes-optimal
action values Q_b(a) (commit to goal k pays 1 if G = k; a safe action pays SAFE), and the
posterior-predictive distribution of the next observation. Position t of a sequence
[BOS, o_1..o_T] has seen o_1..o_t, so position 0 carries the prior.

Actions do not affect observations in this design, so b_t = P(G | o_<=t, a_<t) = P(G | o_<=t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIT, NEIGH = 0.30, 0.20          # iid: P(own token), P(next goal's token)
HIT_ON, NEIGH_ON = 0.55, 0.15    # channel, when on
STAY = 0.95                      # channel persistence
SAFE = 0.5                       # reward of the safe action
TAU = 0.1                        # temperature of the soft action targets
T_OBS = 24


@dataclass
class Env:
    kind: str                    # 'iid' or 'channel'
    K: int
    prior: np.ndarray            # [K]
    L: np.ndarray                # [C, K, M] emission per channel state and goal
    A: np.ndarray                # [C, C] channel transitions
    c0: np.ndarray               # [C] initial channel distribution

    @property
    def M(self):                 # number of observation tokens
        return self.L.shape[-1]

    @property
    def V(self):                 # input vocabulary incl. BOS
        return self.M + 1

    @property
    def C(self):
        return self.L.shape[0]


def _goal_emission(K: int, hit: float, neigh: float) -> np.ndarray:
    M = K + 2
    E = np.zeros((K, M))
    for g in range(K):
        E[g, g] = hit
        E[g, (g + 1) % K] += neigh
        rest = [j for j in range(M) if E[g, j] == 0]
        E[g, rest] = (1 - E[g].sum()) / len(rest)
    return E


def make_env(kind: str = "iid", K: int = 4, prior=None) -> Env:
    prior = np.full(K, 1.0 / K) if prior is None else np.asarray(prior, float)
    M = K + 2
    if kind == "iid":
        L = _goal_emission(K, HIT, NEIGH)[None]
        A = np.ones((1, 1)); c0 = np.ones(1)
    elif kind == "channel":
        L = np.stack([np.full((K, M), 1.0 / M), _goal_emission(K, HIT_ON, NEIGH_ON)])   # c = 0 off, 1 on
        A = np.array([[STAY, 1 - STAY], [1 - STAY, STAY]]); c0 = np.array([0.5, 0.5])
    else:
        raise ValueError(kind)
    assert np.allclose(L.sum(-1), 1)
    return Env(kind, K, prior, L, A, c0)


def sample(env: Env, n: int, T: int = T_OBS, seed: int = 0):
    """Returns X [n, T+1] tokens (BOS first), G [n] goals, Cs [n, T] channel states."""
    rng = np.random.default_rng(seed)
    G = rng.choice(env.K, n, p=env.prior)
    Cs = np.zeros((n, T), int)
    c = rng.choice(env.C, n, p=env.c0)
    O = np.zeros((n, T), int)
    for t in range(T):
        if t > 0:
            stay = rng.random(n) < env.A[c, c]
            c = np.where(stay, c, 1 - c) if env.C == 2 else c
        Cs[:, t] = c
        P = env.L[c, G]                                     # [n, M]
        O[:, t] = (rng.random((n, 1)) < P.cumsum(1)).argmax(1)
    X = np.concatenate([np.zeros((n, 1), int), O + 1], 1)
    return X, G, Cs


def filter_joint(env: Env, X: np.ndarray):
    """Exact forward filter. Returns the joint posterior P(G, c_t | o_<=t) for every position:
    [n, T+1, K, C]. Position 0 (BOS) holds prior x initial channel distribution, read as the
    channel state 'about to emit o_1'."""
    n, T1 = X.shape
    out = np.zeros((n, T1, env.K, env.C))
    alpha = np.broadcast_to(env.prior[:, None] * env.c0[None, :], (n, env.K, env.C)).copy()
    out[:, 0] = alpha
    for t in range(1, T1):
        if t > 1:
            alpha = alpha @ env.A                           # c_{t-1} -> c_t
        o = X[:, t] - 1
        lik = env.L[:, :, :].transpose(1, 0, 2)[:, :, o]    # [K, C, n]
        alpha = alpha * lik.transpose(2, 0, 1)
        alpha /= alpha.sum((1, 2), keepdims=True)
        out[:, t] = alpha
    return out


def posterior(env: Env, X: np.ndarray) -> np.ndarray:
    """b_t(g) for every position: [n, T+1, K]."""
    return filter_joint(env, X).sum(-1)


def log_odds(b: np.ndarray) -> np.ndarray:
    """K-1 log-odds coordinates against the last goal: log b(g_k) / b(g_K)."""
    lb = np.log(np.clip(b, 1e-300, None))
    return lb[..., :-1] - lb[..., -1:]


def next_obs(env: Env, X: np.ndarray, J: np.ndarray | None = None) -> np.ndarray:
    """Posterior predictive of the next observation token at every position: [n, T+1, M]."""
    J = filter_joint(env, X) if J is None else J
    Jn = J.copy()
    Jn[:, 1:] = J[:, 1:] @ env.A                            # advance the channel one step (not at BOS)
    return np.einsum("ntkc,ckm->ntm", Jn, env.L)


def q_values(b: np.ndarray) -> np.ndarray:
    """Expected reward of [commit to goal 1..K, safe]: [..., K+1]."""
    return np.concatenate([b, np.full(b.shape[:-1] + (1,), SAFE)], -1)


def act_hard(b: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    q = q_values(b)
    best = q >= q.max(-1, keepdims=True) - tol
    return best / best.sum(-1, keepdims=True)


def act_soft(b: np.ndarray, tau: float = TAU) -> np.ndarray:
    z = q_values(b) / tau
    z = np.exp(z - z.max(-1, keepdims=True))
    return z / z.sum(-1, keepdims=True)


OBJECTIVES = ("goal", "act_soft", "act_hard", "next_obs")


def out_dim(env: Env, objective: str) -> int:
    return {"goal": env.K, "act_soft": env.K + 1, "act_hard": env.K + 1, "next_obs": env.M}[objective]


def targets(env: Env, X: np.ndarray, objective: str) -> np.ndarray:
    """Exact per-position targets [n, T+1, out_dim]. Every objective's minimiser is a function
    of the belief state; only 'goal' asks for b_t itself."""
    J = filter_joint(env, X); b = J.sum(-1)
    if objective == "goal":
        return b
    if objective == "act_soft":
        return act_soft(b)
    if objective == "act_hard":
        return act_hard(b)
    if objective == "next_obs":
        return next_obs(env, X, J)
    raise ValueError(objective)


# ---- held-out posterior region ----------------------------------------------------------
CONFLICT = {"iid": 0.45, "channel": 0.35}     # ~15-20 % of K=4 sequences enter the region


def in_conflict(b: np.ndarray, thr: float) -> np.ndarray:
    """Two goals both strongly supported: the second-largest posterior is >= thr."""
    return np.sort(b, -1)[..., -2] >= thr


def sample_excluding(env: Env, n: int, T: int = T_OBS, seed: int = 0, thr: float | None = None):
    """Training pool that never passes through the conflict region at any position."""
    thr = CONFLICT[env.kind] if thr is None else thr
    Xs, Gs = [], []; got = 0; k = 0
    while got < n:
        X, G, _ = sample(env, 2 * n, T, seed=seed * 1000 + 17 + k); k += 1
        keep = ~in_conflict(posterior(env, X), thr).any(1)
        Xs.append(X[keep]); Gs.append(G[keep]); got += keep.sum()
    return np.concatenate(Xs)[:n], np.concatenate(Gs)[:n]


def counts(X: np.ndarray, M: int) -> np.ndarray:
    """Cumulative observation counts per position: [n, T+1, M]."""
    oh = np.zeros(X.shape + (M,))
    obs = X > 0
    oh[obs, X[obs] - 1] = 1
    return oh.cumsum(1)
