"""Ground-truth goal-conditioned discounted occupancy and competing geometries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridworld import GridWorld
from .planning import absorbing_P


@dataclass
class Occupancy:
    gamma: float
    rho_s: np.ndarray    # [S, K]     rho(g|s)
    rho_sa: np.ndarray   # [S, A, K]  rho(g|s,a)

    @property
    def Z_state(self) -> np.ndarray:
        return self.rho_s

    @property
    def Z_sa(self) -> np.ndarray:
        """[S, A*K]: concatenation over actions of z(s,a)."""
        S, A, K = self.rho_sa.shape
        return self.rho_sa.reshape(S, A * K)

    @property
    def Q(self) -> np.ndarray:
        """[K, S, A]: goal-specific action relevance rho(g|s,·)."""
        return np.transpose(self.rho_sa, (2, 0, 1))


def goal_occupancy(env: GridWorld, pi: np.ndarray, gamma: float) -> Occupancy:
    """rho(g|s) and rho(g|s,a) = (1-γ) Σ_t γ^t P(s_t=g | s_0=s, a_0=a, π*_g)."""
    S, A = env.n_states, env.n_actions
    K = env.n_goals
    rho_s = np.zeros((S, K))
    rho_sa = np.zeros((S, A, K))
    I = np.eye(S)
    for gi, g in enumerate(env.goal_states):
        P = absorbing_P(env, g)                       # [S, A, S']
        P_pi = np.einsum("sa,sat->st", pi[gi], P)     # [S, S']
        D = (1 - gamma) * np.linalg.solve(I - gamma * P_pi, I)  # normalised occupancy
        rho_s[:, gi] = D[:, g]
        first = (1 - gamma) * (np.arange(S) == g).astype(float)  # t = 0 term
        rho_sa[:, :, gi] = first[:, None] + gamma * (P @ D[:, g])
    return Occupancy(gamma, rho_s, rho_sa)


def successor_representation(env: GridWorld, gamma: float) -> np.ndarray:
    """Normalised SR under the uniform random policy (no absorbing goals)."""
    P_rand = env.P.mean(axis=1)
    I = np.eye(env.n_states)
    return (1 - gamma) * np.linalg.solve(I - gamma * P_rand, I)


def shortest_path_occupancy(env: GridWorld, gamma: float) -> Occupancy:
    """Deterministic-shortest-path surrogate γ^T, computed from the geodesic
    of the *intended* moves; used as the competitor in the stochastic extension."""
    D = env.geodesic()
    S, A, K = env.n_states, env.n_actions, env.n_goals
    rho_s = np.zeros((S, K)); rho_sa = np.zeros((S, A, K))
    det = (env.P == env.P.max(axis=2, keepdims=True))  # most likely next cell
    for gi, g in enumerate(env.goal_states):
        rho_s[:, gi] = gamma ** D[:, g]
        for s in range(S):
            for a in range(A):
                if s == g:
                    rho_sa[s, a, gi] = 1.0
                else:
                    s_next = int(np.argmax(det[s, a]))
                    rho_sa[s, a, gi] = gamma ** (1 + D[s_next, g])
    return Occupancy(gamma, rho_s, rho_sa)
