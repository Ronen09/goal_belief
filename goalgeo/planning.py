"""Exact goal-conditioned planning: value iteration per goal, optimal policies,
and rollouts. The goal cell is absorbing; reward 1 is paid on the arrival step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gridworld import GridWorld


@dataclass
class Solution:
    gamma: float
    V: np.ndarray   # [n_goals, S]
    Q: np.ndarray   # [n_goals, S, A]
    pi: np.ndarray  # [n_goals, S, A] uniform over argmax actions


def absorbing_P(env: GridWorld, g: int) -> np.ndarray:
    """Transition tensor with the goal made absorbing."""
    P = env.P.copy()
    P[g] = 0.0
    P[g, :, g] = 1.0
    return P


def solve_goal(env: GridWorld, g: int, gamma: float, tol: float = 1e-12, max_iter: int = 10_000):
    P = absorbing_P(env, g)
    S, A = env.n_states, env.n_actions
    reward = np.zeros(S)
    reward[g] = 1.0  # earned on arriving at g
    V = np.zeros(S)
    for _ in range(max_iter):
        Q = P @ (reward + gamma * V)  # [S, A]
        Q[g] = 0.0  # absorbing: nothing more to earn once at the goal
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    Q = P @ (reward + gamma * V)
    Q[g] = 0.0
    best = np.isclose(Q, Q.max(axis=1, keepdims=True), atol=1e-9)
    pi = best / best.sum(axis=1, keepdims=True)
    return V, Q, pi


def solve_all_goals(env: GridWorld, gamma: float = 0.9) -> Solution:
    Vs, Qs, pis = [], [], []
    for g in env.goal_states:
        V, Q, pi = solve_goal(env, g, gamma)
        Vs.append(V); Qs.append(Q); pis.append(pi)
    return Solution(gamma, np.stack(Vs), np.stack(Qs), np.stack(pis))


@dataclass
class Trajectory:
    states: list[int]
    actions: list[int]
    goal: int

    @property
    def reached(self) -> bool:
        return self.states[-1] == self.goal


def rollout(env: GridWorld, pi_g: np.ndarray, start: int, goal: int, rng: np.random.Generator,
            max_steps: int = 200) -> Trajectory:
    s = start
    states, actions = [s], []
    for _ in range(max_steps):
        if s == goal:
            break
        a = int(rng.choice(env.n_actions, p=pi_g[s]))
        s = int(rng.choice(env.n_states, p=env.P[s, a]))
        actions.append(a); states.append(s)
    return Trajectory(states, actions, goal)


def optimal_dataset(env: GridWorld, sol: Solution, targets: str = "argmax", temperature: float = 0.02):
    """All (s, g) pairs with s != g. Target = uniform over optimal actions
    (``argmax``) or the Boltzmann policy softmax(Q / temperature) (``boltzmann``),
    which varies smoothly with the dynamics."""
    S_idx, G_idx, T = [], [], []
    if targets == "boltzmann":
        z = sol.Q / temperature
        soft = np.exp(z - z.max(-1, keepdims=True)); soft /= soft.sum(-1, keepdims=True)
    for gi, g in enumerate(env.goal_states):
        for s in range(env.n_states):
            if s == g:
                continue
            S_idx.append(s); G_idx.append(gi); T.append(sol.pi[gi, s] if targets == "argmax" else soft[gi, s])
    return np.array(S_idx), np.array(G_idx), np.stack(T)
