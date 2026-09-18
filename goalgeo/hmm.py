"""A small HMM whose cue beliefs are one-step equivalent but k-step distinct
(TASK4), with exact forward inference, sampling and objective targets."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

TOK = {"p": 0, "q": 1, "x": 2, "y": 3}
S = {"P": 0, "Q": 1, "A": 2, "B": 3, "C": 4, "D": 5}
V = 4


@dataclass
class HMM:
    T: np.ndarray      # [S, S] transition
    E: np.ndarray      # [S, V] emission of the state entered
    pi: np.ndarray     # stationary distribution
    name: str


def make_hmm(kind: str = "clean") -> HMM:
    n = 6
    T = np.zeros((n, n)); E = np.zeros((n, V))
    T[S["P"], S["A"]] = 1; T[S["Q"], S["B"]] = 1; T[S["A"], S["C"]] = 1; T[S["B"], S["D"]] = 1
    E[S["A"], TOK["x"]] = E[S["A"], TOK["y"]] = 0.5
    E[S["B"], TOK["x"]] = E[S["B"], TOK["y"]] = 0.5
    E[S["C"], TOK["x"]], E[S["C"], TOK["y"]] = 0.9, 0.1
    E[S["D"], TOK["x"]], E[S["D"], TOK["y"]] = 0.1, 0.9
    if kind == "clean":
        E[S["P"], TOK["p"]] = 1; E[S["Q"], TOK["q"]] = 1
        T[S["C"], S["P"]] = T[S["C"], S["Q"]] = 0.5; T[S["D"], S["P"]] = T[S["D"], S["Q"]] = 0.5
    elif kind == "noisy":
        E[S["P"], TOK["p"]], E[S["P"], TOK["q"]] = 0.8, 0.2
        E[S["Q"], TOK["q"]], E[S["Q"], TOK["p"]] = 0.8, 0.2
        T[S["C"], S["P"]], T[S["C"], S["Q"]] = 0.7, 0.3; T[S["D"], S["P"]], T[S["D"], S["Q"]] = 0.3, 0.7
    else:
        raise ValueError(kind)
    w, v = np.linalg.eig(T.T); pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = np.abs(pi) / np.abs(pi).sum()
    return HMM(T, E, pi, kind)


def belief(m: HMM, tokens, prior: np.ndarray | None = None) -> np.ndarray:
    """Posterior over the current hidden state after the tokens (forward algorithm).
    The prior is over the state *before* the first token (default: stationary)."""
    b = m.pi if prior is None else prior
    for x in tokens:
        b = (b @ m.T) * m.E[:, x]
        s = b.sum()
        b = b / s if s > 0 else np.full_like(b, 1 / len(b))
    return b


def predictive(m: HMM, b: np.ndarray, k: int) -> np.ndarray:
    """Joint distribution over the next k tokens given belief b: [V**k], row-major."""
    out = np.zeros(V ** k)
    for i, seq in enumerate(itertools.product(range(V), repeat=k)):
        bb = b; p = 1.0
        for x in seq:
            bb = (bb @ m.T) * m.E[:, x]; s = bb.sum(); p *= s
            if s == 0:
                break
            bb = bb / s
        out[i] = p
    return out


def marginals(m: HMM, b: np.ndarray, k: int) -> np.ndarray:
    """P(X_{t+j} | b) for j = 1..k, concatenated: [k*V]."""
    out = []; bb = b
    for _ in range(k):
        bb = bb @ m.T                 # unconditional state distribution j steps ahead
        out.append(bb @ m.E)
    return np.concatenate(out)


def sample(m: HMM, n: int, T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T), int); Z = np.zeros((n, T), int)
    n_s = len(m.pi)
    s = rng.choice(n_s, size=n, p=m.pi)
    for t in range(T):
        s = np.array([rng.choice(n_s, p=m.T[si]) for si in s])
        Z[:, t] = s
        X[:, t] = np.array([rng.choice(V, p=m.E[si]) for si in s])
    return X, Z


_CACHE: dict = {}


def window_targets(m: HMM, W: np.ndarray, k: int, kind: str = "joint") -> np.ndarray:
    """Exact k-step targets given only the window (stationary prior); cached per distinct window."""
    out = []
    for w in W:
        key = (m.name, tuple(int(v) for v in w), k, kind)
        if key not in _CACHE:
            b = belief(m, w)
            _CACHE[key] = predictive(m, b, k) if kind == "joint" else marginals(m, b, k)
        out.append(_CACHE[key])
    return np.stack(out)


def sequential_targets(m: HMM, X: np.ndarray) -> np.ndarray:
    """Exact P(X_{t+1} | X_{1:t}) at every position t (position T-1 predicts beyond the end)."""
    n, T_ = X.shape; out = np.zeros((n, T_, V))
    for i in range(n):
        b = m.pi
        for t in range(T_):
            b = (b @ m.T) * m.E[:, X[i, t]]; b = b / b.sum()
            out[i, t] = (b @ m.T) @ m.E
    return out


def beliefs_of(m: HMM, X: np.ndarray) -> np.ndarray:
    return np.stack([belief(m, x) for x in X])
