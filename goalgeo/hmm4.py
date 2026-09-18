"""Parametrised HMM family for TASK4 (geometric prominence).

A delayed cue branch: cue p/q -> k filler states (x/y 0.5) -> with prob r the
branch-specific state C/D (x: 0.5 +/- delta), else a neutral state N (emits n).
A fixed control branch: cue u/v -> A'/B' (x 0.9 / 0.1), relevant immediately.
Everything returns uniformly to the four cue states."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import hmm as Hm

TOK = {"p": 0, "q": 1, "u": 2, "v": 3, "x": 4, "y": 5, "n": 6}
V = 7


@dataclass
class HMM4(Hm.HMM):
    r: float = 1.0
    delta: float = 0.4
    k: int = 1
    S: dict | None = None               # state name -> index
    mirror: np.ndarray | None = None    # swaps the A branch with the B branch
    relevant: np.ndarray | None = None  # states that emit the relevant (t*) token: C, D, N


def make_hmm4(r: float = 1.0, delta: float = 0.4, k: int = 1, ctrl: float = 0.9) -> HMM4:
    """ctrl: P(x | A') = P(y | B') of the immediate-relevance control branch (TASK6 varies it)."""
    assert 0 <= r <= 1 and 0 <= delta <= 0.5 and k >= 1 and 0.5 <= ctrl <= 1
    S = {"P": 0, "Q": 1, "U": 2, "V": 3}
    for i in range(k):
        S[f"FA{i + 1}"] = 4 + i
    for i in range(k):
        S[f"FB{i + 1}"] = 4 + k + i
    base = 4 + 2 * k
    S.update({"C": base, "D": base + 1, "N": base + 2, "A'": base + 3, "B'": base + 4})
    n = base + 5
    T = np.zeros((n, n)); E = np.zeros((n, V))
    ret = [S["P"], S["Q"], S["U"], S["V"]]
    T[S["P"], S["FA1"]] = 1; T[S["Q"], S["FB1"]] = 1
    for i in range(1, k):
        T[S[f"FA{i}"], S[f"FA{i + 1}"]] = 1; T[S[f"FB{i}"], S[f"FB{i + 1}"]] = 1
    T[S[f"FA{k}"], S["C"]] = r; T[S[f"FA{k}"], S["N"]] = 1 - r
    T[S[f"FB{k}"], S["D"]] = r; T[S[f"FB{k}"], S["N"]] = 1 - r
    T[S["U"], S["A'"]] = 1; T[S["V"], S["B'"]] = 1
    for s in ("C", "D", "N", "A'", "B'"):
        T[S[s], ret] = 0.25
    E[S["P"], TOK["p"]] = E[S["Q"], TOK["q"]] = E[S["U"], TOK["u"]] = E[S["V"], TOK["v"]] = 1
    xy = [TOK["x"], TOK["y"]]
    for i in range(k):
        E[S[f"FA{i + 1}"], xy] = 0.5; E[S[f"FB{i + 1}"], xy] = 0.5
    E[S["C"], xy] = [0.5 + delta, 0.5 - delta]; E[S["D"], xy] = [0.5 - delta, 0.5 + delta]
    E[S["N"], TOK["n"]] = 1
    E[S["A'"], xy] = [ctrl, 1 - ctrl]; E[S["B'"], xy] = [1 - ctrl, ctrl]
    w, v = np.linalg.eig(T.T); pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = np.abs(pi) / np.abs(pi).sum()
    mirror = np.arange(n)
    mirror[[S["P"], S["Q"]]] = [S["Q"], S["P"]]; mirror[[S["C"], S["D"]]] = [S["D"], S["C"]]
    for i in range(k):
        a, b = S[f"FA{i + 1}"], S[f"FB{i + 1}"]; mirror[[a, b]] = [b, a]
    relevant = np.zeros(n, bool); relevant[[S["C"], S["D"], S["N"]]] = True
    return HMM4(T, E, pi, f"hmm4(r={r},delta={delta},k={k},ctrl={ctrl})", r, delta, k, S, mirror, relevant)


def relevant_fraction(m: HMM4) -> float:
    """Stationary fraction of positions whose token is the relevant one."""
    return float(m.pi[m.relevant].sum())


def sample(m: Hm.HMM, n: int, T: int, seed: int = 0):
    """Vectorised sampling of n sequences of length T from the stationary chain."""
    rng = np.random.default_rng(seed)
    cT, cE = np.cumsum(m.T, 1), np.cumsum(m.E, 1)
    n_s, n_v = m.T.shape[0], m.E.shape[1]
    s = rng.choice(n_s, size=n, p=m.pi)
    X = np.zeros((n, T), int); Z = np.zeros((n, T), int)
    for t in range(T):
        s = np.minimum((rng.random(n)[:, None] > cT[s]).sum(1), n_s - 1)
        Z[:, t] = s
        X[:, t] = np.minimum((rng.random(n)[:, None] > cE[s]).sum(1), n_v - 1)
    return X, Z


def beliefs_seq(m: Hm.HMM, X: np.ndarray) -> np.ndarray:
    """Posterior over the state that emitted X[:, t], for every t: [n, T, S]."""
    n, T_ = X.shape; B = np.zeros((n, T_, len(m.pi)))
    b = np.tile(m.pi, (n, 1))
    for t in range(T_):
        b = (b @ m.T) * m.E[:, X[:, t]].T
        b = b / np.maximum(b.sum(1, keepdims=True), 1e-300)
        B[:, t] = b
    return B


def next_token(m: Hm.HMM, B: np.ndarray) -> np.ndarray:
    """P(next token | belief) for beliefs in the last axis."""
    return (B @ m.T) @ m.E


def joint_predictive(m: Hm.HMM, B: np.ndarray, k: int) -> np.ndarray:
    """Joint distribution over the next k tokens for each belief row: [N, V**k], row-major."""
    B = np.atleast_2d(B); n_v = m.E.shape[1]
    out = np.zeros((len(B), n_v ** k))
    for i, seq in enumerate(itertools.product(range(n_v), repeat=k)):
        bb = B; p = np.ones(len(B))
        for x in seq:
            bb = (bb @ m.T) * m.E[:, x]; s = bb.sum(1); p = p * s
            bb = bb / np.maximum(s[:, None], 1e-300)
        out[:, i] = p
    return out


def marginals(m: Hm.HMM, B: np.ndarray, k: int) -> np.ndarray:
    """P(X_{t+j} | belief) for j = 1..k, concatenated: [N, k*V]."""
    B = np.atleast_2d(B); out = []
    for _ in range(k):
        B = B @ m.T; out.append(B @ m.E)
    return np.concatenate(out, axis=1)


def forgetting_cost(m: HMM4, X: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    """Increase in optimal next-token loss if the filter forgets which branch it is on
    (belief symmetrised across the branch mirror after every update): mean KL per
    position, mean KL per relevant event, and the fraction of relevant positions."""
    B = beliefs_seq(m, X)
    Bf = 0.5 * (B + B[..., m.mirror])
    P, Pf = next_token(m, B)[:, :-1], next_token(m, Bf)[:, :-1]
    kl = (P * (np.log(P + 1e-12) - np.log(Pf + 1e-12))).sum(-1)      # [n, T-1]
    rel = m.relevant[Z[:, 1:]]
    seen = np.zeros_like(rel); seen[:, m.k:] = True                    # cue inside the sequence (t >= k)
    ev = rel & seen
    return {"per_step": float(kl.mean()), "per_event": float(kl[ev].mean()) if ev.any() else 0.0,
            "relevant_frac": float(rel.mean())}
