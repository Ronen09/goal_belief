"""Round 18: the belief graph of the maze task, as arrays.

Every posterior the task can produce is a node. Prefix nodes (no goal yet) are indexed by prefix length and
posterior; decision nodes by goal, remaining moves and posterior. Transitions are node x action x symbol -> node,
so an environment can track its exact belief, action values and optimal actions by integer lookups, whatever the
policy does. Values are computed by backward induction over the graph and are checked against `mazebelief.Solver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goalgeo import mazebelief as MB

MAX_PREFIX = 4
ROUND = 9
_SALT = np.random.default_rng(12345).integers(1, 2 ** 63, (2, 64), dtype=np.int64).astype(np.uint64) | np.uint64(1)


def _unique(rows):
    """Distinct rows (after rounding) and the index of each input row among them."""
    q = np.round(rows * 10 ** ROUND).astype(np.int64).astype(np.uint64)
    h = (q * _SALT[0, : q.shape[1]]).sum(1) ^ ((q * _SALT[1, : q.shape[1]]).sum(1) >> np.uint64(29))     # 64-bit hash of the row
    _, first, inv = np.unique(h, return_index=True, return_inverse=True)
    return rows[first], inv.reshape(-1)


def _step(maze, B, goal):
    """From beliefs B [m, n]: for every action and symbol the successor posterior, its probability, and the
    probability of entering the goal. goal=None: the prefix (nothing terminates)."""
    m = len(B)
    P = np.stack([B[:, np.argsort(maze.nxt[:, a], kind="stable")] * 0 for a in range(4)], 1)      # [m, 4, n] location after the move
    for a in range(4):
        np.add.at(P[:, a].T, maze.nxt[:, a], B.T)
    if goal is None:
        r = np.zeros((m, 4))
    else:
        r = P[:, :, goal].copy(); P[:, :, goal] = 0
    W = P[:, :, None, :] * maze.E.T[None, None]                    # [m, 4, symbols, n]
    z = W.sum(-1)
    S = np.divide(W, z[..., None], out=np.zeros_like(W), where=z[..., None] > 1e-12)
    return S, z, r


@dataclass
class Graph:
    maze: MB.Maze
    belief: np.ndarray            # [nodes, n]
    k: np.ndarray                 # [nodes] moves left; -1 for prefix nodes
    goal: np.ndarray              # [nodes] goal index 0..2; -1 for prefix nodes
    level: np.ndarray             # [nodes] prefix length for prefix nodes; -1 otherwise
    nxt: np.ndarray               # [nodes, 4, symbols] successor node, -1 if impossible or terminal
    p_obs: np.ndarray             # [nodes, 4, symbols] probability of (continuing and) that symbol
    r: np.ndarray                 # [nodes, 4] probability of entering the goal
    Q: np.ndarray                 # [nodes, 4]; 0 for prefix nodes
    start: np.ndarray             # [symbols] prefix node after the first symbol, -1 if impossible
    reveal: np.ndarray            # [prefix nodes, goals] decision node with H moves left
    n_prefix: int

    @property
    def V(self):
        return self.Q.max(1)

    def optimal(self, tol=MB.TIE):
        return self.Q >= self.Q.max(1, keepdims=True) - tol


def build(maze: MB.Maze, max_prefix=MAX_PREFIX) -> Graph:
    ns = maze.n_sym
    B, K, G, LV, NX, PO, R = [], [], [], [], [], [], []
    offset = 0

    def add(bel, k, g, lv):
        nonlocal offset
        B.append(bel); K.append(np.full(len(bel), k)); G.append(np.full(len(bel), g)); LV.append(np.full(len(bel), lv))
        ids = offset + np.arange(len(bel)); offset += len(bel)
        return ids

    # prefix
    first = np.array([maze.observe(maze.prior(), o) for o in range(ns)], dtype=object)
    ok = np.array([z > 0 for _, z in first])
    b0, inv = _unique(np.array([b for b, _ in first[ok]]))
    start = np.full(ns, -1); ids = add(b0, -1, -1, 0); start[ok] = ids[inv]
    levels = [(ids, b0)]
    for lv in range(1, max_prefix + 1):
        pid, pb = levels[-1]
        S, z, r = _step(maze, pb, None)
        live = z > 1e-12
        nb, inv = _unique(S[live])
        ids = add(nb, -1, -1, lv)
        nx = np.full(z.shape, -1); nx[live] = ids[inv]
        NX.append(nx); PO.append(z); R.append(r)
        levels.append((ids, nb))
    NX.append(np.full((len(levels[-1][0]), 4, ns), -1)); PO.append(np.zeros((len(levels[-1][0]), 4, ns))); R.append(np.zeros((len(levels[-1][0]), 4)))
    n_prefix = offset
    pre_belief = np.concatenate([b for _, b in levels])
    reveal = np.zeros((n_prefix, len(maze.goals)), dtype=int)
    # decisions, goal by goal, level by level in remaining moves
    for gi, g in enumerate(maze.goals):
        bel, inv = _unique(pre_belief)
        ids = add(bel, maze.H, gi, -1)
        reveal[:, gi] = ids[inv]
        for k in range(maze.H, 0, -1):
            S, z, r = _step(maze, bel, g)
            if k > 1:
                live = z > 1e-12
                nb, inv = _unique(S[live])
                nid = add(nb, k - 1, gi, -1)
                nx = np.full(z.shape, -1); nx[live] = nid[inv]
            else:
                nx = np.full(z.shape, -1); nb = None
            NX.append(nx); PO.append(z); R.append(r)
            bel = nb
    belief, k, goal, level = np.concatenate(B), np.concatenate(K), np.concatenate(G), np.concatenate(LV)
    nxt, p_obs, r = np.concatenate(NX), np.concatenate(PO), np.concatenate(R)
    Q = np.zeros((offset, 4)); V = np.zeros(offset + 1)                # V[-1] = 0 serves the missing successors
    for kk in range(1, maze.H + 1):
        sel = np.nonzero(k == kk)[0]
        Q[sel] = r[sel] + maze.gamma * (p_obs[sel] * V[nxt[sel]]).sum(-1)
        V[sel] = Q[sel].max(1)
    return Graph(maze, belief, k, goal, level, nxt, p_obs, r, Q, start, reveal, n_prefix)


def cached(maze: MB.Maze, path, max_prefix=MAX_PREFIX) -> Graph:
    """build(), stored at `path` (.npz). The file is a cache: it is rebuilt if missing."""
    from pathlib import Path
    path = Path(path)
    if path.exists():
        z = np.load(path)
        return Graph(maze, *[z[k] for k in ("belief", "k", "goal", "level", "nxt", "p_obs", "r", "Q", "start", "reveal")], int(z["n_prefix"]))
    g = build(maze, max_prefix)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, belief=g.belief, k=g.k, goal=g.goal, level=g.level, nxt=g.nxt.astype(np.int32), p_obs=g.p_obs.astype(np.float32),
             r=g.r, Q=g.Q, start=g.start, reveal=g.reveal, n_prefix=g.n_prefix)
    return g
