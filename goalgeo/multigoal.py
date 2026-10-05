"""Multi-goal collection in the maze-belief maze: fixed goal cells, a value per goal drawn every episode and shown at the
reveal; the agent collects as much discounted value as it can in H moves. Exact solver.

Picking up a goal is observed (whatever its value), so the belief over location depends on which goals have been
collected but not on their values: one belief graph serves every value setting. Entering an uncollected goal cell
collects it (reward = its value) and reveals the location exactly; entering any other cell rules out the uncollected
goal cells. Q* for every value setting comes from one backward induction over the shared graph.

Nodes: prefix nodes (goal-free filter, as the maze-belief experiment) and decision nodes keyed by (moves left k, collected mask C, belief).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazegraph as MG


@dataclass
class MGraph:
    maze: MB.Maze
    belief: np.ndarray        # [N, n]
    k: np.ndarray             # [N] moves left; -1 prefix
    C: np.ndarray             # [N] collected mask; -1 prefix
    level: np.ndarray         # [N] prefix length; -1 decision nodes
    nxt: np.ndarray           # [N, 4, S] successor without a pickup, by symbol (-1 none)
    p_obs: np.ndarray         # [N, 4, S] probability of no pickup and that symbol
    pc: np.ndarray            # [N, 4, K] probability of picking up goal j
    nxt_c: np.ndarray         # [N, 4, K] successor after picking up goal j (-1 none or last move)
    start: np.ndarray         # [S] prefix node after the first symbol
    reveal: np.ndarray        # [n_prefix] decision node (k = H, C = 0)
    n_prefix: int


def _move(maze, B):
    P = np.zeros((len(B), 4, maze.n))
    for a in range(4):
        np.add.at(P[:, a].T, maze.nxt[:, a], B.T)
    return P


def build(maze: MB.Maze, max_prefix=4) -> MGraph:
    S, K, n, goals = maze.n_sym, len(maze.goals), maze.n, list(maze.goals)
    rows = dict(belief=[], k=[], C=[], level=[], nxt=[], p_obs=[], pc=[], nxt_c=[])
    off = 0

    def add(bel, k, C, lv):
        nonlocal off
        m = len(bel)
        rows["belief"].append(bel); rows["k"].append(np.full(m, k)); rows["C"].append(np.full(m, C)); rows["level"].append(np.full(m, lv))
        ids = off + np.arange(m); off += m
        return ids

    # prefix: goal-free filter, nothing is collected
    first = [maze.observe(maze.prior(), o) for o in range(S)]
    ok = np.array([z > 0 for _, z in first])
    b0, inv = MG._unique(np.array([b for (b, z) in first if z > 0]))
    start = np.full(S, -1); ids = add(b0, -1, -1, 0); start[ok] = ids[inv]
    levels = [(ids, b0)]
    for lv in range(1, max_prefix + 1):
        pid, pb = levels[-1]
        P = _move(maze, pb)
        W = P[:, :, None, :] * maze.E.T[None, None]
        z = W.sum(-1); Sb = np.divide(W, z[..., None], out=np.zeros_like(W), where=z[..., None] > 1e-12)
        live = z > 1e-12
        nb, inv = MG._unique(Sb[live]); nid = add(nb, -1, -1, lv)
        nx = np.full(z.shape, -1); nx[live] = nid[inv]
        rows["nxt"].append(nx); rows["p_obs"].append(z)
        rows["pc"].append(np.zeros((len(pb), 4, K))); rows["nxt_c"].append(np.full((len(pb), 4, K), -1))
        levels.append((nid, nb))
    last = levels[-1][0]
    rows["nxt"].append(np.full((len(last), 4, S), -1)); rows["p_obs"].append(np.zeros((len(last), 4, S)))
    rows["pc"].append(np.zeros((len(last), 4, K))); rows["nxt_c"].append(np.full((len(last), 4, K), -1))
    n_prefix = off
    pre_b = np.concatenate([b for _, b in levels])

    # decisions, level by level in moves left; frontier: collected mask -> beliefs at this level
    bel, inv = MG._unique(pre_b)
    ids = add(bel, maze.H, 0, -1)
    reveal = ids[inv]
    frontier = {0: (ids, bel)}
    delta = np.eye(n)
    for k in range(maze.H, 0, -1):
        nxt_frontier = {}                                     # C -> list of beliefs (non-pickup and pickup successors)
        pending = []                                          # per group: (ids, C, live mask, beliefs of successors, z, pc)
        for C, (gids, B) in frontier.items():
            P = _move(maze, B)
            pc = np.zeros((len(B), 4, K))
            for j, g in enumerate(goals):
                if not (C >> j) & 1:
                    pc[:, :, j] = P[:, :, g]; P[:, :, g] = 0
            W = P[:, :, None, :] * maze.E.T[None, None]
            z = W.sum(-1); Sb = np.divide(W, z[..., None], out=np.zeros_like(W), where=z[..., None] > 1e-12)
            pending.append((gids, C, z, Sb, pc))
            if k > 1:
                live = z > 1e-12
                nxt_frontier.setdefault(C, []).append(Sb[live])
                for j, g in enumerate(goals):
                    if not (C >> j) & 1 and (pc[:, :, j] > 1e-12).any():
                        nxt_frontier.setdefault(C | (1 << j), []).append(delta[g][None])
        # add the next level's nodes
        new = {}
        for C, parts in nxt_frontier.items():
            nb, _ = MG._unique(np.concatenate(parts))
            new[C] = (add(nb, k - 1, C, -1), nb)
        # successors of this level's nodes
        lookup = {}
        for C, (nid, nb) in new.items():
            q = np.round(nb * 10 ** MG.ROUND).astype(np.int64)
            lookup[C] = {r.tobytes(): i for r, i in zip(q, nid)}
        def find(C, Bs):
            q = np.round(Bs * 10 ** MG.ROUND).astype(np.int64)
            d = lookup[C]
            return np.array([d[r.tobytes()] for r in q], dtype=np.int64)
        for gids, C, z, Sb, pc in pending:
            nx = np.full(z.shape, -1); nxc = np.full(pc.shape, -1)
            if k > 1:
                live = z > 1e-12
                nx[live] = find(C, Sb[live])
                for j, g in enumerate(goals):
                    if not (C >> j) & 1:
                        m = pc[:, :, j] > 1e-12
                        if m.any():
                            nxc[:, :, j][m] = find(C | (1 << j), delta[g][None])[0]
            rows["nxt"].append(nx); rows["p_obs"].append(z); rows["pc"].append(pc); rows["nxt_c"].append(nxc)
        frontier = new
    # the nodes of the last added level (k = 0) are never added: k = 1 has no successors
    cat = {k_: np.concatenate(v) for k_, v in rows.items()}
    assert len(cat["nxt"]) == off, (len(cat["nxt"]), off)
    return MGraph(maze, cat["belief"], cat["k"], cat["C"], cat["level"], cat["nxt"], cat["p_obs"], cat["pc"], cat["nxt_c"],
                  start, reveal, n_prefix)


def value_settings(K, values=(0, 1, 2, 3)):
    return torch.tensor(list(itertools.product(values, repeat=K)), dtype=torch.float32)       # [256, K]


@torch.no_grad()
def solve(g: MGraph, vals: torch.Tensor, device="cuda", dtype=torch.float32):
    """Q* [N, n_settings, 4] for every value setting (rows of vals [n_settings, K]). Prefix nodes get 0."""
    N, nv = len(g.k), len(vals)
    gamma = g.maze.gamma
    Q = torch.zeros(N, nv, 4, dtype=dtype, device=device)
    V = torch.zeros(N + 1, nv, dtype=dtype, device=device)                    # V[-1] = 0 for missing successors
    vals = vals.to(device, dtype)
    nxt = torch.as_tensor(g.nxt, device=device); p_obs = torch.as_tensor(g.p_obs, device=device, dtype=dtype)
    pc = torch.as_tensor(g.pc, device=device, dtype=dtype); nxc = torch.as_tensor(g.nxt_c, device=device)
    k = torch.as_tensor(g.k, device=device)
    for kk in range(1, g.maze.H + 1):
        sel = torch.nonzero(k == kk).squeeze(1)
        for s in range(0, len(sel), 200_000):
            i = sel[s:s + 200_000]
            cont = (p_obs[i][..., None] * V[nxt[i]]).sum(2)                    # [m, 4, nv]
            pick = (pc[i][..., None] * (vals.T[None, None] + gamma * V[nxc[i]])).sum(2)
            q = (pick + gamma * cont).permute(0, 2, 1)                        # [m, nv, 4]
            Q[i] = q; V[i] = q.max(-1).values
    return Q
