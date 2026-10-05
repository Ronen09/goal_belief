"""Multi-goal collection solver (hard cases): exact against brute-force recursion, and identical to the maze-belief experiment's graph
when there is a single goal worth 1."""

import functools

import numpy as np
import pytest
import torch

from goalgeo import mazebelief as MB, mazegraph as MG, multigoal as MGL

X = ["..L...b....", ".aaaaabbbbb", "..a...b...."]
X_START = [(1, 1), (1, 2), (1, 8), (1, 9)]


def make(goals_rc, H):
    cells, sym = [], []
    for r, row in enumerate(X):
        for c, ch in enumerate(row):
            if ch != ".":
                cells.append((r, c)); sym.append({"a": 0, "b": 1, "L": 2}[ch])
    ix = {c: i for i, c in enumerate(cells)}
    return MB.Maze(cells, np.array(sym), tuple(ix[g] for g in goals_rc), 0.4, 0.9, H, tuple(ix[s] for s in X_START))


def brute_q(maze, b, k, C, v):
    goals = list(maze.goals)

    @functools.lru_cache(None)
    def V(bkey, k, C):
        return 0.0 if k == 0 else max(Qf(np.frombuffer(bkey), k, C))

    def Qf(b, k, C):
        out = []
        for a in range(4):
            p = b @ maze.T[a]
            tot = 0.0
            for j, gc in enumerate(goals):
                if not (C >> j) & 1 and p[gc] > 1e-15:
                    d = np.zeros(maze.n); d[gc] = 1
                    tot += p[gc] * (v[j] + maze.gamma * V(d.tobytes(), k - 1, C | (1 << j)))
                    p = p.copy(); p[gc] = 0
            for o in range(maze.n_sym):
                w = p * maze.E[:, o]; z = w.sum()
                if z > 1e-15:
                    tot += maze.gamma * z * V(np.round(w / z, 12).tobytes(), k - 1, C)
            out.append(tot)
        return out

    return Qf(b, k, C)


def test_against_brute_force():
    maze = make([(2, 2), (1, 5), (0, 6), (1, 10)], H=4)
    g = MGL.build(maze, max_prefix=2)
    vals = MGL.value_settings(4)
    Q = MGL.solve(g, vals, device="cpu", dtype=torch.float64).numpy()
    rng = np.random.default_rng(0)
    dec = np.nonzero(g.k > 0)[0]
    for i in rng.choice(dec, 40, replace=False):
        for vi in rng.choice(len(vals), 3, replace=False):
            bq = brute_q(maze, g.belief[i], int(g.k[i]), int(g.C[i]), vals[vi].numpy().astype(float))
            assert np.abs(np.array(bq) - Q[i, vi]).max() < 1e-9
    assert (g.k[g.reveal] == maze.H).all() and (g.C[g.reveal] == 0).all()


@pytest.mark.parametrize("gi,cell", [(0, (1, 5)), (1, (2, 6)), (2, (1, 10))])
def test_single_goal_is_round18(gi, cell):
    r18 = MG.build(MB.cross_maze(H=5), max_prefix=2)
    g = MGL.build(make([cell], H=5), max_prefix=2)
    Q = MGL.solve(g, torch.tensor([[1.0]]), device="cpu", dtype=torch.float64).numpy()[:, 0]
    assert g.n_prefix == r18.n_prefix and np.allclose(g.belief[:g.n_prefix], r18.belief[:r18.n_prefix])
    assert np.abs(Q[g.reveal] - r18.Q[r18.reveal[:, gi]]).max() < 1e-12
    assert int(((g.k > 0) & (g.C == 0)).sum()) == int((r18.goal == gi).sum())
