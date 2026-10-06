"""The random-spawns experiment: the maze-belief task with a chosen set of spawn cells, and its exact belief graph.

    four    the maze-belief experiment's four corridor cells
    all     every cell that is not a goal (11 cells: the corridor, the landmark and the left stub)
Both arms use a passive prefix of 0-2 random moves and 8 moves after the reveal (the maze-belief experiment: 0-4 and
12). With 11 spawn cells the exact belief graph of the original setting does not fit: the build was stopped above
115 GB of memory. At 8 moves and a prefix of 0-2 it has 11.9 M nodes (four spawns: 0.15 M).

    .venv/bin/python studies/6_hard_cases_and_tasks/random_spawns/task.py            # builds and caches both graphs
"""
import sys, time
from pathlib import Path

import numpy as np

from goalgeo import mazebelief as MB, mazegraph as MG, mazeppo as P

HERE = Path(__file__).resolve().parent
X = ["..L...b....", ".aaaaabbbbb", "..a...b...."]                      # the maze-belief layout (goal cells keep their symbols)
GOALS = [(1, 5), (2, 6), (1, 10)]
SPAWNS = {"four": [(1, 1), (1, 2), (1, 8), (1, 9)],
          "all": [(r, c) for r, row in enumerate(X) for c, ch in enumerate(row) if ch != "." and (r, c) not in GOALS]}
MAX_PREFIX, H = 2, 8


def make(spawn, H=H, eps=0.4):
    cells, sym = [], []
    for r, row in enumerate(X):
        for c, ch in enumerate(row):
            if ch != ".":
                cells.append((r, c)); sym.append({"a": 0, "b": 1, "L": 2}[ch])
    ix = {c: i for i, c in enumerate(cells)}
    return MB.Maze(cells, np.array(sym), tuple(ix[g] for g in GOALS), eps, 0.9, H, tuple(ix[s] for s in SPAWNS[spawn]))


def graph(spawn, quick=False):
    if quick:
        return MG.build(make(spawn, H=5), max_prefix=MAX_PREFIX)
    return MG.cached(make(spawn), HERE / "cache" / f"graph_{spawn}.npz", max_prefix=MAX_PREFIX)


def tables(spawn, device, quick=False):
    return P.Tables(graph(spawn, quick), device, max_prefix=MAX_PREFIX)


if __name__ == "__main__":
    for s in sys.argv[1:] or list(SPAWNS):
        t0 = time.time()
        g = graph(s)
        print(f"{s}: {len(SPAWNS[s])} spawn cells, {len(g.k)} nodes, {g.n_prefix} prefix nodes, {time.time() - t0:.0f}s", flush=True)
