"""Small deterministic (optionally slippery) gridworld with fully known dynamics.

States are the free cells of an ASCII map. Four actions U, D, L, R. Moving into a
wall or off the grid leaves the agent in place. A cell may be a *portal*: every
action taken there moves the agent to the portal's destination (a one-way
transition). A *hazard* cell knocks the agent back to a fixed cell with
probability ``p_fail`` whatever action is taken. With ``slip > 0`` the executed action is replaced by a uniformly
random one with probability ``slip``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import shortest_path

ACTIONS = {"U": 0, "D": 1, "L": 2, "R": 3}
ACTION_NAMES = ["U", "D", "L", "R"]
DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


@dataclass
class GridWorld:
    grid: np.ndarray  # bool, True = free
    goal_coords: list[tuple[int, int]]
    portals: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    slip: float | np.ndarray = 0.0  # scalar, or (H, W) per-cell slip probability
    name: str = "grid"
    blocked_edges: set[frozenset] = field(default_factory=set)
    special_pairs: list[tuple[int, int]] = field(default_factory=list)  # designed test pairs
    hazards: dict[tuple[int, int], tuple[float, tuple[int, int]]] = field(default_factory=dict)  # cell -> (p_fail, knock-back cell)

    def __post_init__(self) -> None:
        self.height, self.width = self.grid.shape
        free = np.argwhere(self.grid)
        self._coords = [tuple(int(v) for v in rc) for rc in free]
        self._index = {rc: i for i, rc in enumerate(self._coords)}
        self.n_states = len(self._coords)
        self.n_actions = 4
        for src, dst in self.portals.items():
            if src not in self._index or dst not in self._index:
                raise ValueError(f"portal {src}->{dst} must join free cells")
        for g in self.goal_coords:
            if g not in self._index:
                raise ValueError(f"goal {g} is not a free cell")
        self.goal_states = [self._index[g] for g in self.goal_coords]
        self.n_goals = len(self.goal_states)
        self.P = self._build_transitions()

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_ascii(
        cls,
        text: str,
        portals: dict[tuple[int, int], tuple[int, int]] | None = None,
        slip: float | np.ndarray = 0.0,
        name: str = "grid",
        goals: list[tuple[int, int]] | None = None,
        blocked_edges: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
        hazards: dict[tuple[int, int], tuple[float, tuple[int, int]]] | None = None,
    ) -> "GridWorld":
        rows = [r for r in text.strip("\n").splitlines() if r.strip()]
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("ragged map")
        grid = np.array([[c != "#" for c in r] for r in rows], dtype=bool)
        goal_coords = goals if goals is not None else [
            (i, j) for i, r in enumerate(rows) for j, c in enumerate(r) if c == "G"
        ]
        edges = {frozenset(e) for e in (blocked_edges or [])}
        return cls(grid, goal_coords, portals or {}, slip, name, edges, hazards=hazards or {})

    def _next_cell(self, rc: tuple[int, int], a: int) -> tuple[int, int]:
        if rc in self.portals:
            return self.portals[rc]
        di, dj = DELTAS[a]
        nxt = (rc[0] + di, rc[1] + dj)
        if (0 <= nxt[0] < self.height and 0 <= nxt[1] < self.width and self.grid[nxt]
                and frozenset((rc, nxt)) not in self.blocked_edges):
            return nxt
        return rc

    def _build_transitions(self) -> np.ndarray:
        S, A = self.n_states, self.n_actions
        det = np.zeros((S, A, S))
        for s, rc in enumerate(self._coords):
            for a in range(A):
                det[s, a, self._index[self._next_cell(rc, a)]] = 1.0
        slip = np.asarray(self.slip, dtype=float)
        P = det
        if not np.all(slip == 0.0):
            per_state = (np.full(S, float(slip)) if slip.ndim == 0
                         else np.array([slip[rc] for rc in self._coords]))[:, None, None]
            mixed = det.mean(axis=1, keepdims=True)  # uniform random action
            P = (1 - per_state) * det + per_state * mixed
        for cell, (p_fail, dest) in self.hazards.items():
            s, d = self._index[cell], self._index[dest]
            P = P.copy()
            P[s] = (1 - p_fail) * P[s]
            P[s, :, d] += p_fail
        return P

    # ---- queries ----------------------------------------------------------
    def coords_of(self, s: int) -> tuple[int, int]:
        return self._coords[s]

    def state_of(self, i: int, j: int) -> int:
        return self._index[(i, j)]

    @property
    def coords(self) -> np.ndarray:
        """(n_states, 2) array of (row, col)."""
        return np.array(self._coords, dtype=float)

    def geodesic(self) -> np.ndarray:
        """Shortest-path length between states under the deterministic moves
        (portals count as one step; inf if unreachable)."""
        adj = (self.P > 0).any(axis=1).astype(float)
        np.fill_diagonal(adj, 0.0)
        return shortest_path(adj, directed=True, unweighted=True)

    def render(self, values: np.ndarray | None = None) -> np.ndarray:
        """Return an (H, W) float array with NaN on walls, for plotting."""
        out = np.full(self.grid.shape, np.nan)
        if values is None:
            values = np.zeros(self.n_states)
        for s, rc in enumerate(self._coords):
            out[rc] = values[s]
        return out
