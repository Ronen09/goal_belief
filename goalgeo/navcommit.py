"""The navigate-commit experiment: navigate, investigate, commit. The environment and its exact belief-state solver.

An n x n grid (n = 5). One of the four corners holds a hidden reward, G uniform. G = 2 * tb + lr
with lr in {0 = left, 1 = right} and tb in {0 = top, 1 = bottom}; cells are row * n + col with row 0
at the top. Two distinct non-corner cells hold clue stations: the horizontal one reports lr, the
vertical one reports tb, each correct with its own probability q, independently given G. A station
answers one QUERY. COMMIT is legal only on a corner, ends the episode and pays 1 if that corner is
G. Every move or query costs c. After H actions without a commit the episode ends with no terminal
reward.

Belief. The prior is uniform and the two reports are independent given G, so the posterior always
factorises, b(g) = P(lr) P(tb), and each factor takes one of three values: 1/2 (station unused),
1 - q or q. A station's state s in {0 = unused, 1 = reported 0, 2 = reported 1} therefore carries
both the used flag and the belief, and there are 9 reachable beliefs per (q_h, q_v). The marginal
probability of either report is exactly 1/2 whatever the other station said.

Solver. Backward induction over (remaining actions k, cell, s_h, s_v) for a batch of public
configurations (station cells, reliabilities). Q is kept for every action, -inf where illegal, so
every tied optimal action can be read off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UP, DOWN, LEFT, RIGHT, QUERY, COMMIT = range(6)
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "QUERY", "COMMIT")
NA = 6
TIE = 1e-9                       # two action values closer than this are a tie


def corners(n):
    """Cell of each goal g = 2 * tb + lr."""
    return np.array([0, n - 1, n * (n - 1), n * n - 1])


def station_cells(n):
    return np.array([p for p in range(n * n) if p not in set(corners(n).tolist())])


def move_table(n):
    """nxt[p, a] for the four moves, -1 where the move leaves the grid."""
    nxt = -np.ones((n * n, 4), dtype=int)
    for p in range(n * n):
        r, c = divmod(p, n)
        for a, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            if 0 <= r + dr < n and 0 <= c + dc < n:
                nxt[p, a] = (r + dr) * n + c + dc
    return nxt


def factor(q):
    """P(coordinate = 1 | station state) for states (unused, reported 0, reported 1). [..., 3]"""
    q = np.asarray(q, dtype=float)
    return np.stack([np.full_like(q, 0.5), 1 - q, q], -1)


def belief(qh, qv):
    """b[..., s_h, s_v, g]: the exact posterior over the four corners."""
    ph, pv = factor(qh), factor(qv)                                  # P(right), P(bottom)
    lr = np.stack([1 - ph, ph], -1)                                  # [..., s_h, lr]
    tb = np.stack([1 - pv, pv], -1)                                  # [..., s_v, tb]
    b = tb[..., None, :, :, None] * lr[..., :, None, None, :]        # [..., s_h, s_v, tb, lr]
    return b.reshape(*b.shape[:-2], 4)


def legal_mask(n, hcell, vcell):
    """legal[B, cell, s_h, s_v, action]; it does not depend on the remaining horizon."""
    hcell, vcell = np.atleast_1d(hcell), np.atleast_1d(vcell)
    B, P = len(hcell), n * n
    m = np.zeros((B, P, 3, 3, NA), dtype=bool)
    m[..., :4] = (move_table(n) >= 0)[None, :, None, None, :]
    m[:, corners(n), :, :, COMMIT] = True
    i = np.arange(B)
    m[i, hcell, 0, :, QUERY] = True
    m[i, vcell, :, 0, QUERY] = True
    return m


@dataclass
class Solution:
    n: int
    H: int
    c: float
    hcell: np.ndarray            # [B]
    vcell: np.ndarray            # [B]
    qh: np.ndarray               # [B]
    qv: np.ndarray               # [B]
    Q: np.ndarray                # [B, H + 1, cell, s_h, s_v, action]; index 1 = remaining actions; -inf illegal
    V: np.ndarray                # [B, H + 1, cell, s_h, s_v]

    def optimal(self, tol=TIE):
        """Every tied optimal action. [B, H + 1, cell, s_h, s_v, action], all False at k = 0."""
        return self.Q >= self.V[..., None] - tol

    def regret(self):
        """Local decision regret Q*(s, a*) - Q*(s, a); +inf for illegal actions."""
        return self.V[..., None] - self.Q


def solve(n, H, c, hcell, vcell, qh, qv, allow_query=True, max_queries=None):
    """Backward induction for a batch of configurations. `max_queries` (0, 1 or 2) restricts the
    policy class, for measuring what the queries are worth."""
    hcell, vcell = np.atleast_1d(hcell), np.atleast_1d(vcell)
    qh, qv = np.broadcast_to(np.asarray(qh, float), hcell.shape), np.broadcast_to(np.asarray(qv, float), hcell.shape)
    assert np.all(hcell != vcell)
    if max_queries is None:
        max_queries = 2 if allow_query else 0
    B, P = len(hcell), n * n
    nxt, cor, legal = move_table(n), corners(n), legal_mask(n, hcell, vcell)
    b = belief(qh, qv)                                               # [B, 3, 3, 4]
    used = (np.arange(3)[:, None] > 0).astype(int) + (np.arange(3)[None, :] > 0)     # queries made so far
    i = np.arange(B)
    Q = np.full((B, H + 1, P, 3, 3, NA), -np.inf)
    V = np.zeros((B, H + 1, P, 3, 3))
    for k in range(1, H + 1):
        Vn, Qk = V[:, k - 1], Q[:, k]
        for a in range(4):
            ok = np.where(nxt[:, a] >= 0)[0]
            Qk[..., a][:, ok] = -c + Vn[:, nxt[ok, a]]
        for g in range(4):
            Qk[..., COMMIT][:, cor[g]] = b[..., g]
        can = used < max_queries                                     # [3, 3]
        qH = -c + 0.5 * (Vn[i, hcell, 1, :] + Vn[i, hcell, 2, :])     # [B, s_v]
        qV = -c + 0.5 * (Vn[i, vcell, :, 1] + Vn[i, vcell, :, 2])     # [B, s_h]
        Qk[..., QUERY][i, hcell, 0, :] = np.where(can[0][None], qH, -np.inf)
        Qk[..., QUERY][i, vcell, :, 0] = np.where(can[:, 0][None], qV, -np.inf)
        Qk[~legal] = -np.inf
        V[:, k] = Qk.max(-1)
    return Solution(n, H, c, hcell, vcell, qh, qv, Q, V)


def all_layouts(n):
    """Every ordered pair of distinct non-corner cells (horizontal station, vertical station)."""
    s = station_cells(n)
    h, v = np.meshgrid(s, s, indexing="ij")
    keep = h != v
    return h[keep], v[keep]


# ------------------------------------------------------------------ the environment, for sampling

def step_state(n, hcell, vcell, qh, qv, g, cell, sh, sv, a, rng):
    """One transition of the true environment. Returns (cell, sh, sv, reward, done, report) where
    report is -1 unless a query was answered. The caller checks legality and the horizon."""
    if a < 4:
        return move_table(n)[cell, a], sh, sv, 0.0, False, -1
    if a == COMMIT:
        return cell, sh, sv, float(corners(n)[g] == cell), True, -1
    if cell == hcell:
        truth, q = g % 2, qh
        y = truth if rng.random() < q else 1 - truth
        return cell, 1 + y, sv, 0.0, False, y
    truth, q = g // 2, qv
    y = truth if rng.random() < q else 1 - truth
    return cell, sh, 1 + y, 0.0, False, y


# ------------------------------------------------------------------ regimes of the optimal policy

def rollout_tree(sol, bi, start, prefer=(COMMIT, QUERY, UP, DOWN, LEFT, RIGHT), opt=None):
    """Follow one optimal policy from `start` with the full horizon, branching on reports.
    Ties are broken by `prefer`. Returns the leaves: dicts with the probability, number of queries,
    the order of stations queried, the reports, the path, the committed corner (or None) and
    the state (k, cell, s_h, s_v) each query was made in."""
    opt = (sol.optimal() if opt is None else opt)[bi]
    leaves, stack = [], [(start, 0, 0, sol.H, 1.0, [], [start], [])]
    nxt = move_table(sol.n)
    while stack:
        p, sh, sv, k, pr, ev, path, qs = stack.pop()
        if k == 0:
            leaves.append(dict(p=pr, queries=len(ev), evidence=ev, path=path, commit=None, steps=sol.H, qstates=qs)); continue
        a = next(a for a in prefer if opt[k, p, sh, sv, a])
        if a == COMMIT:
            leaves.append(dict(p=pr, queries=len(ev), evidence=ev, path=path, commit=p, steps=sol.H - k + 1, qstates=qs))
        elif a == QUERY:
            for y in (0, 1):
                if p == sol.hcell[bi]:
                    stack.append((p, 1 + y, sv, k - 1, pr / 2, ev + [("h", y)], path, qs + [(k, p, sh, sv)]))
                else:
                    stack.append((p, sh, 1 + y, k - 1, pr / 2, ev + [("v", y)], path, qs + [(k, p, sh, sv)]))
        else:
            stack.append((nxt[p, a], sh, sv, k - 1, pr, ev, path + [nxt[p, a]], qs))
    return leaves
