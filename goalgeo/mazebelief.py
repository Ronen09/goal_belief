"""The maze-belief experiment: a small aliased maze with a hidden location, its exact Bayesian filter and its exact solver.

World. n traversable cells inside a rows x cols box; moves N, S, E, W are deterministic and a move into a wall
leaves the agent in place. Each cell emits a symbol. Ordinary symbols come in sibling pairs (0, 1), (2, 3), ...:
a cell emits its own symbol with probability 1 - eps and the sibling with probability eps, and several cells
share a symbol. One landmark cell emits a unique symbol without noise.

Episode. The goal is one of a fixed set of goal cells, shown at the start. The start cell is uniform over a fixed
set of start cells that contains no goal, so the start prior does not depend on the goal. The agent observes a symbol at the start and after
every move. Entering the goal pays gamma^(t - 1) for the t-th move and ends the episode. After H moves the
episode ends without reward.

Belief. b_t(s) = P(location = s | symbols, actions, episode not over). Not having terminated is evidence: mass
that a move carries into the goal is removed. That is the only way the goal enters the posterior.

Solver. Exact finite-horizon expectimax over the beliefs reachable from the start prior, memoised on
(remaining moves, belief). Transitions are deterministic and the emission noise is binary, so the reachable set is
finite and small (about 10^5 beliefs for 13 cells and 12 moves).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES = ("N", "S", "W", "E")
TIE = 1e-9


@dataclass
class Maze:
    cells: list                   # (row, col) of each traversable cell
    sym: np.ndarray               # [n] symbol of each cell; the landmark has the largest symbol
    goals: tuple                  # goal cells
    eps: float
    gamma: float
    H: int
    starts: tuple = ()            # start cells (uniform); empty: every cell that is not a goal
    nxt: np.ndarray = field(init=False)
    E: np.ndarray = field(init=False)      # [n, symbols]
    T: np.ndarray = field(init=False)      # [4, n, n]

    def __post_init__(self):
        n, idx = len(self.cells), {c: i for i, c in enumerate(self.cells)}
        self.nxt = np.array([[idx.get((r + dr, c + dc), i) for dr, dc in MOVES] for i, (r, c) in enumerate(self.cells)])
        self.n_sym = int(self.sym.max()) + 1
        self.E = np.zeros((n, self.n_sym))
        for s in range(n):
            if self.sym[s] == self.n_sym - 1:
                self.E[s, self.sym[s]] = 1
            else:
                self.E[s, self.sym[s]] = 1 - self.eps
                self.E[s, self.sym[s] ^ 1] = self.eps
        self.T = np.zeros((4, n, n))
        for a in range(4):
            self.T[a, np.arange(n), self.nxt[:, a]] = 1

    @property
    def n(self):
        return len(self.cells)

    def prior(self):
        p = np.zeros(self.n)
        p[list(self.starts) if self.starts else [i for i in range(self.n) if i not in self.goals]] = 1
        return p / p.sum()

    def observe(self, b, o):
        """Posterior after symbol o, and the probability of o."""
        w = b * self.E[:, o]
        z = w.sum()
        return (w / z if z > 0 else w), z

    def move(self, b, a, goal):
        """(probability of entering the goal, belief over location given the episode continues, its probability)."""
        p = b @ self.T[a]
        r = p[goal]
        q = p.copy(); q[goal] = 0
        z = q.sum()
        return r, (q / z if z > 1e-15 else q), z

    def filter(self, obs, acts, goal=None):
        """Beliefs b_0..b_T for symbols o_0..o_T and actions a_0..a_{T-1}. goal=None: the goal-free filter."""
        b, _ = self.observe(self.prior(), obs[0])
        out = [b]
        for a, o in zip(acts, obs[1:]):
            if goal is None:
                b = b @ self.T[a]
            else:
                _, b, _ = self.move(b, a, goal)
            b, _ = self.observe(b, o)
            out.append(b)
        return np.array(out)

    def mdp_values(self, goal):
        """Fully observed problem: Q[k, s, a] with k moves left."""
        Q = np.zeros((self.H + 1, self.n, 4)); V = np.zeros((self.H + 1, self.n))
        for k in range(1, self.H + 1):
            s2 = self.nxt
            Q[k] = np.where(s2 == goal, 1.0, self.gamma * V[k - 1][s2])
            Q[k, goal] = 0
            V[k] = Q[k].max(1)
        return Q


def key(b, k):
    return (k, np.round(b, 9).tobytes())


class Solver:
    """Exact values for one goal. q(b, k): the four action values with k moves left, after the symbol at b has been
    folded in. policy=None is the optimum; otherwise policy(b, k) -> action is evaluated exactly."""

    def __init__(self, maze: Maze, goal, policy=None, cap=None):
        self.m, self.goal, self.policy, self.memo, self.cap = maze, goal, policy, {}, cap

    def q(self, b, k):
        kk = key(b, k)
        if kk in self.memo:
            return self.memo[kk]
        m, out = self.m, np.zeros(4)
        if k > 0:
            for a in range(4):
                r, q, z = m.move(b, a, self.goal)
                v = r
                if z > 1e-12 and k > 1:
                    for o in range(m.n_sym):
                        b2, zo = m.observe(q, o)
                        if zo > 1e-12:
                            v += m.gamma * z * zo * self.v(b2, k - 1)
                out[a] = v
        self.memo[kk] = out
        if self.cap and len(self.memo) > self.cap:
            raise MemoryError("more reachable beliefs than the cap")
        return out

    def v(self, b, k):
        q = self.q(b, k)
        return q.max() if self.policy is None else q[self.policy(b, k)]

    def start_value(self):
        m, tot = self.m, 0.0
        for o in range(m.n_sym):
            b, z = m.observe(m.prior(), o)
            if z > 0:
                tot += z * self.v(b, m.H)
        return tot


def map_policy(maze, goal):
    """Act as if the most likely cell were certainly the location."""
    Q = maze.mdp_values(goal)
    return lambda b, k: int(Q[k, int(np.argmax(b))].argmax())


def qmdp_policy(maze, goal):
    """Belief-weighted fully observed values: uncertainty is assumed to vanish after one move."""
    Q = maze.mdp_values(goal)
    return lambda b, k: int((b @ Q[k]).argmax())


def reach(maze, goal, solver, policy=None):
    """Decision states (belief, k) reached under a policy (default: the optimum, first optimal action), with their
    probabilities."""
    out, stack = {}, []
    for o in range(maze.n_sym):
        b, z = maze.observe(maze.prior(), o)
        if z > 0:
            stack.append((b, maze.H, z))
    while stack:
        b, k, p = stack.pop()
        if k == 0 or p < 1e-12:
            continue
        kk = key(b, k)
        if kk in out:
            out[kk] = (b, k, out[kk][2] + p)
        else:
            out[kk] = (b, k, p)
        a = int(solver.q(b, k).argmax()) if policy is None else policy(b, k)
        _, q, z = maze.move(b, a, goal)
        if z > 1e-12:
            for o in range(maze.n_sym):
                b2, zo = maze.observe(q, o)
                if zo > 1e-12:
                    stack.append((b2, k - 1, p * z * zo))
    return list(out.values())


def random_maze(seed, n=13, rows=4, cols=5, pairs=2, n_goals=4, eps=0.15, gamma=0.9, H=12):
    rng = np.random.default_rng(seed)
    cells = {(rows // 2, cols // 2)}
    while len(cells) < n:
        r, c = sorted(cells)[rng.integers(len(cells))]
        dr, dc = MOVES[rng.integers(4)]
        if 0 <= r + dr < rows and 0 <= c + dc < cols:
            cells.add((r + dr, c + dc))
    cells = sorted(cells)
    sym = rng.integers(0, 2 * pairs, n)
    lm = rng.integers(n)
    sym[lm] = 2 * pairs
    goals = tuple(sorted(rng.choice([i for i in range(n) if i != lm], n_goals, replace=False).tolist()))
    return Maze(cells, sym, goals, eps, gamma, H)


def draw(maze):
    rows = max(r for r, _ in maze.cells) + 1; cols = max(c for _, c in maze.cells) + 1
    names = "abcdefgh"
    g = [["  ·  " for _ in range(cols)] for _ in range(rows)]
    for i, (r, c) in enumerate(maze.cells):
        s = "L" if maze.sym[i] == maze.n_sym - 1 else names[maze.sym[i]]
        g[r][c] = f"{i:2d}{s}{'*' if i in maze.goals else ' '} "
    return "\n".join("".join(row) for row in g)


def open_loop_policy(maze, goal):
    """Plan without future observations: the first action of the best fixed action sequence for the current belief,
    re-planned at every step from the true posterior (open-loop feedback control). The planner knows that the episode
    ends on entering the goal, and nothing else about what it will see."""
    memo = {}

    def best(b, k):
        kk = key(b, k)
        if kk not in memo:
            out = np.zeros(4)
            if k > 0:
                for a in range(4):
                    r, q, z = maze.move(b, a, goal)
                    out[a] = r + (maze.gamma * z * best(q, k - 1).max() if z > 1e-12 and k > 1 else 0.0)
            memo[kk] = out
        return memo[kk]
    return lambda b, k: int(best(b, k).argmax())


def cross_maze(eps=0.4, gamma=0.9, H=12, starts=4):
    """The maze-belief maze. Left arm: symbol a; right arm: symbol b; a landmark above the left arm; goals G1 (centre),
    G2 (below the right arm's first cell) and G3 (right end)."""
    lay = ["..L...b....",
           ".aaaa1bbbb3",
           "..a...2...."]
    start_cols = {2: (1, 9), 4: (1, 2, 8, 9), 6: (1, 2, 3, 7, 8, 9)}[starts]
    cells, sym, goals, st = [], [], {}, []
    for r, row in enumerate(lay):
        for c, ch in enumerate(row):
            if ch == ".":
                continue
            cells.append((r, c))
            sym.append(2 if ch == "L" else 0 if ch == "a" or (ch == "1") else 1)
            if ch in "123":
                goals[int(ch)] = len(cells) - 1
            if r == 1 and c in start_cols:
                st.append(len(cells) - 1)
    return Maze(cells, np.array(sym), tuple(goals[i] for i in (1, 2, 3)), eps, gamma, H, tuple(st))


def prefix_beliefs(maze, length):
    """Every posterior the goal-free filter can reach after a passive prefix of `length` moves, with its probability
    under uniformly random prefix actions. Returns a list of (belief, probability, actions, symbols)."""
    out = {}
    stack = []
    for o in range(maze.n_sym):
        b, z = maze.observe(maze.prior(), o)
        if z > 0:
            stack.append((b, z, (), (o,)))
    while stack:
        b, p, acts, obs = stack.pop()
        if len(acts) == length:
            kk = np.round(b, 9).tobytes()
            if kk in out:
                out[kk][1] += p
            else:
                out[kk] = [b, p, acts, obs]
            continue
        for a in range(4):
            q = b @ maze.T[a]
            for o in range(maze.n_sym):
                b2, z = maze.observe(q, o)
                if z > 1e-12:
                    stack.append((b2, p * z / 4, acts + (a,), obs + (o,)))
    return [tuple(v) for v in out.values()]


def effective_dimension(B, w):
    """Weighted PCA of beliefs B [N, n]: participation ratio and the number of components for 95 % of the variance."""
    w = np.asarray(w) / np.sum(w)
    X = B - w @ B
    ev = np.clip(np.linalg.eigvalsh((X * w[:, None]).T @ X)[::-1], 0, None)
    return dict(participation=float(ev.sum() ** 2 / (ev ** 2).sum()), n95=int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1),
                spectrum=(ev / ev.sum())[:6].round(4).tolist())
