"""Bump maze: a MiniGrid maze with a known map and goal, an unknown start cell and heading, and one bit of observation
per step (whether a forward move bumped a wall). The most partially observed navigation task that still carries
information, with an exact belief (uniform over the states consistent with the history, since transitions and
observations are deterministic) and an exact optimal policy (value iteration over the reachable belief sets).

State: (cell, heading) with heading 0 E, 1 S, 2 W, 3 N (MiniGrid's convention). Actions: 0 turn left, 1 turn right,
2 forward (MiniGrid's left / right / forward). Observation after a forward move: 1 if the agent bumped (stayed), 0 if
it moved; after a turn: 0. Entering the goal cell ends the episode with reward gamma^(t-1) at move t; after H moves
it ends without reward. The start is uniform over (non-goal free cell, heading).

The MiniGrid environment (`MiniGridBump`) is the same world through minigrid's own dynamics; `test_bumpmaze` checks
the two agree move for move. Everything else (filter, solver, baselines, batched environment, text rendering for a
language model) is here and needs only numpy / torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

DIRS = np.array([(1, 0), (0, 1), (-1, 0), (0, -1)])        # E, S, W, N as (dx, dy); y grows downward as in MiniGrid
DIR_NAMES = ("east", "south", "west", "north")
ACTIONS = ("left", "right", "forward")


# ------------------------------------------------------------------ layout

def layout(seed=0, size=9, loops=4):
    """A perfect maze on the odd cells of a size x size grid (size odd; the border is wall), plus `loops` extra
    openings. Returns a bool array free[y, x]."""
    rng = np.random.default_rng(seed)
    free = np.zeros((size, size), dtype=bool)
    rooms = [(y, x) for y in range(1, size - 1, 2) for x in range(1, size - 1, 2)]
    start = rooms[rng.integers(len(rooms))]
    seen, stack = {start}, [start]
    free[start] = True
    while stack:
        y, x = stack[-1]
        nb = [(y + dy, x + dx) for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)) if 1 <= y + dy < size - 1 and 1 <= x + dx < size - 1 and (y + dy, x + dx) not in seen]
        if not nb:
            stack.pop(); continue
        ny, nx = nb[rng.integers(len(nb))]
        free[(y + ny) // 2, (x + nx) // 2] = True; free[ny, nx] = True
        seen.add((ny, nx)); stack.append((ny, nx))
    walls = [(y, x) for y in range(1, size - 1) for x in range(1, size - 1) if not free[y, x]
             and ((free[y - 1, x] and free[y + 1, x]) or (free[y, x - 1] and free[y, x + 1]))]
    for i in rng.permutation(len(walls))[:loops]:
        free[walls[i]] = True
    return free


# ------------------------------------------------------------------ the world

@dataclass
class World:
    free: np.ndarray                    # [size, size] bool
    goal: tuple                         # (x, y)
    gamma: float = 0.97
    H: int = 40
    cells: list = field(default_factory=list)          # index -> (x, y)
    index: dict = field(default_factory=dict)          # (x, y) -> index

    def __post_init__(self):
        self.cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(self.free))]
        self.index = {c: i for i, c in enumerate(self.cells)}
        n = len(self.cells); self.n_cells = n; self.n_states = 4 * n
        self.goal_idx = self.index[self.goal]
        # transitions over states s = 4 * cell + heading
        nxt = np.zeros((self.n_states, 3), dtype=np.int64); bump = np.zeros(self.n_states, dtype=bool)
        for c, (x, y) in enumerate(self.cells):
            for d in range(4):
                s = 4 * c + d
                nxt[s, 0] = 4 * c + (d - 1) % 4; nxt[s, 1] = 4 * c + (d + 1) % 4
                tx, ty = x + DIRS[d][0], y + DIRS[d][1]
                if self.free[ty, tx]:
                    nxt[s, 2] = 4 * self.index[(tx, ty)] + d
                else:
                    nxt[s, 2] = s; bump[s] = True
        self.nxt, self.bump = nxt, bump
        self.at_goal = np.array([s // 4 == self.goal_idx for s in range(self.n_states)])
        self.start_states = np.array([s for s in range(self.n_states) if not self.at_goal[s]])
        # shortest number of moves from every state to the goal (turns count), by BFS on the reverse graph
        dist = np.full(self.n_states, np.inf)
        goal_states = [s for s in range(self.n_states) if self.at_goal[s]]
        rev = [[] for _ in range(self.n_states)]
        for s in range(self.n_states):
            for a in range(3):
                if nxt[s, a] != s: rev[nxt[s, a]].append(s)
        frontier = goal_states
        for g in goal_states: dist[g] = 0
        while frontier:
            new = []
            for t in frontier:
                for s in rev[t]:
                    if dist[s] == np.inf: dist[s] = dist[t] + 1; new.append(s)
            frontier = new
        self.dist = dist
        # Q* of the fully observed problem in discounted return: entering the goal at move t pays gamma^(t-1)
        self.q_known = np.stack([np.where(self.at_goal[nxt[:, a]] & (nxt[:, a] != np.arange(self.n_states)), 1.0, self.gamma ** (dist[nxt[:, a]])) for a in range(3)], 1)
        self.q_known[self.at_goal] = 0

    @property
    def size(self): return self.free.shape[0]

    def ascii(self, agent=None, mark_goal=True):
        """The map as text: '#' wall, '.' free, 'G' goal, and the agent as > v < ^ if given (x, y, d)."""
        rows = []
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                ch = "." if self.free[y, x] else "#"
                if mark_goal and (x, y) == self.goal: ch = "G"
                if agent is not None and (x, y) == (agent[0], agent[1]): ch = ">v<^"[agent[2]]
                row += ch
            rows.append(row)
        return "\n".join(rows)


def cell_distances(free):
    """BFS move distance between free cells (turns not counted): dict cell -> dict cell -> steps."""
    cells = [(int(x), int(y)) for y, x in zip(*np.nonzero(free))]; out = {}
    for c in cells:
        d = {c: 0}; q = [c]
        while q:
            x, y = q.pop(0)
            for dx, dy in DIRS:
                n = (x + int(dx), y + int(dy))
                if free[n[1], n[0]] and n not in d: d[n] = d[(x, y)] + 1; q.append(n)
        out[c] = d
    return out


def pick_goals(free, k, first=None):
    """k goal cells spread out: the first as given (or the cell farthest from the centre of mass), then repeatedly
    the free cell farthest (in moves) from the goals chosen so far."""
    D = cell_distances(free); cells = list(D)
    if first is None:
        cy, cx = np.mean([c[1] for c in cells]), np.mean([c[0] for c in cells]); first = max(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    goals = [tuple(int(v) for v in first)]
    while len(goals) < k:
        goals.append(max((c for c in cells if c not in goals), key=lambda c: min(D[g][c] for g in goals)))
    return goals


def make(seed=0, size=9, loops=4, gamma=0.97, H=40, goal=None):
    free = layout(seed, size, loops)
    if goal is None:                                                   # the free cell farthest (in BFS steps) from the centre of mass
        cells = [(x, y) for y, x in zip(*np.nonzero(free))]
        cy, cx = np.mean([c[1] for c in cells]), np.mean([c[0] for c in cells])
        goal = max(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
    return World(free, (int(goal[0]), int(goal[1])), gamma, H)


# ------------------------------------------------------------------ exact belief (a set) and the exact solver

def step_belief(w, mask, a, obs, remove_goal=True):
    """mask: bool [n_states] of consistent states; after action a and observation obs (0 moved / 1 bumped; for
    turns obs is ignored). Returns the new mask (states that terminated at the goal are removed: the episode would
    have ended; remove_goal=False gives the goal-free belief shared by every goal)."""
    if a < 2:
        new = np.zeros_like(mask); new[w.nxt[mask, a]] = True; return new
    keep = mask & (w.bump == bool(obs))
    new = np.zeros_like(mask); new[w.nxt[keep, 2]] = True
    if remove_goal: new &= ~w.at_goal
    return new


def common_mask(w, acts, obss):
    """The goal-free belief after a history: every free cell x heading is a possible start."""
    m = np.ones(w.n_states, dtype=bool)
    for a, o in zip(acts, obss): m = step_belief(w, m, a, o, remove_goal=False)
    return m


def solve(w, max_sets=2_000_000, tol=1e-10, iters=10_000):
    """Exact optimal values over the reachable belief sets (uniform over the set). V(S) = max_a [P(goal | S, a) +
    gamma * sum_o P(o | S, a) V(S_o)], the value of a belief in discounted return per remaining move. Returns the
    dict set -> index, V [n_sets], Q [n_sets, 3], and the transition tables."""
    key = lambda m: np.packbits(m).tobytes()
    full = np.zeros(w.n_states, dtype=bool); full[w.start_states] = True
    sets, masks, queue = {key(full): 0}, [full], [full]
    trans = []                                                          # per set: per action: list of (prob, next index or -1 for goal)
    while queue:
        m = queue.pop()
        i = sets[key(m)]
        while len(trans) <= i: trans.append(None)
        n = m.sum(); row = []
        for a in range(3):
            outs = []
            if a < 2:
                nm = step_belief(w, m, a, 0)
                k = key(nm)
                if k not in sets:
                    sets[k] = len(masks); masks.append(nm); queue.append(nm)
                    if len(masks) > max_sets: raise RuntimeError("too many belief sets")
                outs.append((1.0, sets[k]))
            else:
                reach = m & ~w.bump & w.at_goal[w.nxt[:, 2]]
                p_goal = reach.sum() / n
                if p_goal > 0: outs.append((p_goal, -1))
                for obs in (1, 0):
                    cons = m & (w.bump == bool(obs))
                    if obs == 0: cons = cons & ~w.at_goal[w.nxt[:, 2]]
                    p = cons.sum() / n
                    if p == 0: continue
                    nm = step_belief(w, m, 2, obs)
                    k = key(nm)
                    if k not in sets:
                        sets[k] = len(masks); masks.append(nm); queue.append(nm)
                        if len(masks) > max_sets: raise RuntimeError("too many belief sets")
                    outs.append((p, sets[k]))
            row.append(outs)
        trans[i] = row
    S = len(masks); V = np.zeros(S); Q = np.zeros((S, 3))
    for _ in range(iters):
        for i in range(S):
            for a in range(3):
                Q[i, a] = sum(p * (1.0 if j < 0 else w.gamma * V[j]) for p, j in trans[i][a])
        Vn = Q.max(1)
        if np.abs(Vn - V).max() < tol: V = Vn; break
        V = Vn
    return dict(sets=sets, masks=masks, V=V, Q=Q, trans=trans)


class Filter:
    """The exact belief of a batch of episodes as bool masks [N, n_states]; posterior = mask / mask.sum()."""
    def __init__(self, w, N, device="cpu", remove_goal=True):
        self.w, self.dev, self.remove_goal = w, device, remove_goal
        self.nxt = torch.as_tensor(w.nxt, device=device); self.bump = torch.as_tensor(w.bump, device=device); self.at_goal = torch.as_tensor(w.at_goal, device=device)
        full = torch.zeros(w.n_states, dtype=torch.bool, device=device); full[torch.as_tensor(w.start_states, device=device)] = True
        if not remove_goal: full[:] = True                                # the goal-free belief: any cell is a possible start
        self.mask = full[None].repeat(N, 1)

    def update(self, a, obs):
        """a [N] long, obs [N] long (ignored for turns)."""
        N = len(a); m = self.mask
        keep = torch.where((a == 2)[:, None], m & (self.bump[None] == (obs[:, None] == 1)), m)
        tgt = self.nxt[torch.arange(self.w.n_states, device=self.dev)[None].expand(N, -1), a[:, None].expand(N, self.w.n_states)]     # [N, n_states]
        new = torch.zeros(m.shape, dtype=torch.int32, device=self.dev).scatter_add_(1, tgt, keep.int()) > 0     # true where any kept state maps there
        if self.remove_goal: new &= ~self.at_goal[None]
        self.mask = new

    @property
    def posterior(self):
        m = self.mask.double(); return m / m.sum(1, keepdim=True).clamp(min=1)


# ------------------------------------------------------------------ batched environment and reference policies

class Env:
    """N episodes in parallel on torch. step(a) returns (reward [N], obs [N]); done [N] tracks ended episodes."""
    def __init__(self, w, N, gen, device="cpu", start=None):
        self.w, self.N, self.dev = w, N, device
        self.nxt = torch.as_tensor(w.nxt, device=device); self.bump = torch.as_tensor(w.bump, device=device); self.at_goal = torch.as_tensor(w.at_goal, device=device)
        starts = torch.as_tensor(w.start_states, device=device)
        self.s = starts[torch.randint(len(starts), (N,), device=device, generator=gen)] if start is None else start.clone()
        self.done = torch.zeros(N, dtype=torch.bool, device=device); self.t = 0
        self.filter = Filter(w, N, device)
        self.acts, self.obss = [], []

    def step(self, a):
        a = a.clone(); a[self.done] = 0
        self.t += 1
        s2 = self.nxt[self.s, a]
        obs = ((a == 2) & (s2 == self.s)).long()
        reached = (a == 2) & self.at_goal[s2] & ~self.done
        rew = torch.where(reached, self.w.gamma ** (self.t - 1), 0.0)
        self.filter.update(a, obs)
        self.s = s2
        self.done |= reached
        if self.t >= self.w.H: self.done[:] = True
        self.acts.append(a); self.obss.append(obs)
        return rew, obs


def qmdp_action(w, post):
    """QMDP: argmax_a sum_s b(s) Q_known(s, a). post [N, n_states] double."""
    q = post @ torch.as_tensor(w.q_known, device=post.device, dtype=post.dtype)
    return q.argmax(1)


def optimal_action(w, sol, masks, tie="first"):
    """The exact optimum for a batch of belief masks (numpy bool [N, n_states]); each mask must be a reachable set."""
    key = lambda m: np.packbits(m).tobytes()
    idx = np.array([sol["sets"][key(m)] if m.any() else -1 for m in masks])        # an empty mask: the episode has ended
    Q = np.where((idx >= 0)[:, None], sol["Q"][np.maximum(idx, 0)], 0.0)
    return Q.argmax(1), Q


# ------------------------------------------------------------------ text for a language model

def describe(w):
    """The rules and the map, for a prompt."""
    return (
        "You are a robot in a maze. The map below is known to you: '#' is a wall, '.' is a free cell, 'G' is the goal cell. "
        "Rows are numbered from the top starting at 0 and columns from the left starting at 0. "
        "You do NOT know which free cell you start in, nor which direction you face (east, south, west or north); every non-goal free cell and every direction is equally likely. "
        "Each turn you choose one action: 'left' (turn 90 degrees left, you stay in your cell), 'right' (turn 90 degrees right), or 'forward' (move one cell in the direction you face; if there is a wall ahead you stay where you are). "
        "After 'forward' you are told 'moved' or 'bumped'. After a turn you are told nothing. The episode ends as soon as you enter the goal cell. "
        "You want to reach the goal in as few actions as possible, which may require actions that tell you where you are.\n\n"
        f"Map:\n{w.ascii()}\n"
    )


def history_text(acts, obss):
    if not acts: return "History: (no actions yet)\n"
    items = [f"{ACTIONS[a]}" + (f" -> {'bumped' if o else 'moved'}" if a == 2 else "") for a, o in zip(acts, obss)]
    return "History: " + "; ".join(items) + "\n"


# ------------------------------------------------------------------ the MiniGrid environment (optional dependency)

def minigrid_env(w, start_state=None, render_mode=None):
    """The same world as a minigrid environment: walls from `w.free`, the goal cell, the agent at `start_state`
    (index into states; None: uniform over the start states). Observations are replaced by the bump bit."""
    import gymnasium as gym
    from minigrid.core.grid import Grid
    from minigrid.core.mission import MissionSpace
    from minigrid.core.world_object import Goal, Wall
    from minigrid.minigrid_env import MiniGridEnv

    class BumpMaze(MiniGridEnv):
        def __init__(self, world, start, **kw):
            self.world, self.start_state = world, start
            super().__init__(mission_space=MissionSpace(mission_func=lambda: "reach the goal"), width=world.size, height=world.size,
                             max_steps=world.H, see_through_walls=False, agent_view_size=3, **kw)
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Discrete(2)

        def _gen_grid(self, width, height):
            self.grid = Grid(width, height)
            for y in range(height):
                for x in range(width):
                    if not self.world.free[y, x]: self.grid.set(x, y, Wall())
            self.put_obj(Goal(), *self.world.goal)
            s = self.start_state if self.start_state is not None else int(self.np_random.choice(self.world.start_states))
            self.agent_pos = tuple(self.world.cells[s // 4]); self.agent_dir = s % 4
            self.mission = "reach the goal"

        def state(self):
            return 4 * self.world.index[tuple(int(v) for v in self.agent_pos)] + int(self.agent_dir)

        def step(self, action):
            before = self.state()
            _, reward, terminated, truncated, info = super().step(action)
            bump = int(action == 2 and self.state() == before)
            return bump, reward, terminated, truncated, info

        def reset(self, **kw):
            super().reset(**kw); return 0, {}

    return BumpMaze(w, start_state, render_mode=render_mode)



# ------------------------------------------------------------------ the additive ceiling (no network)

def additive_ceiling(decisions, K, H, steps=3000, lr=0.05, seed=0, device="cpu"):
    """decisions: list of (common-set key, goal, step, optimal-set bool [3]). Fits argmax_a h[set, a] + g[goal, step, a]
    by gradient on the log of the softmax mass on the optimal set (as a readout fit), with a free vector per
    distinct common set, and returns the share of decisions whose argmax is optimal (the additive ceiling for this
    sample) and the same for a goal-blind rule h[set, a] alone and a history-blind rule g[goal, step, a] alone."""
    keys = {}; ks = torch.tensor([keys.setdefault(k, len(keys)) for k, _, _, _ in decisions], device=device)
    gs = torch.tensor([g for _, g, _, _ in decisions], device=device); st = torch.tensor([min(t, H - 1) for _, _, t, _ in decisions], device=device)
    opt = torch.tensor(np.array([o for _, _, _, o in decisions]), device=device)
    out = {}
    for name, use_h, use_g in (("additive", True, True), ("goal_blind", True, False), ("history_blind", False, True)):
        torch.manual_seed(seed)
        h = torch.zeros(len(keys), 3, device=device, requires_grad=True); g = torch.zeros(K, H, 3, device=device, requires_grad=True)
        o_ = torch.optim.Adam([h, g], lr=lr)
        for i in range(steps):
            z = (h[ks] if use_h else 0) + (g[gs, st] if use_g else 0)
            loss = -(torch.logsumexp(torch.log_softmax(z, -1).masked_fill(~opt, -1e9), -1)).mean()
            o_.zero_grad(); loss.backward(); o_.step()
        with torch.no_grad():
            z = (h[ks] if use_h else 0) + (g[gs, st] if use_g else 0)
            out[name] = float(opt.gather(1, z.argmax(1, keepdim=True)).double().mean())
    out["sets"] = len(keys)
    return out
