"""A larger aliased maze without a solver: layout, vectorised environment with the exact Bayesian filter, baselines.

World. Open cells of a size x size box: a perfect maze on the lattice of rooms at even coordinates, plus a few extra
openings that make loops. Moves N, S, W, E are deterministic; a move into a wall leaves the agent in place. Ordinary
symbols come in sibling pairs and are noisy (own symbol with probability 1 - eps, sibling with eps); several cells
share a symbol. Landmark cells emit unique symbols without noise.

Episode (no passive prefix). The goal, one of a fixed set of cells, is shown with the first symbol; the agent starts
in a uniformly random cell that is no goal and acts from the first move. Entering the goal pays gamma^(t - 1) for the
t-th move and ends the episode; after H moves it ends without reward. Tokens as in mazemodel with a prefix of length
0: [OBS o_0] [GOAL g] [EVT a o] ...; the next move is predicted at the goal token and at every event token.

Ground truth without a solver. The exact posterior over the location (a vector of one number per cell, updated in
closed form) is carried by every environment. Action values are not computed: the state space of beliefs is too
large for the exact expectimax of mazebelief / mazegraph. Reference policies use the filter and shortest paths only:
    oracle   knows the cell (an upper bound no agent can reach)
    qmdp     belief-weighted shortest-path values: ignores the value of information
    qmdp2    one-step lookahead on qmdp: values the information of the next symbol only
    mls      acts on the most likely cell
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazemodel as MM


def layout(seed, size=10, loops=6):
    rng = np.random.default_rng(seed)
    open_, seen, stack = {(0, 0)}, {(0, 0)}, [(0, 0)]
    while stack:
        r, c = stack[-1]
        nb = [(r + dr, c + dc, dr, dc) for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
              if 0 <= r + dr < size and 0 <= c + dc < size and (r + dr, c + dc) not in seen]
        if not nb:
            stack.pop(); continue
        r2, c2, dr, dc = nb[rng.integers(len(nb))]
        open_ |= {(r + dr // 2, c + dc // 2), (r2, c2)}; seen.add((r2, c2)); stack.append((r2, c2))
    walls = [(r, c) for r in range(size - 1) for c in range(size - 1) if (r, c) not in open_ and (r % 2) + (c % 2) == 1]
    for i in rng.choice(len(walls), min(loops, len(walls)), replace=False):
        open_.add(walls[i])
    return sorted(open_), rng


def distances(nxt, target):
    """Moves needed to reach `target` from every cell (reverse breadth-first search)."""
    n = len(nxt)
    d = np.full(n, np.inf); d[target] = 0
    q = deque([target])
    while q:
        u = q.popleft()
        for s in range(n):
            if d[s] == np.inf and u in nxt[s]:
                d[s] = d[u] + 1; q.append(s)
    return d


def make(seed=0, size=10, loops=6, pairs=2, n_land=2, n_goals=4, eps=0.2, gamma=0.97, H=40):
    """The maze: random symbols from `pairs` sibling pairs, `n_land` landmarks, `n_goals` goals spread out by
    farthest-point sampling on path distance. Start: every cell that is not a goal."""
    cells, rng = layout(seed, size, loops)
    n = len(cells)
    sym = rng.integers(0, 2 * pairs, n)
    m0 = MB.Maze(cells, sym.copy(), (0,), eps, gamma, H)
    D = np.stack([distances(m0.nxt, s) for s in range(n)])
    goals = [int(rng.integers(n))]
    while len(goals) < n_goals:
        goals.append(int(D[goals].min(0).argmax()))
    land = [int(i) for i in rng.permutation([i for i in range(n) if i not in goals])[:n_land]]
    for j, i in enumerate(land):
        sym[i] = 2 * pairs + j
    m = MB.Maze(cells, sym, tuple(goals), eps, gamma, H)
    for i in land:                                                    # every landmark is noiseless (Maze does it for the last symbol only)
        m.E[i] = 0; m.E[i, sym[i]] = 1
    for s in range(n):
        if sym[s] < 2 * pairs:
            m.E[s] = 0; m.E[s, sym[s]] = 1 - eps; m.E[s, sym[s] ^ 1] = eps
    m.landmarks = tuple(land)
    return m


def draw(maze):
    R, C = max(r for r, _ in maze.cells) + 1, max(c for _, c in maze.cells) + 1
    ix = {c: i for i, c in enumerate(maze.cells)}
    rows = []
    for r in range(R):
        s = ""
        for c in range(C):
            if (r, c) not in ix:
                s += "#"
            elif ix[(r, c)] in maze.goals:
                s += "ABCDEFGH"[maze.goals.index(ix[(r, c)])]
            elif ix[(r, c)] in maze.landmarks:
                s += "*"
            else:
                s += str(int(maze.sym[ix[(r, c)]]))
        rows.append(s)
    return "\n".join(rows)


class Sim:
    """The maze on a device. Stands where mazeppo.Tables stands for the small maze, without a belief graph."""

    def __init__(self, maze: MB.Maze, device):
        t = lambda x, dt=None: torch.tensor(np.asarray(x), device=device, dtype=dt)
        self.maze, self.dev, self.H, self.gamma, self.n, self.n_sym, self.max_prefix = maze, device, maze.H, maze.gamma, maze.n, maze.n_sym, 0
        self.nxt_cell, self.E = t(maze.nxt), t(maze.E, torch.float32)
        self.goal_cell = t(maze.goals)
        self.starts = t([i for i in range(maze.n) if i not in maze.goals])
        self.prior = t(maze.prior(), torch.float32)
        self.dist = t(np.stack([distances(maze.nxt, g) for g in maze.goals]), torch.float32)          # [goals, n]
        self.L = MM.seq_len(maze.H, 0)
        self.cells = list(maze.cells)

    def q_mdp(self, goal, k):
        """Fully observed action values [N, n, 4] for goals [N] with k [N] moves left."""
        dn = self.dist[goal][:, self.nxt_cell]                                                # [N, n, 4] distance after the move
        return torch.where(dn <= (k[:, None, None] - 1), self.gamma ** dn, torch.zeros_like(dn))


class Env:
    """A batch of episodes; every one carries its true cell and the exact posterior over the location."""

    def __init__(self, t: Sim, N, gen=None, goal=None, cell=None):
        self.t, self.N, self.gen = t, N, gen
        dev = t.dev
        r = lambda hi: torch.randint(hi, (N,), device=dev, generator=gen)
        self.cell = t.starts[r(len(t.starts))] if cell is None else cell
        self.goal = r(len(t.goal_cell)) if goal is None else goal
        self.tok = torch.zeros(N, t.L, MM.NF, dtype=torch.long, device=dev)
        o = self.sample_symbol()
        self.tok[:, 0, MM.F_TYPE], self.tok[:, 0, MM.F_SYM] = MM.OBS, o + 1
        self.tok[:, 1, MM.F_TYPE], self.tok[:, 1, MM.F_GOAL] = MM.GOAL, self.goal + 1
        b = t.prior[None] * t.E[:, o].T
        self.belief = b / b.sum(1, keepdim=True)
        self.idx = torch.ones(N, dtype=torch.long, device=dev)        # token index of the current decision
        self.k = torch.full((N,), t.H, device=dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)

    def sample_symbol(self):
        return torch.multinomial(self.t.E[self.cell], 1, generator=self.gen).squeeze(1)

    def moved(self, a):
        """The posterior after move a [N], before the next symbol: (probability of entering the goal [N], the
        unnormalised location distribution given the episode continues [N, n])."""
        t = self.t
        p = torch.zeros_like(self.belief).scatter_add_(1, t.nxt_cell[:, a].T, self.belief)
        n = torch.arange(self.N, device=t.dev)
        g = t.goal_cell[self.goal]
        r = p[n, g].clone()
        p[n, g] = 0
        return r, p

    def step(self, a):
        t, live = self.t, ~self.done
        _, p = self.moved(a)
        self.cell = torch.where(live, t.nxt_cell[self.cell, a], self.cell)
        hit = live & (self.cell == t.goal_cell[self.goal])
        o = self.sample_symbol()
        self.k = self.k - live.long()
        cont = live & ~hit & (self.k > 0)
        b = p * t.E[:, o].T
        b = b / b.sum(1, keepdim=True).clamp(min=1e-30)
        self.belief = torch.where(cont[:, None], b, self.belief)
        self.idx = self.idx + 1
        n = torch.nonzero(cont).squeeze(1)
        self.tok[n, self.idx[n]] = torch.stack([torch.full_like(a, MM.EVT), o + 1, a + 1, torch.zeros_like(a), self.k + 1], -1)[n]
        self.done = self.done | ~cont
        return hit.float()


@dataclass
class Roll:
    tok: torch.Tensor            # [N, L, NF]
    pos: torch.Tensor            # [N, H] token index of each decision
    act: torch.Tensor            # [N, H]
    logp: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    alive: torch.Tensor          # [N, H]
    goal: torch.Tensor           # [N]
    cell: torch.Tensor           # [N, H] true cell at each decision
    belief: torch.Tensor | None  # [N, H, n] exact posterior at each decision (if recorded)


@torch.no_grad()
def rollout(net, t: Sim, N, gen=None, greedy=False, behaviour=None, goal=None, cell=None, record=False):
    """Episodes under a network (sampled or greedy) or a behaviour f(env) -> actions."""
    env = Env(t, N, gen, goal, cell)
    H, dev = t.H, t.dev
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=dev)
    pos, act, cells = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
    logp, val, rew, alive = z(), z(), z(), z(dt=torch.bool)
    bel = z(t.n) if record else None
    n = torch.arange(N, device=dev)
    for s in range(H):
        live = ~env.done
        if behaviour is None:
            lg, v = net(env.tok[:, : s + 2], None)
            lg, v = lg[n, env.idx], v[n, env.idx]
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
            logp[:, s], val[:, s] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1), v
        else:
            a = behaviour(env)
        pos[:, s], act[:, s], alive[:, s], cells[:, s] = env.idx, a, live, env.cell
        if record:
            bel[:, s] = env.belief
        rew[:, s] = env.step(a)
    return Roll(env.tok, pos, act, logp, val, rew, alive, env.goal, cells, bel)


# ------------------------------------------------------------------ reference policies (filter and shortest paths only)

def oracle(env):
    t = env.t
    return t.dist[env.goal[:, None], t.nxt_cell[env.cell]].argmin(-1)


def mls(env):
    t = env.t
    return t.dist[env.goal[:, None], t.nxt_cell[env.belief.argmax(-1)]].argmin(-1)


def qmdp_values(env):
    return torch.einsum("ns,nsa->na", env.belief, env.t.q_mdp(env.goal, env.k))


def qmdp(env):
    return qmdp_values(env).argmax(-1)


def qmdp2_values(env):
    """One-step lookahead: r(a) + gamma * sum_o max_a' (w_{a,o} . Q_mdp(k - 1)), w the unnormalised posterior."""
    t = env.t
    Q1 = t.q_mdp(env.goal, env.k - 1)                                                          # [N, n, 4]
    out = []
    for a in range(4):
        r, p = env.moved(torch.full((env.N,), a, device=t.dev))
        w = p[:, None, :] * t.E.T[None]                                                        # [N, symbols, n]
        out.append(r + t.gamma * torch.einsum("nos,nsb->nob", w, Q1).max(-1).values.sum(-1))
    return torch.stack(out, 1)


def qmdp2(env):
    return qmdp2_values(env).argmax(-1)


def uniform(gen):
    return lambda env: torch.randint(4, (env.N,), device=env.t.dev, generator=gen)


def entropy(b):
    return -(b * b.clamp(min=1e-30).log()).sum(-1)


def expected_entropy(env):
    """For each move [N, 4]: the expected posterior entropy after the move and the next symbol (0 where the move
    ends the episode with certainty)."""
    t = env.t
    out = []
    for a in range(4):
        _, p = env.moved(torch.full((env.N,), a, device=t.dev))
        w = p[:, None, :] * t.E.T[None]                                                        # [N, symbols, n]
        z = w.sum(-1)                                                                          # [N, symbols]
        out.append((z * entropy(w / z[..., None].clamp(min=1e-30))).sum(-1))
    return torch.stack(out, 1)


def summarize(b: Roll, t: Sim, M=1):
    """Per model of a model-major batch: discounted return, success, length, by goal."""
    N, H = b.act.shape
    disc = t.gamma ** torch.arange(H, device=t.dev)
    out = []
    for m in range(M):
        s = slice(m * N // M, (m + 1) * N // M)
        ret = (b.rew[s] * disc).sum(1)
        row = dict(ret=ret.mean().item(), success=b.rew[s].sum(1).mean().item(), length=b.alive[s].sum(1).float().mean().item())
        for g in range(len(t.goal_cell)):
            row[f"ret_G{g + 1}"] = ret[b.goal[s] == g].mean().item()
        out.append(row)
    return out
