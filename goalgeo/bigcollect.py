"""Collection in the larger maze: several goals per episode, no solver. Builds on bigmaze.

Episode. Two of the maze's goal cells are active and shown at the start, as two goal tokens (in the goals' order).
Entering an active goal that has not been collected pays gamma^(t - 1) for the t-th move and is announced: the event
token of that move names the goal. The episode ends when both are collected or after H moves. The agent starts in a
uniformly random cell that is no goal cell.
Tokens: [OBS o_0] [GOAL g] [GOAL g'] [EVT a o (goal collected) (moves left)] ...; the next move is predicted at the
second goal token and at every event token.

Ground truth without a solver. The exact posterior over the location: a pickup names the goal, so the posterior
collapses onto its cell; otherwise the mass that would have entered an uncollected active goal is removed. Reference
policies use the filter and the fully observed problem only. The fully observed problem (state: cell and the set of
goals left) is small and solved exactly here by finite-horizon value iteration; it is not the partially observed
optimum.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM


class Sim(BM.Sim):
    def __init__(self, maze, device, m=2):
        super().__init__(maze, device)
        K = len(maze.goals)
        self.K, self.m, self.max_prefix = K, m, m - 1
        self.L = 1 + m + maze.H
        self.pairs = torch.tensor(list(itertools.combinations(range(K), m)), device=device)       # [P, m] goal indices, increasing
        self.pair_mask = (1 << self.pairs).sum(1)                                                # [P] bit mask of the active goals
        gidx = np.full(maze.n, -1)
        gidx[list(maze.goals)] = np.arange(K)
        self.goal_of_cell = torch.tensor(gidx, device=device)                                    # [n] goal index of a cell, -1 if none
        self.bits = 1 << torch.arange(K, device=device)
        # fully observed values Q[mask, k, cell, move] for every set of goals left with at most m members
        H, n, g = maze.H, maze.n, maze.gamma
        Q = np.zeros((1 << K, H + 1, n, 4)); V = np.zeros((1 << K, H + 1, n))
        masks = sorted((mk for mk in range(1, 1 << K) if bin(mk).count("1") <= m), key=lambda mk: bin(mk).count("1"))
        nxt = maze.nxt
        gn = gidx[nxt]                                                                           # [n, 4] goal entered by the move, -1 if none
        for mk in masks:
            hit = (gn >= 0) & (((mk >> np.maximum(gn, 0)) & 1) == 1)
            rest = np.where(hit, mk & ~(1 << np.maximum(gn, 0)), mk)                             # [n, 4] goals left after the move
            for k in range(1, H + 1):
                Q[mk, k] = hit + g * V[rest, k - 1, nxt]
                V[mk, k] = Q[mk, k].max(-1)
        self.Qc = torch.tensor(Q, dtype=torch.float32, device=device)
        self.Vc = torch.tensor(V, dtype=torch.float32, device=device)


class Env:
    def __init__(self, t: Sim, N, gen=None, pair=None, cell=None):
        self.t, self.N, self.gen = t, N, gen
        dev = t.dev
        r = lambda hi: torch.randint(hi, (N,), device=dev, generator=gen)
        self.cell = t.starts[r(len(t.starts))] if cell is None else cell
        self.pair = r(len(t.pairs)) if pair is None else pair
        self.goal = self.pair                                                                    # the instruction, for bigmaze.summarize
        self.mask = t.pair_mask[self.pair].clone()                                               # goals left
        self.tok = torch.zeros(N, t.L, MM.NF, dtype=torch.long, device=dev)
        o = self.sample_symbol()
        self.tok[:, 0, MM.F_TYPE], self.tok[:, 0, MM.F_SYM] = MM.OBS, o + 1
        for j in range(t.m):
            self.tok[:, 1 + j, MM.F_TYPE], self.tok[:, 1 + j, MM.F_GOAL] = MM.GOAL, t.pairs[self.pair, j] + 1
        b = t.prior[None] * t.E[:, o].T
        self.belief = b / b.sum(1, keepdim=True)
        self.idx = torch.full((N,), t.m, dtype=torch.long, device=dev)                           # token index of the current decision
        self.k = torch.full((N,), t.H, device=dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)

    def sample_symbol(self):
        return torch.multinomial(self.t.E[self.cell], 1, generator=self.gen).squeeze(1)

    def left(self):
        """[N, K]: which goals are still to collect."""
        return (self.mask[:, None] & self.t.bits[None]) != 0

    def moved(self, a):
        """After move a [N], before the next symbol: (probability of collecting each goal [N, K], the unnormalised
        location distribution given that nothing is collected [N, n])."""
        t = self.t
        p = torch.zeros_like(self.belief).scatter_add_(1, t.nxt_cell[:, a].T, self.belief)
        pick = p[:, t.goal_cell] * self.left()
        p = p.clone()
        p[:, t.goal_cell] = p[:, t.goal_cell] * (~self.left())
        return pick, p

    def step(self, a):
        t, live = self.t, ~self.done
        _, p = self.moved(a)
        self.cell = torch.where(live, t.nxt_cell[self.cell, a], self.cell)
        gj = t.goal_of_cell[self.cell]
        hit = live & (gj >= 0) & ((self.mask >> gj.clamp(min=0)) & 1).bool()
        self.mask = torch.where(hit, self.mask & ~(1 << gj.clamp(min=0)), self.mask)
        o = self.sample_symbol()
        self.k = self.k - live.long()
        cont = live & (self.mask != 0) & (self.k > 0)
        b = p * t.E[:, o].T
        b = b / b.sum(1, keepdim=True).clamp(min=1e-30)
        b = torch.where(hit[:, None], torch.nn.functional.one_hot(self.cell, t.n).to(b.dtype), b)
        self.belief = torch.where(live[:, None], b, self.belief)
        self.idx = self.idx + 1
        n = torch.nonzero(cont).squeeze(1)
        self.tok[n, self.idx[n]] = torch.stack([torch.full_like(a, MM.EVT), o + 1, a + 1, torch.where(hit, gj + 1, torch.zeros_like(gj)), self.k + 1], -1)[n]
        self.done = self.done | ~cont
        return hit.float()


@torch.no_grad()
def rollout(net, t: Sim, N, gen=None, greedy=False, behaviour=None, pair=None, cell=None, record=False):
    env = Env(t, N, gen, pair, cell)
    H, dev = t.H, t.dev
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=dev)
    pos, act, cells = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
    logp, val, rew, alive = z(), z(), z(), z(dt=torch.bool)
    bel = z(t.n) if record else None
    n = torch.arange(N, device=dev)
    for s in range(H):
        live = ~env.done
        if behaviour is None:
            lg, v = net(env.tok[:, : s + t.m + 1], None)
            lg, v = lg[n, env.idx], v[n, env.idx]
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
            logp[:, s], val[:, s] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1), v
        else:
            a = behaviour(env)
        pos[:, s], act[:, s], alive[:, s], cells[:, s] = env.idx, a, live, env.cell
        if record:
            bel[:, s] = env.belief
        rew[:, s] = env.step(a)
    return BM.Roll(env.tok, pos, act, logp, val, rew, alive, env.pair, cells, bel)


# ------------------------------------------------------------------ reference policies

def q_full(env):
    """Fully observed action values [N, n, 4] for each episode's goals left and moves left."""
    return env.t.Qc[env.mask, env.k]


def oracle(env):
    return q_full(env)[torch.arange(env.N, device=env.t.dev), env.cell].argmax(-1)


def mls(env):
    return q_full(env)[torch.arange(env.N, device=env.t.dev), env.belief.argmax(-1)].argmax(-1)


def qmdp_values(env):
    return torch.einsum("ns,nsa->na", env.belief, q_full(env))


def qmdp(env):
    return qmdp_values(env).argmax(-1)


def qmdp2_values(env):
    """One-step lookahead on qmdp: the pickups and the next symbol are the outcomes."""
    t = env.t
    Q1 = t.Qc[env.mask, (env.k - 1).clamp(min=0)]                                              # [N, n, 4] nothing collected
    out = []
    for a in range(4):
        pick, p = env.moved(torch.full((env.N,), a, device=t.dev))
        rest = env.mask[:, None] & ~t.bits[None]                                               # [N, K] goals left after collecting each
        v_pick = 1 + t.gamma * t.Vc[rest, (env.k - 1).clamp(min=0)[:, None], t.goal_cell[None]]
        w = p[:, None, :] * t.E.T[None]                                                        # [N, symbols, n]
        out.append((pick * v_pick).sum(1) + t.gamma * torch.einsum("nos,nsb->nob", w, Q1).max(-1).values.sum(-1))
    return torch.stack(out, 1)


def qmdp2(env):
    return qmdp2_values(env).argmax(-1)


def expected_entropy(env):
    """For each move [N, 4]: the expected posterior entropy after the move and its outcome (a pickup leaves none)."""
    t = env.t
    out = []
    for a in range(4):
        _, p = env.moved(torch.full((env.N,), a, device=t.dev))
        w = p[:, None, :] * t.E.T[None]
        z = w.sum(-1)
        out.append((z * BM.entropy(w / z[..., None].clamp(min=1e-30))).sum(-1))
    return torch.stack(out, 1)


def summarize(b: BM.Roll, t: Sim, M=1):
    """Per model of a model-major batch: discounted return, goals collected, both collected, length."""
    N, H = b.act.shape
    disc = t.gamma ** torch.arange(H, device=t.dev)
    out = []
    for m in range(M):
        s = slice(m * N // M, (m + 1) * N // M)
        c = b.rew[s].sum(1)
        out.append(dict(ret=(b.rew[s] * disc).sum(1).mean().item(), collected=c.mean().item(), success=(c >= t.m).float().mean().item(),
                        first=(c >= 1).float().mean().item(), length=b.alive[s].sum(1).float().mean().item()))
    return out
