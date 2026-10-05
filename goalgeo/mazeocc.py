"""The maze-occupancy experiment: goal-conditioned occupancy in the maze-belief maze, under the solver's policy and under a model's own.

    d_g^pi(s | b) = E[ sum_{k=1..H} gamma^(k-1) 1{S_{t+k} = s} ]

the expected discounted future visits to cell s, from the current posterior b, pursuing goal g with policy pi. The
sum starts at k = 1, so the current cell is not counted. Entering the goal ends the episode; that visit is counted.

Solver's occupancy: exact, by backward induction over the belief graph, with the solver choosing uniformly among
its tied optimal actions.

A model's occupancy: the model is a function of the token history, not of the posterior, so its occupancy is
defined per history. It is estimated by rollouts of the model from the history, the true cell drawn from the exact
posterior. Each rollout contributes, at every step, the exact distribution of the next cell given the rollout's own
history (not the sampled cell), which removes the sampling noise of the location.
"""

from __future__ import annotations

import numpy as np
import torch

from goalgeo import mazegraph as MG, mazemodel as MM, mazeppo as P


def solver_occupancy(g: MG.Graph):
    """D [nodes, n] (float32); zero for prefix nodes. Also the per-action occupancies at the nodes with H moves left,
    as a function D_a(node_ids) -> [m, 4, n]."""
    m = g.maze
    n, N = m.n, len(g.k)
    D = np.zeros((N + 1, n), dtype=np.float32)                       # row N serves the missing successors
    opt = g.optimal()
    pi = opt / opt.sum(1, keepdims=True)

    def per_action(sel):
        b = g.belief[sel]
        out = np.zeros((len(sel), 4, n), dtype=np.float32)
        for a in range(4):
            pa = np.zeros((len(sel), n)); np.add.at(pa.T, m.nxt[:, a], b.T)
            fut = (g.p_obs[sel, a][:, :, None] * D[g.nxt[sel, a]]).sum(1)
            out[:, a] = pa + m.gamma * fut
        return out
    for k in range(1, m.H + 1):
        sel = np.nonzero(g.k == k)[0]
        for s in range(0, len(sel), 200_000):
            c = sel[s:s + 200_000]
            D[c] = (pi[c][:, :, None] * per_action(c)).sum(1)
    return D[:N], per_action


def env_from(t: P.Tables, tok, prefix, goal, reveal_node, gen):
    """Environments that continue the given histories from the reveal; the true cell is drawn from the posterior."""
    e = object.__new__(P.Env)
    e.t, e.N, e.gen = t, len(tok), gen
    e.tok, e.prefix, e.goal = tok.clone(), prefix, goal
    e.cell = torch.multinomial(t.belief[reveal_node].clamp(min=0), 1, generator=gen).squeeze(1)
    e.idx = 1 + prefix
    e.node = reveal_node
    e.k = torch.full((e.N,), t.H, device=t.dev)
    e.done = torch.zeros(e.N, dtype=torch.bool, device=t.dev)
    e.pre_node = None
    return e


def next_cell_dist(t: P.Tables, node, a):
    """The exact distribution of the cell after move a, given the history that reached `node`. [N, n]"""
    b = t.belief[node]
    return torch.zeros_like(b).scatter_add_(1, t.nxt_cell.T[a], b)


@torch.no_grad()
def model_occupancy(net, t: P.Tables, tok, prefix, goal, reveal_node, K=32, seed=0, chunk=120_000, hook=None):
    """Occupancy of the model's own (sampled) policy for each history, its discounted return, and its action
    distribution at the reveal. hook: an optional patch for the network's forward passes (see intervene)."""
    gen = torch.Generator(device=t.dev); gen.manual_seed(seed)
    N = len(tok)
    D = torch.zeros(N, t.n, device=t.dev); R = torch.zeros(N, device=t.dev); pi0 = torch.zeros(N, 4, device=t.dev)
    rows = torch.arange(N, device=t.dev).repeat_interleave(K)
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        env = env_from(t, tok[r], prefix[r], goal[r], reveal_node[r], gen)
        n = torch.arange(len(r), device=t.dev)
        d = torch.zeros(len(r), t.n, device=t.dev); ret = torch.zeros(len(r), device=t.dev)
        for k in range(t.H):
            live = ~env.done
            top = int(env.idx.max()) + 1
            lg = (net(env.tok[:, :top]) if hook is None else net(env.tok[:, :top], patch=hook(r, env)))[0][n, env.idx.clamp(max=top - 1)]
            p = lg.softmax(-1)
            if k == 0:
                pi0.index_add_(0, r, p)
            a = torch.multinomial(p, 1, generator=gen).squeeze(1)
            d += (t.gamma ** k) * live[:, None] * next_cell_dist(t, env.node, a)
            ret += (t.gamma ** k) * env.step(a)
        D.index_add_(0, r, d); R.index_add_(0, r, ret)
    return D / K, R / K, pi0 / K
