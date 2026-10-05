"""The maze-belief experiment: the maze task vectorised on the GPU, with exact bookkeeping from the belief graph, and PPO.

Every environment carries its true cell and the node of the belief graph its history has reached, so the exact
posterior, action values and optimal actions of every decision are lookups. Regret is exact given the trajectory:
V*(s_0) - E[discounted return] equals the expected sum of gamma^t (V*(s_t) - Q*(s_t, a_t)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from goalgeo import mazegraph as MG, mazemodel as MM
from goalgeo import navppo
from goalgeo.navppo import PPOConfig, clip_per_model  # noqa: F401  (re-exported)


class Stacked(navppo.Stacked):
    """navppo.Stacked for a network whose only input is the token array."""

    def __call__(self, tok, q=None):
        f = lambda p, t: torch.func.functional_call(self.base, p, (t,))
        lg, v = torch.func.vmap(f)(self.params, tok.view(self.M, -1, *tok.shape[1:]))
        return lg.flatten(0, 1), v.flatten(0, 1)


class Tables:
    def __init__(self, g: MG.Graph, device, max_prefix=MG.MAX_PREFIX):
        m = g.maze
        t = lambda x, dt=None: torch.tensor(np.asarray(x), device=device, dtype=dt)
        self.dev, self.H, self.gamma, self.n, self.n_sym, self.max_prefix = device, m.H, m.gamma, m.n, m.n_sym, max_prefix
        self.nxt_cell, self.E = t(m.nxt), t(m.E, torch.float32)
        self.goal_cell, self.starts = t(m.goals), t(m.starts if m.starts else [i for i in range(m.n) if i not in m.goals])
        self.belief, self.Q = t(g.belief, torch.float32), t(g.Q, torch.float32)
        self.node_nxt, self.start, self.reveal = t(g.nxt, torch.long), t(g.start), t(g.reveal)
        self.V = self.Q.max(1).values
        self.L = MM.seq_len(m.H, max_prefix)
        self.cells = list(m.cells)


@dataclass
class Batch:
    tok: torch.Tensor            # [N, L, NF]
    pos: torch.Tensor            # [N, H] token index of each decision
    node: torch.Tensor           # [N, H] belief-graph node of each decision
    act: torch.Tensor            # [N, H]
    logp: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    alive: torch.Tensor          # [N, H]
    regret: torch.Tensor         # [N, H] local decision regret (undiscounted)
    goal: torch.Tensor           # [N] goal index
    prefix: torch.Tensor         # [N] prefix length
    pre_node: torch.Tensor       # [N, max_prefix + 1] prefix node after each prefix token (-1 beyond the prefix)
    cell: torch.Tensor           # [N, H] true cell at each decision


class Env:
    """A batch of episodes. The prefix is played in the constructor."""

    def __init__(self, t: Tables, N, gen=None, prefix=None, goal=None):
        self.t, self.N, self.gen = t, N, gen
        dev = t.dev
        r = lambda hi: torch.randint(hi, (N,), device=dev, generator=gen)
        self.cell = t.starts[r(len(t.starts))]
        self.prefix = r(t.max_prefix + 1) if prefix is None else prefix
        self.goal = r(len(t.goal_cell)) if goal is None else goal
        self.tok = torch.zeros(N, t.L, MM.NF, dtype=torch.long, device=dev)
        o = self.sample_symbol()
        self.tok[:, 0, MM.F_TYPE], self.tok[:, 0, MM.F_SYM] = MM.OBS, o + 1
        node = t.start[o]
        self.pre_node = torch.full((N, t.max_prefix + 1), -1, device=dev)
        self.pre_node[:, 0] = node
        n = torch.arange(N, device=dev)
        for i in range(t.max_prefix):
            on = i < self.prefix
            a = r(4)
            self.cell = torch.where(on, t.nxt_cell[self.cell, a], self.cell)
            o = self.sample_symbol()
            node = torch.where(on, t.node_nxt[node, a, o], node)
            self.tok[on, 1 + i] = torch.stack([torch.full_like(a, MM.EVT), o + 1, a + 1, torch.zeros_like(a), torch.zeros_like(a)], -1)[on]
            self.pre_node[on, i + 1] = node[on]
        self.idx = 1 + self.prefix                                     # token index of the current decision
        self.tok[n, self.idx, MM.F_TYPE], self.tok[n, self.idx, MM.F_GOAL] = MM.GOAL, self.goal + 1
        self.node = t.reveal[node, self.goal]
        self.k = torch.full((N,), t.H, device=dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)

    def sample_symbol(self):
        return torch.multinomial(self.t.E[self.cell], 1, generator=self.gen).squeeze(1)

    def step(self, a):
        t, live = self.t, ~self.done
        self.cell = torch.where(live, t.nxt_cell[self.cell, a], self.cell)
        hit = live & (self.cell == t.goal_cell[self.goal])
        rew = hit.float()
        o = self.sample_symbol()
        self.k = self.k - live.long()
        cont = live & ~hit & (self.k > 0)
        self.node = torch.where(cont, t.node_nxt[self.node, a, o], self.node)
        self.idx = self.idx + 1
        n = torch.nonzero(cont).squeeze(1)
        self.tok[n, self.idx[n]] = torch.stack([torch.full_like(a, MM.EVT), o + 1, a + 1, torch.zeros_like(a), self.k + 1], -1)[n]
        self.done = self.done | ~cont
        return rew


@torch.no_grad()
def rollout(net, t: Tables, N, gen=None, greedy=False, behaviour=None, prefix=None, goal=None):
    env = Env(t, N, gen, prefix, goal)
    H, dev = t.H, t.dev
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=dev)
    pos, node, act, cell = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
    logp, val, rew, reg, alive = z(), z(), z(), z(), z(dt=torch.bool)
    n = torch.arange(N, device=dev)
    for s in range(H):
        live = ~env.done
        if behaviour is None:
            top = int(env.idx.max()) + 1
            lg, v = net(env.tok[:, :top], None)
            lg, v = lg[n, env.idx], v[n, env.idx]
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
            logp[:, s], val[:, s] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1), v
        else:
            a = behaviour(env)
        q = t.Q[env.node]
        reg[:, s] = torch.where(live, q.max(1).values - q.gather(1, a[:, None]).squeeze(1), 0.0)
        pos[:, s], node[:, s], act[:, s], alive[:, s], cell[:, s] = env.idx, env.node, a, live, env.cell
        rew[:, s] = env.step(a)
    return Batch(env.tok, pos, node, act, logp, val, rew, alive, reg, env.goal, env.prefix, env.pre_node, cell)


def solver_behaviour(t: Tables, eps, gen):
    def f(env):
        q = t.Q[env.node]
        best = (q >= q.max(1, keepdim=True).values - 1e-6).float()
        a_opt = torch.multinomial(best, 1, generator=gen).squeeze(1)
        a_rnd = torch.randint(4, (env.N,), device=t.dev, generator=gen)
        return torch.where(torch.rand(env.N, device=t.dev, generator=gen) < eps, a_rnd, a_opt)
    return f


def summarize(b: Batch, t: Tables, M=1):
    """Per model of a model-major batch: discounted return, exact regret, success, length, by goal."""
    N, H = b.act.shape
    disc = t.gamma ** torch.arange(H, device=t.dev)
    out = []
    for m in range(M):
        s = slice(m * N // M, (m + 1) * N // M)
        ret, reg = (b.rew[s] * disc).sum(1), (b.regret[s] * disc).sum(1)
        row = dict(ret=ret.mean().item(), regret=reg.mean().item(), success=b.rew[s].sum(1).mean().item(),
                   length=b.alive[s].sum(1).float().mean().item(), v_star=t.V[b.node[s, 0]].mean().item(),
                   opt_rate=((b.regret[s] <= 1e-6) & b.alive[s]).sum().item() / b.alive[s].sum().item())
        for g in range(len(t.goal_cell)):
            row[f"regret_G{g + 1}"] = reg[b.goal[s] == g].mean().item()
        out.append(row)
    return out


def gae(rew, val, alive, gamma, lam):
    N, H = rew.shape
    adv = torch.zeros_like(rew); last = torch.zeros(N, device=rew.device); nextv = torch.zeros(N, device=rew.device)
    for s in reversed(range(H)):
        delta = rew[:, s] + gamma * nextv - val[:, s]
        last = torch.where(alive[:, s], delta + gamma * lam * last, torch.zeros_like(last))
        adv[:, s] = last
        nextv = torch.where(alive[:, s], val[:, s], torch.zeros_like(nextv))
    return adv, adv + val


def ppo_update(stk, opt, b: Batch, cfg: PPOConfig, ent_coef, params, gamma):
    M = stk.M
    N, H = b.act.shape[0] // M, b.act.shape[1]
    dev = b.act.device
    adv, target = gae(b.rew, b.val, b.alive, gamma, cfg.lam)
    adv, m = adv.view(M, N, H), b.alive.view(M, N, H)
    cnt = m.sum((1, 2)).clamp(min=1)
    mean = (adv * m).sum((1, 2)) / cnt
    std = (((adv - mean[:, None, None]) ** 2 * m).sum((1, 2)) / cnt).sqrt()
    adv = ((adv - mean[:, None, None]) / (std[:, None, None] + 1e-8)).view(M * N, H)
    off = (torch.arange(M, device=dev) * N)[:, None]
    for _ in range(cfg.epochs):
        perm = torch.argsort(torch.rand(M, N, device=dev), 1)
        for ch in perm.chunk(cfg.minibatches, 1):
            idx = (ch + off).reshape(-1)
            lg, v = stk(b.tok[idx])
            p = b.pos[idx].clamp(max=b.tok.shape[1] - 1)
            lg, v = lg.gather(1, p[..., None].expand(-1, -1, 4)), v.gather(1, p)
            lp = torch.log_softmax(lg, -1)
            mm = b.alive[idx].view(M, -1, H).float()
            per = lambda x: (x.view(M, -1, H) * mm).sum((1, 2)) / mm.sum((1, 2)).clamp(min=1)
            ratio = (lp.gather(2, b.act[idx][..., None]).squeeze(2) - b.logp[idx]).exp()
            a = adv[idx]
            pl = -per(torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a))
            vl = per((v - target[idx]) ** 2)
            ent = per(-(lp.exp() * lp).sum(-1))
            loss = (pl + cfg.vf * vl - ent_coef * ent).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            clip_per_model(params, cfg.max_grad)
            opt.step()
