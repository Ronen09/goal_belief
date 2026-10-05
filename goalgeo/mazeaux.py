"""The observation-prediction experiment: next-symbol prediction as an auxiliary objective for the maze transformer.

The head reads the final residual stream and gives, at every token, a distribution over the next symbol for each of
the four moves. The target at a token is the symbol held by the next token, for the move that token holds, wherever
the next token is an event (in the prefix the move was imposed, after the reveal it was the model's own).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from goalgeo import mazemodel as MM, mazeppo as P


class NetAux(MM.Net):
    """mazemodel.Net with the prediction head. Without aux=True it behaves exactly as mazemodel.Net."""

    def __init__(self, n_sym=3, n_goals=3, H=12, max_prefix=4, **kw):
        super().__init__(n_sym, n_goals, H, max_prefix, **kw)             # the shared parameters are initialised first, as in the maze-belief experiment
        self.n_sym = n_sym
        self.obs = nn.Linear(self.d, 4 * n_sym)
        nn.init.normal_(self.obs.weight, std=0.01); nn.init.zeros_(self.obs.bias)

    def forward(self, tok, q=None, record=False, patch=None, aux=False):
        if not aux:
            return super().forward(tok, q, record, patch)
        x = self.embed(tok)
        for b in self.blocks:
            mid = x + b.attn(x)[0]
            x = mid + b.mlp(b.ln2(mid))
        h = self.ln(x)
        return self.pi(h), self.v(h).squeeze(-1), self.obs(h).view(*h.shape[:-1], 4, self.n_sym)


class StackedAux(P.Stacked):
    def aux(self, tok):
        f = lambda p, t: torch.func.functional_call(self.base, p, (t,), dict(aux=True))
        lg, v, ob = torch.func.vmap(f)(self.params, tok.view(self.M, -1, *tok.shape[1:]))
        return lg.flatten(0, 1), v.flatten(0, 1), ob.flatten(0, 1)


def obs_targets(tok):
    """valid [N, L], move [N, L], symbol [N, L]: the next token's move and symbol where the next token is an event."""
    nxt = torch.roll(tok, -1, 1)
    valid = (nxt[..., MM.F_TYPE] == MM.EVT) & (tok[..., MM.F_TYPE] != MM.PAD)
    valid[:, -1] = False
    return valid, (nxt[..., MM.F_ACT] - 1).clamp(min=0), (nxt[..., MM.F_SYM] - 1).clamp(min=0)


def obs_nll(ob, tok):
    """Cross-entropy per token [N, L] (zero where there is no target) and the mask."""
    valid, a, o = obs_targets(tok)
    lp = torch.log_softmax(ob, -1).gather(2, a[..., None, None].expand(-1, -1, 1, ob.shape[-1])).squeeze(2)
    return -lp.gather(2, o[..., None]).squeeze(2) * valid, valid


def exact_nll(b: P.Batch, p_obs):
    """The exact predictive cross-entropy at the same targets, from the belief graph: [N, L]. p_obs [nodes, 4, symbols]
    is the probability of (continuing and) each symbol; a target exists only if the episode continued."""
    N, L = b.tok.shape[:2]
    dev = b.tok.device
    node = torch.full((N, L), -1, dtype=torch.long, device=dev)
    mp = b.pre_node.shape[1]
    node[:, :mp] = b.pre_node
    rows = torch.arange(N, device=dev)[:, None].expand_as(b.pos)
    node[rows[b.alive], b.pos[b.alive]] = b.node[b.alive]                # the goal token overwrites nothing: it follows the prefix
    valid, a, o = obs_targets(b.tok)
    p = p_obs[node.clamp(min=0), a]                                       # [N, L, symbols]
    p = p / p.sum(-1, keepdim=True).clamp(min=1e-12)
    return -p.gather(2, o[..., None]).squeeze(2).clamp(min=1e-12).log() * valid, valid


class EnvCF(P.Env):
    """mazeppo.Env that also keeps the true cell at every prefix token, for counterfactual symbols."""

    def __init__(self, t: P.Tables, N, gen=None, prefix=None, goal=None):
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
        self.pre_cell = torch.zeros(N, t.max_prefix + 1, dtype=torch.long, device=dev)
        self.pre_cell[:, 0] = self.cell
        n = torch.arange(N, device=dev)
        for i in range(t.max_prefix):
            on = i < self.prefix
            a = r(4)
            self.cell = torch.where(on, t.nxt_cell[self.cell, a], self.cell)
            o = self.sample_symbol()
            node = torch.where(on, t.node_nxt[node, a, o], node)
            self.tok[on, 1 + i] = torch.stack([torch.full_like(a, MM.EVT), o + 1, a + 1, torch.zeros_like(a), torch.zeros_like(a)], -1)[on]
            self.pre_node[on, i + 1] = node[on]
            self.pre_cell[:, i + 1] = self.cell
        self.idx = 1 + self.prefix
        self.tok[n, self.idx, MM.F_TYPE], self.tok[n, self.idx, MM.F_GOAL] = MM.GOAL, self.goal + 1
        self.node = t.reveal[node, self.goal]
        self.k = torch.full((N,), t.H, device=dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)


def _cf(t, cell, gen):
    """A symbol for each candidate move from `cell`: [n, 4], and the cells reached."""
    c2 = t.nxt_cell[cell]                                                # [n, 4]
    return torch.multinomial(t.E[c2.reshape(-1)], 1, generator=gen).view(-1, 4), c2


@torch.no_grad()
def rollout_cf(net, t: P.Tables, N, gen=None, one=False):
    """mazeppo.rollout (sampled actions) that also returns counterfactual next symbols: sym [N, L, 4] and valid
    [N, L, 4]. Targets sit at the tokens that carry one under selected-action supervision: prefix tokens followed
    by an event, and decisions with at least two moves left; a move that enters the goal has none. one=True keeps
    one candidate move per token, drawn uniformly."""
    env = EnvCF(t, N, gen)
    H, dev = t.H, t.dev
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=dev)
    pos, node, act, cell = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
    logp, val, rew, reg, alive = z(), z(), z(), z(), z(dt=torch.bool)
    sym = torch.zeros(N, t.L, 4, dtype=torch.long, device=dev); valid = torch.zeros(N, t.L, 4, dtype=torch.bool, device=dev)
    n = torch.arange(N, device=dev)
    for i in range(t.max_prefix):
        on = i < env.prefix
        sym[:, i], _ = _cf(t, env.pre_cell[:, i], gen)
        valid[:, i] = on[:, None]
    for s in range(H):
        live = ~env.done
        top = int(env.idx.max()) + 1
        lg, v = net(env.tok[:, :top], None)
        lg, v = lg[n, env.idx], v[n, env.idx]
        a = torch.distributions.Categorical(logits=lg).sample()
        logp[:, s], val[:, s] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1), v
        o, c2 = _cf(t, env.cell, gen)
        ok = (live & (env.k > 1))[:, None] & (c2 != t.goal_cell[env.goal][:, None])
        sym[n[live], env.idx[live]] = o[live]; valid[n[live], env.idx[live]] = ok[live]
        q = t.Q[env.node]
        reg[:, s] = torch.where(live, q.max(1).values - q.gather(1, a[:, None]).squeeze(1), 0.0)
        pos[:, s], node[:, s], act[:, s], alive[:, s], cell[:, s] = env.idx, env.node, a, live, env.cell
        rew[:, s] = env.step(a)
    if one:
        pick = torch.randint(4, (N, t.L), device=dev, generator=gen)
        valid &= torch.nn.functional.one_hot(pick, 4).bool()
    return P.Batch(env.tok, pos, node, act, logp, val, rew, alive, reg, env.goal, env.prefix, env.pre_node, cell), sym, valid


def ppo_update(stk: StackedAux, opt, b: P.Batch, cfg: P.PPOConfig, ent_coef, params, gamma, aux_coef, cf=None):
    """mazeppo.ppo_update with aux_coef times the prediction loss added, computed in the same forward passes.
    cf = (sym, valid) from rollout_cf: counterfactual targets for the candidate moves; None: the symbol that
    followed the move taken."""
    M = stk.M
    N, H = b.act.shape[0] // M, b.act.shape[1]
    dev = b.act.device
    adv, target = P.gae(b.rew, b.val, b.alive, gamma, cfg.lam)
    adv, m = adv.view(M, N, H), b.alive.view(M, N, H)
    cnt = m.sum((1, 2)).clamp(min=1)
    mean = (adv * m).sum((1, 2)) / cnt
    std = (((adv - mean[:, None, None]) ** 2 * m).sum((1, 2)) / cnt).sqrt()
    adv = ((adv - mean[:, None, None]) / (std[:, None, None] + 1e-8)).view(M * N, H)
    off = (torch.arange(M, device=dev) * N)[:, None]
    last = None
    for _ in range(cfg.epochs):
        perm = torch.argsort(torch.rand(M, N, device=dev), 1)
        for ch in perm.chunk(cfg.minibatches, 1):
            idx = (ch + off).reshape(-1)
            lg, v, ob = stk.aux(b.tok[idx])
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
            if cf is None:
                nll, valid = obs_nll(ob, b.tok[idx])
            else:
                valid = cf[1][idx]
                nll = -torch.log_softmax(ob, -1).gather(3, cf[0][idx][..., None]).squeeze(3) * valid
            al = nll.view(M, -1).sum(1) / valid.view(M, -1).sum(1).clamp(min=1)
            loss = (pl + cfg.vf * vl - ent_coef * ent).sum()
            if aux_coef > 0:
                loss = loss + aux_coef * al.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            P.clip_per_model(params, cfg.max_grad)
            opt.step()
            last = al.detach()
    return last
