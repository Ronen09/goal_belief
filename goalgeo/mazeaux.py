"""Round 23: next-symbol prediction as an auxiliary objective for the maze transformer.

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
        super().__init__(n_sym, n_goals, H, max_prefix, **kw)             # the shared parameters are initialised first, as in round 18
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


def ppo_update(stk: StackedAux, opt, b: P.Batch, cfg: P.PPOConfig, ent_coef, params, gamma, aux_coef):
    """mazeppo.ppo_update with aux_coef times the prediction loss added, computed in the same forward passes."""
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
            nll, valid = obs_nll(ob, b.tok[idx])
            al = nll.view(M, -1).sum(1) / valid.view(M, -1).sum(1).clamp(min=1)
            loss = (pl + cfg.vf * vl - ent_coef * ent).sum()
            if aux_coef > 0:
                loss = loss + aux_coef * al.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            P.clip_per_model(params, cfg.max_grad)
            opt.step()
            last = al.detach()
    return last
