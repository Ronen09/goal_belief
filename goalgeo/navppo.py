"""Round 17: the vectorised environment on the GPU, exact regret bookkeeping, and PPO.

Every environment runs one episode of at most H actions, so a rollout is a batch of complete
episodes and the policy is evaluated on whole token sequences. Regret is exact given the sampled
trajectory: the episode-level regret V*(s_0) - E[return] equals the expected sum along the
trajectory of the local decision regrets V*(s_t) - Q*(s_t, a_t), so the logged regret carries the
sampling noise of the path and the reports but not that of the reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from goalgeo import navcommit as NC
from goalgeo import navmodel as NM


class Tables:
    """Exact Q* for every layout and every reliability pair of a grid, on the device.
    Q[config, k, cell, s_h, s_v, action], config = (layout * nq + iq_h) * nq + iq_v."""

    def __init__(self, n, H, c, qs, device):
        self.n, self.H, self.c, self.qs, self.dev = n, H, c, tuple(qs), device
        h, v = NC.all_layouts(n)
        nq = len(qs)
        self.nlay, self.nq = len(h), nq
        hh = np.repeat(h, nq * nq); vv = np.repeat(v, nq * nq)
        qh = np.tile(np.repeat(np.array(qs), nq), len(h)); qv = np.tile(np.array(qs), len(h) * nq)
        Q = []
        for s in range(0, len(hh), 2048):
            Q.append(torch.tensor(NC.solve(n, H, c, hh[s:s + 2048], vv[s:s + 2048], qh[s:s + 2048], qv[s:s + 2048]).Q,
                                  dtype=torch.float32))
        self.Q = torch.cat(Q).to(device)
        self.V = self.Q.max(-1).values
        self.V[:, 0] = 0
        self.hcell, self.vcell = torch.tensor(hh, device=device), torch.tensor(vv, device=device)
        self.qh = torch.tensor(qh, dtype=torch.float32, device=device)
        self.qv = torch.tensor(qv, dtype=torch.float32, device=device)
        self.nxt = torch.tensor(NC.move_table(n), device=device)
        self.corner = torch.tensor(NC.corners(n), device=device)
        self.is_corner = torch.zeros(n * n, dtype=torch.bool, device=device); self.is_corner[self.corner] = True

    def config(self, layout, iqh, iqv):
        return (layout * self.nq + iqh) * self.nq + iqv


@dataclass
class Batch:
    tok: torch.Tensor            # [N, L, NF]
    q: torch.Tensor              # [N, L]
    legal: torch.Tensor          # [N, H, 6]
    act: torch.Tensor            # [N, H]
    logp: torch.Tensor           # [N, H]
    val: torch.Tensor            # [N, H]
    rew: torch.Tensor            # [N, H]
    alive: torch.Tensor          # [N, H] the decision was made
    regret: torch.Tensor         # [N, H] local decision regret of the action taken
    state: torch.Tensor          # [N, H, 4] k, cell, s_h, s_v at each decision
    cfg: torch.Tensor            # [N]
    goal: torch.Tensor           # [N]
    optimal: torch.Tensor        # [N, H, 6] every tied optimal action


class Env:
    def __init__(self, tab: Tables, cfg, start, goal=None, gen=None):
        self.t, self.cfg, self.gen = tab, cfg, gen
        N, H, dev = len(cfg), tab.H, tab.dev
        self.N = N
        self.goal = torch.randint(4, (N,), device=dev, generator=gen) if goal is None else goal
        self.cell, self.sh, self.sv = start.clone(), torch.zeros_like(start), torch.zeros_like(start)
        self.k = torch.full((N,), H, device=dev)
        self.done = torch.zeros(N, dtype=torch.bool, device=dev)
        L = NM.seq_len(H)
        self.tok = torch.zeros(N, L, NM.NF, dtype=torch.long, device=dev)
        self.q = torch.zeros(N, L, device=dev)
        self.tok[:, 0, NM.F_TYPE], self.tok[:, 1, NM.F_TYPE] = NM.CFG_H, NM.CFG_V
        self.tok[:, 0, NM.F_CELL], self.tok[:, 1, NM.F_CELL] = tab.hcell[cfg] + 1, tab.vcell[cfg] + 1
        self.q[:, 0], self.q[:, 1] = tab.qh[cfg], tab.qv[cfg]
        self.step_i = 0
        self.write_dec()

    def write_dec(self):
        p = NM.dec_pos(self.step_i)
        live = ~self.done
        rec = torch.stack([torch.full_like(self.cell, NM.DEC), self.cell + 1, self.k + 1, (self.sh > 0) + 1,
                           (self.sv > 0) + 1, torch.zeros_like(self.cell), torch.zeros_like(self.cell)], -1)
        self.tok[:, p] = rec * live[:, None]

    def legal(self):
        t = self.t
        m = torch.zeros(self.N, 6, dtype=torch.bool, device=t.dev)
        m[:, :4] = t.nxt[self.cell] >= 0
        m[:, NC.QUERY] = ((self.cell == t.hcell[self.cfg]) & (self.sh == 0)) | ((self.cell == t.vcell[self.cfg]) & (self.sv == 0))
        m[:, NC.COMMIT] = t.is_corner[self.cell]
        return m

    def qstar(self):
        return self.t.Q[self.cfg, self.k, self.cell, self.sh, self.sv]

    def state(self):
        return torch.stack([self.k, self.cell, self.sh, self.sv], -1)

    def step(self, a, err=None, y=None):
        """a [N]. err [N] bool, optional: whether a queried station errs (else sampled). y [N], optional:
        the report itself, whatever the goal (for replaying a history with chosen reports). Returns the reward."""
        t, live = self.t, ~self.done
        move, query, commit = live & (a < 4), live & (a == NC.QUERY), live & (a == NC.COMMIT)
        rew = torch.where(live & ~commit, -t.c, 0.0) + (commit & (t.corner[self.goal] == self.cell)).float()
        at_h = self.cell == t.hcell[self.cfg]
        if err is None:
            qq = torch.where(at_h, t.qh[self.cfg], t.qv[self.cfg])
            err = torch.rand(self.N, device=t.dev, generator=self.gen) >= qq
        truth = torch.where(at_h, self.goal % 2, self.goal // 2)
        y = truth ^ err.long() if y is None else y
        rep = torch.where(query, torch.where(at_h, NM.REP_L + y, NM.REP_T + y), 0)
        self.sh = torch.where(query & at_h, 1 + y, self.sh)
        self.sv = torch.where(query & ~at_h, 1 + y, self.sv)
        self.cell = torch.where(move, t.nxt[self.cell, a.clamp(max=3)], self.cell)
        p = NM.dec_pos(self.step_i) + 1
        self.tok[:, p, NM.F_TYPE] = NM.EVT * live
        self.tok[:, p, NM.F_ACT] = (a + 1) * live
        self.tok[:, p, NM.F_REP] = rep
        self.k = self.k - live.long()
        self.done = self.done | commit | (self.k == 0)
        self.step_i += 1
        if self.step_i < t.H:
            self.write_dec()
        return rew


def sample_configs(tab: Tables, N, q_pairs, gen):
    """Uniform layout and start; (q_h, q_v) index pairs uniform over the allowed list [M, 2]."""
    dev = tab.dev
    lay = torch.randint(tab.nlay, (N,), device=dev, generator=gen)
    qi = q_pairs[torch.randint(len(q_pairs), (N,), device=dev, generator=gen)]
    start = torch.randint(tab.n * tab.n, (N,), device=dev, generator=gen)
    return tab.config(lay, qi[:, 0], qi[:, 1]), start


@torch.no_grad()
def rollout(net, tab, cfg, start, gen=None, greedy=False, behaviour=None):
    """Complete episodes under the network's policy (or `behaviour(env, legal) -> actions`)."""
    env = Env(tab, cfg, start, gen=gen)
    H, N, dev = tab.H, len(cfg), tab.dev
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=dev)
    legal, act, logp, val, rew, alive, reg = z(6, dt=torch.bool), z(dt=torch.long), z(), z(), z(), z(dt=torch.bool), z()
    state, opt = z(4, dt=torch.long), z(6, dt=torch.bool)
    for t in range(H):
        p = NM.dec_pos(t)
        m = env.legal(); live = ~env.done
        m[~live] = True
        if behaviour is None:
            lg, v = net(env.tok[:, : p + 1], env.q[:, : p + 1])
            lg = lg[:, -1].masked_fill(~m, -1e9)
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
            logp[:, t] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1); val[:, t] = v[:, -1]
        else:
            a = behaviour(env, m)
        qs = env.qstar()
        best = qs.max(-1).values
        reg[:, t] = torch.where(live, best - qs.gather(1, a[:, None]).squeeze(1), 0.0)
        opt[:, t] = (qs >= best[:, None] - 1e-6) & live[:, None]
        state[:, t], legal[:, t], act[:, t], alive[:, t] = env.state(), m, a, live
        rew[:, t] = env.step(a)
    return Batch(env.tok, env.q, legal, act, logp, val, rew, alive, reg, state, cfg, env.goal, opt)


def error_split(b: Batch):
    """Regret per episode split by the kind of decision that lost it.
    query: an unwarranted query, or a move / commit where only a query was optimal.
    commit: a commit that was not optimal, or a move / query where only a commit was optimal.
    route: a move that was not optimal where some other move was, or where several kinds were."""
    a, opt, r = b.act, b.optimal, b.regret
    bad = r > 1e-6
    only_q = opt[..., NC.QUERY] & ~opt[..., :4].any(-1) & ~opt[..., NC.COMMIT]
    only_c = opt[..., NC.COMMIT] & ~opt[..., :4].any(-1) & ~opt[..., NC.QUERY]
    qerr = bad & ((a == NC.QUERY) | ((a != NC.QUERY) & only_q & (a != NC.COMMIT)))
    cerr = bad & ~qerr & ((a == NC.COMMIT) | only_c)
    rerr = bad & ~qerr & ~cerr
    n = b.act.shape[0]
    return dict(query=(r * qerr).sum().item() / n, commit=(r * cerr).sum().item() / n, route=(r * rerr).sum().item() / n)


def summarize(b: Batch):
    n = b.act.shape[0]
    ret = b.rew.sum(1)
    committed = ((b.act == NC.COMMIT) & b.alive).any(1)
    out = dict(ret=ret.mean().item(), regret=b.regret.sum(1).mean().item(),
               queries=((b.act == NC.QUERY) & b.alive).sum(1).float().mean().item(),
               committed=committed.float().mean().item(), length=b.alive.sum(1).float().mean().item(),
               opt_rate=((b.regret <= 1e-6) & b.alive).sum().item() / b.alive.sum().item())
    out.update({"err_" + k: v for k, v in error_split(b).items()})
    return out


@dataclass
class PPOConfig:
    n_env: int = 4096
    epochs: int = 3
    minibatches: int = 4
    lr: float = 3e-4
    clip: float = 0.2
    vf: float = 0.5
    ent: float = 0.01
    ent_final: float = 0.001
    lam: float = 0.95
    max_grad: float = 0.5
    warmup: int = 50


def gae(rew, val, alive, lam):
    """gamma = 1. The value after the last decision of an episode is 0."""
    N, H = rew.shape
    adv = torch.zeros_like(rew); last = torch.zeros(N, device=rew.device)
    nextv = torch.zeros(N, device=rew.device)
    for t in reversed(range(H)):
        delta = rew[:, t] + nextv - val[:, t]
        last = torch.where(alive[:, t], delta + lam * last, torch.zeros_like(last))
        adv[:, t] = last
        nextv = torch.where(alive[:, t], val[:, t], torch.zeros_like(nextv))
    return adv, adv + val


def ppo_update(net, opt, b: Batch, cfg: PPOConfig, ent_coef, params):
    adv, target = gae(b.rew, b.val, b.alive, cfg.lam)
    m = b.alive
    adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-8)
    N, H = b.act.shape
    pos = NM.dec_pos(torch.arange(H, device=b.act.device))
    stats = []
    for _ in range(cfg.epochs):
        for idx in torch.randperm(N, device=b.act.device).chunk(cfg.minibatches):
            lg, v = net(b.tok[idx], b.q[idx])
            lg = lg[:, pos].masked_fill(~b.legal[idx], -1e9); v = v[:, pos]
            lp = torch.log_softmax(lg, -1)
            mm = m[idx]
            ratio = (lp.gather(2, b.act[idx][..., None]).squeeze(2) - b.logp[idx]).exp()
            a = adv[idx]
            pl = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a)[mm].mean()
            vl = F.mse_loss(v[mm], target[idx][mm])
            ent = -(lp.exp() * lp.clamp(min=-30)).sum(-1)[mm].mean()
            loss = pl + cfg.vf * vl - ent_coef * ent
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad)
            opt.step()
            stats.append((pl.item(), vl.item(), ent.item()))
    return dict(zip(("pi_loss", "v_loss", "entropy"), np.mean(stats, 0).tolist()))


# ------------------------------------------------------------------ several models trained side by side

class Stacked:
    """M independent networks evaluated in one call (torch.func.vmap over stacked parameters). A batch is
    model-major: rows [m * N, (m + 1) * N) belong to model m. Nothing is shared between the models: the loss
    is a sum of per-model losses and Adam is per-coordinate, so each model gets exactly its own update."""

    def __init__(self, nets):
        import copy
        self.M = len(nets)
        self.base = copy.deepcopy(nets[0])
        params, _ = torch.func.stack_module_state(nets)
        self.params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}

    def __call__(self, tok, q):
        M = self.M
        f = lambda p, t, qq: torch.func.functional_call(self.base, p, (t, qq))
        lg, v = torch.func.vmap(f)(self.params, tok.view(M, -1, *tok.shape[1:]), q.view(M, -1, q.shape[-1]))
        return lg.flatten(0, 1), v.flatten(0, 1)

    def state_dict(self, m):
        return {k: v[m].detach().clone() for k, v in self.params.items()}

    def trainable(self, heads_only=False):
        return [v for k, v in self.params.items() if not heads_only or k.startswith(("pi.", "v."))]


def clip_per_model(params, max_norm):
    sq = sum((p.grad.flatten(1) ** 2).sum(1) for p in params)
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale.view(-1, *([1] * (p.grad.dim() - 1))))


def ppo_update_stacked(stk: Stacked, opt, b: Batch, cfg: PPOConfig, ent_coef, params):
    M = stk.M
    N, H = b.act.shape[0] // M, b.act.shape[1]
    dev = b.act.device
    adv, target = gae(b.rew, b.val, b.alive, cfg.lam)
    adv, m = adv.view(M, N, H), b.alive.view(M, N, H)
    cnt = m.sum((1, 2)).clamp(min=1)
    mean = (adv * m).sum((1, 2)) / cnt
    std = (((adv - mean[:, None, None]) ** 2 * m).sum((1, 2)) / cnt).sqrt()
    adv = ((adv - mean[:, None, None]) / (std[:, None, None] + 1e-8)).view(M * N, H)
    pos = NM.dec_pos(torch.arange(H, device=dev))
    off = (torch.arange(M, device=dev) * N)[:, None]
    for _ in range(cfg.epochs):
        perm = torch.argsort(torch.rand(M, N, device=dev), 1)
        for ch in perm.chunk(cfg.minibatches, 1):
            idx = (ch + off).reshape(-1)
            lg, v = stk(b.tok[idx], b.q[idx])
            lg = lg[:, pos].masked_fill(~b.legal[idx], -1e9); v = v[:, pos]
            lp = torch.log_softmax(lg, -1)
            mm = b.alive[idx].view(M, -1, H).float()
            per = lambda x: (x.view(M, -1, H) * mm).sum((1, 2)) / mm.sum((1, 2)).clamp(min=1)     # per-model mean over decisions
            ratio = (lp.gather(2, b.act[idx][..., None]).squeeze(2) - b.logp[idx]).exp()
            a = adv[idx]
            pl = -per(torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a))
            vl = per((v - target[idx]) ** 2)
            ent = per(-(lp.exp() * lp.clamp(min=-30)).sum(-1))
            loss = (pl + cfg.vf * vl - ent_coef * ent).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            clip_per_model(params, cfg.max_grad)
            opt.step()


def summarize_each(b: Batch, M):
    """summarize() for each of the M models of a model-major batch."""
    N = b.act.shape[0] // M
    out = []
    for m in range(M):
        s = slice(m * N, (m + 1) * N)
        out.append(summarize(Batch(**{k: getattr(b, k)[s] for k in Batch.__dataclass_fields__})))
    return out
