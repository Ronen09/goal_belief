"""The navigate-commit experiment: fixed evaluation histories, replayed unchanged at every checkpoint.

A bank is a set of complete legal histories produced by exploratory behaviour (never by the models
under study) with, for every decision, the exact public state, belief and action values. Two parts:

  broad   many layouts, a few histories each: for probes. Split by layout, so every held-out
          decision comes from a layout (hence layout x start x report combinations) the probe never saw.
  dense   few layouts and starts, many histories each: for matched comparisons and interventions,
          where several histories must share a public decision state.

Behaviours. 'solver': the optimal policy with epsilon-random legal actions. 'wander': a random walk
that queries a station it stands on with probability 1/2 and commits at a corner with probability
0.15, which gives long histories, detours and revisits. 'investigator': walks to one station, then the
other, in a fixed order (both orders are used), then follows the solver.
"""

from __future__ import annotations

import numpy as np
import torch

from goalgeo import navcommit as NC, navmodel as NM, navppo as P


def solver_behaviour(eps, gen):
    def f(env, m):
        qs = env.qstar()
        best = (qs >= qs.max(-1, keepdim=True).values - 1e-6).float()
        a_opt = torch.multinomial(best, 1, generator=gen).squeeze(1)
        a_rnd = torch.multinomial(m.float(), 1, generator=gen).squeeze(1)
        return torch.where(torch.rand(len(a_opt), device=a_opt.device, generator=gen) < eps, a_rnd, a_opt)
    return f


def wander_behaviour(gen, p_query=0.5, p_commit=0.15):
    def f(env, m):
        w = m.float()
        w[:, :4] = w[:, :4] * (1 - p_query * m[:, NC.QUERY, None].float()) * (1 - p_commit * m[:, NC.COMMIT, None].float())
        w[:, :4] = w[:, :4] / w[:, :4].sum(1, keepdim=True)
        w[:, NC.QUERY] = p_query * m[:, NC.QUERY]
        w[:, NC.COMMIT] = p_commit * m[:, NC.COMMIT] * (1 - p_query * m[:, NC.QUERY].float())
        return torch.multinomial(w, 1, generator=gen).squeeze(1)
    return f


def investigator_behaviour(gen, first, eps=0.25, after_eps=0.4):
    """Walks to the station `first` ('h' or 'v'), queries it, walks to the other one, queries it, then follows
    the solver with epsilon-random actions. A fraction eps of its moves are random, so routes have detours."""
    def f(env, m):
        t, n = env.t, env.t.n
        hc, vc = t.hcell[env.cfg], t.vcell[env.cfg]
        h_open, v_open = env.sh == 0, env.sv == 0
        to_h = h_open & ((first == "h") | ~v_open)
        target = torch.where(to_h, hc, vc)
        searching = h_open | v_open
        dr, dc = target // n - env.cell // n, target % n - env.cell % n
        w = torch.zeros(env.N, 6, device=t.dev)
        w[:, NC.UP], w[:, NC.DOWN], w[:, NC.LEFT], w[:, NC.RIGHT] = (dr < 0).float(), (dr > 0).float(), (dc < 0).float(), (dc > 0).float()
        w = w * (1 - eps) / w.sum(1, keepdim=True).clamp(min=1) + eps * m.float() * (torch.arange(6, device=t.dev) < 4) / m[:, :4].sum(1, keepdim=True)
        w[m[:, NC.QUERY]] = 0; w[m[:, NC.QUERY], NC.QUERY] = 1
        a = torch.multinomial(w * m + 1e-12 * m, 1, generator=gen).squeeze(1)
        return torch.where(searching, a, solver_behaviour(after_eps, gen)(env, m))
    return f


def _cat(batches):
    out = {}
    for k in P.Batch.__dataclass_fields__:
        out[k] = torch.cat([getattr(b, k) for b in batches])
    return P.Batch(**out)


def generate(tab, cfg, start, gen):
    """A sixth each: solver with eps = 0.15 and 0.5, the wanderer (twice), and the two investigators."""
    parts = torch.arange(len(cfg), device=tab.dev).chunk(6)
    beh = [solver_behaviour(0.15, gen), solver_behaviour(0.5, gen), wander_behaviour(gen), wander_behaviour(gen, 0.9, 0.05),
           investigator_behaviour(gen, "h"), investigator_behaviour(gen, "v")]
    return _cat([P.rollout(None, tab, cfg[i], start[i], gen=gen, behaviour=b) for i, b in zip(parts, beh)])


@torch.no_grad()
def replay(tab, cfg, start, act, alive, yh, yv):
    """The same action sequences with the reports set to yh, yv [N] (0 / 1) whenever a station is queried."""
    env = P.Env(tab, cfg, start)
    N, H, dev = len(cfg), tab.H, tab.dev
    state = torch.zeros(N, H, 4, dtype=torch.long, device=dev)
    opt = torch.zeros(N, H, 6, dtype=torch.bool, device=dev); legal = torch.zeros_like(opt)
    reg = torch.zeros(N, H, device=dev); rew = torch.zeros(N, H, device=dev)
    for t in range(H):
        live = ~env.done
        assert torch.equal(live, alive[:, t])
        m = env.legal(); m[~live] = True
        qs = env.qstar(); best = qs.max(-1).values
        a = act[:, t]
        assert m.gather(1, a[:, None]).all()
        reg[:, t] = torch.where(live, best - qs.gather(1, a[:, None]).squeeze(1), 0.0)
        opt[:, t] = (qs >= best[:, None] - 1e-6) & live[:, None]
        state[:, t], legal[:, t] = env.state(), m
        at_h = env.cell == tab.hcell[cfg]
        rew[:, t] = env.step(a, y=torch.where(at_h, yh, yv))
    z = torch.zeros(N, H, device=dev)
    return P.Batch(env.tok, env.q, legal, act, z, z, rew, alive, reg, state, cfg, env.goal, opt)


def decisions(tab, b: P.Batch):
    """One row per decision actually made. Returns a dict of aligned tensors."""
    hi, t = torch.nonzero(b.alive, as_tuple=True)
    st = b.state[hi, t]
    k, cell, sh, sv = st.unbind(-1)
    cfg = b.cfg[hi]
    qh, qv = tab.qh[cfg], tab.qv[cfg]
    f = lambda q, s: torch.where(s == 0, torch.full_like(q, 0.5), torch.where(s == 2, q, 1 - q))
    pr, pb = f(qh, sh), f(qv, sv)                                   # P(right), P(bottom)
    belief = torch.stack([(1 - pb) * (1 - pr), (1 - pb) * pr, pb * (1 - pr), pb * pr], -1)
    # clue order: 0 none, 1 h only, 2 v only, 3 h then v, 4 v then h
    rep = b.tok[:, :, NM.F_REP]
    L = rep.shape[1]
    idx = torch.arange(L, device=rep.device)
    first = lambda lo: torch.where((rep >= lo) & (rep < lo + 2), idx, L).min(1).values
    th, tv = first(NM.REP_L)[hi], first(NM.REP_T)[hi]
    p = NM.dec_pos(t)
    hseen, vseen = th < p, tv < p
    order = torch.where(hseen & vseen, torch.where(th < tv, 3, 4), hseen.long() + 2 * vseen.long())
    return dict(hist=hi, t=t, pos=p, k=k, cell=cell, sh=sh, sv=sv, cfg=cfg, layout=cfg // (tab.nq * tab.nq), qh=qh, qv=qv,
                belief=belief, p_right=pr, p_bottom=pb, order=order, q_star=tab.Q[cfg, k, cell, sh, sv],
                optimal=b.optimal[hi, t], legal=b.legal[hi, t], action=b.act[hi, t],
                public=((cfg * 13 + k) * 25 + cell) * 4 + (sh > 0) * 2 + (sv > 0))


def make_bank(tab, q_pairs, seed, n_broad=24000, dense_layouts=40, dense_starts=3, dense_per=400):
    gen = torch.Generator(device=tab.dev); gen.manual_seed(seed)
    dev = tab.dev
    cfg, start = P.sample_configs(tab, n_broad, q_pairs, gen)
    broad = generate(tab, cfg, start, gen)
    lay = torch.randperm(tab.nlay, device=dev, generator=gen)[:dense_layouts]
    qi = q_pairs[torch.randint(len(q_pairs), (dense_layouts,), device=dev, generator=gen)]
    c1 = tab.config(lay, qi[:, 0], qi[:, 1])
    st = torch.stack([torch.randperm(tab.n * tab.n, device=dev, generator=gen)[:dense_starts] for _ in range(dense_layouts)])
    cfg = c1[:, None, None].expand(-1, dense_starts, dense_per).reshape(-1)
    start = st[:, :, None].expand(-1, -1, dense_per).reshape(-1)
    perm = torch.randperm(len(cfg), device=dev, generator=gen)       # so that each behaviour sees every layout
    dense = generate(tab, cfg[perm], start[perm], gen)
    # layouts: the dense ones are all held out from probe fitting; of the rest, 30 % are held out too
    rest = torch.tensor([l for l in torch.randperm(tab.nlay, generator=torch.Generator().manual_seed(seed)).tolist()
                         if l not in set(lay.tolist())], device=dev)
    n_fit = int(0.7 * len(rest))
    split = dict(fit_layouts=rest[:n_fit], test_layouts=rest[n_fit:], dense_discovery=lay[: dense_layouts // 2],
                 dense_eval=lay[dense_layouts // 2:])
    return dict(broad=broad, dense=dense, split=split)


def twins(tab, b: P.Batch, idx):
    """For histories idx: the four replays with reports (yh, yv) in {0, 1}^2. Returns {(yh, yv): Batch}."""
    out = {}
    for yh in (0, 1):
        for yv in (0, 1):
            f = lambda v: torch.full((len(idx),), v, device=tab.dev)
            out[(yh, yv)] = replay(tab, b.cfg[idx], b.state[idx, 0, 1], b.act[idx], b.alive[idx], f(yh), f(yv))
    return out
