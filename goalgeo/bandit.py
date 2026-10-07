"""A hidden goal, noisy cues, and rewards that are evidence: the reward-trained hidden-goal task (the reward-bandit
experiment), with its exact Bayesian filter and the exact belief-MDP solution.

Episode. A goal G ~ uniform over K is hidden. The agent sees N_CUE cue tokens o ~ L[G] (the hidden-goal experiment's
i.i.d. emission: token k favours goal k), then makes T decisions. Action a pays a Bernoulli reward with probability
R[G, a]; the reward is seen, so it is evidence about G too. Return = the sum of rewards (no discount).

Tokens.  [BOS] [CUE o_1] ... [CUE o_n] [EVT a_0 r_0] ... [EVT a_{T-2} r_{T-2}]
The next action is predicted at the last cue token and at every event token. The goal, the posterior and the solver's
values are never inputs; the number of decisions is fixed, so the position gives the steps left.

Ground truth. Every likelihood is a product over independent draws, so the posterior depends on the history only
through its counts: the cue counts [M] and the outcome counts [A, 2]. The belief MDP therefore has finitely many
states, (cue counts, outcome counts), and value iteration over them gives Q* exactly, information value included.
References: optimal (argmax Q*), myopic (argmax_a sum_g b(g) R[g, a], which ignores the value of information), random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from goalgeo import latentgoal as LG
from goalgeo.navmodel import Block

PAD, BOS, CUE, EVT = range(4)
F_TYPE, F_SYM, F_ACT, F_REW = range(4)        # 0 = none in every field but type
NF = 4
R_DEFAULT = np.array([[1.0, 0.6, 0.1, 0.0], [0.1, 1.0, 0.4, 0.0], [0.0, 0.2, 0.7, 1.0]])


def compositions(n_cat, total, exact=False):
    """All vectors of n_cat non-negative ints with sum <= total (exact: == total), as an array [count, n_cat]."""
    if n_cat == 1:
        v = np.arange(total + 1) if not exact else np.array([total])
        return v[:, None]
    rows = []
    for first in range(total + 1):
        rest = compositions(n_cat - 1, total - first, exact)
        rows.append(np.c_[np.full(len(rest), first), rest])
    return np.concatenate(rows)


@dataclass
class Spec:
    K: int = 3
    n_cue: int = 4
    T: int = 6
    R: np.ndarray = None                  # [K, A] success probabilities
    L: np.ndarray = None                  # [K, M] cue emission
    clip: float = 0.0                     # success probabilities clipped to [clip, 1 - clip]: no outcome is conclusive
    channel: bool = False                 # cues through the hidden-goal experiment's sticky reliability channel (the order of the cues matters)
    stay: float = LG.STAY                 # the channel's persistence, and its emission when on
    hit_on: float = LG.HIT_ON
    neigh_on: float = LG.NEIGH_ON

    def __post_init__(self):
        self.R = R_DEFAULT.copy() if self.R is None else np.asarray(self.R, float)
        if self.clip:
            self.R = np.clip(self.R, self.clip, 1 - self.clip)
        self.L = LG._goal_emission(self.K, LG.HIT, LG.NEIGH) if self.L is None else np.asarray(self.L, float)
        self.A, self.M = self.R.shape[1], self.L.shape[1]
        assert self.R.shape[0] == self.K == self.L.shape[0]
        if self.channel:                  # c = 0 off (uniform cues), 1 on; P(stay) = STAY; starts at (1/2, 1/2)
            self.L_c = np.stack([np.full((self.K, self.M), 1.0 / self.M), LG._goal_emission(self.K, self.hit_on, self.neigh_on)])
            self.A_c = np.array([[self.stay, 1 - self.stay], [1 - self.stay, self.stay]]); self.c0 = np.array([0.5, 0.5])

    @property
    def L_seq(self):
        return 1 + self.n_cue + self.T - 1

    def decision_positions(self):
        return np.arange(self.n_cue, self.n_cue + self.T)


class Graph:
    """The belief MDP: every (cue counts, outcome counts) state, its posterior, and Q* by value iteration."""

    def __init__(self, s: Spec):
        self.s = s
        K, A, M, T = s.K, s.A, s.M, s.T
        self.cue = compositions(M, s.n_cue, exact=True)                                      # [nc, M]
        self.out = compositions(2 * A, T)                                                   # [no, 2A]: index 2a + r
        self.cue_base, self.out_base = s.n_cue + 1, T + 1
        self.cue_key = (self.cue * self.cue_base ** np.arange(M)).sum(1)
        self.out_key = (self.out * self.out_base ** np.arange(2 * A)).sum(1)
        oc, oo = np.argsort(self.cue_key), np.argsort(self.out_key)
        self.cue, self.cue_key, self.out, self.out_key = self.cue[oc], self.cue_key[oc], self.out[oo], self.out_key[oo]
        lg = lambda x: np.log(np.clip(x, 1e-300, None))                                    # an impossible outcome: log 1e-300
        logL, logR1, logR0 = lg(s.L), lg(s.R), lg(1 - s.R)
        lp = self.cue @ logL.T                                                              # [nc, K]
        lo = self.out[:, 0::2] @ logR1.T + self.out[:, 1::2] @ logR0.T                       # [no, K]
        z = lp[:, None, :] + lo[None, :, :] - np.log(K)
        z = z - z.max(-1, keepdims=True)
        b = np.exp(z); self.B = b / b.sum(-1, keepdims=True)                                 # [nc, no, K]
        self.level = self.out.sum(1)
        self.Q = np.zeros((len(self.cue), len(self.out), A))
        self.V = np.zeros((len(self.cue), len(self.out)))
        nxt = np.full((len(self.out), A, 2), -1)
        for i, o in enumerate(self.out):
            if self.level[i] < T:
                for a in range(A):
                    for r in (1, 0):
                        nxt[i, a, r] = np.searchsorted(self.out_key, self.out_key[i] + self.out_base ** (2 * a + (1 - r)))
        self.nxt = nxt
        for lev in range(T - 1, -1, -1):
            idx = np.nonzero(self.level == lev)[0]
            p1 = np.einsum("cik,ka->cia", self.B[:, idx], s.R)                                # P(r = 1 | b, a)
            v1, v0 = self.V[:, nxt[idx, :, 1]], self.V[:, nxt[idx, :, 0]]                      # [nc, len(idx), A]
            self.Q[:, idx] = p1 * (1 + v1) + (1 - p1) * v0
            self.V[:, idx] = self.Q[:, idx].max(-1)
        self.Qmy = np.einsum("cik,ka->cia", self.B, s.R)                                      # myopic values
        self.start = np.searchsorted(self.out_key, 0)                                         # the empty outcome count

    def cue_index(self, counts):
        key = (np.asarray(counts) * self.cue_base ** np.arange(self.s.M)).sum(-1)
        return np.searchsorted(self.cue_key, key)


class Sim:
    """The graph's tables on a device."""

    def __init__(self, g: Graph, device):
        t = lambda x, dt=None: torch.tensor(np.asarray(x), device=device, dtype=dt)
        self.g, self.s, self.dev = g, g.s, device
        self.R, self.L = t(g.s.R, torch.float32), t(g.s.L, torch.float32)
        self.B, self.Q, self.V, self.Qmy = t(g.B, torch.float32), t(g.Q, torch.float32), t(g.V, torch.float32), t(g.Qmy, torch.float32)
        self.nxt = t(g.nxt, torch.long)
        self.cue_key, self.cue_base = t(g.cue_key, torch.long), g.cue_base
        self.A, self.K, self.M, self.T, self.n_cue, self.L_seq = g.s.A, g.s.K, g.s.M, g.s.T, g.s.n_cue, g.s.L_seq
        self.channel = g.s.channel
        if self.channel:
            self.L_c, self.A_c, self.c0 = t(g.s.L_c, torch.float32), t(g.s.A_c, torch.float32), t(g.s.c0, torch.float32)
        lg = lambda x: np.log(np.clip(x, 1e-300, None))
        self.out_loglik = t(g.out[:, 0::2] @ lg(g.s.R).T + g.out[:, 1::2] @ lg(1 - g.s.R).T, torch.float32)   # [n_out, K]
        self.level = t(g.level, torch.long)


def channel_cues(t: Sim, goal, gen):
    """Cue tokens [N, n_cue] from the sticky channel, and the exact posterior over the goal after them [N, K]."""
    N, dev = len(goal), t.dev
    c = (torch.rand(N, device=dev, generator=gen) < t.c0[1]).long()
    cues = torch.zeros(N, t.n_cue, dtype=torch.long, device=dev)
    for i in range(t.n_cue):
        if i > 0:
            stay = torch.rand(N, device=dev, generator=gen) < t.A_c[c, c]
            c = torch.where(stay, c, 1 - c)
        cues[:, i] = torch.multinomial(t.L_c[c, goal], 1, generator=gen).squeeze(1)
    return cues, channel_posterior(t, cues)


def channel_posterior(t: Sim, cues):
    """The exact joint filter over (goal, channel) run over the cue tokens [N, n_cue]; returns P(goal | cues) [N, K]."""
    N = cues.shape[0]
    alpha = (torch.full((t.K,), 1.0 / t.K, device=t.dev)[:, None] * t.c0[None, :]).expand(N, -1, -1).clone()     # [N, K, C]
    for i in range(cues.shape[1]):
        if i > 0:
            alpha = alpha @ t.A_c
        lik = t.L_c[:, :, cues[:, i]].permute(2, 1, 0)                                                            # [N, K, C]
        alpha = alpha * lik
        alpha = alpha / alpha.sum((1, 2), keepdim=True)
    return alpha.sum(-1)


def episode_tables(t: Sim, b0):
    """Per-episode decision-phase tables from a belief after the cues b0 [N, K]: the belief at every outcome-count
    state B [N, n_out, K] and Q* [N, n_out, A] by value iteration (the Graph's computation, batched over episodes)."""
    N = b0.shape[0]
    z = torch.log(b0.clamp(min=1e-30))[:, None, :] + t.out_loglik[None]                                           # [N, n_out, K]
    z = z - z.max(-1, keepdim=True).values
    B = torch.exp(z); B = B / B.sum(-1, keepdim=True)
    Q = torch.zeros(N, B.shape[1], t.A, device=t.dev); V = torch.zeros(N, B.shape[1], device=t.dev)
    for lev in range(t.T - 1, -1, -1):
        idx = torch.nonzero(t.level == lev).squeeze(1)
        p1 = B[:, idx] @ t.R                                                                                      # [N, len(idx), A]
        v1, v0 = V[:, t.nxt[idx, :, 1]], V[:, t.nxt[idx, :, 0]]
        Q[:, idx] = p1 * (1 + v1) + (1 - p1) * v0
        V[:, idx] = Q[:, idx].max(-1).values
    return B, Q


class Env:
    """N episodes. Common random numbers: the reward of (episode, step, action) is decided by one uniform draw, so two
    policies on the same episodes face the same outcomes for the same actions."""

    def __init__(self, t: Sim, N, gen=None, goal=None, cues=None, u=None):
        self.t, self.N = t, N
        dev = t.dev
        self.goal = torch.randint(t.K, (N,), device=dev, generator=gen) if goal is None else goal
        if cues is None:
            if t.channel:
                cues, _ = channel_cues(t, self.goal, gen)
            else:
                cues = torch.multinomial(t.L[self.goal], t.n_cue, replacement=True, generator=gen) if t.n_cue else torch.zeros(N, 0, dtype=torch.long, device=dev)
        self.cues = cues
        self.u = torch.rand(N, t.T, t.A, device=dev, generator=gen) if u is None else u
        self.tok = torch.zeros(N, t.L_seq, NF, dtype=torch.long, device=dev)
        self.tok[:, 0, F_TYPE] = BOS
        self.tok[:, 1: 1 + t.n_cue, F_TYPE] = CUE
        self.tok[:, 1: 1 + t.n_cue, F_SYM] = self.cues + 1
        counts = torch.nn.functional.one_hot(self.cues, t.M).sum(1)
        self.ci = torch.searchsorted(t.cue_key, (counts * t.cue_base ** torch.arange(t.M, device=dev)).sum(1))
        self.n = torch.arange(N, device=dev)
        if t.channel:                                                 # the belief after the cues depends on their order: tables per episode
            self.Bep, self.Qep = episode_tables(t, channel_posterior(t, self.cues))
            self.cue_id = (self.cues * t.M ** torch.arange(t.n_cue, device=dev)).sum(1)
        else:
            self.Bep = self.Qep = None
            self.cue_id = self.ci
        self.oi = torch.full((N,), t.g.start, device=dev)
        self.step_i = 0

    @property
    def belief(self):
        return self.Bep[self.n, self.oi] if self.t.channel else self.t.B[self.ci, self.oi]

    def q_star(self):
        return self.Qep[self.n, self.oi] if self.t.channel else self.t.Q[self.ci, self.oi]

    def q_myopic(self):
        return self.belief @ self.t.R

    @property
    def pos(self):
        return self.t.n_cue + self.step_i

    def step(self, a):
        t, n = self.t, torch.arange(self.N, device=self.t.dev)
        r = (self.u[n, self.step_i, a] < t.R[self.goal, a]).long()
        self.oi = t.nxt[self.oi, a, r]
        p = self.pos + 1
        if p < t.L_seq:
            self.tok[:, p] = torch.stack([torch.full_like(a, EVT), torch.zeros_like(a), a + 1, r + 1], -1)
        self.step_i += 1
        return r.float()


@dataclass
class Batch:
    tok: torch.Tensor            # [N, L, NF]
    pos: torch.Tensor            # [N, T]
    act: torch.Tensor            # [N, T]
    logp: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    alive: torch.Tensor          # [N, T] all True (fixed-length episodes)
    regret: torch.Tensor         # [N, T] V* - Q*(a)
    goal: torch.Tensor           # [N]
    ci: torch.Tensor             # [N] cue state
    oi: torch.Tensor             # [N, T] outcome state at each decision
    info_pays: torch.Tensor      # [N, T] the myopic action is not optimal here
    b: torch.Tensor = None       # [N, T, K] exact belief at each decision
    q: torch.Tensor = None       # [N, T, A] Q* at each decision
    cue_id: torch.Tensor = None  # [N] the cue state: counts (i.i.d.) or the sequence (channel)


@torch.no_grad()
def rollout(net, t: Sim, N, gen=None, greedy=False, behaviour=None, goal=None, cues=None, u=None):
    env = Env(t, N, gen, goal, cues, u)
    T, dev = t.T, t.dev
    z = lambda *s, dt=torch.float32: torch.zeros(N, T, *s, dtype=dt, device=dev)
    pos, act, oi = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
    logp, val, rew, reg, pays = z(), z(), z(), z(), z(dt=torch.bool)
    bel, qs = z(t.K), z(t.A)
    for s in range(T):
        p = env.pos
        if behaviour is None:
            lg, v = net(env.tok[:, : p + 1])
            lg, v = lg[:, p], v[:, p]
            a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
            logp[:, s], val[:, s] = torch.log_softmax(lg, -1).gather(1, a[:, None]).squeeze(1), v
        else:
            a = behaviour(env)
        q, qm = env.q_star(), env.q_myopic()
        best = q.max(1).values
        reg[:, s] = best - q.gather(1, a[:, None]).squeeze(1)
        pays[:, s] = q.gather(1, qm.argmax(1, keepdim=True)).squeeze(1) < best - 1e-6
        pos[:, s], act[:, s], oi[:, s], bel[:, s], qs[:, s] = p, a, env.oi, env.belief, q
        rew[:, s] = env.step(a)
    return Batch(env.tok, pos, act, logp, val, rew, torch.ones(N, T, dtype=torch.bool, device=dev), reg, env.goal, env.ci, oi, pays, bel, qs, env.cue_id)


def optimal(gen):
    def f(env):
        q = env.q_star()
        best = (q >= q.max(1, keepdim=True).values - 1e-6).float()
        return torch.multinomial(best, 1, generator=gen).squeeze(1)
    return f


def myopic(gen):
    def f(env):
        q = env.q_myopic()
        best = (q >= q.max(1, keepdim=True).values - 1e-6).float()
        return torch.multinomial(best, 1, generator=gen).squeeze(1)
    return f


def uniform(gen):
    return lambda env: torch.randint(env.t.A, (env.N,), device=env.t.dev, generator=gen)


def summarize(b: Batch, t: Sim, M=1):
    """Per model of a model-major batch: return, exact regret, V*, optimal-action rate, behaviour where information
    pays (the myopic action is not optimal): the share of optimal actions and of myopic actions there."""
    N = b.act.shape[0]
    out = []
    for m in range(M):
        s = slice(m * N // M, (m + 1) * N // M)
        q = b.q[s]                                                                             # [n, T, A]
        qm = b.b[s] @ t.R
        opt = (q.gather(2, b.act[s][..., None]).squeeze(2) >= q.max(-1).values - 1e-6)
        my = (qm.gather(2, b.act[s][..., None]).squeeze(2) >= qm.max(-1).values - 1e-6)
        pays = b.info_pays[s]
        out.append(dict(ret=b.rew[s].sum(1).mean().item(), regret=b.regret[s].sum(1).mean().item(),
                        v_star=q[:, 0].max(-1).values.mean().item(), opt_rate=opt.float().mean().item(),
                        info_pays_share=pays.float().mean().item(), opt_where_pays=opt[pays].float().mean().item() if pays.any() else float("nan"),
                        myopic_where_pays=my[pays].float().mean().item() if pays.any() else float("nan"), myopic_rate=my.float().mean().item()))
    return out


# ------------------------------------------------------------------ policies

class Embed(nn.Module):
    def __init__(self, d, M, A, L):
        super().__init__()
        self.f = nn.ModuleList([nn.Embedding(s, d) for s in (4, M + 1, A + 1, 3)])
        self.pos = nn.Embedding(L, d)
        for e in list(self.f) + [self.pos]:
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, tok):
        return sum(e(tok[..., i]) for i, e in enumerate(self.f)) + self.pos.weight[: tok.shape[1]]


class Net(nn.Module):
    """A pre-LN causal transformer (navmodel's blocks); logits [N, L, A] and value [N, L] at every position.
    record: the residual stream entering each block and after the last ('resid'), as in mazemodel.Net.
    patch: {("attn" | "mlp" | "resid" | "resid_mid", layer): fn(tensor) -> tensor}."""

    def __init__(self, s: Spec, d=128, layers=2, heads=4):
        super().__init__()
        self.d, self.nl, self.nh, self.A = d, layers, heads, s.A
        self.embed = Embed(d, s.M, s.A, s.L_seq)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.pi, self.v = nn.Linear(d, s.A), nn.Linear(d, 1)
        nn.init.normal_(self.pi.weight, std=0.01); nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.v.weight, std=0.1); nn.init.zeros_(self.v.bias)

    def forward(self, tok, record=False, patch=None):
        patch = patch or {}
        x = self.embed(tok)
        rec = dict(resid=[], attn=[], mlp=[]) if record else None
        for l, b in enumerate(self.blocks):
            if ("resid", l) in patch:
                x = patch[("resid", l)](x)
            a, _, _, _ = b.attn(x)
            if ("attn", l) in patch:
                a = patch[("attn", l)](a)
            mid = x + a
            if ("resid_mid", l) in patch:
                mid = patch[("resid_mid", l)](mid)
            m = b.mlp(b.ln2(mid))
            if ("mlp", l) in patch:
                m = patch[("mlp", l)](m)
            if record:
                rec["resid"].append(x); rec["attn"].append(a); rec["mlp"].append(m)
            x = mid + m
        if ("resid", self.nl) in patch:
            x = patch[("resid", self.nl)](x)
        if record:
            rec["resid"].append(x)
        h = self.ln(x)
        out = (self.pi(h), self.v(h).squeeze(-1))
        return out + (rec,) if record else out


class GRUNet(nn.Module):
    """A one-layer GRU written out as tensor operations (so that torch.func.vmap can stack models); the same
    embedding and heads as Net. record: the state after every position ('resid' = [state]); patch: {("resid", 1):
    fn(states [N, L, d]) -> states} replaces the states the heads read (the future is not re-run)."""

    def __init__(self, s: Spec, d=128, layers=1, heads=0):
        super().__init__()
        self.d, self.nl, self.A = d, 1, s.A
        self.embed = Embed(d, s.M, s.A, s.L_seq)
        self.wi, self.wh = nn.Linear(d, 3 * d), nn.Linear(d, 3 * d)
        self.h0 = nn.Parameter(torch.zeros(d))
        self.ln = nn.LayerNorm(d)
        self.pi, self.v = nn.Linear(d, s.A), nn.Linear(d, 1)
        nn.init.normal_(self.pi.weight, std=0.01); nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.v.weight, std=0.1); nn.init.zeros_(self.v.bias)

    def states(self, tok, h=None):
        x = self.embed(tok)
        N, L, d = x.shape
        gi = self.wi(x)                                                                           # [N, L, 3d]
        h = self.h0.expand(N, d) if h is None else h
        hs = []
        for p in range(L):
            gh = self.wh(h)
            r = torch.sigmoid(gi[:, p, :d] + gh[:, :d])
            z = torch.sigmoid(gi[:, p, d: 2 * d] + gh[:, d: 2 * d])
            n = torch.tanh(gi[:, p, 2 * d:] + r * gh[:, 2 * d:])
            h = (1 - z) * n + z * h
            hs.append(h)
        return torch.stack(hs, 1)

    def forward(self, tok, record=False, patch=None):
        patch = patch or {}
        H = self.states(tok)
        if ("resid", 1) in patch:
            H = patch[("resid", 1)](H)
        h = self.ln(H)
        out = (self.pi(h), self.v(h).squeeze(-1))
        return out + (dict(resid=[H]),) if record else out


def build(arch, s: Spec, d=128, layers=2, heads=4):
    return (Net if arch == "tfm" else GRUNet)(s, d, layers, heads)


# ------------------------------------------------------------------ PPO (mazeppo's update, A actions, no discount)

from goalgeo.navppo import PPOConfig, clip_per_model          # noqa: E402
from goalgeo.mazeppo import Stacked, gae                       # noqa: E402


def ppo_update(stk, opt, b: Batch, cfg: PPOConfig, ent_coef, params, A):
    M = stk.M
    N, T = b.act.shape[0] // M, b.act.shape[1]
    dev = b.act.device
    adv, target = gae(b.rew, b.val, b.alive, 1.0, cfg.lam)
    adv = adv.view(M, N, T)
    mean, std = adv.mean((1, 2)), adv.std((1, 2))
    adv = ((adv - mean[:, None, None]) / (std[:, None, None] + 1e-8)).view(M * N, T)
    off = (torch.arange(M, device=dev) * N)[:, None]
    for _ in range(cfg.epochs):
        perm = torch.argsort(torch.rand(M, N, device=dev), 1)
        for ch in perm.chunk(cfg.minibatches, 1):
            idx = (ch + off).reshape(-1)
            lg, v = stk(b.tok[idx])
            p = b.pos[idx]
            lg, v = lg.gather(1, p[..., None].expand(-1, -1, A)), v.gather(1, p)
            lp = torch.log_softmax(lg, -1)
            per = lambda x: x.view(M, -1, T).mean((1, 2))
            ratio = (lp.gather(2, b.act[idx][..., None]).squeeze(2) - b.logp[idx]).exp()
            a = adv[idx]
            pl = -per(torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a))
            vl = per((v - target[idx]) ** 2)
            ent = per(-(lp.exp() * lp).sum(-1))
            loss = (pl + cfg.vf * vl - ent_coef * ent).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            clip_per_model(params, cfg.max_grad)
            opt.step()
