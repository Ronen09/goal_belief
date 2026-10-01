"""Round 26: prediction-only backbones for the maze task, and small goal-conditioned heads on frozen representations.

Predictor. mazemodel.Net with a prediction head, trained only on goal-free random walks (no goal token, no reward).
At every token the head gives, for each supplied move sequence of length k (k = 1: the four moves; k = 2: the
sixteen move pairs), a distribution over the symbols that follow (k = 2: the joint pair, nine values). Targets are
sampled by the simulator from the true cell for every supplied sequence (round 25's balanced targets), so every
candidate is covered at every token. The posterior, the cell and the exact predictive distribution are never given.

Heads. Many small MLPs trained side by side (one per backbone, input site and head seed) on precomputed features: the
layer-normalised representation at one token, concatenated with a one-hot goal, to four action logits. The loss is
-log of the probability given to the solver's optimal action set.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from goalgeo import mazemodel as MM, mazeppo as P


# ------------------------------------------------------------------ exact k-step predictions (solver tables only)

def predictive(t: P.Tables, belief, k):
    """Exact goal-free prediction from beliefs [N, n]: k = 1 -> [N, 4, symbols]; k = 2 -> [N, 16, symbols^2]
    (move pair a1 * 4 + a2, symbol pair o1 * symbols + o2)."""
    nx, E = t.nxt_cell, t.E.to(belief.dtype)                              # [n, 4], [n, symbols]
    p1 = torch.einsum("ns,sao->nao", belief, E[nx])                       # E[nx]: [n, 4, symbols]
    if k == 1:
        return p1
    n, ns = belief.shape[1], t.n_sym
    c1 = nx                                                               # [n, 4]
    c2 = nx[c1]                                                           # [n, 4, 4]
    joint = E[c1][:, :, None, :, None] * E[c2][:, :, :, None, :]          # [n, a1, a2, o1, o2]
    return torch.einsum("ns,sabxy->nabxy", belief, joint).reshape(len(belief), 16, ns * ns)


# ------------------------------------------------------------------ the predictor

class NetPred(MM.Net):
    """mazemodel.Net with a k-step prediction head. The shared parameters are initialised first, so a seed gives
    the same backbone as mazemodel.Net and mazeaux.NetAux (round 23's models)."""

    def __init__(self, n_sym=3, n_goals=3, H=12, max_prefix=4, k=1, **kw):
        super().__init__(n_sym, n_goals, H, max_prefix, **kw)
        self.k, self.n_sym = k, n_sym
        self.n_seq, self.n_out = 4 ** k, n_sym ** k
        self.obs = nn.Linear(self.d, self.n_seq * self.n_out)
        nn.init.normal_(self.obs.weight, std=0.01); nn.init.zeros_(self.obs.bias)

    def forward(self, tok, q=None, record=False, patch=None, pred=False):
        if not pred:
            return super().forward(tok, q, record, patch)
        x = self.embed(tok)
        for b in self.blocks:
            mid = x + b.attn(x)[0]
            x = mid + b.mlp(b.ln2(mid))
        return self.obs(self.ln(x)).view(*x.shape[:-1], self.n_seq, self.n_out)


class StackedPred(P.Stacked):
    def pred(self, tok):
        f = lambda p, t: torch.func.functional_call(self.base, p, (t,), dict(pred=True))
        return torch.func.vmap(f)(self.params, tok.view(self.M, -1, *tok.shape[1:])).flatten(0, 1)

    def trainable(self, heads_only=False):
        return [v for k, v in self.params.items() if not k.startswith(("pi.", "v."))]       # the policy and value heads are unused


def walks(t: P.Tables, N, k, gen):
    """Goal-free random walks filling the context: [OBS o_0] [EVT a_0 o_1] ... (t.L tokens, no goal token).
    Returns tok [N, L, NF], the true cell at each token [N, L], and targets [N, L, 4^k]: for every supplied move
    sequence, symbols sampled from the cells it leads to (k = 2: o1 * symbols + o2)."""
    dev, L = t.dev, t.L
    cell = t.starts[torch.randint(len(t.starts), (N,), device=dev, generator=gen)]
    tok = torch.zeros(N, L, MM.NF, dtype=torch.long, device=dev)
    cells = torch.zeros(N, L, dtype=torch.long, device=dev)
    sample = lambda c: torch.multinomial(t.E[c.reshape(-1)], 1, generator=gen).view(c.shape)
    tok[:, 0, MM.F_TYPE], tok[:, 0, MM.F_SYM] = MM.OBS, sample(cell) + 1
    cells[:, 0] = cell
    for i in range(1, L):
        a = torch.randint(4, (N,), device=dev, generator=gen)
        cell = t.nxt_cell[cell, a]
        tok[:, i, MM.F_TYPE], tok[:, i, MM.F_SYM], tok[:, i, MM.F_ACT] = MM.EVT, sample(cell) + 1, a + 1
        cells[:, i] = cell
    c1 = t.nxt_cell[cells]                                                # [N, L, 4]
    o1 = sample(c1)
    if k == 1:
        return tok, cells, o1
    c2 = t.nxt_cell[c1]                                                   # [N, L, 4, 4]
    o1b = sample(c1[..., None].expand_as(c2))                             # an independent first symbol for every pair
    return tok, cells, (o1b * t.n_sym + sample(c2)).flatten(2)


def pred_loss(logits, target):
    """Mean cross-entropy per model; logits [M * N, L, seq, out], target [M * N, L, seq]."""
    nll = -torch.log_softmax(logits, -1).gather(3, target[..., None]).squeeze(3)
    return nll


# ------------------------------------------------------------------ features at a fixed token

@torch.no_grad()
def features(net, tok, pos, chunk=8192):
    """Residual stream entering each block and after the last ([layers + 1, N, d]) at token `pos` of each row."""
    out = torch.zeros(net.nl + 1, len(tok), net.d, device=tok.device)
    for s in range(0, len(tok), chunk):
        sl = slice(s, s + chunk)
        top = int(pos[sl].max()) + 1
        rec = net(tok[sl, :top], record=True)[2]
        n = torch.arange(len(tok[sl]), device=tok.device)
        for l, x in enumerate(rec["resid"]):
            out[l, sl] = x[n, pos[sl]]
    return out


def layer_norm(x):
    return torch.nn.functional.layer_norm(x, x.shape[-1:])


# ------------------------------------------------------------------ heads trained side by side

class Heads:
    """K independent MLPs [d_in + goals] -> hidden -> 4 (hidden = 0: a linear map). Weights [K, ...]."""

    def __init__(self, K, d_in, n_goals, hidden, seeds, device):
        self.hidden = hidden
        ps = []
        for s in seeds:
            g = torch.Generator().manual_seed(int(s))
            def lin(i, o):
                bound = 1 / math.sqrt(i)
                return (torch.rand(i, o, generator=g) * 2 - 1) * bound, (torch.rand(o, generator=g) * 2 - 1) * bound
            ps.append(lin(d_in + n_goals, hidden or 4) + (lin(hidden, 4) if hidden else ()))
        self.params = [torch.stack([p[i] for p in ps]).to(device).requires_grad_() for i in range(len(ps[0]))]

    def __call__(self, x, goal_onehot):
        h = torch.cat([x, goal_onehot], -1)                               # [K, B, d_in + goals]
        h = torch.baddbmm(self.params[1][:, None], h, self.params[0])
        if self.hidden:
            h = torch.baddbmm(self.params[3][:, None], torch.relu(h), self.params[2])
        return h


def optimal_set_nll(logits, opt):
    lp = torch.log_softmax(logits, -1)
    return -torch.logsumexp(lp.masked_fill(~opt, -1e9), -1)


def train_heads(X, xmap, ex_hist, ex_goal, ex_opt, idx, hidden, seeds, steps, batch, lr, n_goals=3):
    """X [backbones, histories, d_in]: normalised features; head j reads X[xmap[j]]. Examples are (history, goal,
    optimal set) rows ex_hist [E], ex_goal [E], ex_opt [E, 4]; idx [K, n]: the examples of each head. Every head
    takes `steps` Adam steps on minibatches of min(batch, n) of its own examples. Returns the trained Heads."""
    K, n = idx.shape
    dev = X.device
    hd = Heads(K, X.shape[-1], n_goals, hidden, seeds, dev)
    o = torch.optim.Adam(hd.params, lr=lr)
    gen = torch.Generator(device=dev); gen.manual_seed(7919 + n)
    eye = torch.eye(n_goals, device=dev)
    B = min(batch, n)
    for _ in range(steps):
        pick = idx.gather(1, torch.randint(n, (K, B), device=dev, generator=gen))      # [K, B] example ids
        lg = hd(X[xmap[:, None], ex_hist[pick]], eye[ex_goal[pick]])
        loss = optimal_set_nll(lg, ex_opt[pick]).mean(1).sum()
        o.zero_grad(set_to_none=True); loss.backward(); o.step()
    return hd


@torch.no_grad()
def apply_heads(hd, X, xmap, hist, n_goals=3, chunk=16384):
    """Logits [K, len(hist), goals, 4] for every listed history under every goal."""
    eye = torch.eye(n_goals, device=X.device)
    out = []
    for s in range(0, len(hist), chunk):
        x = X[xmap[:, None], hist[s:s + chunk][None]]                       # [K, c, d]
        K, c = x.shape[:2]
        xg = x[:, :, None].expand(-1, -1, n_goals, -1).reshape(K, c * n_goals, -1)
        g = eye.repeat(c, 1)[None].expand(K, -1, -1)
        out.append(hd(xg, g).view(K, c, n_goals, 4))
    return torch.cat(out, 1)
