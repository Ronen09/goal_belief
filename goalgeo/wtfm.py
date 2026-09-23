"""TASK13 (round 14): a windowed transformer with an optional recurrent carry.

Each position attends only to the last `window` positions (itself included) in every layer, with a
learned per-head relative-position bias and no absolute positions, so nothing in the network
records elapsed time. With `carry`, the readout interface of the previous position,
u_{t-1} = LN_f(x_top(t-1)), is projected and added to the input of position t. With window 1 and
the carry, u_t is the only path from the past to the future: a complete cut, as in a GRU. Without
the carry, a 2-layer model sees at most 2(window - 1) + 1 tokens.

The computation runs position by position with a per-layer key/value cache, so the carried state
is exact. `run(X, start=t, u0=...)` restarts from an edited state at position t."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WBlock(nn.Module):
    def __init__(self, d, n_heads, mlp, window):
        super().__init__()
        self.h, self.dh, self.w = n_heads, d // n_heads, window
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.rel = nn.Parameter(torch.zeros(n_heads, window))      # bias by offset 0..window-1

    def step(self, x, cache):
        """x [B, d] at the current position; cache: list of (k, v) [B, H, dh] for earlier positions."""
        B = x.shape[0]
        q, k, v = self.qkv(self.ln1(x)).view(B, 3, self.h, self.dh).unbind(1)
        cache = (cache + [(k, v)])[-self.w:]
        K = torch.stack([c[0] for c in cache], 2); V = torch.stack([c[1] for c in cache], 2)   # [B, H, n, dh]
        n = K.shape[2]
        s = (q[:, :, None] * K).sum(-1) / self.dh ** 0.5 + self.rel[:, :n].flip(-1)          # offset n-1 .. 0
        a = (torch.softmax(s, -1)[..., None] * V).sum(2).reshape(B, -1)
        x = x + self.o(a)
        return x + self.mlp(self.ln2(x)), cache


class WindowTransformer(nn.Module):
    def __init__(self, n_vocab, out_dim, window, carry, d=64, n_heads=4, n_layers=2, mlp=128, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.window, self.carry, self.d = window, carry, d
        self.emb = nn.Embedding(n_vocab, d)
        self.blocks = nn.ModuleList(WBlock(d, n_heads, mlp, window) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d); self.out = nn.Linear(d, out_dim)
        self.cw = nn.Linear(d, d) if carry else None

    def run(self, X, start=0, u0=None, caches=None, keep=False):
        """Positions start..T-1 of X [B, T]. u0 is the carried state entering position `start`
        (the readout interface of position start-1). Returns logits [B, T-start, out] and the
        interfaces u [B, T-start, d]; with keep, also every layer's residual stream."""
        B, T = X.shape; dev = self.emb.weight.device; X = X.to(dev)
        caches = caches or [[] for _ in self.blocks]
        u_prev = u0; zs, us, res = [], [], []
        for t in range(start, T):
            x = self.emb(X[:, t])
            if self.carry and u_prev is not None:
                x = x + self.cw(u_prev)
            r = [x]
            for i, b in enumerate(self.blocks):
                x, caches[i] = b.step(x, caches[i]); r.append(x)
            u = self.ln_f(x); u_prev = u
            zs.append(self.out(u)); us.append(u)
            if keep:
                res.append(torch.stack(r, 1))
        out = (torch.stack(zs, 1), torch.stack(us, 1))
        return out + (torch.stack(res, 1),) if keep else out

    def forward_all(self, X):
        return self.run(X)[0]

    @torch.no_grad()
    def run_edited(self, X, t, u_new=None, res1_new=None):
        """Run X, replacing at position t either the interface u_t (which is what the carry takes
        forward) or the block-1 residual (which later positions' block-2 attention reads).
        Returns output distributions at positions t..T-1."""
        B, T = X.shape; dev = self.emb.weight.device; X = X.to(dev)
        caches = [[] for _ in self.blocks]; u_prev = None; ps = []
        for s in range(T):
            x = self.emb(X[:, s])
            if self.carry and u_prev is not None:
                x = x + self.cw(u_prev)
            for i, b in enumerate(self.blocks):
                x, caches[i] = b.step(x, caches[i])
                if s == t and i == 0 and res1_new is not None:           # block 2 (and its cache) reads the new res1(t)
                    x = torch.as_tensor(res1_new, dtype=x.dtype, device=dev)
            u = self.ln_f(x)
            if s == t and u_new is not None:
                u = torch.as_tensor(u_new, dtype=u.dtype, device=dev)
            u_prev = u
            if s >= t:
                ps.append(torch.softmax(self.out(u), -1))
        return torch.stack(ps, 1).double().cpu().numpy()
