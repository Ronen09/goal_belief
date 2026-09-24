"""TASK13 (round 14): a windowed transformer with an optional recurrent carry.

Each position attends only to the last `window` positions (itself included) in every layer, with a
learned per-head relative-position bias and no absolute positions, so nothing in the network
records elapsed time. With `carry`, the readout interface of the previous position,
u_{t-1} = LN_f(x_top(t-1)), is projected and added to the input of position t. With window 1 and
the carry, u_t is the only path from the past to the future: a complete cut, as in a GRU. Without
the carry, a 2-layer model sees at most 2(window - 1) + 1 tokens.

The computation runs position by position with a per-layer key/value cache, so the carried state
is exact. `run(X, start=t, u0=...)` restarts from an edited state at position t.

With `gated`, every block carries a read gate (TASK16, `readgate.py`): a closed gate leaves the
query its own position and the carry."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import readgate as RG


class WBlock(nn.Module):
    def __init__(self, d, n_heads, mlp, window, gated=False):
        super().__init__()
        self.h, self.dh, self.w = n_heads, d // n_heads, window
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.rel = nn.Parameter(torch.zeros(n_heads, window))      # bias by offset 0..window-1
        self.gate = None
        if gated:                                                   # TASK16: one read gate per query position
            self.gate = nn.Linear(d, 1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, RG.B_INIT)

    def step(self, x, cache, visible=None, gate="open", gen=None):
        """x [B, d] at the current position; cache: list of (k, v) [B, H, dh] for earlier positions.
        visible: optional bool [B, n] over the window's entries after appending (the last is the
        current position), for K/V dropout (TASK15). gate: 'open' | 'own' | 'train' (TASK16); with a
        closed gate the query attends to its own position only. Returns (x, cache, p) with p [B] the
        open probability, or None when no gate was evaluated."""
        B = x.shape[0]
        a_in = self.ln1(x)
        q, k, v = self.qkv(a_in).view(B, 3, self.h, self.dh).unbind(1)
        cache = (cache + [(k, v)])[-self.w:]
        K = torch.stack([c[0] for c in cache], 2); V = torch.stack([c[1] for c in cache], 2)   # [B, H, n, dh]
        n = K.shape[2]

        def attend(K, V):
            m = K.shape[2]
            s = (q[:, :, None] * K).sum(-1) / self.dh ** 0.5 + self.rel[:, :m].flip(-1)      # offset m-1 .. 0
            if visible is not None:
                s = s.masked_fill(~visible[:, None, -m:], float("-inf"))
            return (torch.softmax(s, -1)[..., None] * V).sum(2).reshape(B, -1)

        a = attend(K, V); p = None
        if self.gate is not None and gate != "open" and n > 1:
            g, p = RG.gate(self.gate(a_in).squeeze(-1), gate == "train", gen=gen)             # [B]
            a = RG.mix(g, a, attend(K[:, :, -1:], V[:, :, -1:]))
        x = x + self.o(a)
        return x + self.mlp(self.ln2(x)), cache, p


class WindowTransformer(nn.Module):
    def __init__(self, n_vocab, out_dim, window, carry, d=64, n_heads=4, n_layers=2, mlp=128, seed=0, gated=False):
        super().__init__(); torch.manual_seed(seed)
        self.window, self.carry, self.d, self.gated = window, carry, d, gated
        self.emb = nn.Embedding(n_vocab, d)
        self.blocks = nn.ModuleList(WBlock(d, n_heads, mlp, window, gated) for _ in range(n_layers))
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
                x, caches[i], _ = b.step(x, caches[i]); r.append(x)
            u = self.ln_f(x); u_prev = u
            zs.append(self.out(u)); us.append(u)
            if keep:
                res.append(torch.stack(r, 1))
        out = (torch.stack(zs, 1), torch.stack(us, 1))
        return out + (torch.stack(res, 1),) if keep else out

    def forward_all(self, X, drop: float = 0.0, gen=None):
        if drop <= 0:
            return self.run(X)[0]
        B, T = X.shape; dev = self.emb.weight.device; X = X.to(dev)
        caches = [[] for _ in self.blocks]; u = None; zs = []
        for t in range(T):
            n = min(t + 1, self.window)
            vis = torch.rand(B, n, generator=gen, device=dev) >= drop; vis[:, -1] = True     # history dropped, self kept
            z, u, caches = self.advance(X[:, t], caches, u, vis)
            zs.append(z)
        return torch.stack(zs, 1)

    def advance_g(self, tok, caches, u_prev, visible=None, gate="open", gen=None):
        """One position: token tok [B], caches from earlier positions, carried state u_prev.
        Returns (logits, u, caches, p) with p [B, L] the blocks' open probabilities (1 where none)."""
        x = self.emb(tok.to(self.emb.weight.device))
        if self.carry and u_prev is not None:
            x = x + self.cw(u_prev)
        new, ps = [], []
        for i, b in enumerate(self.blocks):
            x, c, p = b.step(x, list(caches[i]), visible, gate, gen); new.append(c)
            ps.append(torch.ones(x.shape[0], device=x.device, dtype=x.dtype) if p is None else p)
        u = self.ln_f(x)
        return self.out(u), u, new, torch.stack(ps, -1)

    def advance(self, tok, caches, u_prev, visible=None, gate="open", gen=None):
        return self.advance_g(tok, caches, u_prev, visible, gate, gen)[:3]

    @torch.no_grad()
    def prefix(self, X, upto, gate="open"):
        """Caches (per layer, one (k, v) per position 0..upto) and the carried state after position upto,
        computed under the given gate mode (TASK16: 'own' for the model's actual state)."""
        caches = [[] for _ in self.blocks]; u = None
        for t in range(upto + 1):
            _, u, caches = self.advance(torch.as_tensor(X[:, t]), caches, u, None, gate)
        return caches, u

    def forward_gated(self, X, gate="own", gen=None):
        """Logits [B, T, out] and open probabilities [B, T, L] under the given gate mode (TASK16)."""
        B, T = X.shape; dev = self.emb.weight.device; X = X.to(dev)
        caches = [[] for _ in self.blocks]; u = None; zs, ps = [], []
        for t in range(T):
            z, u, caches, p = self.advance_g(X[:, t], caches, u, None, gate, gen)
            zs.append(z); ps.append(p)
        return torch.stack(zs, 1), torch.stack(ps, 1)

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
                x, caches[i], _ = b.step(x, caches[i])
                if s == t and i == 0 and res1_new is not None:           # block 2 (and its cache) reads the new res1(t)
                    x = torch.as_tensor(res1_new, dtype=x.dtype, device=dev)
            u = self.ln_f(x)
            if s == t and u_new is not None:
                u = torch.as_tensor(u_new, dtype=u.dtype, device=dev)
            u_prev = u
            if s >= t:
                ps.append(torch.softmax(self.out(u), -1))
        return torch.stack(ps, 1).double().cpu().numpy()
