"""TASK9: a small pre-LN causal transformer for the HMM next-token task, exposing the same
interface the GRU analyses use (states / forward_all / out) plus residual-stream patching,
which is the transformer's version of 'push a state through the remaining computation'."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seqmodels import FixedGainLinear
from . import readgate as RG


class Block(nn.Module):
    def __init__(self, d, n_heads, mlp, gated=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))
        self.gate = None
        if gated:                                                   # TASK16: one read gate per query position
            self.gate = nn.Linear(d, 1); nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, RG.B_INIT)

    def forward(self, x, mask, need_weights=False, closed_mask=None, gate="open", gen=None):
        """closed_mask: the attention mask a closed gate applies (self and previous position only).
        Returns (x, w) with need_weights, else x; with a gate evaluated, also appends p [B, T]
        through self._p (read by CausalTransformer.residuals_g)."""
        a = self.ln1(x)
        o, w = self.attn(a, a, a, attn_mask=mask, need_weights=need_weights, average_attn_weights=True)
        self._p = None
        if self.gate is not None and gate != "open":
            g, self._p = RG.gate(self.gate(a).squeeze(-1), gate == "train", gen=gen)        # [B, T]
            o_closed, _ = self.attn(a, a, a, attn_mask=closed_mask, need_weights=False)
            o = RG.mix(g, o, o_closed)
        x = x + o; x = x + self.mlp(self.ln2(x))
        return (x, w) if need_weights else x


class CausalTransformer(nn.Module):
    """Pre-LN decoder. `cut` is the residual stream the analyses treat as 'the representation'
    (after block `cut`, default 1 of 2); `readout_input` is the post-final-norm vector the
    unembedding reads."""

    def __init__(self, n_vocab=7, d=64, n_heads=4, n_layers=2, mlp=128, T=48, final_ln=True, seed=0,
                 gain: float | None = None, out_scale: float = 1.0, cut: int = 1,
                 final_norm: str | None = None, attn_diag=(), gated=False):
        """final_norm overrides final_ln: 'learn' (LayerNorm with trainable gamma/beta),
        'frozen' (LayerNorm with gamma == 1, beta == 0, not trainable) or 'none'.
        attn_diag: indices of blocks whose attention is hard-masked to the diagonal, so that
        block moves no information between positions (TASK10 route restriction).
        gated: give every block a read gate (TASK16, `readgate.py`)."""
        super().__init__(); torch.manual_seed(seed)
        self.final_norm = final_norm if final_norm is not None else ("learn" if final_ln else "none")
        assert self.final_norm in ("learn", "frozen", "none")
        self.emb = nn.Embedding(n_vocab, d); self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.gated = gated
        self.blocks = nn.ModuleList(Block(d, n_heads, mlp, gated) for _ in range(n_layers))
        self.ln_f = {"learn": lambda: nn.LayerNorm(d), "frozen": lambda: nn.LayerNorm(d, elementwise_affine=False),
                     "none": nn.Identity}[self.final_norm]()
        self.out = nn.Linear(d, n_vocab) if gain is None else FixedGainLinear(d, n_vocab, gain)
        if out_scale != 1.0:
            with torch.no_grad():
                for p in self.out.parameters():
                    p.mul_(out_scale)
        self.hidden_size = d; self.cut = cut; self.final_ln = self.final_norm != "none"
        self.attn_diag = tuple(attn_diag)

    def _mask(self, T, dev, block: int | None = None):
        if block is not None and block in self.attn_diag:                 # self-attention only
            return torch.where(torch.eye(T, dtype=torch.bool, device=dev), 0.0, float("-inf"))
        return torch.triu(torch.full((T, T), float("-inf"), device=dev), diagonal=1)

    def residuals(self, X, need_weights=False):
        """List of residual streams: [embedding, after block 1, ..., after block L], each [B, T, d].
        With need_weights, also returns the per-block head-averaged attention [B, T, T]."""
        X = X.to(self.pos.device); T = X.shape[1]
        x = self.emb(X) + self.pos[:T]; res = [x]; attn = []
        for i, b in enumerate(self.blocks):
            out = b(x, self._mask(T, x.device, i), need_weights=need_weights)
            x, w = out if need_weights else (out, None)
            res.append(x); attn.append(w)
        return (res, attn) if need_weights else res

    def _closed_mask(self, T, dev):
        """Self and the previous position only (round 16's rule for a hidden history)."""
        keep = torch.eye(T, dtype=torch.bool, device=dev) | torch.diag(torch.ones(T - 1, dtype=torch.bool, device=dev), -1)
        return torch.where(keep, 0.0, float("-inf"))

    def residuals_g(self, X, gate="open", gen=None):
        """Residual streams and the blocks' open probabilities p [B, T, L] (ones where no gate ran)."""
        X = X.to(self.pos.device); T = X.shape[1]
        x = self.emb(X) + self.pos[:T]; res = [x]; ps = []
        cm = self._closed_mask(T, x.device)
        for i, b in enumerate(self.blocks):
            x = b(x, self._mask(T, x.device, i), closed_mask=cm, gate=gate, gen=gen)
            res.append(x); ps.append(torch.ones(X.shape[0], T, device=x.device) if b._p is None else b._p)
        return res, torch.stack(ps, -1)

    def run_from(self, R, layer, return_res=False):
        """Continue the computation from residual stream R taken after block `layer`."""
        x = R; res = [R]
        for i, b in enumerate(self.blocks[layer:], start=layer):
            x = b(x, self._mask(R.shape[1], R.device, i)); res.append(x)
        z = self.out(self.ln_f(x))
        return (z, res) if return_res else z

    def states(self, X):
        return self.residuals(X)[self.cut]

    def hidden(self, X):
        with torch.no_grad():
            return self.states(X)[:, -1].cpu()

    def readout_input(self, X):
        return self.ln_f(self.residuals(X)[-1])

    def forward_all(self, X, gate="open", gen=None):
        return self.out(self.ln_f(self.residuals_g(X, gate, gen)[0][-1]))

    def forward(self, X):
        return self.forward_all(X)[:, -1]

    def patched(self, X, idx, t, h_new, return_res=False):
        """Run sequences X[idx] with the cut residual at position t[i] replaced by h_new[i]."""
        R = self.residuals(torch.as_tensor(X)[idx])[self.cut].clone()
        ar = torch.arange(len(idx), device=R.device)
        R[ar, torch.as_tensor(t, device=R.device)] = h_new.to(R.dtype)
        return self.run_from(R, self.cut, return_res=return_res)
