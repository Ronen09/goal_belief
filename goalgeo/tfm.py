"""TASK9: a small pre-LN causal transformer for the HMM next-token task, exposing the same
interface the GRU analyses use (states / forward_all / out) plus residual-stream patching,
which is the transformer's version of 'push a state through the remaining computation'."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seqmodels import FixedGainLinear


class Block(nn.Module):
    def __init__(self, d, n_heads, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))

    def forward(self, x, mask):
        a = self.ln1(x); x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class CausalTransformer(nn.Module):
    """Pre-LN decoder. `cut` is the residual stream the analyses treat as 'the representation'
    (after block `cut`, default 1 of 2); `readout_input` is the post-final-norm vector the
    unembedding reads."""

    def __init__(self, n_vocab=7, d=64, n_heads=4, n_layers=2, mlp=128, T=48, final_ln=True, seed=0,
                 gain: float | None = None, out_scale: float = 1.0, cut: int = 1):
        super().__init__(); torch.manual_seed(seed)
        self.emb = nn.Embedding(n_vocab, d); self.pos = nn.Parameter(torch.randn(T, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, n_heads, mlp) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d) if final_ln else nn.Identity()
        self.out = nn.Linear(d, n_vocab) if gain is None else FixedGainLinear(d, n_vocab, gain)
        if out_scale != 1.0:
            with torch.no_grad():
                for p in self.out.parameters():
                    p.mul_(out_scale)
        self.hidden_size = d; self.cut = cut; self.final_ln = final_ln

    def _mask(self, T, dev):
        return torch.triu(torch.full((T, T), float("-inf"), device=dev), diagonal=1)

    def residuals(self, X):
        """List of residual streams: [embedding, after block 1, ..., after block L], each [B, T, d]."""
        X = X.to(self.pos.device); T = X.shape[1]
        x = self.emb(X) + self.pos[:T]; res = [x]; mask = self._mask(T, x.device)
        for b in self.blocks:
            x = b(x, mask); res.append(x)
        return res

    def run_from(self, R, layer, return_res=False):
        """Continue the computation from residual stream R taken after block `layer`."""
        x = R; res = [R]; mask = self._mask(R.shape[1], R.device)
        for b in self.blocks[layer:]:
            x = b(x, mask); res.append(x)
        z = self.out(self.ln_f(x))
        return (z, res) if return_res else z

    def states(self, X):
        return self.residuals(X)[self.cut]

    def hidden(self, X):
        with torch.no_grad():
            return self.states(X)[:, -1].cpu()

    def readout_input(self, X):
        return self.ln_f(self.residuals(X)[-1])

    def forward_all(self, X):
        return self.out(self.readout_input(X))

    def forward(self, X):
        return self.forward_all(X)[:, -1]

    def patched(self, X, idx, t, h_new, return_res=False):
        """Run sequences X[idx] with the cut residual at position t[i] replaced by h_new[i]."""
        R = self.residuals(torch.as_tensor(X)[idx])[self.cut].clone()
        ar = torch.arange(len(idx), device=R.device)
        R[ar, torch.as_tensor(t, device=R.device)] = h_new.to(R.dtype)
        return self.run_from(R, self.cut, return_res=return_res)
