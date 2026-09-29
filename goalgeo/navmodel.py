"""Round 17: token format and the causal transformer for navigate / investigate / commit.

Sequence.  [CFG_H] [CFG_V] [DEC_0] [EVT_0] [DEC_1] [EVT_1] ...
  CFG_H, CFG_V   one public record per station: its cell and its reliability (a number).
  DEC_t          the observation before action t: own cell, remaining actions, station-used flags.
                 The next action is predicted here.
  EVT_t          the action taken and the report it returned (none, LEFT, RIGHT, TOP, BOTTOM).
Every token is the sum of the same set of field embeddings; a field that does not apply holds its
NONE index (0). A report is written once, in the event token of the query that produced it, so the
belief at a later decision token has to be assembled from earlier tokens. The posterior, the hidden
goal and solver values are never inputs.

The transformer is pre-LN with learned absolute positions and full causal attention. `forward` can
return every sublayer's contribution and every head's output, and can patch them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, CFG_H, CFG_V, DEC, EVT = range(5)
F_TYPE, F_CELL, F_REM, F_UH, F_UV, F_ACT, F_REP = range(7)      # integer fields; 0 = none everywhere but type
NF = 7
REP_NONE, REP_L, REP_R, REP_T, REP_B = range(5)


def seq_len(H):
    return 2 + 2 * H


def dec_pos(t):
    return 2 + 2 * t


class Embed(nn.Module):
    def __init__(self, d, n_cells, H, L):
        super().__init__()
        sizes = (5, n_cells + 1, H + 2, 3, 3, 7, 5)
        self.f = nn.ModuleList([nn.Embedding(s, d) for s in sizes])
        self.q = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Embedding(L, d)
        for e in self.f:
            nn.init.normal_(e.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, tok, q):
        """tok [N, L, NF] long; q [N, L] float, 0 where no reliability is reported."""
        x = sum(e(tok[..., i]) for i, e in enumerate(self.f))
        x = x + self.q(q[..., None]) * (q > 0)[..., None]
        return x + self.pos.weight[: tok.shape[1]]


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.d = heads, d
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attn(self, x, want=False):
        """Returns the attention sublayer's output and, if wanted, the pattern [N, h, L, L] and the
        per-head outputs [N, h, L, d] (they sum, with the output bias, to the sublayer's output)."""
        N, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(N, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h)
        s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1), -torch.inf)
        A = s.softmax(-1)
        z = A @ v                                                     # [N, h, L, d/h]
        if not want:
            return self.o(z.transpose(1, 2).reshape(N, L, d)), None, None, None
        W = self.o.weight.view(d, self.h, d // self.h)                # out, head, in
        heads = torch.einsum("nhld,ohd->nhlo", z, W)
        return heads.sum(1) + self.o.bias, A, heads, v

    def source_terms(self, A, v, p):
        """What each source token sends to position p[n] through each head: [N, h, L, d]; summed over sources
        it is that head's output at p."""
        n = torch.arange(A.shape[0], device=A.device)
        W = self.o.weight.view(self.d, self.h, self.d // self.h)
        return torch.einsum("nhs,nhsd,ohd->nhso", A[n, :, p], v, W)


class Net(nn.Module):
    def __init__(self, n_cells=25, H=12, d=128, layers=4, heads=4):
        super().__init__()
        self.H, self.d, self.nl, self.nh = H, d, layers, heads
        self.embed = Embed(d, n_cells, H, seq_len(H))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.pi = nn.Linear(d, 6)
        self.v = nn.Linear(d, 1)
        nn.init.normal_(self.pi.weight, std=0.01); nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.v.weight, std=0.1); nn.init.zeros_(self.v.bias)

    def backbone(self):
        return list(self.embed.parameters()) + list(self.blocks.parameters()) + list(self.ln.parameters())

    def forward(self, tok, q, legal=None, record=False, patch=None):
        """logits [N, L, 6], value [N, L]. With record=True also a dict of activations:
            value[l]  per-head values [N, h, L, d / h]
            resid[l]  residual stream entering block l (l = layers: the final stream)      [N, L, d]
            mid[l]    after block l's attention, before its MLP
            attn[l], mlp[l]  the two contributions;  heads[l] [N, h, L, d];  pattern[l] [N, h, L, L]
        patch: {("attn" | "mlp" | "head" | "resid_mid" | "resid", layer[, head]): fn(tensor) -> tensor}."""
        patch = patch or {}
        x = self.embed(tok, q)
        rec = dict(resid=[], mid=[], attn=[], mlp=[], heads=[], pattern=[], value=[]) if record else None
        for l, b in enumerate(self.blocks):
            if ("resid", l) in patch:
                x = patch[("resid", l)](x)
            need = record or any(k[0] == "head" and k[1] == l for k in patch)
            a, A, hs, val = b.attn(x, want=need)
            for k, fn in patch.items():
                if k[0] == "head" and k[1] == l:
                    new = fn(hs[:, k[2]])
                    a = a + new - hs[:, k[2]]
                    hs = torch.cat([hs[:, :k[2]], new[:, None], hs[:, k[2] + 1:]], 1)
            if ("attn", l) in patch:
                a = patch[("attn", l)](a)
            mid = x + a
            if ("resid_mid", l) in patch:
                mid = patch[("resid_mid", l)](mid)
            m = b.mlp(b.ln2(mid))
            if ("mlp", l) in patch:
                m = patch[("mlp", l)](m)
            if record:
                rec["resid"].append(x); rec["mid"].append(mid); rec["attn"].append(a); rec["mlp"].append(m)
                rec["heads"].append(hs); rec["pattern"].append(A); rec["value"].append(val)
            x = mid + m
        if ("resid", self.nl) in patch:
            x = patch[("resid", self.nl)](x)
        if record:
            rec["resid"].append(x)
        h = self.ln(x)
        logits = self.pi(h)
        if legal is not None:
            logits = logits.masked_fill(~legal, -1e9)
        out = (logits, self.v(h).squeeze(-1))
        return out + (rec,) if record else out
