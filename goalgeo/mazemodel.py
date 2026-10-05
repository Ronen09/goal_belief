"""The maze-belief experiment: token format and transformer for the maze task.

Sequence.  [OBS o_0] [EVT a_0 o_1] ... [EVT a_{L-1} o_L] [GOAL g] [EVT a o] [EVT a o] ...
  OBS    the first symbol.
  EVT    a move and the symbol seen after it. In the prefix the move was imposed; after the goal token it is the
         model's own. The format is the same; events after the reveal also carry the number of moves left.
  GOAL   the goal's name. It appears once.
The next move is predicted at the goal token and at every event token after it. The true cell, the posterior and
the map are never inputs. The transformer is the navigate-commit experiment's (pre-LN, learned positions, full causal attention).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from goalgeo.navmodel import Block

PAD, OBS, EVT, GOAL = range(4)
F_TYPE, F_SYM, F_ACT, F_GOAL, F_REM = range(5)      # 0 = none in every field but type
NF = 5


def seq_len(H, max_prefix):
    return 1 + max_prefix + 1 + H


class Embed(nn.Module):
    def __init__(self, d, n_sym, n_goals, H, L):
        super().__init__()
        self.f = nn.ModuleList([nn.Embedding(s, d) for s in (4, n_sym + 1, 5, n_goals + 1, H + 2)])
        self.pos = nn.Embedding(L, d)
        for e in list(self.f) + [self.pos]:
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, tok):
        return sum(e(tok[..., i]) for i, e in enumerate(self.f)) + self.pos.weight[: tok.shape[1]]


class Net(nn.Module):
    def __init__(self, n_sym=3, n_goals=3, H=12, max_prefix=4, d=128, layers=4, heads=4):
        super().__init__()
        self.H, self.d, self.nl, self.nh = H, d, layers, heads
        self.embed = Embed(d, n_sym, n_goals, H, seq_len(H, max_prefix))
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.pi = nn.Linear(d, 4)
        self.v = nn.Linear(d, 1)
        nn.init.normal_(self.pi.weight, std=0.01); nn.init.zeros_(self.pi.bias)
        nn.init.normal_(self.v.weight, std=0.1); nn.init.zeros_(self.v.bias)

    def forward(self, tok, q=None, record=False, patch=None):
        """logits [N, L, 4], value [N, L]; with record=True also the activations (as in navmodel.Net).
        patch: {("attn" | "mlp" | "head" | "resid_mid" | "resid", layer[, head]): fn(tensor) -> tensor}. `q` is unused."""
        patch = patch or {}
        x = self.embed(tok)
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
        out = (self.pi(h), self.v(h).squeeze(-1))
        return out + (rec,) if record else out
