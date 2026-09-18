"""Architectures for TASK3 (all share the interface: activations(s, g) -> dict with
hidden layers and 'logits'; hidden_layers; all_activations()) and a synthetic task."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .model import PolicyNet


class _Base(nn.Module):
    def _setup(self, env, emb_dim, seed):
        torch.manual_seed(seed)
        self.env = env
        self.register_buffer("goal_cells", torch.tensor(env.goal_states))
        self.state_emb = nn.Embedding(env.n_states, emb_dim)
        self.goal_emb = nn.Embedding(env.n_states, emb_dim)

    def forward(self, s, g):
        return self.activations(s, g)["logits"]

    @torch.no_grad()
    def all_activations(self):
        S, K = self.env.n_states, self.env.n_goals
        s = torch.arange(S).repeat_interleave(K); g = torch.arange(K).repeat(S)
        acts = self.activations(s, g)
        return {k: v.reshape(S, K, -1).cpu().numpy() for k, v in acts.items()}


class DeepMLP(_Base):
    def __init__(self, env, out_dim=4, emb_dim=32, hidden=128, n_hidden=6, seed=0):
        super().__init__(); self._setup(env, emb_dim, seed)
        dims = [2 * emb_dim] + [hidden] * n_hidden
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_hidden))
        self.out = nn.Linear(hidden, out_dim)
        self.hidden_layers = [f"h{i + 1}" for i in range(n_hidden)]

    def activations(self, s, g):
        x = torch.cat([self.state_emb(s), self.goal_emb(self.goal_cells[g])], -1)
        acts = {"emb": x}
        for i, l in enumerate(self.layers):
            x = torch.relu(l(x)); acts[f"h{i + 1}"] = x
        acts["logits"] = self.out(x)
        return acts


class ResMLP(_Base):
    def __init__(self, env, out_dim=4, emb_dim=32, hidden=128, n_blocks=3, seed=0):
        super().__init__(); self._setup(env, emb_dim, seed)
        self.inp = nn.Linear(2 * emb_dim, hidden)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, out_dim)
        self.hidden_layers = [f"h{i + 1}" for i in range(n_blocks)]

    def activations(self, s, g):
        x = torch.relu(self.inp(torch.cat([self.state_emb(s), self.goal_emb(self.goal_cells[g])], -1)))
        acts = {"emb": x}
        for i, b in enumerate(self.blocks):
            x = x + b(x); acts[f"h{i + 1}"] = x
        acts["logits"] = self.out(x)
        return acts


class TinyTransformer(_Base):
    def __init__(self, env, out_dim=4, emb_dim=64, hidden=64, n_layers=2, n_heads=4, seed=0):
        super().__init__(); self._setup(env, hidden, seed)
        self.pos = nn.Parameter(torch.zeros(2, hidden)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True)
        self.enc = nn.ModuleList(nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=2 * hidden, dropout=0.0, batch_first=True) for _ in range(n_layers))
        self.out = nn.Linear(hidden, out_dim)
        self.hidden_layers = [f"h{i + 1}" for i in range(n_layers)]

    def activations(self, s, g):
        x = torch.stack([self.state_emb(s), self.goal_emb(self.goal_cells[g])], 1) + self.pos  # [B, 2, d]
        acts = {"emb": x[:, 0]}
        for i, l in enumerate(self.enc):
            x = l(x); acts[f"h{i + 1}"] = x[:, 0]
        acts["logits"] = self.out(x[:, 0])
        return acts


def make_model(arch: str, env, out_dim: int = 4, hidden: int = 128, emb_dim: int = 32, seed: int = 0):
    if arch == "mlp3":
        net = PolicyNet(env, emb_dim=emb_dim, hidden=hidden, n_hidden=3, seed=seed, out_dim=out_dim)
        net.hidden_layers = ["h1", "h2", "h3"]
        return net
    if arch == "mlp6":
        return DeepMLP(env, out_dim, emb_dim, hidden, 6, seed)
    if arch == "resmlp":
        return ResMLP(env, out_dim, emb_dim, hidden, 3, seed)
    if arch == "transformer":
        return TinyTransformer(env, out_dim, hidden=max(64, hidden // 2), seed=seed)
    raise ValueError(arch)


class SyntheticTask:
    """One-hot 'states' with prescribed Q vectors: ``n_background`` random rows, a
    reference row with margin 0.5 and one probe row per requested margin sharing the
    reference's argmax. A single dummy goal so the models' interface is unchanged."""

    def __init__(self, n_background: int = 200, margins=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), seed: int = 0, n_actions: int = 4):
        rng = np.random.default_rng(seed)
        self.margins = list(margins)
        bg = rng.uniform(0, 1, size=(n_background, n_actions))
        ref = np.array([0.9, 0.4, 0.2, 0.1])
        probes = np.stack([np.array([0.9, 0.9 - m, 0.2, 0.1]) for m in self.margins])
        self.Q = np.vstack([bg, ref[None], probes])
        self.n_states = len(self.Q); self.n_goals = 1; self.n_actions = n_actions
        self.reference = n_background
        self.probes = list(range(n_background + 1, self.n_states))
        self.background = list(range(n_background))
        self.goal_states = [0]                       # dummy; never excluded
        self.coords = np.c_[np.arange(self.n_states), np.zeros(self.n_states)].astype(float)
        self.height, self.width = self.n_states, 1

    def rows(self):
        return np.arange(self.n_states), np.zeros(self.n_states, int)
