"""Goal-conditioned policy network: state embedding + goal embedding -> MLP -> action logits.

``encoding="onehot"`` uses learned embedding tables (no spatial prior);
``encoding="coord"`` feeds normalised (row, col) through a linear layer.
Hidden activations are exposed per layer and the network can be re-run from
any layer with a patched activation (for causal interventions).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .gridworld import GridWorld

LAYERS = ["emb", "h1", "h2", "h3", "logits"]


class PolicyNet(nn.Module):
    def __init__(self, env: GridWorld, encoding: str = "onehot", emb_dim: int = 32,
                 hidden: int = 128, n_hidden: int = 3, seed: int = 0, out_dim: int | None = None):
        super().__init__()
        torch.manual_seed(seed)
        self.env = env
        self.encoding = encoding
        self.n_hidden = n_hidden
        coords = torch.tensor(env.coords, dtype=torch.float32)
        coords = coords / torch.tensor([max(env.height - 1, 1), max(env.width - 1, 1)])
        self.register_buffer("coords", coords)
        self.register_buffer("goal_cells", torch.tensor(env.goal_states))
        if encoding == "onehot":
            self.state_emb = nn.Embedding(env.n_states, emb_dim)
            self.goal_emb = nn.Embedding(env.n_states, emb_dim)
        elif encoding == "coord":
            self.state_emb = nn.Linear(2, emb_dim)
            self.goal_emb = nn.Linear(2, emb_dim)
        else:
            raise ValueError(encoding)
        dims = [2 * emb_dim] + [hidden] * n_hidden
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(n_hidden))
        self.out = nn.Linear(hidden, out_dim or env.n_actions)

    # goal index (0..K-1) -> cell index
    def _embed(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        g_cell = self.goal_cells[g]
        if self.encoding == "onehot":
            return torch.cat([self.state_emb(s), self.goal_emb(g_cell)], dim=-1)
        return torch.cat([self.state_emb(self.coords[s]), self.goal_emb(self.coords[g_cell])], dim=-1)

    def activations(self, s: torch.Tensor, g: torch.Tensor) -> dict[str, torch.Tensor]:
        acts = {"emb": self._embed(s, g)}
        x = acts["emb"]
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x))
            acts[f"h{i + 1}"] = x
        acts["logits"] = self.out(x)
        return acts

    def forward(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.activations(s, g)["logits"]

    def forward_from(self, layer: str, x: torch.Tensor) -> torch.Tensor:
        """Continue the forward pass from a (possibly patched) activation at ``layer``."""
        start = 0 if layer == "emb" else int(layer[1:])
        for i in range(start, self.n_hidden):
            x = torch.relu(self.layers[i](x))
        return self.out(x)

    @torch.no_grad()
    def all_activations(self) -> dict[str, np.ndarray]:
        """Activations for every (state, goal) pair: layer -> [S, K, d]."""
        S, K = self.env.n_states, self.env.n_goals
        s = torch.arange(S).repeat_interleave(K)
        g = torch.arange(K).repeat(S)
        acts = self.activations(s, g)
        return {k: v.reshape(S, K, -1).cpu().numpy() for k, v in acts.items()}
