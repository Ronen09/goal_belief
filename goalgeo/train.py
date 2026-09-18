"""Behavioural cloning of the optimal goal-conditioned policies."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .model import PolicyNet


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    acc: list[float] = field(default_factory=list)
    checkpoints: dict[int, dict] = field(default_factory=dict)  # step -> state_dict


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def train_bc(net: PolicyNet, s: np.ndarray, g: np.ndarray, y: np.ndarray, steps: int = 3000,
             lr: float = 1e-3, checkpoint_steps: list[int] | None = None, seed: int = 0) -> History:
    torch.manual_seed(seed)
    s_t, g_t = torch.as_tensor(s), torch.as_tensor(g)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = History()
    ckpts = set(checkpoint_steps or [])
    if 0 in ckpts:
        hist.checkpoints[0] = copy.deepcopy(net.state_dict())
    for step in range(1, steps + 1):
        logits = net(s_t, g_t)
        loss = soft_cross_entropy(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.loss.append(loss.item())
        with torch.no_grad():
            hist.acc.append(float((y_t.gather(1, logits.argmax(1, keepdim=True)) > 0).float().mean()))
        if step in ckpts:
            hist.checkpoints[step] = copy.deepcopy(net.state_dict())
    return hist


@torch.no_grad()
def evaluate_accuracy(net: PolicyNet, s: np.ndarray, g: np.ndarray, y: np.ndarray) -> float:
    """Fraction of (s,g) pairs whose argmax action is one of the optimal actions."""
    logits = net(torch.as_tensor(s), torch.as_tensor(g))
    y_t = torch.as_tensor(y, dtype=torch.float32)
    return float((y_t.gather(1, logits.argmax(1, keepdim=True)) > 0).float().mean())
