"""TASK16 (round 17): a learned, priced read of the history (rounds/r17_read_cost/THEORY.md).

One binary gate per (block, query position) decides whether the historical keys are visible to
that query. The block computes its attention output as g * open + (1 - g) * closed, so the forward
pass is exactly the masked computation and the gate logit receives the straight-through gradient
of a Gumbel-sigmoid relaxation. P(open) = sigmoid(a) exactly (the noise is standard logistic), so
the price c * mean sigmoid(a) is the expected read rate."""

from __future__ import annotations

import torch

TAU = 0.5        # temperature of the relaxed gate (gradient sharpness only)
B_INIT = 3.0     # gate bias at initialisation: reads open with probability 0.95


def gate(a, train: bool, tau: float = TAU, gen=None):
    """a: gate logits [...]. Returns (g, p): g in {0, 1} with the same shape, p = sigmoid(a).
    train: g = 1[a + logistic noise > 0] with the gradient of sigmoid((a + noise) / tau);
    else g = 1[a > 0] with no gradient."""
    p = torch.sigmoid(a)
    if not train:
        return (a > 0).to(a.dtype), p
    u = torch.rand(a.shape, generator=gen, device=a.device, dtype=a.dtype).clamp(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((a + torch.log(u) - torch.log1p(-u)) / tau)
    hard = (soft > 0.5).to(a.dtype)
    return hard + (soft - soft.detach()), p


def mix(g, o_open, o_closed):
    """g [...] broadcast over the last axis of the two attention outputs."""
    return g[..., None] * o_open + (1 - g[..., None]) * o_closed


def penalty(p, first: int = 2):
    """Mean open probability over blocks and positions >= first; p [B, T, L]."""
    return p[:, first:].mean()
