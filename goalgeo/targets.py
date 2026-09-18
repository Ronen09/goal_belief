"""Supervision targets built from the same underlying Q, and a generic trainer.

Target kinds: hard (uniform over argmax set), boltzmann(τ) = softmax(Q/τ),
advantage (MSE on Q − max Q), q (MSE on Q), occupancy (MSE on Z_sa(s)).
Losses: ce, kl (same gradient as ce), mse_prob, mse_logit, mse.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .gridworld import GridWorld
from .occupancy import Occupancy
from .planning import Solution


@dataclass
class Targets:
    kind: str
    tau: float | None
    s: np.ndarray          # state index per row
    g: np.ndarray          # goal index per row
    Y: np.ndarray          # [n, out_dim]
    out_dim: int
    loss: str              # default loss for this target kind
    Q: np.ndarray          # [n, A] underlying Q rows (for reference)


def boltzmann(Q: np.ndarray, tau: float) -> np.ndarray:
    z = Q / tau
    p = np.exp(z - z.max(-1, keepdims=True))
    return p / p.sum(-1, keepdims=True)


def build_targets(env: GridWorld, sol: Solution, occ: Occupancy, kind: str, tau: float | None = None) -> Targets:
    rows_s, rows_g = [], []
    for gi, g in enumerate(env.goal_states):
        for s in range(env.n_states):
            if s != g:
                rows_s.append(s); rows_g.append(gi)
    s_idx, g_idx = np.array(rows_s), np.array(rows_g)
    Q = sol.Q[g_idx, s_idx]                                    # [n, A]
    if kind == "hard":
        Y, loss = sol.pi[g_idx, s_idx], "ce"
    elif kind == "boltzmann":
        Y, loss = boltzmann(Q, tau), "ce"
    elif kind == "advantage":
        Y, loss = Q - Q.max(-1, keepdims=True), "mse"
    elif kind == "q":
        Y, loss = Q, "mse"
    elif kind == "occupancy":
        Y, loss = occ.Z_sa[s_idx], "mse"
    else:
        raise ValueError(kind)
    return Targets(kind, tau, s_idx, g_idx, np.asarray(Y, dtype=np.float64), Y.shape[1], loss, Q)


def targets_from_Q(Q: np.ndarray, kind: str, tau: float | None = None) -> np.ndarray:
    """Apply the target transformation T to arbitrary Q rows (synthetic tasks)."""
    if kind == "hard":
        best = np.isclose(Q, Q.max(-1, keepdims=True), atol=1e-9)
        return best / best.sum(-1, keepdims=True)
    if kind == "boltzmann":
        return boltzmann(Q, tau)
    if kind == "advantage":
        return Q - Q.max(-1, keepdims=True)
    if kind == "q":
        return Q
    raise ValueError(kind)


# ---- losses ----------------------------------------------------------------
def _ce(logits, p):
    return -(p * F.log_softmax(logits, -1)).sum(-1).mean()


def _kl(logits, p):
    return (p * (torch.log(p.clamp_min(1e-12)) - F.log_softmax(logits, -1))).sum(-1).mean()


def _mse_prob(logits, p):
    return ((torch.softmax(logits, -1) - p) ** 2).sum(-1).mean()


def _mse_logit(logits, target_logits):
    lc = logits - logits.mean(-1, keepdim=True)
    tc = target_logits - target_logits.mean(-1, keepdim=True)
    return ((lc - tc) ** 2).sum(-1).mean()


def _mse(out, y):
    return ((out - y) ** 2).sum(-1).mean()


LOSSES = {"ce": _ce, "kl": _kl, "mse_prob": _mse_prob, "mse_logit": _mse_logit, "mse": _mse}


@dataclass
class History:
    loss: list[float] = field(default_factory=list)
    checkpoints: dict[int, dict] = field(default_factory=dict)


def train_targets(net, tg: Targets, steps: int = 3000, lr: float = 1e-3, loss: str | None = None,
                  checkpoint_steps=None, seed: int = 0, rows: np.ndarray | None = None) -> History:
    """Full-batch Adam on the given targets. ``rows`` restricts training to a subset
    of rows (generalisation experiments). For ``mse_logit`` the target is Q/τ."""
    torch.manual_seed(seed)
    loss = loss or tg.loss
    sel = np.arange(len(tg.s)) if rows is None else rows
    s_t, g_t = torch.as_tensor(tg.s[sel]), torch.as_tensor(tg.g[sel])
    if loss == "mse_logit":
        y_t = torch.as_tensor(tg.Q[sel] / (tg.tau or 1.0), dtype=torch.float32)
    else:
        y_t = torch.as_tensor(tg.Y[sel], dtype=torch.float32)
    fn = LOSSES[loss]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = History(); ck = set(checkpoint_steps or [])
    if 0 in ck:
        hist.checkpoints[0] = copy.deepcopy(net.state_dict())
    for step in range(1, steps + 1):
        out = net(s_t, g_t)
        l = fn(out, y_t)
        opt.zero_grad(); l.backward(); opt.step()
        hist.loss.append(l.item())
        if step in ck:
            hist.checkpoints[step] = copy.deepcopy(net.state_dict())
    return hist


@torch.no_grad()
def argmax_accuracy(net, tg: Targets, rows: np.ndarray | None = None) -> float:
    """Fraction of rows whose predicted argmax is an optimal action (argmax set of Q)."""
    sel = np.arange(len(tg.s)) if rows is None else rows
    out = net(torch.as_tensor(tg.s[sel]), torch.as_tensor(tg.g[sel])).numpy()
    best = np.isclose(tg.Q[sel], tg.Q[sel].max(-1, keepdims=True), atol=1e-9)
    if tg.kind == "advantage" or tg.kind == "q":
        pred = out.argmax(-1)
    elif tg.kind == "occupancy":
        return float("nan")
    else:
        pred = out.argmax(-1)
    return float(best[np.arange(len(sel)), pred].mean())
