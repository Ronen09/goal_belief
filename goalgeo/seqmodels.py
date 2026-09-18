"""GRU and window-MLP models trained under one-step, k-step or sequential objectives (TASK4)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import hmm as Hm


def _dev(module):
    return next(module.parameters()).device


class FixedGainLinear(nn.Module):
    """Linear readout whose weight rows have a fixed norm ``gain``: the direction of each
    row is learned, its scale is not; the bias is free (TASK5, readout-scale experiment)."""

    def __init__(self, d_in, d_out, gain: float):
        super().__init__()
        self.v = nn.Parameter(torch.empty(d_out, d_in).uniform_(-1, 1) / d_in ** 0.5)
        self.bias = nn.Parameter(torch.empty(d_out).uniform_(-1, 1) / d_in ** 0.5)
        self.gain = float(gain)

    @property
    def weight(self):
        return self.gain * self.v / self.v.norm(dim=1, keepdim=True)

    def forward(self, h):
        return F.linear(h, self.weight, self.bias)


class SeqNet(nn.Module):
    def __init__(self, n_vocab=4, hidden=64, emb=16, out_dim=4, seed=0, gain: float | None = None, out_scale: float = 1.0):
        super().__init__(); torch.manual_seed(seed)
        self.emb = nn.Embedding(n_vocab, emb); self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, out_dim) if gain is None else FixedGainLinear(hidden, out_dim, gain)
        if out_scale != 1.0:                       # TASK6: readout initialisation scale
            with torch.no_grad():
                for p in self.out.parameters():
                    p.mul_(out_scale)
        self.hidden_size = hidden

    def states(self, X):                    # [B, T] -> [B, T, H]
        h, _ = self.gru(self.emb(X.to(_dev(self)))); return h

    def hidden(self, X):                    # last hidden state [B, H], on CPU
        with torch.no_grad():
            return self.states(X)[:, -1].cpu()

    def forward(self, X):                   # logits at the last position
        return self.out(self.states(X)[:, -1])

    def forward_all(self, X):               # logits at every position
        return self.out(self.states(X))


class WindowMLP(nn.Module):
    def __init__(self, n_vocab=4, L=6, hidden=64, out_dim=4, seed=0):
        super().__init__(); torch.manual_seed(seed)
        self.n_vocab, self.L = n_vocab, L
        self.net = nn.Sequential(nn.Linear(n_vocab * L, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, out_dim); self.hidden_size = hidden

    def _inp(self, X):
        return F.one_hot(X[:, -self.L:].to(_dev(self)), self.n_vocab).float().reshape(len(X), -1)

    def hidden(self, X):
        with torch.no_grad():
            return self.net(self._inp(X)).cpu()

    def forward(self, X):
        return self.out(self.net(self._inp(X)))


def _soft_ce(logits, p):
    return -(p * F.log_softmax(logits, -1)).sum(-1).mean()


def train_objective(net, m: Hm.HMM, objective: str, k: int = 2, steps: int = 3000, batch: int = 128, seed: int = 0,
                    L: int = 6, T: int = 32, lr: float = 3e-3, targets: str = "exact", kind: str = "joint") -> list[float]:
    """objective: 'one-step' (k=1 window), 'k-step' (joint or marginal window targets),
    'sequential' (next-token loss at every position of length-T sequences)."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    dev = _dev(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist = []
    # pre-sample a pool of sequences and reuse windows from it
    pool_X, _ = Hm.sample(m, n=4000, T=T, seed=seed)
    pool_Y = Hm.sequential_targets(m, pool_X) if (objective == "sequential" and targets == "exact") else None
    for step in range(steps):
        idx = rng.integers(0, len(pool_X), batch)
        if objective == "sequential":
            X = pool_X[idx]
            Xt = torch.as_tensor(X, device=dev)
            logits = net.forward_all(Xt)[:, :-1]
            if targets == "exact":
                Y = torch.as_tensor(pool_Y[idx][:, :-1], dtype=torch.float32, device=dev)
                loss = _soft_ce(logits.reshape(-1, logits.shape[-1]), Y.reshape(-1, Y.shape[-1]))
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), Xt[:, 1:].reshape(-1))
        else:
            kk = 1 if objective == "one-step" else k
            start = rng.integers(0, T - L - kk + 1, batch)
            W = np.stack([pool_X[i, s:s + L] for i, s in zip(idx, start)])
            if targets == "exact":
                Y = torch.as_tensor(Hm.window_targets(m, W, kk, kind), dtype=torch.float32, device=dev)
                logits = net(torch.as_tensor(W, device=dev))
                if kind == "marginals" and kk > 1:
                    loss = sum(_soft_ce(logits[:, j * Hm.V:(j + 1) * Hm.V], Y[:, j * Hm.V:(j + 1) * Hm.V]) for j in range(kk))
                else:
                    loss = _soft_ce(logits, Y)
            else:  # sampled continuation as a class index (joint)
                fut = np.stack([pool_X[i, s + L:s + L + kk] for i, s in zip(idx, start)])
                y = torch.as_tensor(sum(fut[:, j] * Hm.V ** (kk - 1 - j) for j in range(kk)), device=dev)
                loss = F.cross_entropy(net(torch.as_tensor(W, device=dev)), y)
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
    return hist


# --- TASK4 (geometric prominence): weighted sequential training with checkpoints ---
from . import hmm4 as H4  # noqa: E402  (placed here to keep the round 4 section untouched)

CHECKPOINTS = (0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000)


def weighted_seq_loss(net, X, Y, W):
    """Per-position soft cross-entropy, weighted per position.
    X [B, T] long tokens; Y [B, T-1, V] exact targets for tokens 1..T-1; W [B, T-1] weights."""
    logits = net.forward_all(X)[:, :-1]
    ce = -(Y * F.log_softmax(logits, -1)).sum(-1)
    return (W * ce).sum() / W.sum()


def train_weighted(net, m: H4.HMM4, lam: float = 1.0, steps: int = 3000, batch: int = 128, seed: int = 0,
                   T: int = 48, lr: float = 3e-3, pool: int = 4000, checkpoints=CHECKPOINTS, lr_out: float | None = None):
    """Sequential next-token training with exact targets. Positions whose *next* token is
    emitted by a relevant state (C, D, N) get weight ``lam``; all others weight 1.
    Returns (loss history, {step: CPU state_dict copy}) with a copy at every listed step."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed); dev = _dev(net)
    X, Z = H4.sample(m, pool, T, seed=seed)
    Y = H4.next_token(m, H4.beliefs_seq(m, X))[:, :-1]
    W = np.where(m.relevant[Z[:, 1:]], lam, 1.0)
    Xt = torch.as_tensor(X, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
    out_params = list(net.out.parameters()); out_ids = {id(p) for p in out_params}
    groups = [{"params": [p for p in net.parameters() if id(p) not in out_ids], "lr": lr},
              {"params": out_params, "lr": lr if lr_out is None else lr_out}]   # TASK6: separate readout learning rate
    opt = torch.optim.Adam(groups, lr=lr); hist = []; ckpt = {}
    for step in range(steps + 1):
        if step in checkpoints:
            ckpt[step] = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if step == steps:
            break
        idx = torch.as_tensor(rng.integers(0, pool, batch), device=dev)
        loss = weighted_seq_loss(net, Xt[idx], Yt[idx], Wt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(loss.item())
    return hist, ckpt
