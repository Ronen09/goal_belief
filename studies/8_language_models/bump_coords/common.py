"""Shared pieces of the bump coordinates study (PLAN.md): the sites, ridge maps from the state under one goal (plus
the one-hot step) to the state under another, rank-k projectors from the singular directions of the per-history goal
difference, the causal measures (switch / keep with the prompt under the source goal), the soft set probe's Jaccard,
bootstrap helpers, the swap-pair construction of the goal-swap study, and (part 2) the fine-tuned four-goal model with
hooked forward passes and the two decision pools. Scripts: collect.py, compose.py, rank.py, preserve.py, tables.py.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_coords .venv/bin/python studies/8_language_models/bump_coords/<script>.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
SITES = (15, 19, 21, 23)                 # residual entering layer l (index 0 = embeddings), as factorial_llm.Patched records it
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 896)
K, H, D = 4, 40, 896
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


# ------------------------------------------------------------------ bootstrap
def boot_mean(x, n=1000, seed=0):
    """95 % percentile interval of the mean over resamples (nans dropped)."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]; rng = np.random.default_rng(seed)
    if len(x) == 0: return [float("nan"), float("nan")]
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_ratio(num, den, n=1000, seed=0):
    """95 % percentile interval of mean(num) / mean(den) over paired resamples."""
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float); rng = np.random.default_rng(seed)
    if len(num) == 0: return [float("nan"), float("nan")]
    b = []
    for _ in range(n):
        i = rng.integers(len(num), size=len(num)); d = den[i].mean(); b.append(num[i].mean() / d if d != 0 else float("nan"))
    b = np.asarray(b); b = b[~np.isnan(b)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) else [float("nan"), float("nan")]


# ------------------------------------------------------------------ maps
def step_onehot(steps):
    return torch.nn.functional.one_hot(torch.as_tensor(steps).long().clamp(max=H - 1), H).double()


def _design(X, steps):
    X = torch.as_tensor(X).double(); return torch.cat([X, step_onehot(steps), torch.ones(len(X), 1, dtype=torch.float64)], 1)


def ridge(X, steps, Y, lam):
    """W [d + H + 1, d'] with (X1ᵀ X1 + λ I) W = X1ᵀ Y, X1 = [X, one-hot step, 1]."""
    X1 = _design(X, steps); Y = torch.as_tensor(Y).double()
    return torch.linalg.solve(X1.T @ X1 + lam * torch.eye(X1.shape[1], dtype=torch.float64), X1.T @ Y)


def apply_map(W, X, steps):
    return _design(X, steps) @ W


def r2(pred, target):
    pred, target = torch.as_tensor(pred).double(), torch.as_tensor(target).double()
    return float(1 - ((pred - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum())


def tune_lambda(X, steps, Y, lambdas=LAMBDAS, seed=0):
    """λ with the best state R² on a held-out quarter of the rows; returns (λ, {λ: R²})."""
    n = len(X); rng = np.random.default_rng(seed); perm = rng.permutation(n); te = torch.zeros(n, dtype=torch.bool); te[perm[: n // 4]] = True; tr = ~te
    steps = torch.as_tensor(steps); scores = {}
    for lam in lambdas:
        W = ridge(X[tr], steps[tr], Y[tr], lam); scores[lam] = r2(apply_map(W, X[te], steps[te]), Y[te])
    return max(scores, key=scores.get), scores


# ------------------------------------------------------------------ causal measures (prompt under the source goal)
def switch(patched_logits, nat_src, nat_tgt):
    """Among decisions whose natural source and target decisions differ: share whose patched argmax is the target's."""
    nat_src, nat_tgt = torch.as_tensor(nat_src), torch.as_tensor(nat_tgt); dep = nat_src != nat_tgt
    per = (torch.as_tensor(patched_logits).argmax(1) == nat_tgt)[dep].double().numpy()
    return (float(per.mean()) if len(per) else float("nan")), per


def keep(patched_logits, nat_src, nat_tgt):
    """Same selection; share whose patched argmax is still the source's."""
    nat_src, nat_tgt = torch.as_tensor(nat_src), torch.as_tensor(nat_tgt); dep = nat_src != nat_tgt
    per = (torch.as_tensor(patched_logits).argmax(1) == nat_src)[dep].double().numpy()
    return (float(per.mean()) if len(per) else float("nan")), per


# ------------------------------------------------------------------ subspaces
def _centre(Dm, centre_by):
    Dm = torch.as_tensor(Dm).double()
    if centre_by is None: return Dm - Dm.mean(0, keepdim=True)
    s = torch.as_tensor(centre_by).long(); out = Dm.clone()
    for v in s.unique(): m = s == v; out[m] = Dm[m] - Dm[m].mean(0, keepdim=True)
    return out


def _svd(Dm, centre_by):
    return torch.linalg.svd(_centre(Dm, centre_by), full_matrices=False)


def topk_projector(Dm, k, centre_by=None):
    _, _, Vh = _svd(Dm, centre_by); V = Vh[:k]; return V.T @ V


def bottomk_projector(Dm, k, centre_by=None):
    Dm = torch.as_tensor(Dm).double(); d = Dm.shape[1]; _, _, Vh = _svd(Dm, centre_by)
    if Vh.shape[0] < d:                                                     # fewer rows than dimensions: complete the basis
        Q, _ = torch.linalg.qr(torch.cat([Vh.T, torch.eye(d, dtype=torch.float64)], 1)); Vh = Q[:, :d].T
    V = Vh[d - k:]; return V.T @ V


def random_projector(d, k, seed=0):
    g = torch.Generator().manual_seed(seed); Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64)); return Q @ Q.T


def spectrum_ranks(Dm, centre_by=None):
    _, S, _ = _svd(Dm, centre_by); share = (S ** 2).cumsum(0) / (S ** 2).sum()
    rank_for = lambda q: int((share < q).sum().item() + 1)
    return dict(k50=rank_for(0.5), k90=rank_for(0.9), k99=rank_for(0.99), spectrum=[float(x) for x in S])


def subspace_overlap(P1, P2, k):
    """trace(P1 P2) / k: the mean squared cosine of the principal angles between two rank-k projectors."""
    return float(torch.trace(torch.as_tensor(P1).double() @ torch.as_tensor(P2).double()) / k)


# ------------------------------------------------------------------ probes
def jaccard(pred_logits, M):
    P = torch.as_tensor(pred_logits) > 0; M = torch.as_tensor(M).bool()
    per = ((P & M).sum(1).double() / (P | M).sum(1).double().clamp(min=1)).numpy()
    return float(per.mean()), per


# ------------------------------------------------------------------ the swap study's pairs
def swap_pairs(dec, n_fit=6144, n_pairs=1024, seed=1):
    """Same-step pairs of decisions with different histories, as factorial_llm.py builds them: the first n_fit are the
    fit pairs, the next n_pairs the test pairs."""
    rng = np.random.default_rng(seed); by_step = {}
    for i, d in enumerate(dec): by_step.setdefault(d["step"], []).append(i)
    pairs = []
    while len(pairs) < n_pairs + n_fit:
        i = int(rng.integers(len(dec))); c = by_step[dec[i]["step"]]; j = c[int(rng.integers(len(c)))]
        if j != i and (dec[i]["acts"], dec[i]["obss"]) != (dec[j]["acts"], dec[j]["obss"]): pairs.append((i, j))
    return pairs[:n_fit], pairs[n_fit:]


def causal_rank(switch_by_k, full, ks=KS):
    """The smallest k in ks at which the rank-k map switches ≥ 0.9 × the full map; None if none does."""
    for k in ks:
        if k in switch_by_k and switch_by_k[k] >= 0.9 * full: return int(k)
    return None
