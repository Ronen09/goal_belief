"""Pure helpers and the small-model runners of the bump channel study (study 8, `bump_channel/common.py`): pair
drawing, the registered rank, projectors, through ratios, the known / uncertain split, and the from-scratch models'
recording / patching / Jacobians. CPU only; random weights, no checkpoints."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import importlib.util

_P = Path(__file__).resolve().parents[1] / "studies" / "8_language_models" / "bump_channel" / "common.py"
_spec = importlib.util.spec_from_file_location("channel_common", _P); CH = importlib.util.module_from_spec(_spec)   # by path and under its own name: test_bump_coords imports its study's `common`
sys.modules["channel_common"] = CH; _spec.loader.exec_module(CH)


# ------------------------------------------------------------------ task 1: pure helpers

def test_draw_pairs_same_step_different_key():
    steps = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]); keys = np.array([b"a", b"a", b"b", b"c"] * 3, dtype=object)
    pairs = CH.draw_pairs(steps, keys, n_pairs=20, seed=1)
    assert len(pairs) == 20
    for i, j in pairs:
        assert i != j and steps[i] == steps[j] and keys[i] != keys[j]


def test_registered_k_rounds_up_and_drops_none():
    ks = (1, 2, 4, 8, 16, 32, 64, 128)
    assert CH.registered_k([4] * 10 + [8] * 2, ks) == 4
    assert CH.registered_k([4] * 6 + [8, 16, 16, 16, 16, 32], ks) == 8           # median 6 → up to 8
    assert CH.registered_k([4] * 5 + [None] * 7, ks) is None
    assert CH.registered_k([4] * 6 + [None] * 6, ks) == 4
    assert CH.registered_k([], ks) is None


def test_projectors_identities():
    g = torch.Generator().manual_seed(0); n, D, k = 60, 16, 3
    X_fit = {gi: torch.randn(n, D, generator=g, dtype=torch.float64) for gi in range(4)}; steps = torch.randint(0, 5, (n,), generator=g)
    Vh, spec = CH.goal_basis(X_fit, steps)
    assert Vh.shape == (D, D) and torch.allclose(Vh @ Vh.T, torch.eye(D, dtype=torch.float64), atol=1e-8) and spec["k90"] >= 1
    P = CH.projectors(Vh, k, D)
    assert set(P) == {"goal", "comp", "bottom", "random0", "random1", "random2"}
    Pg = P["goal"]; I = torch.eye(D, dtype=torch.float64)
    assert torch.allclose(Pg, Pg.T) and torch.allclose(Pg @ Pg, Pg, atol=1e-8) and abs(float(torch.trace(Pg)) - k) < 1e-8
    assert torch.allclose(Pg + P["comp"], I, atol=1e-8)
    assert torch.allclose(Pg @ P["bottom"], torch.zeros(D, D, dtype=torch.float64), atol=1e-8)
    delta = torch.randn(7, D, generator=g, dtype=torch.float64); x1 = torch.randn(7, D, generator=g, dtype=torch.float64)
    assert torch.allclose(CH.candidate(x1, delta, Pg) + CH.candidate(x1, delta, P["comp"]) - x1, x1 + delta, atol=1e-8)
    assert abs(CH.norm_share(delta, Pg) + CH.norm_share(delta, P["comp"]) - 1) < 1e-8
    assert abs(CH.overlap(Pg, Pg, k) - 1) < 1e-8 and CH.overlap(Pg, P["random0"], k) < 0.6


def test_history_and_readout_bases_shapes():
    g = torch.Generator().manual_seed(1); n, D = 40, 12
    X_fit = {gi: torch.randn(n, D, generator=g, dtype=torch.float64) for gi in range(4)}; steps = torch.randint(0, 3, (n,), generator=g)
    keys = np.array([bytes([i % 5]) for i in range(n)], dtype=object)
    Vh = CH.history_basis(X_fit, steps, keys, n_pairs=30, seed=2); assert Vh.shape == (D, D)
    J = torch.randn(9, 3, D, generator=g, dtype=torch.float64); Vj = CH.readout_basis(J); assert Vj.shape == (D, D)
    Pj = CH.projector(Vj, 2); assert torch.allclose(Pj @ Pj, Pj, atol=1e-8)


def test_through_undefined_below_floor_and_on_empty():
    rng = np.random.default_rng(0); n = 400; groups = np.arange(n) // 4
    ref = (rng.random(n) < 0.8).astype(float); var = ref * (rng.random(n) < 0.7)
    t = CH.through(var, ref, groups)
    assert 0.6 < t["ratio"] < 0.8 and t["ratio_ci"][0] < t["ratio"] < t["ratio_ci"][1] and abs(t["ref"] - ref.mean()) < 1e-12
    low = CH.through(var * 0.0 + (rng.random(n) < 0.05), (rng.random(n) < 0.05).astype(float), groups)
    assert low["ratio"] is None and low["ratio_ci"] is None
    empty = CH.through(np.zeros(0), np.zeros(0), np.zeros(0, dtype=int))
    assert empty["ratio"] is None and np.isnan(empty["switch"])


def test_split_known_diff():
    rng = np.random.default_rng(0); n = 800; groups = np.arange(n) // 2; known = np.arange(n) % 2 == 0
    ref = np.ones(n); var = np.where(known, 1.0, (rng.random(n) < 0.5).astype(float))
    s = CH.split_known(var, ref, groups, known)
    assert abs(s["known"]["ratio"] - 1) < 1e-12 and 0.4 < s["uncertain"]["ratio"] < 0.6 and s["diff_ci"][0] > 0
    s2 = CH.split_known(var, ref * 0.0, groups, known)
    assert s2["known"]["ratio"] is None and s2["diff_ci"] is None
