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


# ------------------------------------------------------------------ task 2: the from-scratch models
def _hists():
    """Six histories of steps 1..6 (acts in {0, 1, 2}, obss in {0, 1}) and their goals."""
    rng = np.random.default_rng(5); hists = [([int(a) for a in rng.integers(0, 3, t)], [int(o) for o in rng.integers(0, 2, t)]) for t in range(1, 7)]
    return hists, 2


@pytest.mark.parametrize("kind", ["tfm", "gru"])
def test_small_patch_with_natural_state_reproduces_logits(kind):
    torch.manual_seed(0); m = CH.Small(kind, None, H=40, device="cpu"); hists, g = _hists()
    z, X = m.record(hists, g)
    assert z.shape == (6, 3) and X.shape == (6, len(m.sites), CH.D_SMALL) and torch.allclose(z.sum(1), torch.zeros(6, dtype=torch.float64), atol=1e-6)
    for si, s in enumerate(m.sites):
        zp = m.patched(hists, g, s, X[:, si])
        assert torch.allclose(zp, z, atol=1e-5), (kind, s)


def test_small_gru_site_below_position_zero_is_the_zero_state():
    torch.manual_seed(0); m = CH.Small("gru", None, H=40, device="cpu"); hists, g = _hists()
    z, X = m.record(hists, g); i2 = m.sites.index(-2)
    assert torch.allclose(X[0, i2], torch.zeros(CH.D_SMALL)) and not torch.allclose(X[1, i2], torch.zeros(CH.D_SMALL))      # step 1: h_{t-2} is before any token


@pytest.mark.parametrize("kind", ["tfm", "gru"])
def test_small_patched_record_matches_natural_when_patch_is_natural(kind):
    torch.manual_seed(0); m = CH.Small(kind, None, H=40, device="cpu"); hists, g = _hists()
    z, X = m.record(hists, g); s0 = m.sites[0]
    zp, Xp = m.patched_record(hists, g, s0, X[:, 0])
    assert torch.allclose(zp, z, atol=1e-5) and torch.allclose(Xp, X, atol=1e-4)


def test_small_gru_jacobian_at_h_t_is_head_rows():
    torch.manual_seed(0); m = CH.Small("gru", None, H=40, device="cpu"); hists, g = _hists()
    J = m.jacobian(hists, g, 0); W = m.model.head.weight.detach().double()
    assert J.shape == (6, 3, CH.D_SMALL) and torch.allclose(J, (W - W.mean(0, keepdim=True))[None].expand(6, -1, -1), atol=1e-5)


def test_small_tfm_jacobian_predicts_a_small_change():
    torch.manual_seed(0); m = CH.Small("tfm", None, H=40, device="cpu"); hists, g = _hists()
    z, X = m.record(hists, g); J = m.jacobian(hists, g, 2); eps = torch.randn(6, CH.D_SMALL) * 1e-3
    zp = m.patched(hists, g, 2, X[:, m.sites.index(2)] + eps)
    assert torch.allclose(zp - z, torch.einsum("njd,nd->nj", J, eps.double()), atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuDNN RNN backward is the GPU path")
def test_small_gru_jacobian_runs_on_cuda_in_eval_mode():
    torch.manual_seed(0); m = CH.Small("gru", None, H=40, device="cuda"); hists, g = _hists()
    J = m.jacobian(hists, g, -1); assert J.shape == (6, 3, CH.D_SMALL) and torch.isfinite(J).all() and not m.model.training


# ------------------------------------------------------------------ review fixes: the scorecard's parts, the bins' history-dependent share, the quick cache
_C = Path(__file__).resolve().parents[1] / "studies" / "8_language_models" / "bump_channel" / "channel.py"
_cspec = importlib.util.spec_from_file_location("channel_script", _C); CHS = importlib.util.module_from_spec(_cspec); sys.modules["channel_script"] = CHS
_prev = sys.modules.get("common"); sys.modules["common"] = CH                       # channel.py's `import common` must bind the study's module, without leaving it under that name
try: _cspec.loader.exec_module(CHS)
finally:
    if _prev is None: sys.modules.pop("common", None)
    else: sys.modules["common"] = _prev


def _thru(ratio, lo, hi, known=None):
    r = dict(switch=ratio, switch_ci=[lo, hi], ref=1.0, ratio=ratio, ratio_ci=[lo, hi], n=100)
    r["known"] = known or dict(known=dict(ratio=ratio), uncertain=dict(ratio=ratio), diff_ci=[-0.1, 0.1]); return r


def _parts():
    ctrl = {n: _thru(0.01, 0.0, 0.02) for n in ("bottom", "random0", "random1", "random2")}
    p2 = {"23": dict(goal=_thru(0.9, 0.85, 0.95), comp=_thru(0.24, 0.19, 0.29), **ctrl)}
    p1 = {"19->23": dict(goal=_thru(0.9, 0.85, 0.95), comp=_thru(0.31, 0.25, 0.37), **ctrl)}
    return p2, p1


def test_scorecard_counts_part1_for_the_fine_tuned_model_only():
    p2, p1 = _parts(); ov = {23: dict(goal_readout=0.2)}
    small = CHS.scorecard("tfm", 128, {23: 16}, ov, p2, p1, 23, 19); llm = CHS.scorecard("llm", 896, {23: 4}, ov, p2, p1, 23, 19)
    assert small["COMP"]["holds"] and small["COMP"]["values"] == [0.24] and small["COMP_part1"]["value"] == 0.31 and small["THRU_U"] is None and not small["part1_decision_bearing"]
    assert not llm["COMP"]["holds"] and llm["COMP"]["values"] == [0.24, 0.31] and llm["THRU_U"]["holds"]
    assert small["CTRL"]["holds"] and len(small["CTRL"]["values"]) == 4 and len(llm["CTRL"]["values"]) == 8


def test_dep_share_by_bin():
    known_pair = np.array([True, True, False, False, False]); cells_p = np.array([0, 0, 0, 1, 2, 2, 4])          # 4 goals per pair
    s = CH.dep_share_by_bin(known_pair, cells_p, 4)
    assert abs(s["known"] - 4 / 8) < 1e-12 and abs(s["uncertain"] - 3 / 12) < 1e-12 and s["n_pairs"] == dict(known=2, uncertain=3)


def test_resolve_cache_prefers_the_quick_layouts(tmp_path):
    out = tmp_path / "_smoke" / "channel"; out.mkdir(parents=True)
    assert CH.resolve_cache("tfm", True, None, out) == CH.CACHE                                         # nothing quick exists: the real cache
    rp = tmp_path / "_smoke" / "subspace_small" / "cache"; rp.mkdir(parents=True); (rp / "states_tfm.pt").write_bytes(b"x")
    assert CH.resolve_cache("tfm", True, None, out) == rp                                              # reproduce.py's layout
    mp = tmp_path / "_smoke" / "subspace_gru" / "cache"; mp.mkdir(parents=True); (mp / "states_gru.pt").write_bytes(b"x")
    assert CH.resolve_cache("gru", True, None, out) == mp                                              # the manual smoke layout
    assert CH.resolve_cache("gru", False, "/elsewhere", out) == Path("/elsewhere") and CH.resolve_cache("gru", False, None, out) == CH.CACHE


# ------------------------------------------------------------------ post hoc: the goal subspace outside the readout
def test_orth_split_separates_a_subspace_against_the_readout():
    g = torch.Generator().manual_seed(2); D, k, kR = 20, 4, 3
    Vh_R = CH.CC.basis(torch.randn(10, D, generator=g, dtype=torch.float64), None); R = Vh_R[:kR].T
    C = torch.linalg.qr(torch.randn(D, k, generator=g, dtype=torch.float64))[0]                      # generic: meets R trivially
    perp, par = CH.orth_split(C, Vh_R, kR)
    assert perp.shape == (D, k) and torch.allclose(perp.T @ perp, torch.eye(k, dtype=torch.float64), atol=1e-8)
    assert torch.allclose(R.T @ perp, torch.zeros(kR, k, dtype=torch.float64), atol=1e-8)            # orthogonal to the readout
    assert par.shape[1] == min(k, kR) and torch.allclose(par - R @ (R.T @ par), torch.zeros_like(par), atol=1e-8)   # inside the readout
    inside = R[:, :2]; perp2, par2 = CH.orth_split(inside, Vh_R, kR)                                  # a subspace inside R has no outside part
    assert perp2.shape[1] == 0 and par2.shape[1] == 2
    Q = CH.random_in_complement(Vh_R, kR, D, 3, seed=0)
    assert Q.shape == (D, 3) and torch.allclose(R.T @ Q, torch.zeros(kR, 3, dtype=torch.float64), atol=1e-8) and torch.allclose(Q.T @ Q, torch.eye(3, dtype=torch.float64), atol=1e-8)
