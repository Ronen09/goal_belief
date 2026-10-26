"""Pure helpers of the bump coordinates study (study 8, `bump_coords/common.py`): ridge maps with the step, rank-k
projectors, switch / keep / Jaccard, swap-pair construction, the probe pool's drop rule. CPU only; no model."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "8_language_models" / "bump_coords"))
import common as C  # noqa: E402


def test_step_onehot_clips():
    oh = C.step_onehot(torch.tensor([0, 39, 40, 57]))
    assert oh.shape == (4, C.H) and oh[2, 39] == 1 and oh[3, 39] == 1 and oh.sum() == 4


def test_ridge_recovers_linear_map_and_composes():
    g = torch.Generator().manual_seed(0); n, d = 2000, 16
    X = torch.randn(n, d, generator=g, dtype=torch.float64); steps = torch.randint(0, 40, (n,), generator=g)
    TAB, TBC = torch.randn(d, d, generator=g, dtype=torch.float64), torch.randn(d, d, generator=g, dtype=torch.float64)
    XB, XC = X @ TAB, X @ TAB @ TBC
    WAB, WBC, WAC = C.ridge(X, steps, XB, 1e-6), C.ridge(XB, steps, XC, 1e-6), C.ridge(X, steps, XC, 1e-6)
    assert C.r2(C.apply_map(WAB, X, steps), XB) > 0.999
    composed = C.apply_map(WBC, C.apply_map(WAB, X, steps), steps); direct = C.apply_map(WAC, X, steps)
    assert C.r2(composed, direct) > 0.999


def test_tune_lambda_returns_candidate():
    g = torch.Generator().manual_seed(1); X = torch.randn(400, 8, generator=g, dtype=torch.float64); steps = torch.zeros(400, dtype=torch.long)
    lam, scores = C.tune_lambda(X, steps, X @ torch.eye(8, dtype=torch.float64), lambdas=(1e-3, 1.0))
    assert lam in (1e-3, 1.0) and set(scores) == {1e-3, 1.0}


def test_switch_and_keep():
    L = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [1., 0, 0]]); src = torch.tensor([0, 0, 0, 0]); tgt = torch.tensor([1, 1, 2, 0])
    s, per = C.switch(L, src, tgt); assert s == pytest.approx(2 / 3) and per.tolist() == [0, 1, 1]
    k, _ = C.keep(L, src, tgt); assert k == pytest.approx(1 / 3)


def test_switch_empty_is_nan():
    s, per = C.switch(torch.zeros(3, 3), torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long))
    assert np.isnan(s) and len(per) == 0


def test_topk_projector_full_rank_is_identity():
    D = torch.randn(50, 12, dtype=torch.float64); steps = torch.randint(0, 40, (50,))
    assert torch.allclose(C.topk_projector(D, 12, steps), torch.eye(12, dtype=torch.float64), atol=1e-8)
    P1 = C.topk_projector(D, 3, steps); assert torch.allclose(P1 @ P1, P1, atol=1e-8) and torch.linalg.matrix_rank(P1) == 3
    assert torch.allclose(C.topk_projector(D, 3, steps) + C.bottomk_projector(D, 9, steps), torch.eye(12, dtype=torch.float64), atol=1e-8)


def test_spectrum_ranks_low_rank_data():
    g = torch.Generator().manual_seed(2); D = torch.randn(300, 2, generator=g, dtype=torch.float64) @ torch.randn(2, 20, generator=g, dtype=torch.float64)
    r = C.spectrum_ranks(D, None); assert r["k99"] == 2 and r["k50"] <= 2 and len(r["spectrum"]) == 20


def test_subspace_overlap_bounds():
    D = torch.randn(100, 10, dtype=torch.float64); P = C.topk_projector(D, 4, None); R = C.random_projector(10, 4, seed=0)
    assert C.subspace_overlap(P, P, 4) == pytest.approx(1.0) and 0 <= C.subspace_overlap(P, R, 4) <= 1


def test_jaccard_empty_prediction():
    M = torch.tensor([[True, False, True], [True, True, True]]); L = torch.tensor([[-1., -1, -1], [1., 1, -1]])
    j, per = C.jaccard(L, M); assert per.tolist() == [0.0, pytest.approx(2 / 3)] and j == pytest.approx(1 / 3)


def test_boot_ratio_and_causal_rank():
    lo, hi = C.boot_ratio(np.ones(100), np.ones(100) * 2); assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    assert C.causal_rank({1: 0.1, 2: 0.5, 4: 0.85, 8: 0.9, 16: 0.95}, 1.0) == 8
    assert C.causal_rank({1: 0.1, 2: 0.2}, 1.0) is None


def test_swap_pairs_same_step_distinct_histories():
    dec = [dict(step=1 + i % 3, acts=[i % 7], obss=[0]) for i in range(60)]
    fit, test = C.swap_pairs(dec, n_fit=20, n_pairs=5, seed=1)
    assert len(fit) == 20 and len(test) == 5
    for i, j in fit + test: assert dec[i]["step"] == dec[j]["step"] and (dec[i]["acts"], dec[i]["obss"]) != (dec[j]["acts"], dec[j]["obss"])
    assert C.swap_pairs(dec, 20, 5, 1) == (fit, test)


def test_probe_decisions_drop_ended():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "studies" / "8_language_models" / "bump_goals"))
    from design import worlds_and_solutions
    from goalgeo import bumpmaze as BMZ
    _, _, worlds, _ = worlds_and_solutions()
    P = C.probe_decisions(worlds, n=512, seed=0)
    assert P["masks"].shape[1] == 140 and P["fit"].sum() + P["test"].sum() == len(P["hists"]) and not (P["fit"] & P["test"]).any()
    assert len(P["hists"]) + P["dropped"] == 512 and len(P["steps"]) == len(P["hists"])
    for acts, obss in P["hists"][:64]:
        for w in worlds:
            m = np.zeros(w.n_states, dtype=bool); m[w.start_states] = True
            for a, o in zip(acts, obss): m = BMZ.step_belief(w, m, a, o, remove_goal=True)
            assert m.any()
