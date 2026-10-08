import numpy as np
import pytest

from goalgeo import maze20 as M


def test_parse_symbols_and_emissions():
    rows = ["A..*",
            "#.##",
            "#..B"]
    m = M.from_rows(rows, eps=0.2)
    assert m.n == 8 and m.goals == (0, 7) and m.landmarks == (3,)
    ix = {c: i for i, c in enumerate(m.cells)}
    assert m.sym[ix[(0, 2)]] == 1                 # E-W corridor
    assert m.sym[ix[(1, 1)]] == 0                 # N-S corridor
    assert m.sym[ix[(0, 1)]] == 3                 # junction (W, E, S)
    assert m.sym[ix[(2, 1)]] == 2                 # corner
    assert m.sym[ix[(0, 3)]] == 4 and m.E[ix[(0, 3)], 4] == 1
    np.testing.assert_allclose(m.E.sum(1), 1)
    assert m.E[ix[(0, 2)], 1] == pytest.approx(0.8) and m.E[ix[(0, 2)], 0] == pytest.approx(0.2)


def test_disconnected_rejected():
    with pytest.raises(AssertionError):
        M.from_rows(["A#B"])


def test_ceiling_additive_corridor_is_one_and_ring_reverses():
    rows = ["A......B"]                           # a corridor with a goal at each end: additive
    m = M.from_rows(rows)
    acc, err = M.ceiling(m, [list(m.goals)], restarts=2, steps=600, device="cpu")
    assert acc[0] == 1 and err.sum() == 0
    assert M.reversals(m, list(m.goals)) == (0, 0)
    ring = ["..A..",                              # goals at the top and bottom of a ring: the sides reverse
            ".###.",
            ".###.",
            ".###.",
            "..B.."]
    m = M.from_rows(ring)
    cnt, cells = M.reversals(m, list(m.goals))   # (0,1): A E, B W; (0,3): A W, B E; (4,1): A W, B E; (4,3): A E, B W
    assert (cnt, cells) == (4, 4)
    acc, _ = M.ceiling(m, [list(m.goals)], restarts=4, steps=1500, device="cpu")
    assert acc[0] < 1


def test_exact_ceiling_matches_small_cases():
    m = M.from_rows(["A......B"])
    r = M.ceiling_exact(m, list(m.goals))
    assert r["exact"] and r["incumbent"] == 1
    ring = ["..A..", ".###.", ".###.", ".###.", "..B.."]
    m = M.from_rows(ring)
    r = M.ceiling_exact(m, list(m.goals))
    # 16 cells, 2 goals, 15 decisions each; the 4 reversal cells force at least 2 errors (one per reversal pair)
    assert r["exact"] and r["incumbent"] == pytest.approx(1 - 2 / 30)
    acc, _ = M.ceiling(m, [list(m.goals)], restarts=4, steps=1500, device="cpu", refine_rounds=3)
    assert acc[0] <= r["incumbent"] + 1e-9
