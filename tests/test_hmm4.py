import numpy as np

from goalgeo import hmm4 as H4


def _after(m, tokens):
    return H4.beliefs_seq(m, np.array([tokens]))[0, -1]


def test_base_condition_reduces_to_round4_chain():
    m = H4.make_hmm4(1.0, 0.4, 1)
    S, T = m.S, H4.TOK
    assert len(m.pi) == 11
    bp, bq = _after(m, [T["p"]]), _after(m, [T["q"]])
    assert np.isclose(bp[S["P"]], 1) and np.isclose(bq[S["Q"]], 1)
    assert np.allclose(H4.next_token(m, bp), H4.next_token(m, bq))
    p2p, p2q = H4.joint_predictive(m, bp[None], 2)[0], H4.joint_predictive(m, bq[None], 2)[0]
    assert not np.allclose(p2p, p2q) and np.isclose(p2p.sum(), 1)
    assert np.isclose(m.E[S["C"], T["x"]], 0.9) and np.isclose(m.E[S["D"], T["y"]], 0.9)


def test_one_step_equivalence_holds_at_every_filler_step_for_all_conditions():
    for r, d, k in [(0.0, 0.4, 1), (0.5, 0.1, 4), (1.0, 0.0, 2), (0.25, 0.4, 8)]:
        m = H4.make_hmm4(r, d, k); T = H4.TOK
        for j in range(k):                           # cue, then j < k filler tokens: next token still a filler
            a = _after(m, [T["p"]] + [T["x"]] * j); b = _after(m, [T["q"]] + [T["x"]] * j)
            assert np.allclose(H4.next_token(m, a), H4.next_token(m, b)), (r, d, k, j)
        a, b = _after(m, [T["p"]] + [T["x"]] * k), _after(m, [T["q"]] + [T["x"]] * k)
        differs = not np.allclose(H4.next_token(m, a), H4.next_token(m, b))
        assert differs == (r > 0 and d > 0), (r, d, k)


def test_control_pair_is_fixed_and_immediate():
    for r, d, k in [(0.0, 0.0, 1), (1.0, 0.4, 16)]:
        m = H4.make_hmm4(r, d, k); T = H4.TOK
        pu, pv = H4.next_token(m, _after(m, [T["u"]])), H4.next_token(m, _after(m, [T["v"]]))
        assert np.isclose(pu[T["x"]] - pv[T["x"]], 0.8)


def test_sample_matches_forward_marginals_and_states():
    m = H4.make_hmm4(0.5, 0.4, 2)
    X, Z = H4.sample(m, n=3000, T=20, seed=0)
    assert X.shape == Z.shape == (3000, 20) and X.max() < H4.V and Z.max() < len(m.pi)
    after_p = Z[:, 1:][X[:, :-1] == H4.TOK["p"]]
    assert np.all(after_p == m.S["FA1"])
    three_after_p = X[:, 3:][X[:, :-3] == H4.TOK["p"]]
    assert abs(np.mean(three_after_p == H4.TOK["n"]) - 0.5) < 0.05
    assert abs(np.mean(three_after_p == H4.TOK["x"]) - 0.45) < 0.05


def test_forgetting_cost_matches_analytic_value():
    for r, d in [(1.0, 0.4), (0.5, 0.2), (0.25, 0.05), (0.0, 0.4), (1.0, 0.0)]:
        m = H4.make_hmm4(r, d, 1)
        X, Z = H4.sample(m, n=500, T=30, seed=1)
        fc = H4.forgetting_cost(m, X, Z)
        h = -((0.5 + d) * np.log(0.5 + d) + (0.5 - d) * np.log(0.5 - d))
        expected = r * (np.log(2) - h)
        assert np.isclose(fc["per_event"], expected, atol=1e-6), (r, d, fc, expected)
        # per-step cost is slightly below per_event * frac: relevant positions whose cue precedes the sequence cost nothing
        assert expected * fc["relevant_frac"] * 0.9 <= fc["per_step"] + 1e-9 <= expected * fc["relevant_frac"] + 1e-9
        assert abs(fc["relevant_frac"] - H4.relevant_fraction(m)) < 0.03


def test_mirror_is_an_involution_that_swaps_branches():
    m = H4.make_hmm4(1.0, 0.4, 3)
    assert np.all(m.mirror[m.mirror] == np.arange(len(m.pi)))
    assert m.mirror[m.S["FA2"]] == m.S["FB2"] and m.mirror[m.S["C"]] == m.S["D"] and m.mirror[m.S["U"]] == m.S["U"]
