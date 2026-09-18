import numpy as np
import torch

from goalgeo import hmm as Hm
from goalgeo import seqmodels as Sm


def test_clean_hmm_cue_beliefs_are_one_step_equivalent_but_two_step_different():
    m = Hm.make_hmm("clean")
    bp = Hm.belief(m, [Hm.TOK["p"]]); bq = Hm.belief(m, [Hm.TOK["q"]])
    assert np.isclose(bp[Hm.S["P"]], 1.0) and np.isclose(bq[Hm.S["Q"]], 1.0)   # posterior over the emitting state; next state is A or B
    assert np.allclose(Hm.predictive(m, bp, 1), Hm.predictive(m, bq, 1))
    assert not np.allclose(Hm.predictive(m, bp, 2), Hm.predictive(m, bq, 2))
    assert np.isclose(Hm.predictive(m, bp, 2).sum(), 1.0)
    assert Hm.predictive(m, bp, 3).shape == (64,)


def test_noisy_hmm_keeps_one_step_equivalence():
    m = Hm.make_hmm("noisy")
    bp = Hm.belief(m, [Hm.TOK["p"]]); bq = Hm.belief(m, [Hm.TOK["q"]])
    assert not np.allclose(bp, bq)
    assert np.allclose(Hm.predictive(m, bp, 1), Hm.predictive(m, bq, 1))
    assert not np.allclose(Hm.predictive(m, bp, 2), Hm.predictive(m, bq, 2))


def test_sampled_sequences_match_forward_marginals():
    m = Hm.make_hmm("clean")
    X, _ = Hm.sample(m, n=2000, T=9, seed=0)
    # a token after p is x or y with equal frequency; two after p it is x ~90%
    after_p = X[:, 1:][X[:, :-1] == Hm.TOK["p"]]
    assert abs(np.mean(after_p == Hm.TOK["x"]) - 0.5) < 0.05
    two_after_p = X[:, 2:][X[:, :-2] == Hm.TOK["p"]]
    assert abs(np.mean(two_after_p == Hm.TOK["x"]) - 0.9) < 0.05


def test_objective_targets_have_right_shapes():
    m = Hm.make_hmm("clean")
    X, _ = Hm.sample(m, n=50, T=12, seed=1)
    t1 = Hm.window_targets(m, X[:, -6:], k=1); t2 = Hm.window_targets(m, X[:, -6:], k=2)
    assert t1.shape == (50, 4) and t2.shape == (50, 16) and np.allclose(t2.sum(1), 1.0)
    seq = Hm.sequential_targets(m, X)
    assert seq.shape == (50, 12, 4) and np.allclose(seq.sum(-1), 1.0)


def test_gru_objectives_train_and_expose_hidden_states():
    m = Hm.make_hmm("clean")
    for obj in ("one-step", "k-step", "sequential"):
        net = Sm.SeqNet(n_vocab=4, hidden=16, emb=8, out_dim=4 if obj != "k-step" else 16, seed=0)
        hist = Sm.train_objective(net, m, obj, k=2, steps=60, batch=32, seed=0)
        assert hist[-1] < hist[0]
        X, _ = Hm.sample(m, n=10, T=12, seed=2)
        h = net.hidden(torch.as_tensor(X))
        assert h.shape == (10, 16)
