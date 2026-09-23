"""Identities the causal tests rely on."""

import numpy as np

from goalgeo import beliefcausal as BC, latentgoal as LG


def test_filter_continue_matches_the_full_filter():
    env = LG.make_env("channel", 4)
    X, _, _ = LG.sample(env, 50, seed=3)
    J = LG.filter_joint(env, X)
    Jc = BC.filter_continue(env, J[:, 6], X[:, 7:])
    assert np.allclose(Jc, J[:, 6:], atol=1e-12)
    assert np.allclose(BC.ideal(env, J[:, 1:], "next_obs"), LG.next_obs(env, X, J)[:, 1:], atol=1e-12)


def test_transplant_keeps_channel_given_goal():
    env = LG.make_env("channel", 4)
    X, _, _ = LG.sample(env, 20, seed=4)
    J = LG.filter_joint(env, X)[:, 8]
    b_new = np.random.default_rng(0).dirichlet(np.ones(4), 20)
    Jt = BC.transplant(J, b_new)
    assert np.allclose(Jt.sum(-1), b_new)
    assert np.allclose(Jt / Jt.sum(-1, keepdims=True), J / J.sum(-1, keepdims=True))


def test_both_probe_moves_set_the_decoded_belief_exactly():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=(2000, 3)); H = Y @ rng.normal(size=(3, 16)) + 0.1 * rng.normal(size=(2000, 16))
    m = BC.BeliefMap(H, Y)
    h = H[:10]; y_star = rng.normal(size=(10, 3))
    assert np.allclose(m.z(m.probe_e(h, y_star)), y_star, atol=1e-8)
    assert np.allclose(m.z(m.probe_d(h, y_star)), y_star, atol=1e-8)
