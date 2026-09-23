"""The exact filter against brute-force enumeration, and the identities the round relies on."""

import itertools

import numpy as np

from goalgeo import latentgoal as LG


def _brute(env, obs):
    """P(G, o_1..o_t) summed over every channel path, for each prefix; also P(o_{t+1} | o_<=t)."""
    post, pred = [], []
    for t in range(len(obs) + 1):
        joint = np.zeros(env.K); nxt = np.zeros(env.M)
        for g in range(env.K):
            for path in itertools.product(range(env.C), repeat=t + 1):
                p = env.prior[g] * env.c0[path[0]]
                for s in range(1, t + 1):
                    p *= env.A[path[s - 1], path[s]]
                for s in range(t):
                    p *= env.L[path[s], g, obs[s]]
                joint[g] += p
                nxt += p * env.L[path[t], g]          # path[t] is the state that emits o_{t+1}
        post.append(joint / joint.sum()); pred.append(nxt / nxt.sum())
    return np.array(post), np.array(pred)


def test_filter_matches_enumeration():
    for kind in ("iid", "channel"):
        env = LG.make_env(kind, K=3)
        X, _, _ = LG.sample(env, 3, T=5, seed=2)
        b = LG.posterior(env, X); p = LG.next_obs(env, X)
        for i in range(3):
            bb, pp = _brute(env, X[i, 1:] - 1)
            assert np.allclose(b[i], bb, atol=1e-12)
            assert np.allclose(p[i], pp, atol=1e-12)


def test_iid_log_odds_are_affine_in_counts():
    env = LG.make_env("iid", K=4)
    X, _, _ = LG.sample(env, 200, seed=0)
    y = LG.log_odds(LG.posterior(env, X))
    lam = np.log(env.L[0, :-1]) - np.log(env.L[0, -1:])        # [K-1, M]
    assert np.allclose(y, LG.counts(X, env.M) @ lam.T, atol=1e-9)


def test_targets_are_distributions_and_neutral_tokens_are_neutral():
    for kind in ("iid", "channel"):
        env = LG.make_env(kind, K=4)
        X, _, _ = LG.sample(env, 50, seed=1)
        for obj in LG.OBJECTIVES:
            Y = LG.targets(env, X, obj)
            assert Y.shape == (50, LG.T_OBS + 1, LG.out_dim(env, obj))
            assert np.allclose(Y.sum(-1), 1)
        on = env.L[-1]
        assert np.allclose(on[:, env.K:], on[0, env.K:])            # same under every goal


def test_excluded_pool_never_enters_the_conflict_region():
    env = LG.make_env("channel", K=4)
    X, _ = LG.sample_excluding(env, 300, seed=0)
    assert not LG.in_conflict(LG.posterior(env, X), LG.CONFLICT["channel"]).any()
