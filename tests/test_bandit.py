"""The reward-bandit task: the count-based filter against sequential Bayes, Q* against brute-force expectimax, the
environment's bookkeeping, and the policies' interface."""

import itertools

import numpy as np
import torch

from goalgeo import bandit as BD


def _spec(n_cue=2, T=2):
    return BD.Spec(n_cue=n_cue, T=T)


def test_belief_from_counts_matches_sequential_bayes():
    s = _spec(3, 3); g = BD.Graph(s)
    rng = np.random.default_rng(0)
    for _ in range(50):
        G = rng.integers(s.K)
        cues = rng.choice(s.M, s.n_cue, p=s.L[G])
        b = np.ones(s.K) / s.K
        for o in cues:
            b = b * s.L[:, o]; b /= b.sum()
        ci = g.cue_index(np.bincount(cues, minlength=s.M))
        oi = g.start
        assert np.allclose(g.B[ci, oi], b)
        for _ in range(s.T):
            a = rng.integers(s.A)
            r = int(rng.random() < s.R[G, a])
            b = b * (s.R[:, a] if r else 1 - s.R[:, a]); b /= b.sum()
            oi = g.nxt[oi, a, r]
            assert np.allclose(g.B[ci, oi], b, atol=1e-9)


def test_value_iteration_matches_expectimax():
    s = _spec(1, 2); g = BD.Graph(s)

    def V(b, k):
        if k == 0:
            return 0.0
        best = -1
        for a in range(s.A):
            p1 = float(b @ s.R[:, a])
            q = 0.0
            for r, pr in ((1, p1), (0, 1 - p1)):
                if pr < 1e-12:
                    continue
                b2 = b * (s.R[:, a] if r else 1 - s.R[:, a]); b2 /= b2.sum()
                q += pr * (r + V(b2, k - 1))
            best = max(best, q)
        return best

    for ci in range(len(g.cue)):
        assert np.isclose(g.V[ci, g.start], V(g.B[ci, g.start].copy(), s.T), atol=1e-9)
    live = g.level < s.T
    assert (g.V[:, live] >= g.Qmy.max(-1)[:, live] - 1e-9).all()      # information has non-negative value
    assert np.allclose(g.V[:, g.level == s.T], 0)


def test_env_rollout_and_policies():
    s = _spec(2, 3); g = BD.Graph(s); t = BD.Sim(g, "cpu")
    gen = torch.Generator(); gen.manual_seed(0)
    b = BD.rollout(None, t, 500, gen, behaviour=BD.optimal(gen))
    assert b.regret.abs().max() < 1e-6
    assert b.tok.shape == (500, s.L_seq, BD.NF) and (b.pos == torch.arange(s.n_cue, s.n_cue + s.T)).all()
    # the reward recorded in the token equals the reward paid, and the outcome state follows the moves
    for n in range(20):
        for st in range(s.T - 1):
            tk = b.tok[n, s.n_cue + st + 1]
            assert tk[BD.F_TYPE] == BD.EVT and tk[BD.F_ACT] == b.act[n, st] + 1 and tk[BD.F_REW] == b.rew[n, st] + 1
    gen.manual_seed(0)
    b2 = BD.rollout(None, t, 500, gen, behaviour=BD.optimal(gen))
    assert (b2.act == b.act).all() and (b2.rew == b.rew).all()      # common random numbers
    for arch in ("tfm", "gru"):
        net = BD.build(arch, s, d=16, layers=1, heads=2)
        b3 = BD.rollout(net, t, 64, gen)
        assert b3.act.shape == (64, s.T)
        lg, v, rec = net(b3.tok, record=True)
        assert lg.shape == (64, s.L_seq, s.A) and v.shape == (64, s.L_seq) and rec["resid"][-1].shape == (64, s.L_seq, 16)
        stk = BD.Stacked([BD.build(arch, s, d=16, layers=1, heads=2) for _ in range(2)])
        lg2, v2 = stk(b3.tok.repeat(2, 1, 1))
        assert lg2.shape == (128, s.L_seq, s.A)
        pc = BD.PPOConfig(n_env=64, minibatches=2, epochs=1)
        params = stk.trainable()
        opt = torch.optim.Adam(params, lr=1e-3)
        b4 = BD.rollout(stk, t, 128, gen)
        BD.ppo_update(stk, opt, b4, pc, 0.01, params, s.A)


def test_channel_posterior_matches_latentgoal_and_tables_match_graph():
    import numpy as np
    from goalgeo import latentgoal as LG
    s = BD.Spec(n_cue=4, T=3, clip=0.1, channel=True); g = BD.Graph(s); t = BD.Sim(g, "cpu")
    gen = torch.Generator(); gen.manual_seed(0)
    goal = torch.randint(3, (200,), generator=gen)
    cues, b0 = BD.channel_cues(t, goal, gen)
    env = LG.make_env("channel", 3)
    X = np.concatenate([np.zeros((200, 1), int), cues.numpy() + 1], 1)
    b_ref = LG.posterior(env, X)[:, -1]
    assert np.allclose(b0.numpy(), b_ref, atol=1e-5)
    # per-episode tables from a count-determined belief equal the Graph's rows
    s2 = BD.Spec(n_cue=2, T=3, clip=0.1); g2 = BD.Graph(s2); t2 = BD.Sim(g2, "cpu")
    ci = torch.arange(len(g2.cue))
    B, Q = BD.episode_tables(t2, t2.B[ci, g2.start])
    assert torch.allclose(B, t2.B[ci], atol=1e-5) and torch.allclose(Q, t2.Q[ci], atol=1e-5)
    # the channel environment runs, with zero regret under the optimal policy
    b = BD.rollout(None, t, 300, gen, behaviour=BD.optimal(gen))
    assert b.regret.abs().max() < 1e-5 and b.b.shape == (300, 3, 3) and (b.cue_id < 5 ** 4).all()
