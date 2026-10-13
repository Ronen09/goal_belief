"""The bump maze: the exact filter against the true state, the solver's values, and minigrid's dynamics against
the module's transition table."""

import numpy as np
import pytest
import torch

from goalgeo import bumpmaze as BMZ


@pytest.fixture(scope="module")
def world():
    return BMZ.make(0, 9, 4)


def test_layout_and_transitions(world):
    w = world
    assert w.n_cells == 35 and w.n_states == 140
    assert not w.at_goal[w.start_states].any()
    for s in range(w.n_states):                      # a turn keeps the cell; forward keeps the heading
        assert w.nxt[s, 0] // 4 == s // 4 and w.nxt[s, 1] // 4 == s // 4 and w.nxt[s, 2] % 4 == s % 4
        assert w.nxt[w.nxt[s, 0], 1] == s
    assert np.isfinite(w.dist[w.start_states]).all()


def test_filter_matches_truth_and_solver_sets(world):
    w = world; sol = BMZ.solve(w, max_sets=100_000)
    gen = torch.Generator().manual_seed(1); e = BMZ.Env(w, 256, gen)
    for t in range(w.H):
        a = torch.randint(3, (256,), generator=gen); e.step(a)
        alive = ~e.done
        if not alive.any(): break
        m = e.filter.mask[alive]
        assert bool(m[torch.arange(int(alive.sum())), e.s[alive]].all())       # the true state is always consistent
        for row in m.numpy()[:8]:                                               # every live belief is a reachable set
            assert np.packbits(row).tobytes() in sol["sets"]
    # the numpy filter agrees with the torch one
    m = np.zeros(w.n_states, dtype=bool); m[w.start_states] = True
    f = BMZ.Filter(w, 1)
    for a, o in ((2, 1), (0, 0), (2, 0), (1, 0), (2, 0)):
        m = BMZ.step_belief(w, m, a, o); f.update(torch.tensor([a]), torch.tensor([o]))
        assert (f.mask[0].numpy() == m).all()


def test_solver_values(world):
    w = world; sol = BMZ.solve(w, max_sets=100_000)
    V = sol["V"]; assert 0 < V[0] < 1 and (V <= 1).all()
    # a singleton belief at state s has value gamma^(dist - 1), the known-state optimum
    for s in w.start_states[:20]:
        m = np.zeros(w.n_states, dtype=bool); m[s] = True
        k = np.packbits(m).tobytes()
        if k in sol["sets"]:
            assert abs(V[sol["sets"][k]] - w.gamma ** (w.dist[s] - 1)) < 1e-6


def test_minigrid_agrees(world):
    pytest.importorskip("minigrid")
    w = world; rng = np.random.default_rng(0)
    for s0 in rng.choice(w.start_states, 5, replace=False):
        env = BMZ.minigrid_env(w, int(s0)); env.reset(seed=0)
        assert env.state() == s0
        s = int(s0)
        for _ in range(30):
            a = int(rng.integers(3))
            bump, reward, term, trunc, _ = env.step(a)
            s2 = int(w.nxt[s, a])
            assert env.state() == s2 and bump == int(a == 2 and s2 == s)
            assert term == bool(w.at_goal[s2])
            if term: break
            s = s2
