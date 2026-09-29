"""Round 18: the maze filter and solver against computations that use no belief state."""

import itertools

import numpy as np

from goalgeo import mazebelief as MB


def small(seed=3, H=4, **kw):
    return MB.random_maze(seed, n=6, rows=3, cols=3, pairs=1, n_goals=2, H=H, **kw)


def _joint(m, obs, acts, goal):
    """Unnormalised P(current cell, symbols, no termination), by following every start cell."""
    w = np.zeros(m.n)
    for s0 in range(m.n):
        p, s = m.prior()[s0] * m.E[s0, obs[0]], s0
        for a, o in zip(acts, obs[1:]):
            s = m.nxt[s, a]
            if goal is not None and s == goal:
                p = 0.0
            p *= m.E[s, o]
        w[s] += p
    return w


def _expectimax(m, goal, obs, acts, k):
    """Optimal expected discounted return from a raw history, as a sum over start cells (unnormalised)."""
    if k == 0:
        return 0.0
    best = -1.0
    for a in range(4):
        v = 0.0
        for s0 in range(m.n):                                   # reward now: histories that enter the goal with this move
            p, s = m.prior()[s0] * m.E[s0, obs[0]], s0
            for a_, o in zip(acts, obs[1:]):
                s = m.nxt[s, a_]
                p *= 0.0 if s == goal else m.E[s, o]
            v += p * (m.nxt[s, a] == goal)
        for o in range(m.n_sym):
            v += m.gamma * _expectimax(m, goal, obs + [o], acts + [a], k - 1)
        best = max(best, v)
    return best


def test_filter_matches_enumeration():
    m = small()
    rng = np.random.default_rng(0)
    for goal in (None, m.goals[0]):
        for _ in range(30):
            acts = rng.integers(0, 4, 4).tolist()
            obs = rng.integers(0, m.n_sym, 5).tolist()
            w = _joint(m, obs, acts, goal)
            if w.sum() < 1e-12:
                continue
            b = m.filter(obs, acts, goal)[-1]
            assert np.isclose(b.sum(), 1) and np.allclose(b, w / w.sum(), atol=1e-12)


def test_goal_enters_the_posterior_only_through_survival():
    m = small()
    obs, acts = [0, 1, 0], [1, 3]
    free = m.filter(obs, acts)
    for g in m.goals:
        b = m.filter(obs, acts, g)
        assert np.allclose(b[0], free[0])                        # before any move the goal is irrelevant
        assert b[-1][g] == 0


def test_solver_matches_history_expectimax():
    for seed in (3, 5):
        m = small(seed, H=3)
        for g in m.goals:
            S = MB.Solver(m, g)
            for o in range(m.n_sym):
                b, z = m.observe(m.prior(), o)
                if z > 0:
                    assert np.isclose(z * S.v(b, m.H), _expectimax(m, g, [o], [], m.H), atol=1e-12)


def test_known_location_is_the_mdp():
    m = small(H=5)
    g = m.goals[0]
    Q, S = m.mdp_values(g), MB.Solver(m, g)
    for s in range(m.n):
        if s != g:
            b = np.zeros(m.n); b[s] = 1
            assert np.allclose(S.q(b, 5), Q[5, s], atol=1e-12)


def test_heuristics_never_beat_the_optimum():
    m = MB.random_maze(1, H=6)
    for g in m.goals:
        v = MB.Solver(m, g).start_value()
        for pol in (MB.map_policy, MB.qmdp_policy):
            assert MB.Solver(m, g, pol(m, g)).start_value() <= v + 1e-12


def test_graph_matches_solver_and_filter():
    from goalgeo import mazegraph as MG
    m = MB.cross_maze(H=5)
    G = MG.build(m, max_prefix=2)
    rng = np.random.default_rng(0)
    for gi, g in enumerate(m.goals):
        S = MB.Solver(m, g)
        for _ in range(40):
            L = rng.integers(0, 3)
            obs = [int(rng.choice(np.nonzero(G.start >= 0)[0]))]
            node, acts = G.start[obs[0]], []
            for _ in range(L):                                          # passive prefix
                a = int(rng.integers(4)); o = int(rng.choice(np.nonzero(G.nxt[node, a] >= 0)[0]))
                node = G.nxt[node, a, o]; acts.append(a); obs.append(o)
            assert np.allclose(G.belief[node], m.filter(obs, acts)[-1], atol=1e-9)
            node = G.reveal[node, gi]
            b = G.belief[node]
            for k in range(m.H, 0, -1):
                assert G.k[node] == k and np.allclose(G.Q[node], S.q(b, k), atol=1e-9)
                a = int(rng.integers(4))
                live = np.nonzero(G.nxt[node, a] >= 0)[0]
                if k == 1 or len(live) == 0:
                    break
                o = int(rng.choice(live))
                _, q, _ = m.move(b, a, g)
                b, _ = m.observe(q, o)
                node = G.nxt[node, a, o]
                assert np.allclose(G.belief[node], b, atol=1e-9)


def test_solver_occupancy_matches_simulation_and_value():
    import torch
    from goalgeo import mazegraph as MG, mazeocc as MO, mazeppo as P
    m = MB.cross_maze(H=5)
    G = MG.build(m, max_prefix=2)
    D, per_action = MO.solver_occupancy(G)
    dec = np.nonzero(G.k == m.H)[0]
    # the occupancy of the goal cell is the value: visits to the goal happen once, and pay gamma^(k-1)
    for gi, g in enumerate(m.goals):
        sel = dec[G.goal[dec] == gi]
        assert np.allclose(D[sel][:, g], G.V[sel], atol=1e-5)
    # total discounted occupancy = expected discounted time alive
    assert np.all(D[dec].sum(1) <= sum(m.gamma ** k for k in range(m.H)) + 1e-5)
    # against simulation of the solver's policy
    t = P.Tables(G, "cpu", max_prefix=2)
    gen = torch.Generator().manual_seed(0)
    node = torch.tensor(dec[[3, 40, 77]]).repeat_interleave(20000)
    goal = torch.tensor(G.goal)[node]
    cell = torch.multinomial(t.belief[node], 1, generator=gen).squeeze(1)
    occ = torch.zeros(len(node), m.n); alive = torch.ones(len(node), dtype=torch.bool)
    opt = torch.tensor(G.optimal())
    for k in range(m.H):
        a = torch.multinomial(opt[node].float(), 1, generator=gen).squeeze(1)
        cell = torch.where(alive, t.nxt_cell[cell, a], cell)
        occ[torch.arange(len(node)), cell] += (m.gamma ** k) * alive
        hit = alive & (cell == t.goal_cell[goal])
        o = torch.multinomial(t.E[cell], 1, generator=gen).squeeze(1)
        nxt = t.node_nxt[node, a, o]
        alive = alive & ~hit & (nxt >= 0)
        node = torch.where(alive, nxt, node)
    est = occ.view(3, 20000, m.n).mean(1).numpy()
    assert np.abs(est - D[dec[[3, 40, 77]]]).max() < 0.02
