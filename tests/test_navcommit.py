"""The navigate-commit experiment: the exact solver against two independent brute-force computations, and the belief identities."""

import itertools

import numpy as np

from goalgeo import navcommit as NC


def _joint(n, hcell, vcell, qh, qv, reports):
    """Unnormalised P(G, reports) from the generative definition; reports = {'h': y, 'v': y}."""
    w = np.full(4, 0.25)
    for g in range(4):
        if "h" in reports:
            w[g] *= qh if reports["h"] == g % 2 else 1 - qh
        if "v" in reports:
            w[g] *= qv if reports["v"] == g // 2 else 1 - qv
    return w


def _legal(n, hcell, vcell, cell, reports):
    nxt = NC.move_table(n)
    acts = [a for a in range(4) if nxt[cell, a] >= 0]
    if (cell == hcell and "h" not in reports) or (cell == vcell and "v" not in reports):
        acts.append(NC.QUERY)
    if cell in NC.corners(n):
        acts.append(NC.COMMIT)
    return acts


def _expectimax(n, c, hcell, vcell, qh, qv, cell, reports, k):
    """Optimal expected return over history-dependent policies, without a belief state: every
    quantity is a sum over goals of the joint weight of (goal, reports so far)."""
    w = _joint(n, hcell, vcell, qh, qv, reports)
    if k == 0:
        return 0.0
    nxt, best = NC.move_table(n), -np.inf
    for a in _legal(n, hcell, vcell, cell, reports):
        if a < 4:
            v = -c + _expectimax(n, c, hcell, vcell, qh, qv, nxt[cell, a], reports, k - 1)
        elif a == NC.COMMIT:
            g = int(np.where(NC.corners(n) == cell)[0][0])
            v = w[g] / w.sum()
        else:
            key = "h" if cell == hcell else "v"
            v = -c
            for y in (0, 1):
                w2 = _joint(n, hcell, vcell, qh, qv, {**reports, key: y})
                v += w2.sum() / w.sum() * _expectimax(n, c, hcell, vcell, qh, qv, cell, {**reports, key: y}, k - 1)
        best = max(best, v)
    return best


def _policies(n, hcell, vcell, cell, reports, k):
    """Every deterministic policy tree from this history: (action, {report: subtree})."""
    if k == 0:
        yield None
        return
    nxt = NC.move_table(n)
    for a in _legal(n, hcell, vcell, cell, reports):
        if a == NC.COMMIT:
            yield (a, {})
        elif a < 4:
            for sub in _policies(n, hcell, vcell, nxt[cell, a], reports, k - 1):
                yield (a, {-1: sub})
        else:
            key = "h" if cell == hcell else "v"
            subs = [list(_policies(n, hcell, vcell, cell, {**reports, key: y}, k - 1)) for y in (0, 1)]
            for s0, s1 in itertools.product(*subs):
                yield (a, {0: s0, 1: s1})


def _evaluate(policy, n, c, hcell, vcell, qh, qv, cell):
    """Exact expected return of one policy tree: a sum over the goal and both stations' noise."""
    nxt, total = NC.move_table(n), 0.0
    for g, eh, ev in itertools.product(range(4), (0, 1), (0, 1)):          # e = 1: that station errs
        pr = 0.25 * (1 - qh if eh else qh) * (1 - qv if ev else qv)
        node, p, ret = policy, cell, 0.0
        while node is not None:
            a, kids = node
            if a == NC.COMMIT:
                ret += float(NC.corners(n)[g] == p); break
            ret -= c
            if a < 4:
                p, node = nxt[p, a], kids[-1]
            else:
                y = (g % 2) ^ eh if p == hcell else (g // 2) ^ ev
                node = kids[y]
        total += pr * ret
    return total


def test_belief_is_bayes_and_normalised():
    rng = np.random.default_rng(0)
    for _ in range(20):
        qh, qv = rng.uniform(0.5, 1, 2)
        b = NC.belief(qh, qv)
        assert np.allclose(b.sum(-1), 1, atol=1e-12)
        for sh, sv in itertools.product(range(3), repeat=2):
            rep = {}
            if sh: rep["h"] = sh - 1
            if sv: rep["v"] = sv - 1
            w = _joint(5, 0, 0, qh, qv, rep)
            assert np.allclose(b[sh, sv], w / w.sum(), atol=1e-12)


def test_report_probability_is_half_whatever_the_other_station_said():
    qh, qv = 0.8, 0.65
    for other in ({}, {"v": 0}, {"v": 1}):
        w = _joint(5, 0, 0, qh, qv, other)
        for y in (0, 1):
            assert np.isclose(_joint(5, 0, 0, qh, qv, {**other, "h": y}).sum() / w.sum(), 0.5)


def test_belief_factorises_and_a_report_moves_only_its_own_marginal():
    b = NC.belief(0.8, 0.7).reshape(3, 3, 2, 2)                            # [s_h, s_v, tb, lr]
    lr, tb = b.sum(2), b.sum(3)
    assert np.allclose(b, tb[..., :, None] * lr[..., None, :])
    assert np.allclose(tb, tb[:1])                                          # s_h does not move P(tb)
    assert np.allclose(lr, lr[:, :1])


def test_solver_matches_expectimax_over_histories():
    n, c = 3, 0.03
    for (hcell, vcell, qh, qv, H) in ((1, 3, 0.8, 0.8, 5), (4, 7, 0.9, 0.6, 6), (5, 1, 0.7, 0.95, 6)):
        sol = NC.solve(n, H, c, hcell, vcell, qh, qv)
        for cell in range(n * n):
            for k in range(H + 1):
                assert np.isclose(sol.V[0, k, cell, 0, 0], _expectimax(n, c, hcell, vcell, qh, qv, cell, {}, k), atol=1e-12)
        assert np.isclose(sol.V[0, H - 2, 4, 2, 1], _expectimax(n, c, hcell, vcell, qh, qv, 4, {"h": 1, "v": 0}, H - 2), atol=1e-12)


def test_solver_matches_exhaustive_policy_enumeration():
    n, c = 3, 0.03
    for (hcell, vcell, qh, qv, start, H) in ((1, 3, 0.8, 0.8, 1, 3), (1, 3, 0.8, 0.8, 4, 4), (1, 5, 0.9, 0.7, 1, 4),
                                             (3, 1, 0.6, 0.8, 0, 4)):
        sol = NC.solve(n, H, c, hcell, vcell, qh, qv)
        best = max(_evaluate(p, n, c, hcell, vcell, qh, qv, start) for p in _policies(n, hcell, vcell, start, {}, H))
        assert np.isclose(sol.V[0, H, start, 0, 0], best, atol=1e-12)


def test_legal_mask_and_ties():
    h, v = NC.all_layouts(5)
    assert len(h) == 21 * 20
    sol = NC.solve(5, 8, 0.02, h[:3], v[:3], 0.8, 0.8)
    legal = NC.legal_mask(5, h[:3], v[:3])
    assert np.all(np.isfinite(sol.Q[:, 1:]) == legal[:, None])
    opt = sol.optimal()
    assert np.all(opt[:, 1:].any(-1)) and not opt[:, 0].any()
    assert np.all(sol.regret()[:, 1:][opt[:, 1:]] <= NC.TIE)
    # centre of the grid, nothing known, no station within reach: all four moves tie by symmetry
    far = NC.solve(5, 4, 0.02, 1, 3, 0.8, 0.8, max_queries=0)
    assert far.optimal()[0, 4, 12, 0, 0, :4].all()


def test_restricting_queries_never_helps():
    h, v = NC.all_layouts(5)
    vals = [NC.solve(5, 12, 0.03, h[::37], v[::37], 0.8, 0.8, max_queries=m).V for m in (0, 1, 2)]
    assert np.all(vals[0] <= vals[1] + 1e-12) and np.all(vals[1] <= vals[2] + 1e-12)
