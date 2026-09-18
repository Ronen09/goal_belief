import numpy as np

from goalgeo.envs import make_env, ENV_NAMES
from goalgeo.planning import solve_all_goals
from goalgeo.occupancy import goal_occupancy


def test_all_envs_are_connected_and_have_enough_goals():
    for name in ENV_NAMES:
        env = make_env(name)
        D = env.geodesic()
        assert np.isfinite(D).all(), f"{name} has unreachable cells"
        assert 10 <= env.n_goals <= 20, (name, env.n_goals)
        assert env.height <= 9 and env.width <= 11


def test_base_env_size():
    env = make_env("base")
    assert env.height == 8 and env.width == 8
    assert 40 <= env.n_states <= 64


def test_portal_pockets_have_identical_occupancy_but_distant_coords():
    env = make_env("portal")
    occ = goal_occupancy(env, solve_all_goals(env, 0.9).pi, 0.9)
    a, b = env.special_pairs[0]
    assert np.allclose(occ.Z_sa[a], occ.Z_sa[b])
    assert np.linalg.norm(env.coords[a] - env.coords[b]) >= 7


def test_barrier_pairs_are_adjacent_but_far():
    env = make_env("barrier")
    D = env.geodesic()
    for a, b in env.special_pairs:
        assert np.linalg.norm(env.coords[a] - env.coords[b]) == 1
        assert D[a, b] >= 6


def test_twins_mirror_pairs_have_identical_state_occupancy():
    env = make_env("twins")
    occ = goal_occupancy(env, solve_all_goals(env, 0.9).pi, 0.9)
    for a, b in env.special_pairs:
        assert np.allclose(occ.Z_state[a], occ.Z_state[b])
        assert a != b


def test_switch_env_has_policy_switch_in_lambda():
    from goalgeo.envs import make_switch_env
    from goalgeo.gridworld import ACTIONS
    e0 = make_switch_env(0.0); e1 = make_switch_env(0.9)
    assert e0.n_goals == 10 and e0.n_states == e1.n_states
    s = e0.state_of(4, 0)                      # entrance of the short route
    g = e0.goal_coords.index((4, 12))          # goal at the far end of the short route
    pi0 = solve_all_goals(e0, 0.9).pi[g, s]; pi1 = solve_all_goals(e1, 0.9).pi[g, s]
    assert pi0[ACTIONS["R"]] == 1.0            # reliable short route: go right
    assert pi1[ACTIONS["U"]] == 1.0            # unreliable: take the long safe route


def test_corridor_env_has_runs_of_identical_policy_tables():
    from goalgeo.envs import make_corridor_env
    e = make_corridor_env()
    pi = solve_all_goals(e, 0.9).pi            # [K, S, A]
    tables = np.transpose(pi, (1, 0, 2)).reshape(e.n_states, -1)
    assert np.allclose(tables[e.state_of(0, 1)], tables[e.state_of(0, 9)])
    assert not np.allclose(tables[e.state_of(0, 9)], tables[e.state_of(0, 11)])
