import numpy as np

from goalgeo.gridworld import GridWorld
from goalgeo.planning import solve_all_goals
from goalgeo.occupancy import goal_occupancy, successor_representation

MAP = """
..#.
.G..
#..G
....
"""


def test_state_action_occupancy_equals_gamma_power_arrival_time():
    env = GridWorld.from_ascii(MAP)
    gamma = 0.9
    sol = solve_all_goals(env, gamma=gamma)
    occ = goal_occupancy(env, sol.pi, gamma=gamma)
    D = env.geodesic()
    # rho(g|s,a) = gamma^T where T = 1 + d(next(s,a), g)
    for gi, g in enumerate(env.goal_states):
        for s in range(env.n_states):
            for a in range(4):
                s_next = int(np.argmax(env.P[s, a]))
                T = 0 if s == g else 1 + D[s_next, g]
                assert np.isclose(occ.rho_sa[s, a, gi], gamma ** T), (s, a, gi)
    # state occupancy rho(g|s) = gamma^d(s,g)
    for gi, g in enumerate(env.goal_states):
        for s in range(env.n_states):
            assert np.isclose(occ.rho_s[s, gi], gamma ** D[s, g])


def test_occupancy_shapes():
    env = GridWorld.from_ascii(MAP)
    sol = solve_all_goals(env, gamma=0.9)
    occ = goal_occupancy(env, sol.pi, gamma=0.9)
    assert occ.Z_sa.shape == (env.n_states, 4 * env.n_goals)
    assert occ.Z_state.shape == (env.n_states, env.n_goals)
    assert occ.Q.shape == (env.n_goals, env.n_states, 4)
    assert np.allclose(occ.Q[0], occ.rho_sa[:, :, 0])


def test_unreachable_goal_has_zero_occupancy():
    env = GridWorld.from_ascii("""
G#.
.#G
""")
    sol = solve_all_goals(env, gamma=0.9)
    occ = goal_occupancy(env, sol.pi, gamma=0.9)
    assert occ.rho_s[env.state_of(0, 0), 1] == 0.0
    assert occ.rho_s[env.state_of(1, 2), 0] == 0.0


def test_successor_representation_rows_sum_to_one():
    env = GridWorld.from_ascii(MAP)
    M = successor_representation(env, gamma=0.9)
    assert M.shape == (env.n_states, env.n_states)
    assert np.allclose(M.sum(1), 1.0)
