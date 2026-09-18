import numpy as np

from goalgeo.gridworld import GridWorld, ACTIONS
from goalgeo.planning import solve_all_goals, rollout

MAP = """
..#.
.G..
#..G
....
"""


def test_value_iteration_gives_gamma_power_distance():
    env = GridWorld.from_ascii(MAP)
    sol = solve_all_goals(env, gamma=0.9)
    g = env.goal_states[0]  # (1,1)
    s = env.state_of(0, 0)  # distance 2
    # V_g(s) = gamma^(d-1) with reward 1 on the arrival step
    assert np.isclose(sol.V[0, s], 0.9 ** 1)
    assert np.isclose(sol.V[0, g], 0.0)  # absorbing goal, no further reward


def test_optimal_policy_is_uniform_over_ties():
    env = GridWorld.from_ascii(MAP)
    sol = solve_all_goals(env, gamma=0.9)
    s = env.state_of(0, 0)  # to (1,1): D then R or R then D
    pi = sol.pi[0, s]
    assert np.isclose(pi[ACTIONS["D"]], 0.5) and np.isclose(pi[ACTIONS["R"]], 0.5)
    assert pi.sum() == 1.0
    s2 = env.state_of(3, 3)  # to (2,3): only U
    assert sol.pi[1, s2, ACTIONS["U"]] == 1.0


def test_rollout_reaches_goal_in_optimal_steps():
    env = GridWorld.from_ascii(MAP)
    sol = solve_all_goals(env, gamma=0.9)
    rng = np.random.default_rng(0)
    traj = rollout(env, sol.pi[1], start=env.state_of(0, 0), goal=env.goal_states[1], rng=rng)
    assert traj.states[-1] == env.goal_states[1]
    assert len(traj.actions) == 5  # (0,0)->(2,3): manhattan 5, no detour needed
