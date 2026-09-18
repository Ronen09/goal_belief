import numpy as np
import pytest

from goalgeo.gridworld import GridWorld, ACTIONS

MAP = """
..#.
.G..
#..G
....
"""


def test_parse_free_cells_and_goals():
    env = GridWorld.from_ascii(MAP)
    assert env.height == 4 and env.width == 4
    assert env.n_states == 14  # 16 cells minus 2 walls
    assert env.coords_of(env.goal_states[0]) == (1, 1)
    assert env.coords_of(env.goal_states[1]) == (2, 3)
    assert len(env.goal_states) == 2


def test_deterministic_transitions_move_and_bump():
    env = GridWorld.from_ascii(MAP)
    s = env.state_of(0, 0)
    P = env.P  # [S, A, S']
    assert P.shape == (env.n_states, 4, env.n_states)
    assert np.allclose(P.sum(-1), 1.0)
    # up from (0,0) bumps the border -> stays
    assert P[s, ACTIONS["U"], s] == 1.0
    # left bumps -> stays
    assert P[s, ACTIONS["L"], s] == 1.0
    # down -> (1,0)
    assert P[s, ACTIONS["D"], env.state_of(1, 0)] == 1.0
    # right from (0,1) into wall at (0,2) -> stays
    s2 = env.state_of(0, 1)
    assert P[s2, ACTIONS["R"], s2] == 1.0


def test_portal_redirects_every_action():
    env = GridWorld.from_ascii(MAP, portals={(3, 3): (0, 0)})
    src = env.state_of(3, 3)
    dst = env.state_of(0, 0)
    for a in range(4):
        assert env.P[src, a, dst] == 1.0


def test_slip_mixes_actions():
    env = GridWorld.from_ascii(MAP, slip=0.2)
    s = env.state_of(1, 2)  # interior-ish cell: up (0,2) is wall, others free
    # intended D with prob 0.8 + 0.2/4 = 0.85; other 3 outcomes get 0.05 each
    row = env.P[s, ACTIONS["D"]]
    assert np.isclose(row[env.state_of(2, 2)], 0.85)
    assert np.isclose(row[s], 0.05)  # up bumps into wall -> stays
    assert np.isclose(row[env.state_of(1, 1)], 0.05)
    assert np.isclose(row[env.state_of(1, 3)], 0.05)


def test_geodesic_distances_respect_walls():
    env = GridWorld.from_ascii(MAP)
    D = env.geodesic()
    a, b = env.state_of(0, 1), env.state_of(0, 3)
    # (0,1)->(0,3) blocked by wall at (0,2): go (1,1)->(1,2)->(1,3)->(0,3)
    assert D[a, b] == 4
    assert D[a, a] == 0


def test_blocked_edge_prevents_move_both_ways():
    env = GridWorld.from_ascii(MAP, blocked_edges=[((0, 0), (0, 1))])
    a, b = env.state_of(0, 0), env.state_of(0, 1)
    assert env.P[a, ACTIONS["R"], a] == 1.0
    assert env.P[b, ACTIONS["L"], b] == 1.0
    assert env.geodesic()[a, b] == 3  # (0,0)->(1,0)->(1,1)->(0,1)


def test_per_cell_slip_map():
    slip_map = np.zeros((4, 4)); slip_map[1, 2] = 0.4
    env = GridWorld.from_ascii(MAP, slip=slip_map)
    icy, dry = env.state_of(1, 2), env.state_of(1, 3)
    # dry cell is deterministic
    assert env.P[dry, ACTIONS["D"], env.state_of(2, 3)] == 1.0
    # icy cell: intended D with prob 0.6 + 0.1 = 0.7
    assert np.isclose(env.P[icy, ACTIONS["D"], env.state_of(2, 2)], 0.7)


def test_hazard_knocks_back_with_probability():
    env = GridWorld.from_ascii(MAP, hazards={(1, 2): (0.3, (3, 0))})
    h, back = env.state_of(1, 2), env.state_of(3, 0)
    row = env.P[h, ACTIONS["D"]]
    assert np.isclose(row[env.state_of(2, 2)], 0.7)
    assert np.isclose(row[back], 0.3)
    assert np.allclose(env.P.sum(-1), 1.0)
