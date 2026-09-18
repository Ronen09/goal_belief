"""Named 8x8 layouts.

base    - Phase 1-5, 7, 8: open grid with an L-shaped wall and a stub, 16 goals.
barrier - Phase 6: thin wall between columns 3 and 4 (rows 0-6), one gap at the
          bottom row. Cells (r,3)/(r,4) are Euclidean-adjacent but geodesically far.
portal  - Phase 6: two corner pocket cells whose every action teleports to the
          same hub cell. The pockets have identical Z_sa and are ~10 cells apart.
ice     - Phase 9: base map with alternating icy rows (per-cell slip), so exact
          probabilistic occupancy and shortest-path γ^T geometry diverge.
twins   - Phase 6 (9x11): two mirror-image rooms above/below a goal row that is
          the mirror axis, so mirror pairs share Z_state exactly (and Z_sa up to
          the U/D block swap) while being up to 8 cells apart.
"""

from __future__ import annotations

from .gridworld import GridWorld

BASE = """
...G..G.
.G..G..G
..###...
G.#..G..
..#.G...
.G...##G
G...G...
G..G...G
"""

BARRIER = """
G..G....
....G..G
G.......
.G...G..
....G..G
G.......
.G...G.G
....G...
"""

PORTAL = """
.G..#..G
....#...
G.......
...G.G.G
G.......
..G.#...
....#.G.
G...#G..
"""

TWINS = """
...#.......
.#...#.#...
...#...#.#.
##.###.###.
GGGGGGGGGGG
##.###.###.
...#...#.#.
.#...#.#...
...#.......
"""

SWITCH = """
...G.....G..G
.###########G
G###########G
.###########G
..G..HHH..G.G
"""

CORRIDOR = "G.........G.........G"

RING = """
G.....G......
.###########.
.###########.
.###########G
.###########.
G###########.
.###########.
.###########.
......G......
"""

ENV_NAMES = ["base", "barrier", "portal", "twins"]


def make_switch_env(lam: float) -> GridWorld:
    """Policy-switch ring (TASK2, Task 1): short route along row 4 through three
    hazard cells that knock the agent back to (4,0) with probability ``lam``;
    long safe route along row 0. Route lengths from (4,0) to (4,12): 12 vs 20."""
    rows = SWITCH.strip().splitlines()
    haz = {(i, j): (lam, (4, 0)) for i, r in enumerate(rows) for j, c in enumerate(r) if c == "H"}
    env = GridWorld.from_ascii(SWITCH, name=f"switch{lam:.2f}", hazards=haz)
    env.hazard_cells = sorted(haz)
    return env


def make_ring_env() -> GridWorld:
    """9x13 ring of 40 cells with 5 goals (TASK3, Task 14): along each arc between
    goal antipodes the optimal action is constant while Q margins vary with distance."""
    return GridWorld.from_ascii(RING, name="ring")


def make_corridor_env() -> GridWorld:
    """1x21 corridor with goals at 0, 10, 20 (TASK2, Task 5): two runs of nine
    cells with identical optimal-action tables but very different occupancy."""
    return GridWorld.from_ascii(CORRIDOR, name="corridor")
ICE_SLIP = 0.9


def make_env(name: str, slip: float = 0.0) -> GridWorld:
    if name == "base":
        return GridWorld.from_ascii(BASE, slip=slip, name=name)
    if name == "ice":
        # base map; every even row is icy (action replaced by a random one w.p. ICE_SLIP)
        import numpy as np
        slip_map = np.zeros((8, 8)); slip_map[::2, :] = ICE_SLIP
        return GridWorld.from_ascii(BASE, slip=slip_map, name=name)
    if name == "barrier":
        edges = [((r, 3), (r, 4)) for r in range(7)]
        env = GridWorld.from_ascii(BARRIER, slip=slip, name=name, blocked_edges=edges)
        env.special_pairs = [(env.state_of(r, 3), env.state_of(r, 4)) for r in range(5)]
        return env
    if name == "portal":
        hub = (3, 4)
        env = GridWorld.from_ascii(PORTAL, slip=slip, name=name,
                                   portals={(0, 0): hub, (7, 7): hub})
        env.special_pairs = [(env.state_of(0, 0), env.state_of(7, 7))]
        return env
    if name == "twins":
        env = GridWorld.from_ascii(TWINS, slip=slip, name=name)
        pairs = []
        for r in range(4):
            for c in range(env.width):
                if env.grid[r, c]:
                    pairs.append((env.state_of(r, c), env.state_of(8 - r, c)))
        env.special_pairs = pairs
        return env
    raise ValueError(name)
