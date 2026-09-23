"""Figure style for rounds 12-16: one categorical order (goal, act_soft, act_hard, next_obs; untrained
networks neutral grey), muted axes, light grid. Rounds 1-11 use `goalgeo.plotting`."""

from __future__ import annotations

COL = {"goal": "#2a78d6", "act_soft": "#eb6834", "act_hard": "#1baf7a", "next_obs": "#eda100", "init": "#9a9a94"}
INK, MUTED, GRID = "#2b2b28", "#6b6a63", "#e4e3dc"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
