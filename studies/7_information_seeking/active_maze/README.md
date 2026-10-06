# Active maze: solver-only screen of the small maze without its passive prefix

Exploratory, no network, no plan. Asked in conversation: "lets remove the prefix moves so that it becomes a true RL
environment where information seeking moves matter".

`screen.py` → `screen.md`: on the maze-belief maze, a policy that ignores the value of information (QMDP) loses 1.9 %
of the optimal value with the passive prefix and 2.0 % without it. **Removing the prefix does not make information
seeking pay in the 14-cell maze.** The larger maze of the [maze10](../maze10/) experiment was the response.
