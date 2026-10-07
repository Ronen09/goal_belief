# Information seeking in a larger maze

A 55-cell maze without a solver: no passive prefix, random spawns, the exact filter as the only ground truth. Trained by reward, the agents reach the level of QMDP (and pass it with a goal from anywhere) and localise faster than it does; the additive policy, run as a policy and enforced as a state, keeps three quarters to four fifths of the goal-directed return, with interior goals as with spread ones; two thirds when two goals are collected or when every cell can be the goal, although the fully observed task is as additive as ever (the additive ceiling).

| experiment | question | report | cost |
|---|---|---|---|
| [Active maze](active_maze/) | Solver-only screen: does removing the passive prefix make information seeking pay in the small maze? (No: QMDP loses 2 % either way.) | [note](active_maze/README.md) | 20 s GPU |
| [maze10](maze10/) | A 55-cell aliased, noisy maze without a solver: competence against QMDP, information seeking from behaviour, the additive code as a policy and under online removals, occupancy decoding | [report](maze10/REPORT.md) | 1.5 h GPU |
| [Interior goals](interior_goals/) | The same maze with six junction goals: one goal per episode, and two goals to collect. Is the additive policy a consequence of goal placement? Does collection break it? | [report](interior_goals/REPORT.md) | 2.5 h GPU |
| [Additive ceiling](additive_ceiling/) | No network: given the cell, how often can the best rule argmax H(cell) + G(goal) pick a shortest-path move, for each maze and goal set? (0.93–0.97 in the larger maze whatever the goal set.) | [tables](additive_ceiling/tables.md) | 15 min CPU |
| [Random goals](random_goals/) | The same maze with every cell a possible goal, one per episode: does a goal from anywhere force the goal × history interaction? | [report](random_goals/REPORT.md) | 2 h GPU |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py maze10`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
