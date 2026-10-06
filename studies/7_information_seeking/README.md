# Information seeking in a larger maze

A 55-cell maze without a solver: no passive prefix, random spawns, the exact filter as the only ground truth. Trained by reward, the agents reach the level of QMDP and localise faster than it does; the additive policy, run as a policy and enforced as a state, keeps three quarters to four fifths of the goal-directed return.

| experiment | question | report | cost |
|---|---|---|---|
| [Active maze](active_maze/) | Solver-only screen: does removing the passive prefix make information seeking pay in the small maze? (No: QMDP loses 2 % either way.) | [note](active_maze/README.md) | 20 s GPU |
| [maze10](maze10/) | A 55-cell aliased, noisy maze without a solver: competence against QMDP, information seeking from behaviour, the additive code as a policy and under online removals, occupancy decoding | [report](maze10/REPORT.md) | 1.5 h GPU |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py maze10`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
