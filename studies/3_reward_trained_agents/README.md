# Reward-trained agents

Agents trained by reward alone, first in a navigate-and-commit grid and then in an aliased maze where the agent never sees its cell. The belief is decodable before training; what training changes is whether the evidence is used. Edits along the decoded belief do not steer the policy, history matters a little beyond the belief, and an observation-prediction objective improves regret without making the policy more belief-consistent.

| experiment | question | report | cost |
|---|---|---|---|
| [Navigate–commit](navigate_commit/) | How does a full-attention transformer come to build and use a belief while its policy improves? | [report](navigate_commit/REPORT.md) | 4 h GPU |
| [Maze belief](maze_belief/) | Does an inferred location belief support goal-dependent decisions? | [report](maze_belief/REPORT.md) | 1.5 h GPU + CPU |
| [Maze occupancy](maze_occupancy/) | Is goal-conditioned occupancy represented beyond the posterior and the action values? | [report](maze_occupancy/REPORT.md) | 25 min GPU |
| [Belief edit](belief_edit/) | Does the policy use the decoded belief? Edits at the goal-free prefix interface | [report](belief_edit/REPORT.md) | 20 min GPU |
| [Pattern specificity](pattern_specificity/) | Is the Sigma-W edit specific? Replication with PCA, random and posterior-matched controls | [report](pattern_specificity/REPORT.md) | 2 min GPU |
| [Pair types](pair_types/) | Three pair types: same posterior; same action under one goal; different action. Whole and PCA patches, all goals | [report](pair_types/REPORT.md) | 2 min GPU |
| [Observation prediction](obs_prediction/) | Reward only against reward plus next-symbol prediction, on the pair-types experiment's fixed pairs | [report](obs_prediction/REPORT.md) | 2.5 h GPU |
| [Head consistency](head_consistency/) | Does the prediction head agree on identical-posterior histories? | [report](head_consistency/REPORT.md) | 1 min GPU |
| [Balanced prediction](balanced_prediction/) | Selected-action against balanced candidate-action supervision of the prediction head | [report](balanced_prediction/REPORT.md) | 1.5 h GPU |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py navigate_commit`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
