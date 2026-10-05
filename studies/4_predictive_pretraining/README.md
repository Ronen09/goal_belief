# Predictive pretraining

A backbone trained only to predict the maze, reused for control by a small goal-conditioned head. It transfers best, and a belief-encoding edit controls the new head; the original policy reads the same change only weakly, because it recomputes the evidence from the raw tokens.

| experiment | question | report | cost |
|---|---|---|---|
| [Predictive transfer](predictive_transfer/) | Prediction-only, reward-only and random backbones, frozen, under a small goal-conditioned head trained on optimal actions | [report](predictive_transfer/REPORT.md) | 30 min GPU |
| [Belief-encoding edit](belief_encoding_edit/) | Belief-encoding edits of the frozen state, read by the predictive-transfer experiment's frozen goal-conditioned heads | [report](belief_encoding_edit/REPORT.md) | 5 min GPU |
| [Policy belief edit](policy_belief_edit/) | One belief-encoding edit at the pre-goal interface, read by the original policy and by the new head | [report](policy_belief_edit/REPORT.md) | 3 min GPU |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py predictive_transfer`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
