# Round 25: arms, measures, decision rule and expectations

Written before any model of this round was trained. Code: `goalgeo/mazeaux.py`, `rounds/r23_obs_prediction/train.py`
(option `--sup`), `rounds/r25_balanced_prediction/measure.py`.

## Arms

Task, transformer, PPO settings, 1 500 updates, 4 096 episodes per update, seeds 0–9 (the same seed has the same
initialisation in every arm) and the auxiliary coefficient (1.0) are round 23's. The arms differ only in what the
head is given as targets.

| arm | targets at a token | labels per token | models |
|---|---|---|---|
| selected | the symbol that followed the move actually taken (round 23) | 1 | round 23's `aux1`, reused: the definition and code path are unchanged |
| balanced, all | for each of the four candidate moves, a symbol sampled by the simulator from the true cell that move leads to | 4 | trained here |
| balanced, one | the same for one candidate move drawn uniformly per token | 1 | trained here |
| reward only | none | 0 | round 23's `reward`, for reference on the policy measures |

* **Counterfactual symbols** are sampled from the emission table at the cell the candidate move would reach from the
  true cell. They are samples of an observation; the posterior, the cell and the exact predictive distribution are
  never given. They are drawn independently of the symbol the episode goes on to show.
* **The same tokens carry targets in every arm**: prefix tokens followed by an event, the goal token and later
  event tokens, at decisions with at least two moves left. After the reveal a candidate move that enters the goal
  has no next symbol and no target (the selected arm has none there either, because the episode ends). The balanced
  arms have about 2 % more tokens with a target: decisions where the move taken entered the goal and another
  candidate would not. Checked before training on 400 000 episodes of an untrained policy: every selected-arm
  token is a balanced-arm token, and the counterfactual symbols' cross-entropy under the exact predictive
  distribution equals its entropy (0.6952 nats), so they are samples from it.
* **Weight and budget.** The auxiliary loss is the mean cross-entropy over the targets present, times 1.0, in every
  arm: the total weight is the same, and in *balanced, all* each label has a quarter of it. Updates, episodes and
  gradient clip are the same. *Balanced, one* has the selected arm's number of labels and the balanced arm's
  coverage of moves; it separates coverage from the number of labels.

## Measures (fixed; round 22's pairs, round 24's code, final checkpoint)

Prediction, at the goal token, in total variation over the three symbols, all four candidate moves weighted
equally (round 24's registered measure):

* **P1 error**: between the head and the exact predictive distribution, on type A histories. The histories are from
  the held-out bank; no arm sees the exact distribution in training.
* **P2 identical-posterior inconsistency**: D_same, between the head's predictions for the two histories of a type A
  pair.
* Secondary: both for the greedy move, for the other three moves and at the last prefix token; D_same relative to
  D_diff (type C); excess cross-entropy on the evaluation episodes (taken move).

Policy, natural access, no patch (round 23's):

* **Q1 policy inconsistency**: type A, TV between the two histories' action distributions (O1). Secondary: relative
  to type C; greedy changes.
* Competence: greedy regret and the number of seeds failing a goal; O2 (greedy changes where the optimal set is the
  same).

## Decision rule

As in round 23: a measure is *reduced* in an arm if its median over the ten seeds is below the selected arm's and a
one-sided exact rank-sum test gives p < 0.05. Primary comparison: *balanced, all* against *selected*.

| question | decided by |
|---|---|
| 1. does balanced supervision reduce prediction error and identical-posterior inconsistency? | P1 and P2 both reduced |
| 2. if so, does policy inconsistency also decrease? | Q1 reduced |

| result | reading |
|---|---|
| P1, P2 reduced; Q1 not reduced, and its median within 10 % of the selected arm's | the dissociation: predictions become more belief-consistent and the policy does not |
| P1, P2 reduced; Q1 reduced | a connection between the two; mediation not established |
| P1, P2 reduced; Q1 not reduced by the rule but its median more than 10 % lower | undecided at ten seeds |
| P1 or P2 not reduced | question 2 does not arise |

Across seeds within the balanced arm, the rank correlation between P2 and Q1 is reported, with no threshold.
Regret is reported for every arm; a reading of Q1 is made only if regret is not worse than the selected arm's by
the same rule.

## Expectations

| | expectation |
|---|---|
| E1 | *balanced, all* reduces P1 from 0.043 to below 0.025 and P2 from 0.026 to below 0.018 |
| E2 | the reduction is in the moves the policy does not take; on the greedy move P1 and P2 change by less than a quarter |
| E3 | Q1 is not reduced (the dissociation) |
| E4 | regret is not worse in the balanced arms |
| E5 | *balanced, one* reduces P1 and P2 by at least half as much as *balanced, all*: coverage matters more than the number of labels |

I hold E3 weakly. Round 24 found the two inconsistencies in the same cells, which points to a shared state; if
balanced targets change that state, Q1 should move too.

Limits known in advance: one maze, one architecture, ten seeds, one coefficient; the selected arm was trained in
round 23 (same code path, earlier run); counterfactual samples add label noise that the exact predictive
distribution would not.
