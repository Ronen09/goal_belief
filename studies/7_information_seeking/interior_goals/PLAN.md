# Interior goals: tasks, measures, decision rule and expectations

Written before any model of this experiment was trained. Reference policies and a fully observed check (no network)
were run first and are given below. Brief: `BRIEF.md`. Code: `goalgeo/bigmaze.py` (`junctions`), `goalgeo/bigcollect.py`,
`train.py`, `measure_collect.py`, and the maze10 experiment's `measure.py` for the single-goal arm.

## Tasks

The maze10 maze, symbols, landmarks, noise (0.4), spawns (any cell that is no goal), 40 moves, discount 0.97. The goals
are moved to **six junction cells** (three or more neighbours, no landmark; the most central junction, then
farthest-point sampling among junctions): none is in a corner or a dead end.

```
a b ★ a b b b █ a
█ █ █ █ █ █ a █ b
a b a a a █ a █ a
a █ █ █ a █ a █ a
3 b b a b b 1 a b
b █ █ █ b █ ★ █ b
b █ b b 4 █ b a 5
a █ b █ b █ b █ b
a a 2 a b b 6 b b
```

| arm | episode | what it isolates |
|---|---|---|
| **single** | one of the six goals, shown at the start (the maze10 task with the goals moved) | goal placement |
| **collect** | two of the six goals (15 pairs), shown at the start as two goal tokens; each pays when first entered, in any order; a pickup is announced with the goal's name; the episode ends when both are collected | collection on top of placement |

A pickup names the goal, so the agent's location is known from then on (moves are deterministic). The uncertain part of
a collection episode is the way to the first goal, and which goal to take first.

Reference policies on 20 000 episodes (return; the oracle knows its cell):

| | oracle | QMDP + lookahead | QMDP | most likely cell | random |
|---|---|---|---|---|---|
| single | 0.847 | 0.678 | 0.662 | 0.608 | 0.137 |
| collect | 1.633 | 1.367 | 1.345 | 1.249 | 0.277 |

**A check without a network.** If the agent knew its cell, how often would a fixed preference per goal pick a
shortest-path move? The most common first move toward a goal covers 0.44 of the cells for the maze10 goals and 0.43
for the junction goals; the best additive fit H(cell) + G(goal) to the shortest-path values picks a shortest-path move
in 0.75 and 0.80. **By this check the junction goals are not harder for an additive policy than the maze10 goals.** My
reading in the maze10 report, that corner goals favour a fixed bias, has no support from it. The trained models decide.

Training: as maze10 (PPO on reward only, the same network, 1 000 updates of 4 096 episodes), six seeds per arm.

## Measures

**Single arm**: the maze10 experiment's `measure.py`, unchanged: behaviour against the references; the additive policy
argmax H + G run in the environment, with its recovery (additive − goal-blind) / (natural − goal-blind); the
interaction removed at the decision token online; information seeking; decoders.

**Collect arm** (`measure_collect.py`), on 4 096 fixed episodes:
* behaviour: return, goals collected, both collected, against the references; **first target**: the share of
  episodes whose first pickup is the goal nearer to the spawn by path, against the references;
* information seeking before the first pickup: deviations from QMDP and their expected information gain, posterior
  entropy by step, as in maze10;
* **the additive policy.** The goal state is the set of goals still to collect (15 pairs before the first pickup, 6
  single goals after it). H(h) = the logits averaged over the alternative goal states for the same history, the goal
  tokens replaced: before a pickup the pairs, after the pickup of X the pairs that contain X. Alternatives the history
  rules out are left out (a pair containing a goal cell the agent has walked over without a pickup). G(goal state,
  step) = the mean deviation on separate episodes, per step before the pickup and per step since it afterwards.
  Run in the environment: natural; **additive** (argmax H + G); goal-blind (argmax H); history-blind (argmax G).
  Recovery as in maze10, on return; and the share of decisions where the additive choice is the model's move, before
  and after the first pickup.

## Decision rule

Six models per arm; medians. Arms are compared with the maze10 models by exact Mann–Whitney tests (the initialisations
differ: the goal embedding has another size), one-sided, p < 0.05.

| | criterion |
|---|---|
| **LEARN** (each arm) | greedy return ≥ the most-likely-cell policy's |
| **AD** (each arm) | additive policy: recovery ≥ 0.9 |
| **PLACE** | single arm: recovery lower than maze10's (0.81) by ≥ 0.15 |
| **COLLECT** | collect arm: recovery lower than the single arm's by ≥ 0.15 |

| result | reading |
|---|---|
| PLACE | **the corner goals were why the additive policy did so well in maze10** |
| not PLACE, single arm's recovery ≥ 0.7 | goal placement is not why: the policy is mostly additive with junction goals too |
| COLLECT | collecting several goals needs the goal × history interaction that single goals did not |
| not COLLECT, collect arm's recovery ≥ 0.7 | the additive form extends to sets of goals |
| not LEARN (an arm) | that arm is not interpreted |

## Expectations

| | expectation |
|---|---|
| E1 | LEARN holds in both arms |
| E2 | single arm: recovery 0.65–0.85; PLACE fails |
| E3 | single arm: interaction removed at the decision token, recovery 0.6–0.8 |
| E4 | collect arm: recovery 0.5–0.75; COLLECT holds |
| E5 | collect arm: the additive choice matches the model's move less often after the first pickup than before it |
| E6 | collect arm: the models take the nearer goal first more often than the most-likely-cell policy, less often than QMDP |

## Limits known in advance

No optimal policy. One maze; the junction goals lie on the lower two thirds of it. In the collect arm a pickup ends
the uncertainty; alternatives replace goal tokens in a history the agent produced under its own goals; no state-level
removal and no decoders for the collect arm. G indexed by goal state and step.
