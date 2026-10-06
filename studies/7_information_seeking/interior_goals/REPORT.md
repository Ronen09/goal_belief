# Interior goals, and two goals to collect (interior goals)

Run date: 2026-10-06/07. Brief: `BRIEF.md` (given in conversation). Tasks, measures and decision rule: `PLAN.md`,
committed (`9b7fcc9`) **before any model was trained**; reference policies and a fully observed check came first and
are given there. Numbers from `tables.md`; data in `results_single.json`, `results_collect.json`, `order.json` (post
hoc), training logs in `runs/`. Reproduce: `reproduce.py interior_goals` (2.5 h on one GPU).

**Setting.** The maze10 maze, symbols, landmarks and noise, with the goals moved to six junction cells, none in a
corner or a dead end (`PLAN.md` shows them). Two arms, six reward-trained models each:
* **single**: one of the six goals per episode. It differs from maze10 in the goals only.
* **collect**: two of the six goals per episode, collected in any order; a pickup is announced with the goal's name.

No solver. References from the exact filter: the oracle knows its cell; QMDP ignores the value of information.

## 1. Both tasks are learned to about the level of QMDP

| | maze10 (four spread goals) | single junction goal | collect two junction goals |
|---|---|---|---|
| **models: return** | 0.546 | 0.665 (0.663–0.671) | 1.327 (1.321–1.338) |
| models: success (collect: both goals) | 0.84 | 0.99 | 1.00 |
| oracle | 0.786 | 0.844 | 1.628 |
| QMDP + lookahead | 0.564 | 0.676 | 1.370 |
| QMDP | 0.551 | 0.661 | 1.346 |
| most likely cell | 0.489 | 0.606 | 1.244 |

* **LEARN holds in both arms** (p 0.016 each). All six seeds of each arm end alike; no goal is left unlearned, unlike
  maze10's dead-end goal.
* Single: level with QMDP (0.665 against 0.661). Collect: between the most-likely-cell policy and QMDP.
* As in maze10, the models localise faster than QMDP on the same episodes (posterior entropy after 20 moves: 0.71
  against 1.39 single; 0.58 against 1.07 before the first pickup in collect), and their deviations from QMDP are the
  more informative moves (+0.13 and +0.10, against −0.00 and −0.03 for a random other move in the same states).

## 2. Goal placement is not why the additive policy worked

The additive policy argmax H + G, run in the environment; recovery = (additive − goal-blind) / (natural − goal-blind).

| | maze10 | single junction goal | collect two junction goals |
|---|---|---|---|
| natural | 0.546 | 0.665 | 1.327 |
| **additive** | 0.480 | 0.585 | 1.171 |
| goal-blind (argmax H) | 0.191 | 0.344 | 0.858 |
| history-blind (argmax G) | 0.251 | 0.322 | 0.546 |
| **recovery** | **0.81** (0.73–0.86) | **0.75** (0.70–0.86) | **0.66** (0.58–0.73) |
| the additive choice is the model's move, where the goal matters | 0.87 | 0.90 | 0.89 |
| additive share of the logits' goal dependence | 0.74 | 0.73 | 0.65 |
| interaction removed at the decision token, online: recovery | 0.74 | 0.69 | — |

* **PLACE fails**: with junction goals the additive policy keeps 0.75 of the goal-directed return, against 0.81 with
  the spread goals (a drop of 0.06; the rule asked for 0.15). The additive choice is the model's move as often as
  before (0.90), and the state-level removal agrees (0.69 against 0.74). **Registered reading: goal placement is not
  why; the policy is mostly additive with junction goals too.** My suggestion in the maze10 report is withdrawn.
* **AD fails in both arms** by its threshold of 0.9, as in maze10. The interaction carries about a quarter of the
  goal-directed return with one junction goal, about a third with two goals to collect.
* **COLLECT fails**: the collect arm's recovery is 0.66, lower than the single arm's by 0.09 (the rule asked for 0.15).
  It is also below the 0.7 that the other reading needed, so neither registered reading applies. The step from maze10
  to collection is 0.15 in two parts of 0.06 and 0.09.

## 3. Collection: the additive form holds before the first pickup better than after it

Goal state: the set of goals still to collect (15 pairs, then 6 single goals).

| | before the first pickup | after it |
|---|---|---|
| the model's move depends on the goal state | 0.78 | 0.71 |
| there: the additive choice is the model's move | 0.90 | 0.87 |
| there: the goal-blind choice is the model's move | 0.67 | 0.63 |
| additive share of the logits' goal dependence | **0.69** | **0.45** |

* Before the first pickup the logits are as additive as in the single-goal tasks (0.69). After it, with the location
  known and one goal left, less than half of their goal dependence is additive (0.45): where to go from a known cell
  to a named goal is a function of both.
* The action still agrees in 0.87, because a goal-blind choice is already the model's move in 0.63: in a maze of
  corridors most cells leave little to choose.

| in the environment | return | goals collected | both collected |
|---|---|---|---|
| models | 1.327 | 2.00 | 1.00 |
| additive policy | 1.171 | 1.79 | 0.82 |
| goal-blind | 0.858 | 1.34 | 0.44 |
| history-blind | 0.546 | 0.80 | 0.25 |

The additive policy collects both goals in 82 % of episodes. The goal-blind policy, which wanders the junctions,
still collects 1.34 goals: these goals lie on the paths most histories favour, which is why the floor is high here.

## 4. Which goal first: the models are less sensitive to position than QMDP (partly post hoc)

| | first pickup is the goal nearer to the spawn | order fixedness per pair | takes its usual first goal although the other is ≥ 4 moves nearer |
|---|---|---|---|
| **models** | 0.70 (0.67–0.73) | 0.72 (0.70–0.74) | 0.41 (0.39–0.43) |
| QMDP | 0.82 | 0.63 | 0.20 |
| most likely cell | 0.89 | 0.63 | 0.11 |
| oracle | 1.00 | 0.65 | 0.00 |

* Registered (E6): the models take the nearer goal first in 0.70 of episodes, less often than QMDP (0.82) and than the
  most-likely-cell policy (0.89). E6 expected them between the two.
* Post hoc (`order.py`): each pair of goals has a usual first goal, and the models keep to it more than any reference
  (fixedness 0.72 against 0.63). When the other goal is clearly nearer they still take the usual one first in 41 % of
  episodes (QMDP 20 %). A preferred order per goal set is what a fixed bias per goal state produces.
* It costs little: their return is 0.019 below QMDP's.

## 5. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **LEARN** (single) | return ≥ the most-likely-cell policy's | 0.665 against 0.606; p 0.016 | **yes** |
| **LEARN** (collect) | return ≥ the most-likely-cell policy's | 1.327 against 1.244; p 0.016 | **yes** |
| **AD** (single) | additive policy: recovery ≥ 0.9 | 0.75 | no |
| **AD** (collect) | additive policy: recovery ≥ 0.9 | 0.66 | no |
| **PLACE** | single arm's recovery lower than maze10's by ≥ 0.15 | 0.75 against 0.81 | no |
| **COLLECT** | collect arm's recovery lower than the single arm's by ≥ 0.15 | 0.66 against 0.75 | no |

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | LEARN holds in both arms | yes | |
| E2 | single: recovery 0.65–0.85; PLACE fails | yes | 0.75 |
| E3 | single: interaction removed, recovery 0.6–0.8 | yes | 0.69 |
| E4 | collect: recovery 0.5–0.75; COLLECT holds | in part: 0.66; COLLECT fails (0.09) | |
| E5 | collect: the additive choice matches less often after the first pickup | yes | 0.87 against 0.90; share 0.45 against 0.69 |
| E6 | collect: nearer goal first more often than most-likely-cell, less often than QMDP | **no**: less often than both | 0.70 |

4 of 6, one in part.

## Conclusions

1. **The additive policy is not a consequence of where the goals are.** With six junction goals it keeps three
   quarters of the goal-directed return and picks the model's move in 0.90 of goal-dependent decisions, as with the
   spread goals. Across the small maze, random spawns, the larger maze and now interior goals, reward training arrives
   at a history profile plus a goal bias, and adds interaction for a part that grows with the task: about a twentieth
   of decisions in the small maze, a quarter of the return here.
2. **Collecting two goals needs more interaction, by a moderate amount** (recovery 0.66). It shows after the first
   pickup, where the logits' goal dependence is less than half additive.
3. **The collection policy has a habit**: a usual first goal per pair, kept in two fifths of the episodes where the
   other goal is clearly nearer. QMDP, which combines belief and goals multiplicatively, does that half as often.
4. Both tasks are learned from reward to about QMDP's level, and the agents again localise faster than QMDP.

Next, if wanted: make the interaction pay. The cases are now concrete: the second leg of a collection, and the choice
of the first goal. Optimal-move supervision is not available without a solver, but QMDP's moves are; fine-tuning on
them where the model departs from its own additive choice would test whether the interaction can be taught, as the
hard-cases experiment did in the small maze.

Limits: no optimal policy; one maze; the junction goals lie on the lower two thirds of it and on well-travelled paths
(the goal-blind floor is high, which lowers what recovery can show); alternatives replace goal tokens in a history
made under the agent's own goals; G per goal state and step; no state-level removal or decoders for the collect arm;
the order analysis is post hoc; maze10's models are compared across different initialisations.
