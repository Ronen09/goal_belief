# Random spawns: task, arms, measures, decision rule and expectations

Written before any model of this experiment was trained, and before the solver-level additive fit was computed for
either arm. Brief: `BRIEF.md`. Code: `task.py`, `train.py`, `measure.py`.

## Task and arms

The maze-belief maze, goals and noise (0.4). Two arms that differ only in the spawn cells:

| arm | spawn | exact belief graph |
|---|---|---|
| **four** (control) | the four corridor cells of the maze-belief experiment | 0.15 M nodes |
| **all** | uniform over the 11 cells that are not goals (corridor, landmark, left stub) | 11.9 M nodes |

**A change forced by the solver, made before training:** both arms use a passive prefix of 0–2 random moves and
**8 moves** after the reveal, where the earlier experiments used 0–4 and 12. With 11 spawn cells the exact graph of the
original setting did not fit (the build was stopped above 115 GB). The control arm is therefore retrained in the shorter
setting, so the two arms differ in the spawns only. Eight moves do not always suffice to reach the goal (the hard-cases
screen: solver success 0.64 for four spawns at 8 moves and a prefix of 0–4).

Training: the observation-prediction experiment's reward arm, unchanged (PPO on reward only, the same network and
initialisation per seed, 1 500 updates of 4 096 episodes), ten seeds per arm.

## Measures (`measure.py`)

As in the hard-cases experiment's later-decisions analysis, per arm: 40 000 episodes of the solver with 20 % random
moves on that arm's task; every decision (the first and the later ones) replayed under all three goals through the
belief graph. Logits at the decision token, centred over actions, l(h, g).

* **Additive code**: G(g; prefix length, step) fitted on one half of the episodes as the mean goal deviation;
  H(h) = mean over goals; additive decision argmax H + G. On the other half, goal-dependent cells (optimal set disjoint
  from another goal's):
  * natural optimal, additive optimal, their ratio, and the share of cells where the two choose the same action;
  * the additive share of the logits' goal dependence;
  * additively unsolvable posteriors (the additive-code experiment's margin under the model's G): their share, and
    natural and additive accuracy inside and outside them.
* **Solver reference**: the same fit applied to the solver's own Q* (the hard-cases experiment's screen): how often the
  best additive move fails on goal-dependent cells. It says how additive the task is, whatever a network does.
* **Competence**: greedy regret and regret / V* on fixed evaluation episodes.

## Decision rule

Ten models per arm; medians; arms compared by exact Wilcoxon signed-rank tests paired by seed (the same initialisation),
one-sided, p < 0.05.

| | criterion |
|---|---|
| **REP** | four arm: additive ≥ 0.9 × natural on goal-dependent cells (the additive-code experiment's AD, in the shorter setting) |
| **AD-all** | all arm: additive ≥ 0.9 × natural, and the same action in ≥ 0.9 of goal-dependent cells |
| **DROP** | the ratio additive / natural is lower in the all arm than in the four arm by ≥ 0.05 |
| **LEARN** | all arm: regret / V* ≤ 2 × the four arm's (the models are comparably competent) |

| result | reading |
|---|---|
| REP, AD-all, not DROP | **the additive code does not depend on the spawns** |
| REP, AD-all, DROP | the policy stays mostly additive, but random spawns make it use more goal × history interaction |
| REP, not AD-all | **the additive code was a consequence of the four spawns**: with random spawns the policy needs the interaction |
| not REP | the shorter setting itself changes the result; the comparison with the earlier experiments is not valid |
| not LEARN | any difference may be a difference in competence; reported with that caveat |

## Expectations

| | expectation |
|---|---|
| E1 | REP holds (ratio ≥ 0.93) |
| E2 | the task is less additive with random spawns: the solver's best additive move fails on ≥ 0.05 more of the goal-dependent cells in the all arm |
| E3 | AD-all holds, more weakly: ratio 0.88–0.95 |
| E4 | DROP holds |
| E5 | LEARN holds |
| E6 | in the all arm the network is still no better than the additive code where additivity is impossible (natural − additive ≤ 0.05 in unsolvable posteriors) |

## Limits known in advance

Shorter prefix and horizon than the earlier experiments; histories from the solver with random moves, not the models'
own; G per prefix length and step, shared by all histories; one maze and one goal placement (the brief's concern about
the goals themselves is not tested here).
