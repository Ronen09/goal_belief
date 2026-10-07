# Random goals: task, measures, decision rule and expectations

Written before any model of this experiment was trained. **Disclosure:** the reference policies (no network) and the
additive ceiling (`../additive_ceiling/`, no network) were run first and are given below; a `--quick` smoke run of
`train.py` (10 updates, 2 seeds, models discarded) checked the code. Brief: `BRIEF.md`. Code: `goalgeo/bigmaze.py`
(`all_goals`), `train.py`, `measure.py`, `tables.py`.

## Task

The maze10 maze, symbols, landmarks, noise (0.4), 40 moves, discount 0.97. **Every one of the 55 cells is a possible
goal**; one is drawn per episode and shown at the start as the goal token (55 goal names). The spawn is uniform over
the other 54 cells, and the filter's prior is uniform over them. Nothing else changes: the single-goal arm of the
interior-goals experiment differs from this task in the goal set only (6 junction cells there, 55 here; maze10 has the
4 spread cells).

Paths to the goal are shorter on average (7.5 moves under the oracle against 9.2) because the goal is anywhere.

Reference policies on the 8 192 evaluation episodes (return; success in brackets):

| | oracle | QMDP + lookahead | QMDP | most likely cell | random |
|---|---|---|---|---|---|
| random goals (55) | 0.826 (1.00) | 0.615 (0.98) | 0.587 (0.95) | 0.527 (0.89) | 0.095 (0.14) |
| maze10 (4 spread goals) | 0.786 (1.00) | 0.564 (0.95) | 0.551 (0.93) | 0.489 (0.84) | 0.067 (0.10) |
| single junction goal (6) | 0.844 | 0.676 | 0.661 | 0.606 | |

**The additive ceiling (no network).** If the agent knew its cell, how often could the best rule of the form
argmax H(cell) + G(goal) pick a shortest-path move? H and G fitted directly to the shortest-path moves
(`../additive_ceiling/run.py`, 8 restarts; `../additive_ceiling/tables.md`):

| maze and goals | best additive rule: argmax H(cell) + G(goal) | best goal-blind rule: argmax H(cell) |
|---|---|---|
| small maze, its 3 goals | 1.000 | 0.872 |
| small maze, every cell a goal (14) | 1.000 | 0.835 |
| larger maze, the 4 spread goals (maze10) | 0.926 | 0.773 |
| larger maze, the 6 junction goals | 0.969 | 0.790 |
| larger maze, 16 goals (every third cell) | 0.949 | 0.829 |
| **larger maze, every cell a goal (55)** | **0.958** | 0.799 |

**Given the cell, the task is about as additive with 55 goals as with 4 or 6** (0.96 against 0.93 and 0.97): in a
maze of corridors a per-cell profile plus a per-goal tie-break picks a shortest-path move almost everywhere, and a
goal-blind per-cell move alone does so in 0.80. (This is higher than the interior-goals plan's 0.75–0.80, which fitted
the shortest-path *values* by least squares; here the rule is fitted to the moves.) So a drop in recovery here could
not be blamed on the fully observed task: it would be the belief-level computation, H over histories rather than
cells and a bias to be fitted for each of 55 goals, that fails to stay additive. My expectation before the ceiling
was a large drop; after it, a moderate one.

Training: as maze10 and the interior-goals arms (PPO on reward only, the same network with 55 goal embeddings,
1 000 updates of 4 096 episodes), six seeds. The number of updates is kept for comparability; the learning curves are
reported.

## Measures (`measure.py`), on each model's own greedy episodes

The maze10 experiment's measures, with the following changes for 55 goals.

**A. Behaviour.** Return, success and length on 8 192 fixed episodes (the same spawns and goals for every model and
reference).

**B. Information seeking.** As maze10: deviations from QMDP, their expected information gain against QMDP's move, the
random-deviation null, posterior entropy by step.

**C. The additive code, tested by behaviour.** Logits at every decision under each of the 55 goals (the goal token
replaced, the history the model's own). H(h) = their mean over goals, centred over actions. G(goal, step) = the mean
goal deviation per step bin on separate fit episodes; **32 768 fit episodes** (four times maze10's, about 600 per
goal), and a (goal, step) bin with fewer than 50 decisions falls back to the goal's mean over all steps.
* offline: the share of decisions where the model's move depends on the goal; there, the share where argmax H + G is
  the model's move; the additive share of the logits' goal dependence;
* **online**: natural; **additive** (argmax H + G); goal-blind (argmax H); history-blind (argmax G), run in the
  environment on the evaluation episodes. **Recovery** = (additive − goal-blind) / (natural − goal-blind), in return.

**D. The additive state, tested by intervention.** As maze10, online, at the decision token, under all 55 goals:
interaction removed; history part removed; goal part removed. On the first 4 096 evaluation episodes.

**E. What the state holds.** Ridge decoders from the final state at decision tokens, held-out episodes: the exact
posterior (R²) and the true cell. The occupancy decoders are dropped: the posterior × goal × step design has
55 × 55 × 8 features here, and maze10's OCC was negative.

**F. The fixed bias per goal (descriptive).** G(goal) averaged over steps gives one preferred move per goal. The share
of cells from which that move is a shortest-path move to the goal, against the best single move per goal (the
most-common-first-move check of the interior-goals plan) and against the additive ceiling.

## Decision rule

Six models; medians. LEARN by the exact one-sided Wilcoxon signed-rank test against the reference's value; DROP by
the exact one-sided Mann–Whitney test against the six recoveries of the single junction arm; p < 0.05.

| | criterion |
|---|---|
| **LEARN** | greedy return ≥ the most-likely-cell policy's (0.527) |
| **SEEK-B** | at deviations from QMDP the model's move has a higher expected information gain than QMDP's move, and by more than the random-deviation null |
| **AD** | additive policy: recovery ≥ 0.9 |
| **DROP** | recovery lower than the single junction arm's (0.75) by ≥ 0.15 |

| result | reading |
|---|---|
| DROP | **a goal from anywhere forces the goal × history interaction**: the additive policy of the earlier arms was a property of a few fixed goals |
| not DROP, recovery ≥ 0.65 | the additive form survives every cell being a goal: reward training keeps a fixed preference per goal and pays for it in the interaction's share |
| not DROP, recovery < 0.65 | in between; reported as such |
| AD | the additive policy holds whatever the goal set (not expected) |
| not LEARN | the models did not learn the task; nothing else is interpreted |

## Expectations

| | expectation |
|---|---|
| E1 | LEARN holds; the models end between the most-likely-cell policy and QMDP, as in maze10 |
| E2 | recovery 0.55–0.75; DROP fails (the drop from the junction arm's 0.75 is under 0.15) |
| E3 | interaction removed at the decision token: recovery 0.5–0.7 |
| E4 | where the model's move depends on the goal, the additive choice is the model's move in 0.80–0.90 (maze10 0.87, junction goals 0.90) |
| E5 | SEEK-B holds, as in both earlier arms |
| E6 | the posterior decodes at R² 0.40–0.55 |

## Limits known in advance

No optimal policy. One maze. 1 000 updates may be short for 55 goals (the curves will say). G indexed by goal and
step bin, with about 600 fit episodes per goal; the fallback to the goal's all-step mean smooths late bins. The
removals act at the decision token only. The comparison with the junction arm is across different initialisations
(the goal embedding has another size).
