# Is the additive policy a consequence of where the agent starts? (random spawns)

Run date: 2026-10-06. Brief: `BRIEF.md` (given in conversation). Task, arms, measures and decision rule: `PLAN.md`,
committed (`1384814`) **before any model of this experiment was trained** and before the solver-level fit was computed.
Numbers from `tables.md`; data in `results.json`, training logs in `runs/`. Reproduce: `reproduce.py random_spawns`
(1.5 h on one GPU; the all-spawn graph needs about 40 GB of memory to build).

**Setting.** The maze-belief maze, goals and noise. Two arms that differ only in the spawn cells: the **four** corridor
cells used so far, and **all** 11 cells that are not goals. Ten reward-trained models per arm, the
observation-prediction experiment's reward recipe. Both arms use **8 moves and a prefix of 0–2** (earlier experiments:
12 and 0–4), because the exact belief graph for 11 spawns did not fit in the original setting. The additive code
H(history) + G(goal) is fitted and tested as in the hard-cases experiment, on every decision of solver episodes with
20 % random moves, each replayed under all three goals.

## 1. With random spawns the policy is still additive

Goal-dependent cells, all decisions:

| | four spawns | all spawns |
|---|---|---|
| natural policy optimal | 0.762 | 0.720 |
| additive decision optimal | 0.706 | 0.685 |
| **additive / natural** | **0.915** (0.896–0.944) | **0.950** (0.931–0.976) |
| same action | 0.895 | 0.916 |
| additive share of the logits' goal dependence | 0.46 | 0.51 |

* **REP holds**: the control arm reproduces the additive code in the shorter setting (0.915; at the first decision
  0.955, the additive-code experiment's 0.96).
* **AD-all holds**: with spawns anywhere, a history profile plus a fixed bias per goal keeps 0.95 of the policy's
  accuracy and picks the same action in 0.92 of goal-dependent cells.
* **DROP fails, in the opposite direction**: over all decisions the all-spawn models are *more* additive (by 0.031;
  p 0.001 for that direction, not registered).

## 2. But the all-spawn models are less competent, and the first decision shows why

| | four spawns | all spawns |
|---|---|---|
| greedy regret / V* | 0.010 (0.004–0.060) | 0.026 (0.022–0.029) |
| optimal-move rate | 0.969 | 0.914 |
| regret on G1 / G2 / G3 | 0.004 / 0.011 / 0.000 | 0.007 / 0.029 / 0.003 |

* **LEARN fails**: relative regret is 2.6 times the control arm's, so the registered reading carries the caveat that
  the arms differ in competence. The all-spawn regret had stopped falling (seed 0: 0.0135 at update 1 206, 0.0132 at
  1 500). Three control seeds (1, 5, 7) had not learned G2 (regret ≈ 0.03), as in the maze-belief experiment.
* Within each arm, a model's regret does not predict how additive it is (Spearman −0.31 and 0.25, both p > 0.3), and
  the seven competent control seeds give the same ratio (0.926). Post hoc.

At the first decision (the reveal), where the spawn matters most:

| first decision, goal-dependent cells | four spawns | all spawns |
|---|---|---|
| solver's Q*: the best additive move is optimal | 0.827 | 0.705 |
| natural policy optimal | 0.893 | 0.721 |
| additive decision optimal | 0.873 | 0.668 |
| additive / natural | 0.955 | 0.920 |
| natural − additive in additively unsolvable posteriors | 0.19 | 0.17 |

* **Random spawns make the task itself less additive at the reveal**: the best additive fit to the solver's own Q*
  goes from 0.83 to 0.71. Over all decisions the change is small (0.703 to 0.671; E2 expected more).
* **The network does not follow.** Its accuracy at the reveal falls from 0.89 to 0.72, close to what an additive code
  can reach on that task (0.71), and its own additive code still accounts for 0.92 of it (four spawns: 0.955; lower by
  0.037, p 0.005, post hoc).
* So the concern in the brief is half right. The four spawns made the task easy *for* an additive policy. But
  removing them did not make the policy stop being additive. It stayed additive and paid for it in regret, mostly on
  G2.

## 3. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **REP** | four arm: additive ≥ 0.9 × natural | 0.915 | **yes** |
| **AD-all** | all arm: additive ≥ 0.9 × natural, same action ≥ 0.9 | 0.950; 0.916 | **yes** |
| **DROP** | ratio lower in the all arm by ≥ 0.05 | −0.031 | no (opposite) |
| **LEARN** | all arm: regret / V* ≤ 2 × the four arm's | 0.026 against 0.010 | no |

**Registered reading: the additive code does not depend on the spawns; LEARN failed, so the arms differ in
competence.** Section 2 says what that difference is: the lost competence is where the task stopped being additive.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | REP holds, ratio ≥ 0.93 | **no** (REP holds at 0.915) | 0.915 |
| E2 | solver's additive move fails on ≥ 0.05 more cells with random spawns | **no** overall (0.032); at the first decision 0.12 | 0.297 / 0.329 |
| E3 | AD-all holds, ratio 0.88–0.95 | yes | 0.950 |
| E4 | DROP holds | **no** (opposite) | −0.031 |
| E5 | LEARN holds | **no** | 2.6 × |
| E6 | all arm, unsolvable posteriors: natural − additive ≤ 0.05 | **no**, narrowly | 0.065 |

1 of 6. I expected random spawns to push the policy toward the interaction. They did not.

## Conclusions

1. **The additive code is not produced by the four spawn cells.** With spawns over all 11 non-goal cells, the trained
   policy is as additive as before (0.95 of its accuracy; same action in 0.92 of goal-dependent cells).
2. **The four spawns did make the task friendlier to an additive policy.** With random spawns the best additive fit to
   the optimal values at the reveal drops from 0.83 to 0.71, and the network's accuracy there drops with it (0.89 to
   0.72). Reward training kept the additive strategy and lost competence instead of learning the interaction.
3. This matches the additive-code experiment's second conclusion: the network's limits are the additive code's limits.

Next, if wanted: the brief's other half, the goals. Random goal cells (the hard-cases screen found the task least
additive with ten goals: unsolvable in 0.60 of goal-dependent cells) would test whether a fixed bias per goal survives
when there are many goals.

Limits: 8 moves and a prefix of 0–2, not the earlier 12 and 0–4; competence differs between the arms; histories from the
solver with random moves, not the models' own; G shared by all histories of a prefix length and step; one maze and one
goal placement; the first-decision comparison and the within-arm checks are post hoc.
