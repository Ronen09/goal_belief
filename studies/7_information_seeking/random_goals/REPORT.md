# Random goals: a goal drawn from every cell (random goals)

Run date: 2026-10-07. Brief: `BRIEF.md` (given in conversation). Task, measures and decision rule: `PLAN.md`, committed
(`f44e390`) **before any model was trained**; the additive ceiling (`../additive_ceiling/`, no network) and the
reference policies came first and are given there. Numbers from `tables.md`; data in `results.json`; training logs in
`runs/`. Reproduce: `reproduce.py random_goals` (2 h on one GPU).

**Setting.** The maze10 maze, symbols, landmarks and noise, with **every one of the 55 cells a possible goal**, one
per episode, shown at the start; the spawn is any other cell. Six reward-trained models (the maze10 network with 55
goal embeddings, PPO, 1 000 updates). No solver: references from the exact filter.

## 1. Learned above QMDP, and the agents still seek information

| | maze10 (4 spread goals) | single junction goal (6) | **random goals (55)** |
|---|---|---|---|
| **models: return** | 0.546 | 0.665 | **0.606** (0.598–0.607) |
| models: success | 0.84 | 0.99 | 0.97 |
| oracle | 0.786 | 0.844 | 0.826 |
| QMDP + lookahead | 0.564 | 0.676 | 0.616 |
| QMDP | 0.551 | 0.661 | 0.587 |
| most likely cell | 0.489 | 0.606 | 0.527 |

* **LEARN holds** (p 0.016). Every seed ends alike (0.598–0.607) and **every seed is above QMDP** (0.587), the
  policy that is handed the exact belief; none reaches QMDP with lookahead (0.616). In maze10 the median model was
  level with QMDP and in the junction arm too; with a goal from anywhere the models pass it. (SEEK-R was not
  registered for this experiment; E1 expected the models below QMDP.)
* **SEEK-B holds**: the models leave QMDP's move in 0.44 of decisions, to the more informative move in 0.70, with an
  advantage of +0.104 against −0.030 for the null (p 0.016). On the same episodes their posterior entropy after 20
  moves is 0.72 against QMDP's 1.56, and they step on a landmark in 0.36 of episodes against 0.22.

## 2. The additive policy keeps two thirds of the goal-directed return

The additive policy argmax H + G run in the environment; recovery = (additive − goal-blind) / (natural − goal-blind).

| | maze10 | single junction goal | **random goals** |
|---|---|---|---|
| natural | 0.546 | 0.665 | 0.606 |
| **additive** | 0.480 | 0.585 | 0.455 |
| goal-blind (argmax H) | 0.191 | 0.344 | 0.181 |
| history-blind (argmax G) | 0.251 | 0.322 | 0.180 |
| **recovery** | **0.81** (0.73–0.86) | **0.75** (0.70–0.86) | **0.645** (0.59–0.65) |
| the additive choice is the model's move, where the goal matters | 0.87 | 0.90 | 0.855 |
| additive share of the logits' goal dependence | 0.74 | 0.73 | 0.665 |
| interaction removed at the decision token, online: recovery | 0.74 | 0.69 | **0.54** (0.48–0.60) |

* **AD fails** (0.645 against 0.9), as in every arm. **DROP fails**: the recovery is 0.10 below the junction arm's
  (the rule asked for 0.15; p 0.71). The median sits a hair under the 0.65 the other registered reading needed, so
  neither reading applies by the letter; by the numbers the policy is **two thirds additive with 55 goals** against
  three quarters with 6 and four fifths with 4, and the state-level removal, which loses more, puts the interaction
  at nearly half (0.54).
* The additive choice is still the model's move in 0.855 of goal-dependent decisions (0.87, 0.90 before): where the
  interaction matters it changes the move rarely but at a cost in return, which is what a return-weighted measure
  shows and an agreement rate hides.
* The additive ceiling says the fully observed task is no less additive with 55 goals (0.958 of decisions solvable
  by argmax H(cell) + G(goal); 4 goals 0.926, 6 goals 0.969). The drop is therefore not the task's: it is the belief-
  level computation, H over histories and a bias fitted per goal, that carries less with 55 goals.
* **The fixed bias per goal is nearly the best single direction.** G(goal)'s preferred move is a shortest-path move
  from 0.435 of the cells; the best single move per goal would be from 0.461, and G picks that move for 0.65 of the
  goals. The goal part is the map's "which way, on average, to this goal", and H is the history's "which ways are
  open"; where neither suffices the interaction decides.

## 3. The posterior is less linearly decodable

| decoder from the final state at decision tokens | maze10 | junction goal | **random goals** |
|---|---|---|---|
| exact posterior, R² | 0.46 | 0.46 | **0.28** (0.27–0.31) |
| true cell from the decoded posterior / from the exact posterior | 0.37 / 0.47 | 0.33 / 0.41 | 0.26 / 0.44 |

* E6 fails (0.28 against the expected 0.40–0.55). The decision token's final state is a goal-specific computation,
  and with 55 goals less of its variance is the location posterior.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **LEARN** | return ≥ the most-likely-cell policy's | 0.606 against 0.527; p 0.016 | **yes** |
| **SEEK-B** | deviations from QMDP more informative than its move and than the null | +0.104 against −0.030; p 0.016 | **yes** |
| **AD** | additive policy: recovery ≥ 0.9 | 0.645 | no |
| **DROP** | recovery lower than the junction arm's by ≥ 0.15 | 0.645 against 0.75; p 0.71 | no |

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | LEARN; models between most-likely-cell and QMDP | in part: LEARN; above QMDP in every seed | 0.606 |
| E2 | recovery 0.55–0.75; DROP fails | yes | 0.645 |
| E3 | interaction removed online: recovery 0.5–0.7 | yes | 0.54 |
| E4 | the additive choice is the model's move in 0.80–0.90 | yes | 0.855 |
| E5 | SEEK-B holds | yes | +0.104 |
| E6 | posterior R² 0.40–0.55 | **no** | 0.28 |

4 of 6, one in part.

## Conclusions

1. **A goal from anywhere does not break the additive policy; it wears it down.** Recovery falls from 0.81 (4
   goals) to 0.75 (6) to 0.645 (55); the state-level removal from 0.74 to 0.69 to 0.54. The fully observed task is
   as additive as ever (ceiling 0.96), so the loss is in the belief-level code: a fixed bias per goal is nearly the
   best single direction to it, and with 55 goals that direction is right from fewer histories.
2. **Reward training in the larger maze now passes QMDP** (0.606 against 0.587 in every seed) while localising
   faster than it. With more goals the training signal covers the maze and the information-seeking habit pays.
3. The posterior is less linearly present in the decision token's state (R² 0.28).

Next, if wanted: the additive code's parts have now been measured across four goal sets; the goal part's "best
single direction" reading suggests fitting G per (goal, region of the belief) to see how little interaction suffices.

Limits: no optimal policy; one maze; G indexed by goal and step bin; removals at the decision token only, in chunks
of 1 024 histories (the goal part is a mean over the chunk); the comparison with the junction arm is across different
initialisations; the posterior decoder is linear.
