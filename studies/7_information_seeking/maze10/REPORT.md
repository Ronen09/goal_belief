# A larger maze without a solver: behaviour and interventions (maze10)

Run date: 2026-10-06. Brief: `BRIEF.md` (given in conversation). Task, measures and decision rule: `PLAN.md`, committed
(`3185235`) **before the models were trained**; a design probe of reference policies and a 200-update learnability
pilot (returns only, models discarded) came first and are disclosed there. Numbers from `tables.md`; data in
`results.json`, training logs in `runs/`. Reproduce: `reproduce.py maze10` (1.5 h on one GPU).

**Setting.** 55 open cells in a 9 × 9 box (`design.md`), four fixed goals, two noiseless landmarks, one symbol pair
read wrongly 40 % of the time. No passive prefix: the goal is shown at the start and every move is the agent's. Spawn
anywhere that is not a goal; 40 moves; discount 0.97. Six reward-trained models (the maze-belief transformer, PPO, 1 000
updates). **No solver**: the exact posterior over the location is known at every step, optimal moves are not.
References use the filter and shortest paths: the oracle knows its cell; QMDP uses the exact belief and ignores the
value of information; QMDP + lookahead values the next symbol's information.

## 1. The models learn the task and end level with QMDP

| policy | return | success | moves |
|---|---|---|---|
| **models (greedy)** | 0.546 (0.535–0.577) | 0.84 (0.83–0.94) | 19.9 |
| oracle | 0.786 | 1.00 | 9.2 |
| QMDP + lookahead | 0.564 | 0.95 | 20.8 |
| QMDP | 0.551 | 0.93 | 21.3 |
| most likely cell | 0.489 | 0.84 | 23.3 |
| random moves | 0.067 | 0.10 | 37.6 |

* **LEARN holds.** From reward alone, with no filter given, every model beats the most-likely-cell policy (p 0.016).
* **SEEK-R fails**: the median model is level with QMDP (0.546 against 0.551). Two models (seeds 2, 4) are above
  QMDP + lookahead (0.577, 0.574).
* By goal (post hoc): the models beat QMDP on goal A (0.63 against 0.58) and match it on B and D. Goal C, at the end
  of the dead-end corridor, separates them: four models get 0.27–0.29 (QMDP 0.36), the two best 0.41–0.42. As in the
  small maze, some seeds have not learned the hardest goal.

## 2. Their deviations from QMDP are informative ones

Per decision, from the exact filter: is the model's move a QMDP-best move, and what is each move's expected information
gain (expected fall of the posterior's entropy)?

| | deviates from QMDP | information gain of its move − QMDP's, at deviations | its move is the more informative | QMDP value given up |
|---|---|---|---|---|
| **models** | 0.45 | **+0.126** | 0.81 | 0.021 |
| a random other move in the same states | | +0.043 | | |
| registered null: QMDP with random moves inserted | 0.45 | −0.024 | 0.38 | 0.041 |
| QMDP + lookahead | 0.17 | +0.111 | 0.71 | 0.006 |
| most likely cell | 0.43 | +0.021 | 0.52 | 0.015 |
| oracle | 0.52 | +0.100 | 0.73 | 0.008 |

| posterior entropy on the same episodes | start | step 5 | step 10 | step 20 | step 30 | episodes stepping on a landmark |
|---|---|---|---|---|---|---|
| **models** | 3.72 | 2.61 | 1.71 | 0.73 | 0.47 | 0.40 |
| QMDP | 3.72 | 2.74 | 2.16 | 1.43 | 0.99 | 0.33 |

* **SEEK-B holds.** In 45 % of decisions the model does not take a QMDP-best move, and in 81 % of those its move is the
  more informative one. The advantage (+0.126) is above the registered null (−0.024, p 0.016), above a random other
  move in the same states (+0.043), and above QMDP + lookahead's own (+0.111).
* **The models localise faster than QMDP**: after 10 moves their exact posterior has entropy 1.71 against QMDP's 2.16,
  after 20 moves 0.73 against 1.43. They step on a landmark in 40 % of episodes against 33 %.
* It costs little QMDP value (0.021 per deviation, half the random null's), and it does not buy return over QMDP.
* **A caution about this measure.** The oracle, which has nothing to find out, also deviates from QMDP with a positive
  advantage (+0.100): QMDP hedges across cells and often pushes against a wall, which is uninformative, and a move
  along the true corridor is not. So "more informative than QMDP's move" also describes any policy that commits to a
  path. The entropy curves are the cleaner evidence: on the same episodes the models' histories carry more information
  than QMDP's.

## 3. The additive policy, run for real, keeps four fifths of the goal-directed return

H(h): the logits averaged over the four goals. G(goal, step): a fixed bias, fitted on separate episodes.

| policy run in the environment | return | success |
|---|---|---|
| natural | 0.546 | 0.84 |
| **additive: argmax H + G** | **0.480** | 0.76 |
| history-blind: argmax G | 0.251 | 0.38 |
| goal-blind: argmax H | 0.191 | 0.33 |
| **recovery** (additive − goal-blind) / (natural − goal-blind) | **0.81** (0.73–0.86) | |

* The model's move depends on the goal in 93 % of decisions. There the additive choice is the model's move in 0.87
  (without the goal: 0.55). The additive share of the logits' goal dependence is 0.74.
* **AD-B fails by the registered threshold** (0.81 against 0.9), and my expectation fails in the other direction: I
  expected under 0.5. A history profile plus a bias per goal and step steers the agent to the goal in 76 % of episodes
  in a two-dimensional maze.

## 4. Removing the interaction at the decision token, while the agent acts

At every decision the network runs under all four goals, and each attention and MLP output at the decision token is
replaced by its parts, recomputed in order (the additive-ablation experiment's scheme, online). 4 096 episodes.

| at the decision token | return | success |
|---|---|---|
| nothing | 0.548 | 0.84 |
| **interaction removed: history part + goal part** | **0.461** | 0.73 |
| history part removed, interaction kept | 0.345 | 0.56 |
| goal part removed, interaction kept | 0.230 | 0.41 |
| reference: goal-blind policy | 0.189 | 0.32 |
| **recovery**, interaction removed | **0.74** (0.68–0.79) | |

* **AD-S fails** (0.74 against 0.9). With no goal × history interaction at the decision token the agent keeps three
  quarters of the goal-directed return.
* **The goal acts through its main effect**: without the goal part the return falls to 0.23, close to the goal-blind
  policy, although the interaction is still in place.
* Without the history part the return is 0.345, above the history-blind policy (0.251). Here every earlier token comes
  after the goal token, so earlier tokens can mix goal and history, and the decision token can read that through the
  interaction. The removal acts at the decision token only.
* The interaction is worth about a quarter of the goal-directed return. In the small maze it changed 4–5 % of
  goal-dependent decisions.

## 5. What the state holds

| decoder from the final state at decision tokens (held-out episodes) | |
|---|---|
| exact posterior, R² | 0.46 (0.44–0.54) |
| true cell: from the decoded posterior / from the exact posterior | 0.37 / 0.47 |
| occupancy (discounted future visitation on the model's own path), R²: from the state | 0.25 |
| from the exact posterior × goal × step | 0.28 |
| from both | 0.28 |

* The posterior is linearly decodable at R² 0.46; the decoded most likely cell is the true cell in 37 % of decisions,
  the exact posterior's in 47 %.
* **OCC fails**: the state predicts where the agent will go no better than the exact belief, goal and step do (0.25
  against 0.28), and adds nothing to them (0.28 together).

## 6. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **LEARN** | return ≥ the most-likely-cell policy's | 0.546 against 0.489; p 0.016 | **yes** |
| **SEEK-R** | return > QMDP's | 0.546 against 0.551 | no |
| **SEEK-B** | deviations from QMDP are more informative, and more than the random-deviation null | +0.126 against −0.024; p 0.016 | **yes** |
| **AD-B** | additive policy: recovery ≥ 0.9 | 0.81 | no |
| **AD-S** | interaction removed at the decision token: recovery ≥ 0.9 | 0.74 | no |
| **OCC** | occupancy R² from the state − from posterior × goal × step ≥ 0.05 | −0.034 | no |

**Registered readings.** Neither AD-B nor AD-S: *the additive policy was a property of the small maze; here the policy
needs goal × history interaction.* SEEK-B: *the agent seeks information.*

The first reading is stronger than the numbers. The rule had one threshold, and both recoveries fall between it and
what I expected: the policy is **mostly additive (0.74–0.81 of the goal-directed return) and needs the interaction for
the rest**. In the small maze the additive code was nearly the whole policy.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | LEARN holds | yes | 0.546 against 0.489 |
| E2 | SEEK-R fails; models between most-likely-cell and QMDP | yes (two seeds above) | 0.546 |
| E3 | SEEK-B fails | **no** | +0.126 |
| E4 | AD-B fails, recovery < 0.5 | in part: fails, at 0.81 | 0.81 |
| E5 | AD-S fails likewise | in part: fails, at 0.74 | 0.74 |
| E6 | posterior R² ≥ 0.5 | **no** | 0.46 |
| E7 | OCC holds | **no** | −0.034 |

2 of 7, two in part.

## Conclusions

1. **Reward alone learns the larger maze to the level of QMDP**, a policy that is handed the exact filter. Two of six
   models pass QMDP with one step of lookahead.
2. **The agents gather information.** On the same episodes their histories localise them faster than QMDP's (entropy
   0.73 against 1.43 after 20 moves), and their departures from QMDP are the more informative moves four times in
   five. It does not yet pay in return.
3. **The additive policy mostly survives a maze where direction depends on position**, as a policy run in the
   environment (0.81) and as a state (0.74). The interaction now carries a fifth to a quarter of the goal-directed
   return, against about a twentieth of decisions in the small maze.
4. **The goal placement may be why.** The four goals were spread out by farthest-point sampling, which put three of
   them in corners. For a corner goal the direction is nearly the same from everywhere ("down and left"), which is what
   a fixed bias needs. This is the concern behind the random-spawns experiment, not resolved here.
5. The state holds no more about the agent's future path than belief, goal and step predict.

Next, if wanted: the same maze with goals in the interior, or a goal drawn from all cells, where no fixed direction
points to a goal. That is the direct test of conclusion 4.

Limits: no optimal policy, so competence and information seeking are relative to QMDP; the information-gain measure
also credits committing to a path (the oracle); one maze, one goal set with three corner goals; G indexed by goal and
step; removals at the decision token only; linear decoders, with single-trajectory occupancy targets; six models, of
which four have not learned goal C well.
