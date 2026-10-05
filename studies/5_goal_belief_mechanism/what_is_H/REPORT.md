# What is the history profile H? (what is H)

Run date: 2026-10-02. Brief: `BRIEF.md`. Candidates, tests and decision rule: `PLAN.md`, committed (`9b5a237`) **before
`run.py` was run on any trained model**. Numbers from `tables.md`; data in `results.json` and `untrained.json`.
Reproduce: `reproduce.py what_is_H` (3 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen. H(h) is the goal-averaged action profile at the goal token, round
42's additive code (logits ≈ H(h) + G(g, L)). The candidates are per-posterior quantities, one value per action:

* from the exact action values Q*:
  * max_g Q* (the brief's hypothesis);
  * mean_g Q*;
  * P(a optimal | random goal);
  * max_g advantage;
* fully observed reachability under the belief, Q_MDP(b, g, a) = Σ_s b(s) γ^d(next(s,a), g), which ignores the value of
  information:
  * max over goals and mean over goals;
  * closeness to the nearest goal.

Each candidate is tested three ways: as a predictor of H (held-out R²); as H's replacement in the additive decision;
and as a causal edit. For the edit, the goal-free state is regressed on the candidate, and the same vector
(C(b_B) − C(b_A))·W is applied under all goals at the goal token entering block 2.

## 1. H is not max_g Q*. It is a goal-averaged usefulness

| candidate | R² for H | top action agrees | causal edit: share of whole's gain |
|---|---|---|---|
| **max_g Q*** (the brief's) | **0.35** | 0.62 | **0.21** |
| max_g Q_MDP | 0.35 | 0.64 | 0.22 |
| nearest goal | 0.42 | 0.42 | 0.27 |
| max_g advantage | 0.49 | **0.94** | 0.42 |
| mean_g Q* | 0.62 | 0.74 | 0.50 |
| **P(optimal \| random goal)** | **0.66** | 0.82 | **0.59** |
| **mean_g Q_MDP** | **0.67** | 0.72 | **0.53** |
| ceiling: affine in b | 0.82 | | 0.63 |
| ceiling: H's own mean per posterior | 0.91 | | 0.68 |
| untrained reference, best candidate | 0.32 | | |

* **MQ fails clearly**: max_g Q* ranks sixth of seven as a predictor, and its edit moves decisions a fifth as far as
  replacing the whole state. Every "max over goals" quantity is among the worst.
* **Every "average over goals" quantity is among the best.** Fully observed reachability averaged over goals (0.67),
  the probability that an action is optimal for a random goal (0.66) and mean_g Q* (0.62) lead. The top two are
  statistically tied (p 0.46).
* **The argmax and the magnitudes tell different things.** H's top action is an action optimal under some goal in 94 % of
  histories (max advantage = 0; the additive-code experiment's U). But how strongly H prefers each action follows how many goals it serves,
  and how well on average.
* **The causal edits agree in ordering.** Editing the goal token's state by the donor's P(optimal | random goal) moves
  0.59 of the way to whole replacement, mean reachability 0.53, max_g Q* 0.21; rotated vectors move none. Edits work
  at the final residual stream too, with the same ordering.

## 2. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **MQ** | max_g Q* best, above the runner-up | ranks 6 of 7 (0.35) | no |
| **EXPL** | best R² ≥ 0.8 × H's posterior ceiling | mean reachability: 0.73 × | no |
| **CE** | best's edit gain ≥ 0.5 × whole's, above rotated | 0.53 ×; p < 0.001 | **yes** |
| **AG** | best's edit gain ≥ every other candidate's − 0.02 | P(optimal) exceeds it by 0.021 | no (by 0.001) |
| **D** | decision with the best candidate ≥ 0.9 × with H | 0.96 × | **yes** |

**Reading.** The brief's algorithm with *max* is rejected. A goal-*averaged* version fits better: score each action by
its usefulness averaged over the possible goals, then add the goal's preference. "Best" by prediction is mean
reachability, and by causation P(optimal | random goal). They are tied in prediction, so AG fails by a hair. EXPL fails:
the best single candidate explains 0.73 of what the posterior explains. H is goal-averaged usefulness, not exactly any
one of these quantities.

## 3. What the edits cannot do

On goal-dependent pairs ("left under G1, right under G2"), every 4-number edit fails (0.07–0.11, against 0.49 for
whole replacement). H is the same for all goals by construction. Turning one belief change into different actions under
different goals needs more than shifting H, as the additive-code experiment found for the policy itself.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | MQ fails | yes | rank 6 of 7 |
| E2 | best is P(optimal) or max advantage | in part | P(optimal) ties for best (0.66 against 0.67) and is best causally |
| E3 | EXPL holds | no | 0.73 × |
| E4 | CE holds | yes | 0.53 × |
| E5 | D holds | yes | 0.96 × |
| E6 | best Q*-based beats best fully observed | no (tie) | 0.66 against 0.67 |
| E7 | affine-in-b edit ≥ best candidate's | yes | 0.63 against 0.53 |

3 of 7, one in part.

## Conclusions

1. **The network does not score actions by their best value across goals.** max_g Q* is among the worst descriptions of
   H, predictively (0.35) and causally (0.21 of whole).
2. **It scores them by their average usefulness across goals.** How likely an action is to be optimal for a random goal,
   or how close it brings the agent to the goals on average under the belief, each explain about two thirds of H.
   Editing the state by either moves decisions half to three fifths of the way to the donor's.
3. **Information value is not visibly part of it.** Fully observed reachability, which ignores what a move would reveal,
   predicts H as well as the Bayes-optimal values do.
4. **The learned algorithm, with the query-swap to additive-code experiments:**
   * infer the posterior;
   * score actions by goal-averaged usefulness (H);
   * add a fixed goal preference (G, via block 0's self value);
   * choose.

   Because H averages over goals, the policy fails where the right action for one goal is poor on average and is not
   rescued by the goal bias: the additive-code experiment's non-additive cases.

Next, if wanted: fit H as a weighted mixture of the top candidates, to see whether usefulness and reachability are two
parts of one code. Or test whether the goal weights inside H's average follow the goals' prior frequencies in training.

Limits: candidates are functions of the posterior at the reveal; 4-number codes edit along at most 3 directions; first
decision; reward models only.
