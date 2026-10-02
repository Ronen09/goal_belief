# Round 42: additive code, solvability, decision rule and expectations

Written before `run.py` was run on any trained model. Its additive-solvability margin was checked on synthetic data only
(G = 0: solvable exactly when all goals share the optimal action; random G: agrees with a brute-force search in 300 of
300 cases). Brief: `BRIEF.md`.

## Additive code (logits)

The natural logits at the goal token, centred over actions, are l(h, g).

* **G(g, L)**: the goal bias, the mean over fit-side histories of l(h, g) − mean over goals of l(h, ·), per goal and
  prefix length.
* **H(h)**: the history profile, the mean over the three goals of l(h, ·).
* **Additive decision**: argmax_a H_a(h) + G_a(g, L).

Held-out histories (length ≥ 1) under each goal. A cell is **goal-dependent** if its optimal set is disjoint from
another goal's. Measured:
* the additive share of the logits' goal dependence;
* optimal rates, natural and additive;
* **need**: cells where the natural choice is optimal and the additive one is not;
* spoil: the reverse.

## Additive solvability (solver and G only)

For a posterior b and length L, with a*(g) the optimal action under each goal, ask whether any H makes a*(g) the argmax
of H + G(g, L) under all three goals. The best margin δ* is the minimum over cycles of the difference-constraint graph;
the posterior is **solvable** if δ* > 0. Prediction: cells needing the interaction lie where no additive code can be
right.

## What H holds

The share of histories (with fewer than four actions optimal under some goal) whose H ranks the union of the three
goals' optimal actions on top. The G matrices are reported.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05.

| | criterion |
|---|---|
| **AD** | goal-dependent cells: additive optimal ≥ 0.9 × natural optimal |
| **EX** | P(need \| unsolvable) ≥ 3 × P(need \| solvable), and above it (p < 0.05) |
| **U** | H ranks the union of the goals' optimal actions on top in ≥ 0.7 of histories |

| result | reading |
|---|---|
| AD, EX | **the policy is additive wherever an additive code can work, and uses the interaction where it cannot** |
| AD, U | the history profile lists the candidate actions, and the goal bias picks among them |
| not EX | the interaction is used, but not specifically where additivity is impossible |

## Expectations

| | expectation |
|---|---|
| E1 | AD holds (round 41: 0.79 of 0.83 at the state level) |
| E2 | EX holds |
| E3 | U holds |
| E4 | additive share of the logits' goal dependence ≥ 0.8 |
| E5 | the majority of solvable posteriors exceed 0.5 of all |
| E6 | in unsolvable posteriors the natural policy is optimal more often than the additive one by ≥ 0.2 |

## Limits known in advance

G is shared by all histories of a length (fixed goal biases); logits only (round 41 removed the interaction at the
state level); first decision; reward models only.
