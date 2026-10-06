# Residual trace: strata, measures, decision rule and expectations

Written before `run.py` was run on any trained model. Brief: `BRIEF.md`.

## Setting

As in the history-mediator experiment. At the goal token's four attention outputs, the history part
d(h) = x_b(h) + r(h), where x_b = E[d | posterior, L] is a table over fit-side histories. That experiment's cell set
(tables covered), random same-length donor h′ (new draw, fixed seed), informative pairs (the recipient's and donor's
natural actions under g differ). Two arms on every informative pair:
* **real**: add r(h′) − r(h) (the history-mediator experiment's swap_r; it changed 0.115 of decisions);
* **rot**: the same vectors rotated to random directions, same norm per component (it changed 0.037).

## Per-pair quantities

* **flip**: the action under the arm differs from the recipient's natural action.
* **shift**: how far the arm moves the logit gap toward the donor's action, at the goal token:
  (l[a′] − l[a])_arm − (l[a′] − l[a])_natural, with a the recipient's and a′ the donor's natural action.
* **excess** = real − rot, for both flip and shift. This is the part of r's effect that is specific to r's content.
  Rot is the same-size generic perturbation.

## Strata (each pair is classified by its recipient cell, except where stated)

| stratum | bins |
|---|---|
| **margin**: recipient's natural logit gap l[a] − l[a′] | quartiles per model |
| **additively impossible**: the posterior is not additively solvable under the model's goal biases G (the additive-code experiment's δ* ≤ 0; G from its `results.json`) | solvable / unsolvable |
| **interaction used**: the natural action differs from the additive decision argmax H + G | yes / no |
| **goal** | G1 / G2 / G3 |
| **posterior geometry**: entropy of the posterior | terciles (over cells; the same for every model) |
| **H reconstruction error**: ‖H(h) − E[H \| posterior, L]‖, H's deviation from its posterior mean | quartiles per model |
| **‖r(h′) − r(h)‖**: the size of the swapped residual (pair) | quartiles per model |

For each stratum, per bin: real and rot flip rates; excess flip; mean excess shift; and the **concentration** of a bin,
its share of the total excess flip divided by its share of pairs.

**Joint**: a least-squares fit of the per-pair excess shift on all strata together (standardized margin, unsolvable,
interaction used, goal dummies, entropy, H error, ‖Δr‖). Coefficients are reported per model.

## Decision rule

Ten models; medians; exact one-sided Wilcoxon signed-rank tests, p < 0.05.

| | criterion |
|---|---|
| **BND** | the lower half of margin holds ≥ 0.75 of the excess flip, **and** the excess shift does not depend on margin: top-quartile excess shift ≥ 0.5 × bottom-quartile's |
| **IMP** | excess flip concentration in unsolvable posteriors ≥ 2, and its excess shift there > solvable's (Wilcoxon) |
| **INT** | excess flip concentration in interaction-used cells ≥ 2 |
| **ERR** | excess flip concentration in the top quartile of H reconstruction error ≥ 2 |

| result | reading |
|---|---|
| BND, not IMP, not INT | **fine-grained correction**: r is a small, broadly spread shift of the logits that changes the decision only near ties |
| IMP or INT | **interaction route**: r's effect is where the additive code fails, the h × g interaction found earlier |
| neither BND nor IMP/INT | r is a separate, localized signal; the strata say where |

BND and IMP can both hold, because the additively impossible cases are near-ties between two goals (hard-cases
experiment). The joint fit separates them: IMP's reading needs the unsolvable coefficient positive with margin in the
fit.

## Expectations

| | expectation |
|---|---|
| E1 | real flip ≈ 0.115 and rot ≈ 0.037 (history-mediator experiment) |
| E2 | BND holds |
| E3 | IMP fails: concentration < 2 |
| E4 | INT holds: r flips decisions where the policy already departs from the additive code |
| E5 | ERR holds (trivially in part: r is what moves H off its posterior mean) |
| E6 | no goal has concentration ≥ 2 |

## Limits known in advance

One draw of donors; the posterior as Z only; excess is a difference of two binary outcomes per pair, so per-bin rates
are noisy where bins are small (unsolvable posteriors are ~14 % of goal-dependent cells).
