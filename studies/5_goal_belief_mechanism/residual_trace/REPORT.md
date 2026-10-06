# Where the residual beyond the posterior matters (residual trace)

Run date: 2026-10-06. Brief: `BRIEF.md` (given in conversation). Strata, measures and decision rule: `PLAN.md`, committed
(`dd1f5c6`) **before `run.py` was run on any trained model**. Numbers from `tables.md`; data in `results.json`.
Reproduce: `reproduce.py residual_trace` (3 min on one GPU).

**Setting.** As in the history-mediator experiment: at the goal token's four attention outputs, the history part
d(h) = x_b(h) + r(h), with x_b the posterior's table. On informative pairs (24 k per model), the real arm swaps r to a
random same-length donor's. The rot arm adds the same vectors rotated, at the same size. **Excess** = real − rot is r's
content-specific effect, as a flip of the decision and as a shift of the logit gap toward the donor's action.
**Concentration** = a bin's share of the total excess flip / its share of pairs.

Overall: flips real 0.116, rot 0.038; shift real 1.37, rot 0.72 (reproducing the history-mediator experiment).

## 1. No stratum holds r's effect: it is spread, with moderate enrichment

| stratum | where the excess flip is most concentrated | concentration | its excess shift (against the rest) |
|---|---|---|---|
| margin to the donor's action | lowest quartile | 1.96 | 0.87 (top quartile 0.58) |
| H's deviation from its posterior mean | top quartile | 1.96 | 1.55 (bottom quartile 0.34) |
| additively unsolvable posterior (7.5 % of pairs) | unsolvable | 1.59 | 1.14 (solvable 0.56) |
| posterior entropy | top tercile | 1.54 | 1.19 (bottom 0.38) |
| natural action ≠ additive decision (8.6 %) | differs | 1.48 | 0.99 (same 0.67) |
| swapped residual's size | top quartile | 1.20 | 0.77 |
| goal | none (G1 0.91, G2 1.08, G3 0.70) | ≤ 1.08 | |

* **Margin.** r's flips favour low-margin decisions, as any perturbation's do: the lower half holds 0.74 of the excess.
  But the shift it causes is nearly as large at the top margin quartile as at the bottom (0.84 ×). r moves the logits
  broadly. It flips decisions mostly where they are close, yet in the top quartile it still flips 4.4 % beyond the
  rotated control. So r is not confined to near-ties.
* **Additively impossible cases, and cells where the policy departs from the additive code**, are enriched (1.59,
  1.48) but hold only 12 % and 13 % of the excess. Most of r's effect is in solvable posteriors whose decision is
  additive.
* **No goal is singled out.**

## 2. Jointly, uncertainty and H's own deviation predict r's effect; the additive failures do not

Least-squares fit of the per-pair excess shift on all strata together (standardized; median, range over ten models):

| margin | unsolvable | interaction used | entropy | H's deviation | residual size |
|---|---|---|---|---|---|
| 0.13 (−0.33–0.34) | 0.13 (−0.02–0.65) | 0.01 (−0.54–0.20) | **0.53 (0.37–0.71)** | **0.65 (0.01–1.17)** | 0.16 (−0.12–0.82) |

* **Only posterior entropy and H's deviation from its posterior mean are positive in all ten models.** With them in the
  fit, unsolvability and interaction use add nothing consistent. Their enrichment in section 1 comes from overlap with
  uncertain posteriors.
* So r matters most where the exact posterior is uncertain, and in histories whose own H the posterior predicts badly.

## 3. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **BND** | lower half of margin holds ≥ 0.75 of the excess flip, and the top quartile's excess shift ≥ 0.5 × the bottom's | 0.74; 0.84 × | no (by 0.01) |
| **IMP** | unsolvable concentration ≥ 2, and excess shift there > solvable's | 1.59; 1.14 against 0.56, p < 0.001 | no |
| **INT** | interaction-used concentration ≥ 2 | 1.48 | no |
| **ERR** | top H-error quartile concentration ≥ 2 | 1.96 | no (by 0.04) |

**Registered reading: neither boundary correction nor the interaction route.** The plan named this outcome "a
separate, localized signal". The strata show it is **not localized**: no bin holds twice its share. The first half of
that reading holds: r is not the h × g interaction found earlier. "Localized" does not.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | real flip ≈ 0.115, rot ≈ 0.037 | yes | 0.116, 0.038 |
| E2 | BND holds | **no** (narrowly) | 0.74 |
| E3 | IMP fails | yes | 1.59 |
| E4 | INT holds | **no** | 1.48 |
| E5 | ERR holds | **no** (narrowly) | 1.96 |
| E6 | no goal concentration ≥ 2 | yes | ≤ 1.08 |

3 of 6.

## Conclusions

1. **r is not the interaction route.** The additively impossible cases and the cells where the policy departs from the
   additive code are mildly enriched, but hold only an eighth of r's effect each. Their enrichment disappears once
   posterior uncertainty is controlled for.
2. **r is not only a near-tie correction either.** It shifts the logits broadly, about as much at high margins as at
   low ones. It flips decisions mostly near ties, but not only there.
3. **What predicts it is uncertainty.** r's effect is largest where the exact posterior has high entropy, and where a
   history's H departs from the posterior's mean H. A plausible reading, not tested here: r is the history-specific
   error of the network's approximate belief. Different histories with the same exact posterior leave slightly
   different internal estimates, and those differences matter most when the belief is spread over many cells. That
   matches the pair-types experiment, where identical-posterior histories differ in which cases the model gets wrong.

Next, if wanted: decode the network's own posterior at the goal token, and test whether r is predicted by that
decoded posterior's error against the exact one.

Limits: one draw of donors; the posterior as Z only; per-pair excess is a difference of binary outcomes, noisy in
small bins; two criteria missed their thresholds by 0.01 and 0.04, so the rule's verdict is close to its boundary.
