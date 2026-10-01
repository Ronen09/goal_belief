# Round 29: routes, measures, decision rule and expectations

Written before any route intervention of this round was applied to a trained model. Code: `run.py` (smoke-tested on
an untrained network only). Brief: `BRIEF.md`.

## Models, pairs, cells

Round 23's ten reward-only models, frozen. Round 27's pairs, unchanged: main (30 000; primary) and one-step agreeing
(20 000; secondary). Every pair under all three goals: a pair under one goal is a cell. With the pair fixed, the
evidence difference is fixed; only the goal changes. 18 159 main pairs have change cells (disjoint optimal sets)
under two or more goals, and 11 936 one-step agreeing pairs (solver tables only).

## Two routes, exactly

The decision at the goal token is a function of the residual stream entering block 1 at the prefix tokens and at the
goal token. Attention is causal, and nothing after the goal token exists at the decision.

* **Interface**: the prefix tokens' states entering block 1 (goal-free; rounds 20–28).
* **Direct route**: the goal token's own state entering block 1. It is computed by block 0 from the raw tokens and the
  goal.

Interventions, per cell, from the hybrid run (the donor's prefix with the same goal):

| intervention | prefix states entering block 1 | goal token's state entering block 1 |
|---|---|---|
| none | recipient | recipient |
| **interface** (whole-interface replacement) | donor | recipient |
| **encoding** (round 28's edit) | h_A,i + E (b_B,i − b_A,i) | recipient |
| **direct** (targeted, the direct route) | recipient | donor |
| both | donor | donor: equals the hybrid exactly (a check) |

## Measures

* **Reliance** on a route: the projection of the intervention's logit change onto the hybrid's,
  a = ⟨ℓ_k − ℓ_none, ℓ_hyb − ℓ_none⟩ / |ℓ_hyb − ℓ_none|² (centred log-probabilities), per cell. a_interface +
  a_direct = 1 if the routes add up; the remainder is their interaction. Also the TV form (moved).
* **Eligible cells**: TV(none, hybrid) ≥ 0.1, where the policy's behaviour differs between the two histories.
* **Controls**, per cell: the unedited behavioural difference TV(none, hybrid) and |ℓ_hyb − ℓ_none|; the oracle action
  gap, the cost of acting on the other history's optimal action, averaged over the two directions (exact values).
* **Direct-route adequacy**, defined without any intervention: an affine probe, per goal, from the goal token's state
  entering block 1 to the exact action values. It is fitted on the fit side with each history under each goal. A cell
  is *direct-adequate* if the probe's greedy action is optimal for both histories under that goal. The same probe from
  the last prefix token's state gives *interface adequacy* (secondary).

## Analysis and decision rule

Per model: within-pair regressions (pair fixed effects) on eligible cells of pairs with at least two eligible goals.
Across the ten models: medians and exact Wilcoxon signed-rank tests, p < 0.05.

| | question | regression of a_interface on | criterion |
|---|---|---|---|
| Q1 | does interface reliance depend on the goal at a fixed pair? | goal dummies (G2, G3) + TV + norm + gap | a goal coefficient with median \|β\| ≥ 0.05 and two-sided p < 0.05 |
| **Q2** | is it higher where the direct route handles the goal's distinction poorly? | direct-adequate + TV + norm + gap | β(direct-adequate) median ≤ −0.05, one-sided p < 0.05 |
| Q2b | the distinction, not just the goal's identity | as Q2 plus goal dummies | β(direct-adequate) < 0, one-sided p < 0.05 |

| result | reading |
|---|---|
| Q2 (and Q2b) | **the goal influences which evidence contributes, appropriately**: where the goal needs a distinction the direct route lacks, the policy takes more from the interface |
| Q2 without Q2b | the shift is carried by which goal it is; within a goal, direct-route adequacy does not add |
| Q1 without Q2 | route dependence varies with the goal, but not in the way the direct route's adequacy predicts |
| neither | route dependence is similar across goals: a shared evidence estimate is transformed into different action preferences |

Reported alongside, without criteria: the same regressions for a_encoding and a_direct; per-goal means (all eligible
cells, and the pairs eligible under all three goals); direct-adequate against inadequate cells per goal; a matched
contrast (two goals of the same pair, one direct-inadequate and one adequate, with TV within 0.1 and gap within 0.02);
the one-step agreeing pairs; probe accuracies.

## Expectations

| | expectation |
|---|---|
| E1 | check: a_both = 1 within 10⁻³ on eligible cells |
| E2 | Q1 holds |
| E3 | Q2 holds |
| E4 | Q2b holds |
| E5 | the routes roughly add up: median interaction \|1 − a_interface − a_direct\| ≤ 0.2 per goal |
| E6 | one-step agreeing pairs: a_interface higher than on main pairs, every goal |
| E7 | the direct-route probe is least accurate under G1 and G2, most under G3 |

I hold E3 and E4 weakly: the interface is read by the goal token's attention in blocks 1–3, whose queries depend on the
goal. That makes goal-dependent reading possible, but does not require it.

## Limits known in advance

Adequacy is a linear probe of the routes' states, not the policy's own read-out; one interface depth; the first
decision after the reveal; reward models only.
