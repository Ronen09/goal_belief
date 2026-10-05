# Maze occupancy: definitions, comparisons and criteria

Written before any occupancy measurement was made on the maze-belief models.

## Definitions

* **Occupancy** d_g^π(s | b): `goalgeo/mazeocc.py`. k starts at 1; the visit that enters the goal is counted and ends
  the sum. Its goal-cell component under the solver's policy equals the optimal value (tested).
* **Solver's occupancy**: exact, from the belief graph, the solver choosing uniformly among tied optimal actions.
* **Model's occupancy**: per history (the model is a function of the tokens, not of the posterior), from 32
  rollouts of the model's sampled policy, the true cell drawn from the exact posterior; each step contributes the
  exact next-cell distribution given the rollout's history.
* **Rows**: 12 000 maze-belief bank histories with a prefix of at least one move, each with all three goals (36 000
  rows), at the reveal. Patches and decoders are at the goal token and at the last prefix token.
* **Held-out sets**. Evidence whose hash falls in the test side: all three goals are test rows (*held-out
  evidence*). For every other evidence one goal, chosen by the hash, is withheld (*held-out combination*): the
  evidence and the goal were both fitted, never together.

## What occupancy has to beat

Occupancy is a function of (b, g). Exact-feature predictors of d, fitted on the fit rows:
* P0: the goal's mean occupancy;
* P1: affine in [b, goal one-hot] (no interaction);
* P2: affine in b, separately per goal;
* P3: affine in [b, Q*], separately per goal.
The **residual occupancy** is d − P3: what neither a linear read-out of the posterior nor the action values give.
Its share of the variance of d is reported; if it is small, the task cannot separate occupancy from the other two.

## Tests

1. Held-out decoding at the 9 sites, of b, Q*, d (solver), d (model), and the residual occupancy; at the goal token,
   and at the last prefix token (where the goal is unknown: d for each goal is then a function of b only). The
   initialisation is the baseline. A decoder fitted without one goal, tested on it (goal means removed).
2. Pairs with the **same evidence, two goals, the same solver-optimal first action set and the same greedy action of
   the model**, whose occupancies differ by at least 0.5 (L1). Slope of the decoded difference on the true one,
   ⟨Δd̂, Δd⟩ / ⟨Δd, Δd⟩, for the activation decoder and for P0 and P1. The same for pairs with the same goal, the same
   first action and different evidence.
3. Only if decoding is convincing.

## Criteria for "convincing" (needed before test 3)

* C1. Residual occupancy is at least 5 % of the variance of d (otherwise stop: not separable here).
* C2. The residual decodes from some site at held-out R² ≥ 0.3 on held-out combinations, and at least 0.15 above
  the same site at initialisation.
* C3. On same-evidence, same-first-action pairs the activation decoder's slope is ≥ 0.5 and above P1's by ≥ 0.1.

## Expectation

From the navigate-commit and maze-belief experiments (the product of marginals was not represented; the posterior's code was goal-specific) I
expect d to decode well in aggregate, mostly through the posterior and the goal, and the residual to decode
poorly: C2 fails. I expect the model's own occupancy to decode no better than the solver's in the three seeds that
learned, and better in the two that did not learn G2.
