# Is goal-conditioned occupancy represented, beyond the posterior and the action values? (round 19)

Run date: 2026-09-29. Brief: `BRIEF.md`. Definitions, comparisons and criteria: `PLAN.md`, committed (`4d76653`)
**before any occupancy measurement on the round-18 models**. No model was trained: the eight round-18 runs are
analysed at their first and last checkpoints. Numbers from `tables.md`, data in `results.json`. Reproduce:
`reproduce.py r19` (25 min on one GPU; needs round 18's models and belief graph).

## Definitions

d_g^π(s | b) = E[Σ_{k=1..12} 0.9^(k−1) 1{S_{t+k} = s}], from the posterior b at the reveal, pursuing goal g.
The current cell is not counted; the visit that enters the goal is, and ends the sum.
* **Solver's occupancy**: exact, from the belief graph (its goal-cell component equals the optimal value, and it
  agrees with simulation to 0.02; `tests/test_mazebelief.py`).
* **A model's occupancy**: per history, from 32 rollouts of the model's sampled policy.
* **Rows**: 12 000 histories × 3 goals at the reveal. Held-out evidence: 11 292 rows. Held-out evidence–goal
  combinations (both fitted, never together): 8 236 rows.

One addition after the plan: a predictor P4 that also has the model's own action distribution at the reveal. A
model's next action is part of its own occupancy and is trivially readable from its activations.

## 1. How much of occupancy is not already the posterior or the action values

| predictor from exact quantities | R² of the solver's occupancy (held-out combinations) |
|---|---|
| P0: the goal's mean | 0.22 |
| P1: affine in [posterior, goal] | 0.52 |
| P2: affine in the posterior, per goal | 0.72 |
| P3: affine in [posterior, action values], per goal | 0.79 |

The residual after P3 is 19 % of the variance (criterion C1, ≥ 5 %: met). The task can separate occupancy from the
other two, with a fifth of occupancy's variance to do it on.

## 2. Where each quantity is accessible

Held-out affine decoding at the goal token, best site, held-out combinations:

| | posterior | action values | solver's occupancy | beyond the posterior | beyond posterior and action values |
|---|---|---|---|---|---|
| untrained | 0.71 | 0.82 | 0.64 | 0.12 | 0.01 |
| reward-trained, 5 seeds | 0.79–0.82 | 0.91–0.94 | 0.74–0.82 | 0.23–0.45 | 0.07, 0.10, 0.14, 0.22, 0.26 |
| solver-supervised | 0.85 | 0.91 | 0.80 | 0.43 | 0.31 |
| frozen backbone | 0.70–0.71 | 0.80–0.82 | 0.63–0.64 | 0.09–0.12 | −0.01–0.01 |

* **Action values are the most accessible quantity and the one training improves most** (0.82 → 0.93). Occupancy
  decodes less well than either of the other two at every site.
* **Most of what occupancy adds to the posterior is action value.** Beyond the posterior alone occupancy decodes at
  0.23–0.45; with the action values also removed, at 0.07–0.26. On held-out evidence: 0.01–0.21.
* **Criterion C2 (residual ≥ 0.3) fails in all five reward-trained models.** It is met, narrowly, by the supervised
  model (0.31; 0.28 on held-out evidence).
* **By depth** (mean of the reward-trained seeds): posterior 0.59 after the first attention layer and 0.80 from
  block 1 on; action values 0.76 and 0.90–0.93; occupancy 0.57, rising slowly to 0.78 at the output. The residual
  is 0 through block 0 and grows to its maximum at the last site.
* **At the last prefix token**, where the goal is not known, decoders fitted per goal recover the posterior at
  0.84–0.87, each goal's action values at 0.86–0.91 and each goal's occupancy at 0.67–0.72. These are functions
  of the posterior alone there.
* **No code is shared across goals.** A decoder fitted without one goal, goal means removed, last site: posterior
  −0.15 to 0.60, action values −0.26 to 0.43, occupancy −0.66 to 0.06, against 0.78–0.92 for decoders fitted on
  the goal. Occupancy transfers worst. A goal-independent map of "cells I will visit" is not what is there.

## 3. The same first action, a different future

Pairs with the same evidence and two goals, the same solver-optimal first action set and the same greedy action
of the model (total variation between the model's two action distributions: 0.01), occupancies 1.9 apart (L1);
4 270–7 012 pairs per model. Slope of the predicted occupancy difference on the true one:

| | slope |
|---|---|
| P0, P1 (goal means; posterior and goal without interaction) | 0.33–0.50 |
| P2, P3 (exact, per goal) | 0.76–0.84, 0.84–0.88 |
| activations, untrained | 0.69 |
| activations, reward-trained | 0.70, 0.73, 0.80, 0.80, 0.84 |
| activations, supervised | 0.69 |

Criterion C3 (≥ 0.5 and 0.1 above P1) is met by every model, **including the untrained one**. The goal token
states the goal and the first attention layer brings in the evidence, and an affine read-out of their sum already
tells these futures apart. Training adds 0.01–0.15. For pairs with the same goal and different evidence the
untrained network gives 0.46 and the trained ones 0.65–0.74, close to what the exact posterior gives per goal
(0.69–0.73). The discrimination asked for is there, and it does not distinguish a representation of occupancy from
a representation of the posterior and the goal.

## 4. The models' own occupancy

| | regret at the reveal, G2 | L1 gap to the solver's occupancy: G1 | G2 | G3 |
|---|---|---|---|---|
| reward-trained, seeds 1, 3, 4 | 0.003–0.004 | 0.48–0.51 | 0.93–1.04 | 0.36–0.39 |
| reward-trained, seeds 0, 2 (G2 not learned) | 0.048–0.053 | 0.52–0.58 | **4.53–4.63** | 0.40–0.41 |
| supervised | 0.001 | 0.29 | 0.49 | 0.29 |

* The two seeds that failed G2 go somewhere else entirely for that goal; for the other goals they match the rest.
* A model's own occupancy decodes better than the solver's in every reward-trained model (0.83–0.90 against
  0.74–0.82), most in the two that failed G2 (0.89–0.90), as expected.
* Its residual beyond the posterior and the optimal action values decodes at 0.28–0.47. With the model's own
  action distribution also removed: **0.13–0.29**, and the frozen backbone reaches 0.11–0.15 by the same measure.
  Below the criterion in every model.

## 5. Test 3 was not run

The plan made the intervention conditional on C1–C3. C2 failed for the reward-trained models, for the solver's
occupancy and for their own. An "occupancy subspace" defined by a decoder that explains 7–26 % of a residual
would mostly contain something else, and its effects could not be attributed.

## Criteria

| | criterion | met | numbers |
|---|---|---|---|
| C1 | residual occupancy ≥ 5 % of occupancy's variance | **yes** | 19 % |
| C2 | the residual decodes at ≥ 0.3 on held-out combinations, ≥ 0.15 above initialisation | no | reward-trained 0.07–0.26; supervised 0.31 |
| C3 | same evidence, same first action: slope ≥ 0.5, ≥ 0.1 above P1 | **yes** | 0.70–0.84 against 0.33–0.50; untrained 0.69 |

My expectation in the plan was that C2 would fail; it did. I did not expect C3 to be met by an untrained network.
I should have required a margin over the initialisation there as I did for C2.

## Conclusions

1. **Occupancy is decodable, and nearly all of that is the posterior, the goal and the action values.** What is
   specific to future visitation decodes at 0.07–0.26 in reward-trained models.
2. **The middle ground between the belief and the decision is better described by action values.** They are the
   most accessible quantity at every site, the one training improves most, and they absorb half of what occupancy
   adds to the posterior.
3. **A weak occupancy-specific signal exists late in the network** in two reward-trained seeds (0.22, 0.26) and the
   supervised model (0.31), absent before training (0.01) and in the frozen backbone. It is largest where the
   policy is best.
4. **Nothing here shows an order of computation.** The posterior, the action values and occupancy all become
   decodable within the first two blocks; whether one is computed from another was not tested.

Limits: affine decoders; the reveal only; one maze, in which a linear function of the posterior and the action
values already gives 79 % of occupancy. A task in which futures that share a first action diverge more, and later,
would give occupancy more to explain.
