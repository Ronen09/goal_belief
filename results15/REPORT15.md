# Inducing recurrence by incentive: random K/V dropout on history (TASK15, round 16)

Run date: 2026-09-23. Brief: `TASK15.md`. Theory and 7 predictions: `docs/task15_incentive_theory.md`,
committed (`57b3e10`) **before any model was trained**. A finer dropout grid was added afterwards
and is reported separately as a follow-up (`results15/fine/`). Numbers from `results15/tables15.md`,
per-model values in `results15.json` and `fine/results15.json`, weights in `models/`, figure
`fig1_incentive.png`. Reproduce: `scripts/run_task15.py` (30 min), then
`scripts/run_task15.py --out results15/fine --ps 0.05 0.1 0.2 0.3 --plain-ps 0.1 0.25`, then
`scripts/task15_tables.py`.

## Design

Next-token language models on the hidden-channel process (round 15's setting: sampled tokens, context
24), trained with **historical K/V dropout**. For each query and each key older than the guaranteed
ones, the entry is removed with probability p, resampled every step. p is a continuous "cost of
recomputation" knob. Two families:
* **plain**: a standard 4-layer full-attention transformer; self and the previous position are
  always visible. Your recipe as stated.
* **carry**: round 14's transformer with full attention plus a carried copy of the previous
  position's readout interface; the carry is always present and every historical K/V entry
  (including the previous position's) can be dropped. p = 0 is full attention with a carry, and
  p = 1 is window 1 with a carry, a complete cut.

The read-out is round 15's. Position t+1 is computed with everything position t exports (its K/V
at every layer, and for the carry family its carried vector) taken from another history B, while A's
tokens are kept. The prior weight λ is where t+1's next-token prediction lands between the exact
Bayes updates A⁺ (0) and B⁺ (1), calibrated per seed on the baseline and the consistent
counterfactual. Measured at t = 4, 8, 16, 23, with full attention at test and again under the
model's own training dropout. 3 seeds per p.

## 1. The transition is continuous, and training incentive alone produces it

Prior weight at t = 16, full attention at test (fig. 1, left):

| training dropout p | 0 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 | 0.75 | 0.9 | 0.97 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **carry** | 0.36 | 0.65 | 0.73 | 0.80 | 0.84 | 0.90 | 0.95 | 0.975 | 0.995 | 1.000 |
| weight on older evidence (R3) | 0.57 | 0.28 | 0.21 | 0.16 | 0.12 | 0.07 | 0.04 | 0.02 | 0.005 | 0.00 |
| **plain** | 0.03 | | 0.18 | | 0.23 (p = 0.25) | 0.25 | 0.25 | 0.24 | 0.24 | 0.14 |

(p = 0.05–0.3 for the carry, and p = 0.1, 0.25 for the plain, are the follow-up grid.)

* **With a carry, the prior's weight is a smooth function of the cost of recomputation.** It
  follows logit(weight) = 2.39 + 0.69·logit(p) (R² 0.97 over 24 seed-level points, 0 < p < 1).
  The weight on the older evidence falls in step, and the two sum to 0.93–1.00: the model splits
  one unit of trust between its previous state and a reread of the history. Your three regimes
  are all present: recomputation available (p = 0; the carry still takes 0.36, see §3),
  recomputation costly (0 < λ < 1 across the whole range), and complete cut (λ = 1.000).
* **The incentive changes the algorithm even when evidence is not missing.** Every row above is
  measured with full attention at test, and removing the dropout at test changes nothing (the
  dropout-at-test column differs by ≤ 0.01 for the carry). A model trained at p = 0.3 reads the
  whole history and still puts 0.84 of its weight on the state it carried.
* **Every carry model solves the task** (KL ≤ 0.006), and the more it relies on its state, the
  better it does (KL 0.006 at p = 0 to 0.002 at p = 0.9). The recursive solution generalises better
  on this process.

## 2. Without same-layer recurrence, the incentive hits a ceiling

The plain transformer's prior weight rises quickly (0.03 → 0.18 at p = 0.1) and then stops at about
0.24 however hard recomputation is made. The only route from position t to t+1 goes up one layer
per step, so a 4-layer model can hand its state forward only a few positions. Dropping history
beyond that point costs accuracy (KL 0.025 at p = 0.97, 0.05 at p = 1), not more recursion. Under its
own dropout at test, the plain model leans on position t more (up to 0.66), because there is less
else to read. But it is being deprived, not recursive. The incentive can move a plain transformer
only part of the way; the carry is what makes the full range reachable.

## 3. Two things the theory did not anticipate

* **A carry is used even with full attention in a next-token model** (0.36 at p = 0). In round 14 a
  carry with full attention was ignored, but that model was trained to output the goal posterior,
  and only its carried vector was swapped. Here the objective is next-token prediction and the
  swap includes position t's K/V. The prior's weight also falls with position (0.47 at t = 4 to
  0.34 at t = 23 for p = 0; fig. 1, right). With little cost to recomputation, the longer the
  history, the more the model rereads it.
* **The transition is front-loaded.** The first 5% of dropout moves the prior's weight from 0.36
  to 0.65. The pre-registered grid put its intermediate points at p ≥ 0.5, where the carry family
  is already at 0.90–0.995.

## Pre-registered predictions

| | prediction | held | numbers |
|---|---|---|---|
| I1 | carry: ≤ 0.1 at p = 0, ≥ 0.95 at p = 1, monotone, ≥ 2 intermediate points in (0.1, 0.95) | no | monotone ✓, 1.000 at p = 1 ✓; **0.36 at p = 0** ✗; only p = 0.5 intermediate on the registered grid ✗ |
| I2 | carry: KL < 0.01 under own dropout at every p | **yes** | ≤ 0.006 |
| I3 | carry p = 0.9: prior weight with full access ≥ 0.3 | **yes** | 0.975 |
| I4 | dropout at test shifts weight to the prior | **yes** | carry +0.01, plain +0.09 to +0.43 |
| I5 | plain: rises from ≤ 0.05 at p = 0 to > 0.2 at p = 0.9 | **yes** | 0.027 → 0.243 |
| I6 | plain p = 1: KL under own dropout > 0.05 | no | 0.023 (0.052 with full access) |
| I7 | carry p ≥ 0.75: R3 ≤ recomputation prediction − 0.5 × prior weight | **yes** | R3 ≤ 0.04 vs bounds ≥ 0.36 |

5 of 7 held. I1 failed at both ends of the registered grid for the same reason: the transition is
front-loaded, and the carry is used at p = 0. The follow-up grid resolves the continuum the
prediction was about. I6 underestimated how much a 4-layer model can infer from the last few
positions of this process.

## Conclusions

1. **An optimisation incentive induces the transition from recomputation to recursion
   continuously.** With a recurrent path available, the weight a transformer gives its previous
   inferred state is a smooth, monotone function of how unreliable the historical evidence was
   during training: logit-linear in the dropout rate, from 0.36 to 1.000, with the task solved at
   every point. The learned algorithm persists when the evidence is restored.
2. **The incentive decides how much the state is used; the architecture decides how much it can
   be.** Without same-layer recurrence, the same incentive saturates at about 0.25 and then only
   costs accuracy.
3. **The general statement these rounds support:** a sufficient state becomes the causal
   computational state to the degree that (a) the architecture gives it a path that does not
   decay with distance, and (b) the alternative, rereading the evidence, is unavailable or
   unreliable during training. (a) sets the ceiling, and (b) sets the position below it.
