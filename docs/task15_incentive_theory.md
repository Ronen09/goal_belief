# Inducing recurrence by incentive: random K/V dropout on history (TASK15, round 16)

Brief: `TASK15.md`. Predictions in §4 were committed **before any model of this round was trained**.

## 1. What an incentive can and cannot do in each architecture

**Plain transformer.** Position t's layer-ℓ residual is read by position t+1 only at layer ℓ+1. There
is no same-layer recurrence, so information travels at most L positions through the "previous
position" route. Dropping every older K/V entry (keeping only self and t) therefore leaves a model
that sees about L+1 tokens: round 14's windowed model without a carry, which cannot compute the
posterior. In a plain transformer, K/V dropout can raise the weight of the previous position's
export, but it cannot reach a complete cut that still solves the task. The continuum is capped by
depth and paid for in accuracy.

**Carry transformer** (round 14: full attention + the previous position's readout interface added to
the input). It spans the full range architecturally: with full attention the carry was ignored
(future control −0.005 to 0.17), and with window 1 it was a complete cut (1.000). Randomly dropping
historical K/V entries with probability p while the carry is always present varies continuously
between those two ends: p = 0 is full attention with a carry, and p = 1 is window 1 with a carry.

## 2. Models

Both next-token language models on sampled tokens of the channel environment (K = 4, T = 24), as in
round 15.
* **plain**: 4-layer pre-LN transformer (`tfm.CausalTransformer`). During training, for every query
  position u and key s < u − 1, the entry is dropped independently with probability p (the same
  mask in every layer); u and u − 1 are always visible. 30k steps, batch 128, on the GPU.
* **carry**: 2-layer `wtfm.WindowTransformer` with full window and carry. Every historical key s < u
  is dropped with probability p; the carry u_{t−1} is always present. 30k steps, on the CPU.

p ∈ {0, 0.5, 0.75, 0.9, 0.97, 1} × 3 seeds for each family.

## 3. Read-out

Round 15's 2×2 at t ∈ {4, 8, 16, 23}. Pairs (A, B) with full filter states ≥ 2 apart, and the same
pair generator and seeds as round 15. Position t+1 is computed from spliced exports:
* R1: all of A's;
* R2: everything position t exports taken from B (plain: res_i(t) for i ≥ 1; carry: every cached
  K/V entry of t and the carried vector u_t), with A's raw tokens kept;
* R3: positions < t from B, position t from A;
* R4: all of B's.

λ = position of the output predictive at t+1 along A⁺ → B⁺ (centred-log coordinates), rescaled per
seed so that R1 = 0 and R4 = 1. **Prior weight** = calibrated λ of R2. R3 is compared with the
parameter-free pure-recomputation prediction of round 15. Every model is measured (a) with full
access at test time and (b) under its own training dropout, where query t+1 sees each older key with
probability 1 − p and always sees t.

Accuracy: KL of the model's predictive to the exact predictive, under full access and under its
training dropout.

## 4. Pre-registered predictions

Prior weight at t = 16 unless stated; seed means.

* **I1 (carry: the continuum exists).** Carry models, full access at test: prior weight ≤ 0.1 at
  p = 0, ≥ 0.95 at p = 1, non-decreasing in p, and strictly between 0.1 and 0.95 for at least two
  intermediate p.
* **I2 (carry: all solve the task).** Carry models reach predictive KL < 0.01 under their training
  dropout at every p.
* **I3 (carry: the incentive changes the algorithm even when evidence is available).** Carry model
  at p = 0.9: prior weight under full access at test ≥ 0.3.
* **I4 (dropping at test shifts weight to the prior).** Both families: prior weight under the
  model's own dropout ≥ prior weight under full access, at every p ∈ (0, 1).
* **I5 (plain: rises, but capped).** Plain models, full access: prior weight increases with p from
  ≤ 0.05 at p = 0, and exceeds 0.2 at p = 0.9.
* **I6 (plain: the price of depth).** Plain models at p = 1 have predictive KL > 0.05 under their
  own dropout (they cannot see the history).
* **I7 (recomputation gives way).** For carry models at p ≥ 0.75, R3 (older evidence from B, t
  from A) departs from the pure-recomputation prediction toward A by at least half the prior
  weight: λ_R3 ≤ prediction − 0.5 × prior weight.
