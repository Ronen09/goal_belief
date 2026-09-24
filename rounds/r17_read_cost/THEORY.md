# Pricing recomputation: a learned, costed read of the history (TASK16, round 17)

Brief (given in conversation, 2026-09-24): round 16 made recomputation *hard* from outside, by dropping
historical K/V entries at random. This round makes it *expensive* and lets the network decide when
it is worth paying: does a next-token transformer, charged per read of the history, learn to trust a
carried belief and look back only when that belief is unreliable? Predictions in §4 are committed
**before any model of this round is trained**.

## 1. What a price adds to round 16

Round 16 established two things. A carry transformer moves continuously from recomputation
(prior weight 0.36 at p = 0) to full recurrence (1.000 at p = 1) as the history is dropped, solving
the task at every p. A plain transformer saturates at a prior weight of ≈ 0.25: position t reaches
t+1 only one layer up, so a 4-layer model can hand its state forward only a few positions, and
dropping the history beyond that costs accuracy rather than buying recursion.

Random dropout cannot ask *when* the network wants the history, because the network never chooses.
A price can. If the carried belief is a sufficient statistic, then in a carry model with a
well-trained carry a read of the history is worth nothing, and any positive price should close the
reads once the carry is trusted. But the carry is not trusted at initialisation, when the reads are
what builds it; and along a sequence the carried belief is more reliable at some positions than at
others (in the channel environment, least reliable just after the reliability channel switches).
A learned gate exposes that choice. Three questions follow:

* **Does the price induce recurrence at all?** The carry model should close its reads as the price
  rises, with the task still solved (round 16's continuum, now chosen rather than imposed).
* **Is the reading selective?** At an intermediate price, the reads should concentrate where the
  carried belief is uncertain or has just moved, not be spread uniformly.
* **Does the plain transformer's ceiling survive a price?** Round 16 attributed the ceiling to depth.
  If a price pushes the plain model's prior weight well past 0.25 while it still solves the task,
  that explanation is wrong. If the plain model instead keeps paying for reads until the price
  exceeds the accuracy they buy, and then loses accuracy, the ceiling is structural.

## 2. Models

Both families are next-token models on sampled tokens of the channel environment (K = 4, T = 24), as
in rounds 15–16, trained 30k steps at batch 128, Adam 1e-3, 3 seeds.

**The gate.** In every block ℓ and at every query position u there is one binary gate g_ℓ(u) ∈ {0, 1},
shared across heads, that decides whether the historical keys are visible to that query:
* plain: keys s < u − 1 are visible iff g_ℓ(u) = 1; u and u − 1 are always visible (round 16's rule);
* carry: keys s < u are visible iff g_ℓ(u) = 1; u is always visible and the carry u_{u−1} is always
  present.

The gate logit is an affine function of the block's normalised input at u, a_ℓ(u) = w_ℓ · LN1(x_ℓ(u))
+ b_ℓ, so in the carry model the gate can consult the carried belief. During training the gate is a
straight-through Gumbel-sigmoid sample (hard 0/1 forward, gradient through the relaxed sigmoid at
temperature 0.5); at test it is deterministic, g = 1[a > 0]. The gate parameters are initialised so
that reads are open (b_ℓ = 3).

**The price.** Loss = next-token cross-entropy + c · mean_{ℓ, u ≥ 2} σ(a_ℓ(u)), so a model that
reads everywhere pays c nats per token on top of its cross-entropy. Grid: c ∈ {0, 0.003, 0.01,
0.03, 0.1, 0.3}. For scale, round 16's plain model lost 0.05 nats per token when denied the history
(p = 1), and the carry model 0.006; the grid straddles both.

* **plain**: 4-layer pre-LN transformer (`tfm.CausalTransformer`, GPU, stacked trainer).
* **carry**: 2-layer `wtfm.WindowTransformer`, full window with carry (CPU workers).

c = 0 with the gate present is the control: it is round 16's p = 0 with an extra (unpriced) gate.

## 3. Read-out

**Read rate.** Fraction of (ℓ, u ≥ 2) with an open deterministic gate over 3000 held-out sequences;
also per position and per layer.

**Prior weight.** Round 15/16's calibrated λ of R2 (position t's exports swapped for history B's;
positions < t and the raw tokens from A) at t ∈ {4, 8, 16, 23}, 1000 pairs, in two regimes: (a)
**forced open** — every gate set to 1, so the network can see the history; (b) **own gates** — the
gates as learned. Under own gates λ_R2 = 1 wherever the gate is closed, by construction; the
informative regime for the algorithm is forced open, as round 16's full-access regime was. Note that
in the carry family the gate at t+1 reads the carried u_t, which the R2 splice replaces, so the gate
decision may itself differ between cells; it is recorded per cell.

**Accuracy.** KL of the predictive to the exact predictive under own gates and forced open.

**Selectivity** (carry family, at every c whose seed-mean read rate lies in [0.1, 0.6]; if no c
qualifies, the nearest one above 0.6). On held-out sequences, with the exact joint filter computed
alongside:
* entropy split: mean entropy of the exact filter over (G, c) at positions where the gate is open
  minus at positions where it is closed, within positions u ≥ 8 (so that the early-sequence
  bootstrapping does not drive it);
* movement split: the same difference for the KL between consecutive exact filter states
  (how far the belief just moved);
* position profile: read rate as a function of u.

**Crossover.** c* = the smallest grid price at which the seed-mean read rate is ≤ 0.5, per family
(∞ if none).

## 4. Pre-registered predictions

Seed means; prior weight at t = 16 unless stated.

* **R1 (carry: the price closes the reads).** Read rate is non-increasing in c, ≥ 0.5 at c = 0 and
  ≤ 0.05 at c = 0.3.
* **R2 (carry: the task is still solved).** Predictive KL under own gates < 0.01 at every c.
* **R3 (carry: the closed model is recurrent).** At every c with read rate ≤ 0.05, prior weight
  under own gates ≥ 0.95 and under forced open ≥ 0.8.
* **R4 (carry: the reads are selective).** At the qualifying intermediate c, the entropy split is
  ≥ +0.05 nats and the movement split is ≥ +0.02 nats: the gate opens where the carried belief is
  less certain and where it has just moved.
* **R5 (carry: bootstrapping).** At the qualifying intermediate c the read rate at positions
  u ≤ 4 exceeds the rate at u ≥ 16 by ≥ 0.2.
* **R6 (plain: the ceiling is structural).** Prior weight under forced open ≤ 0.4 at every c.
* **R7 (plain: closing the reads costs accuracy).** At every c where the plain read rate is
  ≤ 0.1, predictive KL under own gates ≥ 0.03.
* **R8 (the carry makes reads cheaper to give up).** c*_carry < c*_plain; the plain read rate is
  still ≥ 0.5 at c = 0.01.
