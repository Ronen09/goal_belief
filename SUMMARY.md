# goalgeo — summary of results, rounds 1–12

Condensed from the per-round reports; numbers are seed means as reported there. See
`README.md` for the round-to-file map.

## Rounds 1–4: what shapes the learned geometry

**Round 1 — occupancy geometry (`results/REPORT.md`).** A 3-layer MLP trained by behavioural
cloning toward a supplied goal learns a hidden geometry only weakly predicted by the occupancy
vectors (RSA 0.29–0.35 at the last two layers) and strongly predicted by the policy table
(0.73–0.78 in the base grid, 0.77–0.85 in the twins maze, where every other hypothesis is near
zero). States with identical futures merge (portal pockets at 0.35 of the median distance);
mirror states requiring opposite actions do not. The representation is largely additive (≈44 %
state, 46 % goal, 11 % interaction at the middle layer). Goal-difference steering flips 88–100 %
of eligible states versus 41–49 % for random directions. With stochastic transitions the
geometry follows the stochastic-optimal policy table (partial RSA 0.61–0.69 vs 0.12).

**Round 2 — policy quotient vs occupancy (`results2/REPORT2.md`).** Under argmax targets the
policy table leads a six-way RDM regression in all nine environments (β 0.49–0.86). When no
optimal action changes between two λ values, the trained networks are bit-for-bit identical.
Under Boltzmann targets softmax(Q/0.02), hidden change tracks occupancy change smoothly
(Spearman 0.27–0.37 among non-flipping pairs) and advantage geometry leads (β 0.51–0.61). The
interaction component is not causally necessary (removing it costs ≤ 7 pp accuracy). Steering
thresholds are predicted by the linearised action-boundary crossing (Spearman 0.92–1.00);
occupancy predictors add nothing.

**Round 3 — the supervision bottleneck (`results3/REPORT3.md`).** Last-layer policy RSA falls
monotonically with target temperature, from 0.78 (hard) to 0.05 (τ=5), while PR rises from 3.1
to 16. Hidden dimensionality follows target informativeness, not target rank. Under hard
targets the within/between class ratio falls 0.98 → 0.46; held-out rows classify at 92–94 %.
The hard-target quotient replicates across MLP, 6-layer MLP, residual MLP and a 2-layer
transformer (RSA 0.71–0.78); the soft-target geometry replicates in the MLPs but not in the
normalised architectures (RSA ≤ 0.17). Six of eight success criteria met, one in part.

**Round 4 — HMM objectives (`results_hmm/REPORT_HMM.md`).** For two one-step-equivalent
beliefs, a window MLP collapses them under one-step training (0.24 of the control distance)
and separates them under 2-step training (1.05). A GRU does not collapse them (0.85) but stops
amplifying them; the belief stays decodable at R² ≈ 1. Sequential next-token training behaves
like one-step training in geometry, with exact or sampled targets.

## Rounds 5–7: what makes a distinction prominent (GRU, TASK4 HMM family)

**Round 5 — prominence (`results4/REPORT4.md`).** In 27 of 35 conditions the delayed cue is
99 % decodable one step before use while its prominence ranges 0.25–1.0 of the control.
Frequency, loss weight, expected forgetting cost and gradient magnitude do not set prominence;
the required log-odds gap at the time of use does — the frequency, strength, weight and grid
sweeps fall on one curve against it.

**Round 6 — readout scale (`results5/REPORT5.md`).** The required gap is a constraint on the
product of readout gain, hidden separation and alignment. Halving a fixed readout gain doubles
the hidden separation; an eightfold gain shrinks it to a quarter, with identical outputs and
loss. With a free readout, the gain barely changes with δ and the representation absorbs it.

**Round 7 — allocation (`results6/REPORT6.md`).** All 30 runs reach the Bayes floor with an
achieved gap of 4.305. The split among gain, separation and alignment is set by (1) relative
speed during fitting — changing the readout learning rate changes the split in proportion,
(2) initial scale, asymmetrically — a large initial gain is kept, a small one is grown, and
(3) a weak post-floor drift toward an intermediate allocation.

## Rounds 8–11: invariants and implementation freedom

**Round 8 — representation invariants (`results7/REPORT7.md`).** Across 64 functionally
equivalent GRUs:

| measure | coordinate-invariant | stable across equivalent models | sensitive to the function |
|---|---|---|---|
| raw distances, norm, PR, P_metric, gain, cos θ | no | no (CV 0.15–0.9) | weak |
| Euclidean / cosine RSA | no | yes (CV 0.02) | no |
| OLS decodability, rank | yes | yes | no |
| readout contrast (w_x−w_y)·Δh | yes | yes (CV 0.004) | yes |
| ‖JΔh‖, Fisher metric | yes | no (CV 0.27 / 0.22) | weak |
| divergence after finite propagation | yes | yes (CV 0.000) | yes |

Criteria 1–4 and 7–9 hold; 5 and 6 fail (local linearised measures are no more stable than raw
distance).

**Round 9 — intervention equivalence (`results8/REPORT8.md`).** Steering vectors in equivalent
models, matched by effect, give the same output distribution at every strength; matched by
geometry they do not. The geometries are linear images of each other on the reachable set
(R² 0.997; mapped steering vectors cosine 0.997). Local functional metrics disagree across the
same models by CV 0.4–1.2 at arbitrarily small scale, in every direction. Finite-propagation
divergence is invariant at every horizon.

**Round 10 — transformer reproduction (`results9/REPORT9.md`).** 2-layer pre-LN transformer,
with and without final LayerNorm, 60 models (53 at the Bayes floor). Decodability itself tracks
use (carrying a cue forward requires an attention pattern built only when the cue matters). The
required-gap identity holds at the readout interface in every model; with final LayerNorm the
freedom goes into alignment, without it separation regains about half its GRU role. Readout
contrast and interface decodability are stable to CV 0.004; raw geometry, gain, separation,
alignment and local metrics are not. Single-position patching effects vary by seed; patching
all positions between cue and use restores the invariant.

**Round 11 — implementation freedom (`results10/REPORT10.md`, theory and pre-registered
predictions in `docs/task10_theory.md`).**

- *Claim A, cut identifiability.* Over 20 models, 5 routing regimes and 3 complete cuts, the
  patched prediction equals the counterfactual prediction bitwise (deviation 0.0). The
  single-node effect ranges 0 → 0.368 (CV 1.13) across equivalent models; the sum of the two
  route effects ranges 0.367 → 0.798. Claim A holds as stated.
- *Claim B, factorisation freedom.* C = g·D·cos θ is fixed by the function (CV 0.006 across 77
  equivalent models whose factors span 37×; the three log-slopes sum to zero to three
  decimals). The frozen-LayerNorm bound C ≤ 4c√d held in all 49 frozen models but is not
  tight: the attained ceiling is C = 14.4c at δ=0.4 (non-antipodal rows ×0.65, non-antipodal
  states ×0.70), predicting a boundary of 0.306 against an observed 0.4 (0.220 vs 0.3 at
  δ=0.2). A 3× learning rate or 4× steps moves capped models by < 1 %. The separation D
  saturates at 11.3 (δ=0.4) and 5.8 (δ=0.2). Low-gain δ=0.2 frozen models (c ≤ 0.07) are stuck
  (cos θ ≈ 0.3), not capped. The identity half of Claim B holds; the architecture-predicts-
  factorisation half holds only ordinally; the quantitative boundary is falsified.
- 8 of 12 pre-registered predictions held (P1, P2, P3, P3b, P4, P7, P9a, P9c).

## Round 12: a hidden goal

**Round 12 — hidden goal (`results11/REPORT11.md`, theory and pre-registered predictions in
`docs/task11_theory.md`).** A latent goal (K = 4; also 3, 5) is inferred from 24 noisy tokens, and
the exact Bayesian posterior is computed. Two evidence processes are used: i.i.d. (log-odds
exactly affine in token counts) and a latent sticky reliability channel (best count-affine R²
0.869). Four objectives (posterior, soft and hard Bayes-optimal actions, next observation) are
trained on a 2-layer transformer and a GRU, and an affine probe to the K−1 log-odds is scored on
held-out sequences, confident posteriors (EXT), a held-out posterior region (CONF, also withheld
from network training) and later time steps.

- Posterior-trained models: interface R²_y ≥ 0.975 on every held-out split, both architectures,
  K = 3–5. In the channel env the probe removes 94–97 % of the best tally's error. Networks never
  trained on the conflict region decode it at 0.994–1.000 with unchanged output KL.
- In-distribution R² does not discriminate: every trained iid transformer reaches ≥ 0.969 at its
  best site and an untrained one 0.82. Extrapolation does: the `goal` interface is affine in
  log-odds (EXT R²_y 0.98, R²_b −0.97) and the `act_soft` interface in probabilities (EXT R²_y
  0.55, R²_b 0.999).
- Gain over counts at the channel-env interface: goal 0.94 / 0.97, next_obs 0.51 / 0.79,
  act_soft 0.39 / 0.73, untrained −0.78 / −1.28 (transformer / GRU). In the transformer the
  nonlinear filter appears in block 2.
- Hard-target transformers carry the posterior upstream and compress it to the decision at the
  last block. Trained without the conflict region, they fail there (output KL 2.7 nats vs 0.01),
  while posterior-trained models do not. In the channel env hard-target models are capped at
  ≈ 98.5 % optimal-action agreement, and 4× training does not raise it (not pre-registered).
- 4 of 11 pre-registered predictions held (P1, P2, P5, P10). The failures are three threshold
  near-misses, one mis-specified criterion (P8) and three substantive: the GRU carries log-odds
  whatever the target, the first transformer block is not a running mean, and the channel
  hard-target models are capped.

## Cross-round findings

1. Hidden geometry mirrors the distinctions the training target contains (policy quotient for
   argmax targets, advantage for soft targets), across MLP, residual and transformer
   architectures for hard targets.
2. The scale of a distinction is set by the output gap it must produce, divided between readout
   gain, separation and alignment by optimisation dynamics, not by frequency or gradient
   pressure.
3. Raw geometry and local (Jacobian, Fisher) metrics are implementation-dependent; decodability,
   readout contrast and finite-propagation effects on a complete causal cut are invariant
   across functionally equivalent models, in both GRU and transformer.
