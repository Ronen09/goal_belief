# What makes information geometrically prominent? (TASK4)

Run date: 2026-09-18, 8 CPU worker processes, about 20 minutes in total
(78 sweep runs in 815 s, 27 grid runs plus 12 dense-checkpoint runs in 377 s).
3 seeds per condition, 3000 Adam steps, batch 128, sequences of 48 tokens.
Numbers from `rounds/r05_prominence/tables.md`, per-run values in `rounds/r05_prominence/results4.json`.
Design: `rounds/r05_prominence/DESIGN.md`.
Reproduce with `rounds/r05_prominence/run.py --jobs 8 --grid --dynamics`.

## Setup

**HMM family** (`goalgeo/hmm4.py`, parameters r, δ, k). A delayed branch: cue
token p or q, then k filler states emitting x/y at 0.5, then with probability r
the branch-specific state C (x with 0.5+δ) or D (x with 0.5−δ), otherwise a
neutral state N that emits its own token n. A fixed control branch: cue u or v,
then A′ (x 0.9) or B′ (x 0.1) immediately. Every branch returns to one of the
four cue states uniformly. At r=1, δ=0.4, k=1 the delayed branch is the round 4
chain. The delayed cue never affects the next k predictions; it affects the
prediction at t* = t+k+1 only when the gate opened, by a log-odds difference of
ln((0.5+δ)/(0.5−δ)) per token between the two branches. The control cue has the
same difference (0.9 versus 0.1) at the very next step in every condition.

**Model and objective.** The round 4 GRU (64 hidden units, 16-dim embedding,
7-token vocabulary), next-token prediction with exact per-position targets.
Positions whose token is the relevant one (emitted by C, D or N) get loss
weight λ, all others weight 1. Checkpoints at 11 steps from 0 to 3000.

**Measurements** (`goalgeo/prominence.py`, on 2000 held-out sequences):

- P_metric = mean distance between hidden states after p and after q, divided by
  the same for u versus v. Also at the pre-relevant position t*−1, over the
  median pairwise distance, and as variance along the A−B direction over the
  variance along random directions.
- Decodability: ridge-CV R² of the belief from the hidden state, and linear
  branch accuracy at t*−1, where the last token is x or y for both branches.
- RSA against belief, 1-step, 2-step and future-marginal geometries.
- Gradient of the relevant loss ℓ_{t*} with respect to the cue state h_t,
  computed by re-injecting h_t as a leaf and re-running the GRU over the k
  following tokens: norm G_t and projection G_∥ on the A−B direction, at init,
  at the end, and averaged over checkpoints ("integrated").
- KL to the exact targets, overall and at t*.
- ΔL_forget: KL between the true next-token predictive and that of a filter
  whose belief is symmetrised across the two branches after every update; per
  relevant event it equals r·(ln 2 − H(0.5+δ)) exactly.

## Results

### Baseline reproduces round 4

Base condition (r=1, δ=0.4, k=1, λ=1), mean of 3 seeds, init → final:

| quantity | init | final |
|---|---|---|
| delayed-pair distance (raw) | 1.51 | 4.02 |
| control-pair distance (raw) | 1.67 | 4.24 |
| P_metric | 0.94 | 0.95 |
| belief R² | 0.76 | 0.90 |
| branch accuracy at t*−1 | 0.99 | 0.99 |
| RSA with 1-step predictive | 0.13 | 0.61 |
| KL to target | 1.11 | 0.000 |

Both pairs are amplified about 2.5× by training and the delayed pair ends at
0.95 of the control. The belief is decodable throughout.

### Experiment 1, relevance frequency: a step, not a slope

| r | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 1 |
|---|---|---|---|---|---|---|---|
| P_metric | 0.33 | 0.91 | 0.91 | 0.86 | 0.82 | 0.89 | 0.95 |
| P_metric at t*−1 | 0.21 | 0.95 | 0.96 | 0.96 | 0.94 | 1.01 | 0.99 |
| branch accuracy at t*−1 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| belief R² | 0.98 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 | 0.90 |

At r=0 the cue never matters (Control 2): the raw delayed-pair distance stays at
its initial value (1.51 → 1.48) while the control pair grows to 4.56, so the
pair falls to 0.33 of the control and 0.14 of the median distance. Yet the
branch is still 99 % decodable one step later and the belief decodes at R² 0.98.
As soon as r > 0, even at r = 0.05 where the cue matters once in twenty cycles,
the pair is amplified to about 0.9 of the control, and frequency makes no
further difference (Spearman 0.43 across the sweep, driven entirely by r=0).
Decodability does not saturate "before" prominence along r; it is saturated
everywhere, including where prominence is absent.

### Experiment 2, relevance strength: graded

| δ | 0 | 0.02 | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|---|---|
| required gap (norm of the log-probability difference) | 0 | 0.11 | 0.28 | 0.57 | 1.20 | 3.11 |
| ΔL_forget per event (nats) | 0 | 0.0008 | 0.005 | 0.020 | 0.082 | 0.368 |
| P_metric | 0.25 | 0.30 | 0.43 | 0.50 | 0.64 | 0.95 |
| P_metric at t*−1 | 0.17 | 0.19 | 0.24 | 0.33 | 0.52 | 0.99 |
| branch accuracy at t*−1 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| belief R² | 0.84 | 0.87 | 0.88 | 0.89 | 0.91 | 0.90 |

Prominence rises monotonically with δ (Spearman 1.0) while decodability is flat
at 0.99. This is the cleanest dissociation in the study: at δ=0.05 the branch is
as decodable as at δ=0.4 and less than half as prominent.

### Experiment 3, loss weight: only zero matters

| λ | 0 | 0.1 | 0.25 | 0.5 | 1 | 2 | 5 |
|---|---|---|---|---|---|---|---|
| P_metric | 0.36 | 0.90 | 0.89 | 0.89 | 0.95 | 0.92 | 0.99 |
| branch accuracy at t*−1 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 |
| KL at t* | 7.0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Same data, same HMM, same model. With λ=0 the relevant step is never trained,
the pair is not amplified (raw 1.51 → 1.76) and ends at 0.36 of the control
while still 99 % decodable. Any λ ≥ 0.1 gives full amplification; a 50-fold
range of λ moves P_metric from 0.90 to 0.99, within seed noise (std 0.08–0.16).
The gradient measured at the cue is huge at λ=0 (∫G_t = 2.0, the relevant loss
never falls) but it is never applied; λ·∫G_∥ is what enters training.

### Experiment 5, frequency × strength grid: prominence depends on δ, not on r

P_metric, rows r, columns δ (3 seeds each):

| r \ δ | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|
| 0.10 | 0.36 | 0.47 | 0.62 | 0.91 |
| 0.25 | 0.38 | 0.47 | 0.63 | 0.86 |
| 0.50 | 0.37 | 0.47 | 0.62 | 0.82 |
| 1.00 | 0.43 | 0.50 | 0.64 | 0.95 |

Branch accuracy at t*−1 is 0.99 in all 16 cells. The rows are
indistinguishable and the columns track δ. Expected forgetting cost
r·ΔL_forget(δ) does *not* collapse the grid: r=0.1, δ=0.4 (cost 0.0072 nats per
position) gives P_metric 0.91 while r=1, δ=0.1 (cost 0.0039) gives 0.50 and
r=1, δ=0.2 (cost 0.016) gives 0.64. A rare but strong consequence produces far
more prominence than a frequent but weak one of equal or larger expected cost
(Figure 2).

### Experiment 4, delay: P_metric falls, but mostly because the control grows

| k | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| P_metric, λ=1 | 0.95 | 0.87 | 0.66 | 0.57 | 0.62 |
| P_metric, λ matched | – | 0.88 | 0.68 | 0.56 | 0.70 |
| delayed-pair distance / median | 0.65 | 0.66 | 0.59 | 0.60 | 0.85 |
| raw delayed-pair distance | 4.02 | 3.87 | 3.73 | 3.74 | 4.41 |
| raw control-pair distance | 4.24 | 4.47 | 5.76 | 7.18 | 7.79 |
| branch accuracy at t*−1 | 0.99 | 0.98 | 0.97 | 0.92 | 0.84 |
| belief R² | 0.90 | 0.96 | 0.99 | 0.76 | 0.41 |
| ∫G_∥ | 0.004 | 0.012 | 0.009 | 0.013 | 0.029 |
| KL at t* | 0.000 | 0.000 | 0.000 | 0.000 | 0.004 |

Relative to the control pair, prominence falls with delay (Spearman −0.90), and
matching the relevant loss share per sequence (λ_k from 1.15 to 4.0) changes
nothing, so the drop is not a loss-share effect. But the raw delayed-pair
distance does not shrink with k; the control pair grows, because long filler
runs push the network to a larger hidden-state scale. Normalised by the median
distance the delayed pair is flat up to k=8 and larger at k=16. Decodability
does fall with delay (branch accuracy 0.99 → 0.84, belief R² 0.90 → 0.41 with
41 states), and the integrated gradient rises, because the relevant loss stays
high for longer before credit reaches the cue. The one run that failed Control 5
(KL 0.18) is a k=16 matched-λ seed; excluding it changes no conclusion.

### Main analysis: what predicts prominence?

Regression of P_metric on the four TASK4 predictors, all 105 runs, standardised:

| predictor | β | Spearman | R² alone |
|---|---|---|---|
| belief R² | +0.08 | +0.19 | 0.005 |
| expected cost λ·ΔL_forget | +0.44 | +0.70 | 0.20 |
| ∫G_t | −0.03 | +0.33 | 0.03 |
| ∫G_∥ | −0.10 | +0.45 | 0.03 |

Full R² 0.22; leave-one-sweep-out R² is negative for every sweep. Decodability
explains nothing, expected cost explains a fifth, gradient pressure adds 0.02,
and no fit transfers across manipulations.

The quantity the sweeps single out is the **required output separation**: the
norm of the log-probability difference between the two branches' next-token
distributions at t* (0 when λ=0, otherwise √2·ln((0.5+δ)/(0.5−δ)), independent
of r and k; the x-versus-y logit gap is 2·ln((0.5+δ)/(0.5−δ))). Alone it
gives R² 0.61 over all runs and 0.85 over the frequency, strength, weight and
grid sweeps (81 runs), with leave-one-sweep-out R² of 0.64 (frequency), 0.74
(strength), 0.65 (grid) and 0.38 (weight). It fails only on the delay sweep,
where the control-pair growth described above dominates (Figure 6). With this
predictor added, the six-predictor regression reaches R² 0.71 and the target
gap carries the largest weight (β +0.81).

### Training dynamics (dense checkpoints, 3 seeds, Figure 4)

Base condition: KL at t* falls from 1.6 to below 0.05 between steps 30 and 50;
the gradient projection on the A−B direction peaks at step 20–30; the raw
delayed-pair distance makes 90 % of its growth by step 30–60; branch accuracy
at t*−1 is 0.99 from step 0 (with k=1 the cue token is one step back). So:
gradient pulse → loss drop and metric growth together, with decodability
already present.

k=8: decodability at t*−1 starts at 0.87, dips to 0.6–0.8 around step 50 while
the control geometry reorganises, and recovers to 0.92 by step 100–150; the
gradient projection peaks at step 200; the delayed distance grows and the KL at
t* falls at steps 150–200. Decodability is restored some 50–100 steps before
the distance is amplified, and amplification coincides with the gradient pulse
and the loss drop.

r=0 and δ=0.05: the raw delayed distance never grows (r=0) or grows a little
(δ=0.05) while the control triples within 50 steps; P_metric falls in the first
30–60 steps and then drifts slowly down. Decodability stays at 0.99 throughout.

### Controls

1. Immediate-relevance pair: the u/v control, log-odds gap 3.1 in every
   condition; grows 2.5–5× with training in every condition.
2. Permanently irrelevant cue (r=0 or δ=0): not amplified, still decodable
   (branch accuracy 0.99, belief R² 0.84–0.98).
3. Random direction: variance along the A−B direction over variance along
   random directions is 9 at init (the direction is the last-token direction),
   3.6–4.0 after training when the cue matters (δ=0.4), 1.3–1.6 when it does
   not (r=0, λ=0). The direction is amplified relative to random directions only
   when it is used.
4. Initialisation: every table has an init row; P_metric at init is 0.94 in
   all k=1 conditions.
5. Matched performance: KL to the exact targets is below 0.005 nats for every
   run with λ > 0 except one k=16 seed at 0.18; λ=0 runs by design do not learn
   t*.

## Reading against the success criteria

1. Decodable before prominent: partly. Decodability is present at init and in
   every condition; where the distance is amplified (k=8) decodability recovers
   50–100 steps earlier. Prominence is the later and the more selective property.
2. Prominence increases with frequency: **no**. It is a step at r > 0.
3. Prominence increases with strength: **yes**, monotone and graded.
4. Loss weight alone changes prominence: only between λ=0 and λ>0; over a
   50-fold range of positive λ it does not.
5. Expected forgetting cost predicts across manipulations: **no** (R² 0.20,
   negative out-of-sample). Rare-and-strong beats frequent-and-weak at equal cost.
6. Measured gradient relevance predicts prominence: **no** (R² 0.03; in the
   delay sweep it moves the opposite way).
7. Delay reduces prominence without destroying decodability: relative to the
   control yes, relative to the median no, and decodability falls too.
8. The immediate control is more prominent than equally decodable delayed
   information: only when the delayed consequence is weaker (δ < 0.4). At equal
   output gap the delayed pair is as prominent as the control (0.95).

## Conclusion

The dissociation is real and general in this HMM: in 27 of 35 conditions the
delayed branch is 99 % linearly decodable one step before it is needed, while
its geometric prominence ranges from 0.25 to 1.0 of the control. But the
variable that sets prominence is not optimisation pressure in the sense of
frequency, expected loss or gradient magnitude. It is the **size of the output
difference the distinction has to produce when it is used**: the log-odds gap
at t*. That gap is the same whether the gate opens once in ten cycles or every
cycle, and whether the loss at t* is weighted 0.1 or 5, and the geometry is the
same too. When the gap is zero (δ=0, r=0, λ=0) the distinction is retained at
its initial scale and merely not amplified.

This is the reading closest to Outcome B: the generative process, through the
magnitude of the consequence, determines the representational scale, and the
optimiser's job is to reach the readout that produces that consequence. A
natural mechanism is that the readout weights along the A−B direction grow to
a bounded size, so the hidden-state separation must supply the required logit
gap. Two things this round does not settle: why the delay sweep departs from the
gap account (the control pair, not the delayed pair, changes scale with k), and
whether the gap account holds when the readout is deeper than one linear layer.

## Figures

- `fig1_info_vs_prominence.png`: decodability and P_metric against r.
- `fig2_prominence_vs_cost.png`: P_metric against expected forgetting cost, all
  sweeps; the curves do not collapse.
- `fig3_gradient_vs_prominence.png`: P_metric against integrated gradient
  projection; no relationship.
- `fig4_dynamics.png`: dense-checkpoint dynamics for the base, δ=0.05, r=0 and
  k=8 conditions.
- `fig5_delay.png`: the delay sweep with both normalisations.
- `fig6_prominence_vs_target_gap.png`: P_metric against the required log-odds
  gap; the frequency, strength, weight and grid sweeps fall on one curve.
