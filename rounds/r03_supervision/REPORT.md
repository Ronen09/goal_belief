# The supervision bottleneck (TASK3)

Run date: 2026-09-17. Numbers are seed means from `rounds/r03_supervision/tables.md` (5 seeds
for the temperature sweep and the ring, 3 seeds elsewhere). Design:
`rounds/r03_supervision/DESIGN.md`. Theory:
`rounds/r03_supervision/THEORY.md`. Reproduce with `rounds/r03_supervision/run.py` (18 min on CPU).

## Summary

**Hidden geometry preserves the distinctions the training target contains, and
loses them gradually as the target softens.** Six of the eight success criteria
are met in full, one in part, one fails in a specific and informative way.

- **Temperature sweep (Task 1).** With one-hot inputs, RSA of the last layer with
  the policy table falls monotonically from 0.78 (hard) through 0.56, 0.52, 0.46,
  0.41 (τ = 0.02 to 0.2) to 0.05 (τ = 5), while occupancy and advantage RSA first
  rise (0.35 → 0.59 and 0.32 → 0.63 at τ = 0.02) and then decay as the targets
  approach uniform. The participation ratio of the last layer rises from 3.1 to
  16. The transition is gradual, not binary.
- **Target geometry (Task 2).** At the row level the target RDM is the best single
  predictor of hidden geometry at every temperature where the target is
  informative (RSA 0.57 hard, 0.71–0.75 for τ = 0.02 to 0.1, versus 0.06–0.20 for
  the Q rows). At the state level the target adds nothing beyond the environment
  quantities it is computed from, because the concatenated soft-target table is
  nearly a linear image of occupancy there.
- **Fixed-argmax counterfactual (Task 3).** In a synthetic task with no
  environment, probes sharing the reference's argmax stay at 0.35–0.49 of the
  median distance under hard targets regardless of their margin, and under
  τ = 0.05 move from 0.35 (identical margin) to 0.85 (margin 0.005) in a graded
  way. In the base grid the same-argmax row distance grows from 0.46 (hard) to
  0.61 (τ = 0.05) to 0.87 (τ = 0.5), and within such pairs distance tracks target
  difference (Spearman 0.51 at τ = 0.05).
- **Environment held fixed (Task 4).** With identical inputs, changing only T(Q)
  moves the geometry: hard versus τ = 0.05 correlate at 0.56 (seed reliability
  0.85 and 0.66), hard versus τ = 0.5 at 0.33, and soft BC resembles the occupancy
  regression control (0.54) more than hard BC does (0.35). Advantage and Q
  regression with MSE produce weakly structured, unreliable geometries (seed
  reliability 0.11 and 0.27); the 60-dimensional occupancy control is reliable
  (0.88) and reproduces occupancy geometry (RSA 0.93).
- **Bottleneck of the target map (Task 5).** The hard map keeps 15 classes out of
  347 distinct Q rows and lets advantage be recovered at R² 0.57 but not Q (0.06).
  Softmax at any τ keeps all rows distinct and advantage recoverable (0.60–0.76)
  but discards the per-row constant, so Q stays unrecoverable (0.08). Hidden
  dimensionality follows the informativeness of the target rather than its rank:
  all Boltzmann targets have rank 3, yet last-layer PR runs from 3.5 (τ = 0.02) to
  28 (τ = 2).
- **Compression during training (Tasks 6, 7).** The within/between hard-class
  ratio falls from 0.98 to 0.46 under hard targets, to 0.58 under τ = 0.05, and
  only to 0.85 under τ = 0.5. Under hard targets the between-class distance grows
  28× and the within-class distance 13×. Row pairs with identical targets but
  far-apart occupancy sit at 0.49 of the median distance under hard targets;
  pairs with different targets but near-identical occupancy at 1.22.
- **Generalisation (Task 8).** Held-out (state, goal) rows are classified at 92–94%
  accuracy under hard targets and land in their target class by nearest centroid
  50–62% of the time (chance 16–25%), with a within/between ratio of 0.59–0.61
  versus 0.43–0.45 on training rows. With coordinate inputs, unseen corridor cells
  are placed with their own decision run 100% of the time (chance 41%).
- **Architectures (Task 9).** The hard-target quotient replicates exactly across
  the 3-layer MLP, a 6-layer MLP, a residual MLP and a 2-layer transformer (policy
  RSA 0.71–0.78). The soft-target geometry replicates in the MLPs (target RSA
  0.51–0.59 at τ = 0.1) but not in the two normalised architectures, whose
  last layers become unstructured at soft targets (RSA ≤ 0.17 with every model).
  The "softer targets preserve more" half of the claim is therefore
  architecture-dependent.
- **Additive decomposition (Task 10).** The ANOVA decomposition already satisfies
  the centring constraints. The part of the interaction that is orthogonal in
  feature space to the additive effects is 43%, 17% and 2% of the interaction at
  h1, h2, h3 (0.2% of total variance at h3). Removing it changes nothing
  (accuracy 1.000); removing the whole interaction costs 1.5–2.6 points. The
  round-2 subspace ablation was confounded and the interaction is causally inert.
- **Steering (Task 11).** The network's own linearised boundary predicts the flip
  threshold at Spearman 0.98–1.00 at every temperature. As targets soften, the
  learned logit margin's correlation rises (0.81 → 0.90–0.93) and the g1 Q margin's
  correlation rises (0.46 → 0.57–0.66), while the target-probability margin at the
  g2 row stays weak (≤ 0.27). Steering success at α = 1 drops from 0.85 (hard) to
  0.66 for every soft condition.
- **Loss versus target (Task 13).** Cross-entropy and KL give identical networks
  (same-seed RSA 1.0). MSE on probabilities and MSE on logits give geometries as
  similar to the cross-entropy networks as those are to each other across seeds
  (0.73–0.74 versus 0.66). At τ = 0.05 the loss does not matter beyond the target.
- **Ring (Task 14).** Pair A (same hard target, far occupancy): 0.53 hard, 0.96
  soft. Pair B (near occupancy, different hard target): 1.12 hard, 0.87 soft.
  Pair C (same hard target, disproportionately different soft targets): 0.52 hard,
  0.65 soft. Random init: 1.0 for all. The merge/separate pattern matches the
  prediction table except that C separates less than A under soft targets, in
  proportion to its smaller absolute target difference.

Scoped claim: in these small goal-conditioned networks, learned representations
preserve the distinctions demanded by supervision rather than the geometry of the
underlying predictive state. Hard behavioural cloning produces a policy quotient
because the argmax target has already discarded within-action distinctions;
Boltzmann targets restore those distinctions with a strength set by τ, until
the targets become too uniform to carry them. The chain
world → value/occupancy → **target** → representation holds, with the caveat
that the last arrow depends on the architecture once targets are graded.

## Task 1: temperature sweep

Base grid, one-hot inputs, 3×128 MLP, 5 seeds. Accuracy is 100% at every τ
(the argmax of a Boltzmann target is the optimal action). State-level RSA at h3:

| condition | target | policy | occupancy | advantage | SR | spatial | PR h3 |
|---|---|---|---|---|---|---|---|
| hard | 0.78 | 0.78 | 0.35 | 0.32 | 0.49 | 0.47 | 3.1 |
| τ = 0.02 | 0.59 | 0.56 | 0.59 | 0.63 | 0.55 | 0.70 | 4.1 |
| τ = 0.05 | 0.57 | 0.52 | 0.57 | 0.59 | 0.53 | 0.67 | 5.1 |
| τ = 0.1 | 0.51 | 0.46 | 0.52 | 0.55 | 0.48 | 0.61 | 7.9 |
| τ = 0.2 | 0.24 | 0.41 | 0.36 | 0.42 | 0.37 | 0.45 | 11.9 |
| τ = 0.5 | 0.06 | 0.34 | 0.26 | 0.33 | 0.31 | 0.32 | 15.2 |
| τ = 1 | 0.06 | 0.21 | 0.16 | 0.20 | 0.17 | 0.19 | 16.1 |
| τ = 5 | 0.01 | 0.05 | 0.04 | 0.04 | 0.03 | 0.04 | 13.9 |

Two regimes are visible. Up to τ ≈ 0.1 the targets are informative and the
geometry moves from the policy quotient toward a smooth occupancy/advantage
geometry. From τ ≈ 0.2 the targets are nearly uniform (maximum probability 0.33
at τ = 0.2, 0.28 at τ = 0.5), the gradient carries little structure, and the
geometry approaches a random network's (PR 13–16 versus 13–20 at initialisation).
Figure: `task1_temperature_sweep.png`.

## Task 2: target geometry

Row-level RSA at h3: target 0.57 (hard), 0.71, 0.74, 0.75, 0.73, 0.62 (τ = 0.02 to
0.5); Q rows 0.06 (hard) and 0.19–0.21 (soft); policy rows 0.57 (hard) and
0.69–0.43 (soft). Target geometry predicts hidden geometry better than the
environment quantities it was built from, at every informative temperature.
At the state level (concatenated over goals) the picture is different: the
partial correlation of the target given occupancy, SR, spatial, policy and
advantage is 0.25 for hard targets (where target = policy table) and 0.04–0.15
for soft ones, because the soft-target table is almost a linear image of the
occupancy table at that granularity. Figure: `task2_target_geometry.png`.

## Task 3: fixed-argmax counterfactual

Synthetic task (200 background rows with random Q, a reference with margin 0.5,
probes with the same argmax and margins from 0.005 to 0.5; distances divided by
the median background distance, h3):

| condition | m = 0.005 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| hard | 0.47 | 0.42 | 0.47 | 0.49 | 0.36 | 0.35 | 0.43 | 0.41 |
| τ = 0.02 | 0.71 | 0.70 | 0.68 | 0.66 | 0.55 | 0.44 | 0.27 | 0.33 |
| τ = 0.05 | 0.85 | 0.85 | 0.83 | 0.83 | 0.75 | 0.67 | 0.49 | 0.35 |
| τ = 0.1 | 0.69 | 0.70 | 0.69 | 0.71 | 0.63 | 0.53 | 0.41 | 0.35 |
| τ = 0.5 | 0.78 | 0.74 | 0.76 | 0.79 | 0.74 | 0.65 | 0.55 | 0.55 |

Hard targets: flat in the margin. Soft targets at informative τ: graded in the
margin difference, saturating where the softmax saturates. At τ ≥ 0.5 all probes
are far from the reference because the geometry is unstructured, not because it
is graded. The natural version in the base grid agrees (`task3_counterfactual.png`).

## Task 4: same inputs, different T(Q)

RSA between the h3 geometries of conditions (diagonal = between seeds):

| | hard | τ = 0.05 | τ = 0.5 | advantage MSE | Q MSE | occupancy MSE |
|---|---|---|---|---|---|---|
| hard | 0.85 | 0.56 | 0.33 | 0.15 | 0.17 | 0.35 |
| τ = 0.05 | 0.56 | 0.66 | 0.39 | 0.23 | 0.30 | 0.54 |
| τ = 0.5 | 0.33 | 0.39 | 0.21 | 0.26 | 0.21 | 0.24 |
| advantage MSE | 0.15 | 0.23 | 0.26 | 0.11 | 0.20 | 0.19 |
| Q MSE | 0.17 | 0.30 | 0.21 | 0.20 | 0.27 | 0.41 |
| occupancy MSE | 0.35 | 0.54 | 0.24 | 0.19 | 0.41 | 0.88 |

Changing only the target map changes the geometry by as much as changing the
seed or more. Regression heads on the 4-dimensional advantage or Q vectors
(values in [−0.2, 1]) barely shape the representation, a limitation of MSE on
small-magnitude targets rather than evidence about value geometry; the
60-dimensional occupancy control shows that a regression head *can* impose its
geometry when the signal is strong. Figure: `task4_fixed_env.png`.

## Task 5: what each target map destroys

| target | classes (of 840 rows) | rank | RSA with Q | recover Q R² | recover advantage R² | recover Z_sa R² | hidden PR h3 |
|---|---|---|---|---|---|---|---|
| hard | 15 | 3 | 0.10 | 0.06 | 0.57 | 0.11 | 2.8 |
| τ = 0.02 | 333 | 3 | 0.17 | 0.06 | 0.60 | 0.10 | 3.5 |
| τ = 0.05 | 347 | 3 | 0.26 | 0.07 | 0.69 | 0.07 | 4.3 |
| τ = 0.1 | 346 | 3 | 0.33 | 0.08 | 0.73 | 0.06 | 6.4 |
| τ = 0.5 | 344 | 3 | 0.35 | 0.08 | 0.76 | 0.05 | 19.0 |
| τ = 2 | 310 | 3 | 0.34 | 0.08 | 0.76 | 0.05 | 28.0 |
| advantage | 347 | 4 | 0.50 | 0.65 | 1.00 | 0.04 | 21.4 |
| Q | 347 | 4 | 1.00 | 1.00 | 0.91 | 0.05 | 5.3 |
| occupancy | 57 | 56 | 0.02 | −0.12 | −0.02 | 1.00 | 4.6 |

There are only 347 distinct Q rows, so every softmax keeps every distinction that
exists; what changes with τ is how much of the gradient those distinctions
occupy. The argmax map collapses to 15 classes. Hidden dimensionality is not
predicted by the rank or class count of the target but by its signal-to-uniform
ratio. Figure: `task5_bottleneck.png`.

## Tasks 6 and 7: sufficient statistic and training dynamics

Row-level distances / median (h3): identical target, far occupancy 0.49 (hard),
0.51 (τ = 0.05), 0.86 (τ = 0.5); different target, near occupancy 1.22, 1.17,
1.08. Within/between hard-class ratio at h3 over steps 0, 10, 30, 100, 300, 1000,
3000:

| condition | ratio | between (× init) | within (× init) |
|---|---|---|---|
| hard | 0.98 → 0.85 → 0.65 → 0.54 → 0.51 → 0.49 → 0.46 | 28× | 13× |
| τ = 0.05 | 0.98 → 0.86 → 0.67 → 0.57 → 0.59 → 0.60 → 0.58 | 6× | 4× |
| τ = 0.5 | 0.98 → 0.92 → 0.87 → 0.88 → 0.88 → 0.87 → 0.85 | 1.4× | 1.2× |

Geometry does not first reflect the input (one-hot inputs carry none) and then
reorganise; it organises around the target from the first steps, with the
target RSA rising in step with accuracy and continuing to rise after accuracy
saturates for hard targets (0.73 at step 100 to 0.78 at 3000). The compression is
driven by between-class growth, as the theory note predicts, but within-class
distances grow too. Figure: `task7_training.png`.

## Task 8: generalisation

| setting | accuracy train / held-out | ratio train / held-out | nearest-centroid accuracy (chance) |
|---|---|---|---|
| base, hard, 30% rows held out | 1.00 / 0.92 | 0.45 / 0.61 | 0.50 (0.16) |
| base, τ = 0.05 | 1.00 / 0.79 | 0.61 / 0.78 | 0.40 (0.16) |
| twins, hard | 1.00 / 0.94 | 0.43 / 0.59 | 0.62 (0.25) |
| twins, τ = 0.05 | 1.00 / 0.86 | 0.68 / 0.80 | 0.55 (0.25) |
| corridor, coordinates, 4 states unseen, hard | 1.00 / 1.00 | — | 1.00 same-run (0.41) |

Held-out rows are placed in their target class well above chance but are less
compressed than training rows, so the geometry is partly structural and partly
memorised. With coordinate inputs, unseen cells generalise perfectly to their
decision run. Figure: `task8_generalisation.png`.

## Task 9: architectures

Last hidden layer, RSA with target / policy / occupancy:

| architecture | hard | τ = 0.1 | τ = 0.5 |
|---|---|---|---|
| mlp3 | 0.78 / 0.78 / 0.35 | 0.51 / 0.44 / 0.51 | 0.03 / 0.35 / 0.25 |
| mlp6 | 0.73 / 0.73 / 0.37 | 0.59 / 0.49 / 0.62 | 0.29 / 0.49 / 0.60 |
| resmlp | 0.78 / 0.77 / 0.41 | 0.10 / 0.19 / 0.14 | −0.05 / 0.07 / 0.01 |
| transformer | 0.78 / 0.77 / 0.46 | 0.07 / 0.17 / 0.16 | −0.01 / 0.01 / 0.01 |

The hard-target quotient is architecture-independent. The soft-target geometry is
not: the deeper MLP preserves even more occupancy structure than the 3-layer one,
while the residual MLP and the transformer, both with LayerNorm in the residual
stream, show no candidate geometry at the layer probed once targets are graded.
Their synthetic counterfactual curves are flat for both target kinds. Figure:
`task9_architectures.png`.

## Task 10: clean additive decomposition

| layer | interaction fraction | unique to complement of additive span | accuracy: baseline / − unique / − random in complement / − all interaction |
|---|---|---|---|
| h1 | 0.124 | 0.43 (5.3% of total) | 1.000 / 1.000 / 1.000 / 0.985 |
| h2 | 0.106 | 0.17 (1.8%) | 1.000 / 1.000 / 1.000 / 0.975 |
| h3 | 0.115 | 0.02 (0.2%) | 1.000 / 1.000 / 1.000 / 0.974 |

Almost all interaction variance lives inside the feature directions used by the
additive state and goal effects; the genuinely unique part is negligible and its
removal has no behavioural effect. Figure: `task10_unique_interaction.png`.

## Task 11: steering across temperature

Steering success at α = 1 (h3): 0.85 hard, 0.66 at every τ. Spearman of the flip
threshold with predictors, h3: linearised boundary 0.98–1.00 throughout; learned
logit margin 0.81 (hard) → 0.90–0.93 (soft); g1 Q margin 0.46 → 0.57–0.66; g2 Q
margin 0.19 → ≈ 0; target-probability margin 0.27 → 0.05–0.19; ‖Δρ‖ ≤ 0.15.
Steering acts through the network's own logit geometry at all temperatures. As
targets soften the logit margin becomes a faithful copy of the g1 Q margin (the
target's logits are Q/τ), so the value quantity that survives *through the
target* gains predictive power, exactly as the task anticipated; occupancy
contributes nothing independently. Figure: `task11_steering_tau.png`.

## Task 12: theory

See `rounds/r03_supervision/THEORY.md`. In brief: the loss depends on the input only through
the target, so the quotient is always a minimiser; with hard targets, logit growth
on separable data inflates between-class distances while confident points stop
receiving gradient, giving a falling within/between ratio (observed: 0.98 → 0.46
with between 28× and within 13×); with soft targets the linear readout must
reproduce the centred Q/τ, so within-argmax distinctions must survive with
strength set by τ (observed: Tasks 1, 3, 7). Conditions: supervised loss, no
input geometry to destroy, training into the vanishing-gradient regime, and no
normalisation that fixes the scale. The last condition is what Task 9 exposes.

## Task 13: target versus loss

CE and KL are the same objective up to a constant and give identical networks.
At τ = 0.05, MSE on probabilities and MSE on centred logits give hidden
geometries as similar to the CE networks (RSA 0.73–0.74) as CE networks are to
each other across seeds (0.66), with the same RSA to target (0.50–0.52), policy
(0.37–0.45) and occupancy (0.52–0.56). At τ = 0.5 all losses give unstructured
geometry (seed reliability 0.09–0.29). At informative temperatures the geometry
depends on what the target contains, not on how the loss weights it.
Figure: `task13_losses.png`.

## Task 14: one environment, three pairs

Ring, row-level pairs, hidden distance / median at h3:

| pair | prediction hard / soft | hard BC | soft BC (τ = 0.02) | random init |
|---|---|---|---|---|
| A: same hard target, far occupancy (‖ΔQ‖ 0.90) | merge / may separate | 0.53 | 0.96 | 1.00 |
| B: near occupancy (‖ΔQ‖ 0.12), different hard target | separate / separate | 1.12 | 0.87 | 1.01 |
| C: same hard target, soft targets differ beyond ‖ΔQ‖ (‖ΔQ‖ 0.41) | merge / strongly separate | 0.52 | 0.65 | 0.98 |

A and C merge under hard supervision and separate under soft; B stays apart
under both. C separates less than A under soft supervision because its
absolute target difference is smaller (0.32 versus 0.39 in probability units);
the "strongly" in the prediction is not borne out. Figure: `task14_abc.png`.

## Success criteria

| criterion | outcome |
|---|---|
| 1. Hidden geometry changes systematically with target temperature | Met; gradual, two regimes (informative τ ≤ 0.1, near-uniform τ ≥ 0.2). |
| 2. Target geometry explains hidden geometry better than raw occupancy | Met at the row level (0.71–0.75 vs 0.06–0.21); at the state level the target adds nothing beyond occupancy for soft τ. |
| 3. Same-argmax, different-confidence states merge under hard and separate under soft | Met (synthetic, base grid, ring). |
| 4. Effect appears with the environment held fixed | Met (Task 4), with the caveat that MSE regression heads on small targets are weak controls. |
| 5. Within-target-class compression develops during training | Met; driven by between-class growth (28×) outpacing within-class growth (13×). |
| 6. Replicates across architectures | Hard quotient: yes (4 of 4). Soft-target geometry: MLPs only; normalised architectures lose it. |
| 7. Target/loss structure predicts intervention geometry better than occupancy | Met; the linearised boundary and logit margin dominate, occupancy ≤ 0.15. |
| 8. Theoretical equivalence-class account explains the effects | Met qualitatively (falling ratio, τ-graded survival, advantage not Q); refined to a relative collapse. |

## Caveats and next steps

- The τ grid is set by Q ∈ [0, 1]; τ ≥ 0.2 is already near-uniform here, so
  "soft" is τ ≈ 0.02–0.1 in this setting.
- Regression heads with MSE on 4-dimensional targets are poor probes of value
  geometry; a scaled or standardised target, or a larger head, would be needed to
  compare "predict A" fairly with "imitate softmax(A/τ)".
- The normalised architectures should be probed at the pre-LayerNorm residual
  stream and with a wider τ range before concluding that they discard graded
  target structure.
- A weight-decay sweep would test the theory's prediction that regularisation
  turns relative collapse into absolute collapse.
