# Policy quotient versus occupancy geometry (TASK2)

Run date: 2026-09-17. Numbers are means over seeds from `rounds/r02_policy_quotient/tables.md`
(5 seeds per λ in the sweep, 3 warm-start chains, 3 seeds per extra environment)
and `rounds/r02_policy_quotient/boltzmann/tables.md`. Design: `rounds/r02_policy_quotient/DESIGN.md`.
Reproduce with `rounds/r02_policy_quotient/run.py` (8 min) and
`rounds/r02_policy_quotient/run.py --targets boltzmann --sweep-only --out rounds/r02_policy_quotient/boltzmann` (7 min).

## Summary

**The learned geometry mirrors the geometry of the imitation target, and under
argmax targets that target is the policy quotient.**

- With standard behavioural cloning (uniform over optimal actions), the network's
  hidden geometry is organised by decision equivalence classes (H2). Policy-table
  geometry dominates a six-way regression against occupancy, successor
  representation, advantage, action margin and spatial geometry in all nine
  environments (β 0.49–0.86 at h2; partial correlation 0.44–0.79 after controlling
  for the other five). States with far-apart occupancy but identical action tables
  merge (0.27–0.45 of the median distance at h3, versus 1.0 at initialisation), and
  states with near-identical occupancy but different actions stay apart (0.96–1.21).
- Across the policy-switch sweep, representational change with argmax targets is
  concentrated entirely at action flips. This is structural, not just empirical:
  when no optimal action changes between λ and λ+Δ, the training targets are
  identical and the independently trained networks are *bit-for-bit identical*
  (Procrustes distance 0). Occupancy change cannot reach an argmax-cloned network
  except through the policy.
- Replacing the targets with a Boltzmann policy softmax(Q/0.02), still pure action
  prediction, reverses the picture. Hidden change then tracks occupancy change
  smoothly (Spearman 0.27–0.37 among pairs whose action did not flip, versus
  −0.03 to −0.06 with argmax targets), concentrates only weakly at flips
  (1.4× versus 3.5× the floor), and the six-way regression is led by the advantage
  geometry (β 0.51–0.61), which is what softmax(Q/τ) encodes.
- The goal-conditioned interaction component `h_sg` (about 11% of variance) is
  not causally necessary: removing it exactly costs at most 7 percentage points of
  accuracy and *improves* steering success (0.86 → 0.96). Projecting out its
  top-k subspace destroys behaviour, but that subspace also carries 90–97% of the
  additive variance, so the subspace result is confounded and should not be read
  as evidence for the interaction term.
- Steering acts by moving activations across the network's own action boundary.
  The linearised boundary crossing predicts the per-state flip threshold with
  Spearman 0.92–1.00 and absorbs all the regression weight; occupancy margins,
  the occupancy-vector difference and the occupancy interpolation model add
  nothing (β ≈ 0) and correlate at most 0.45 on their own.
- Dimensionality partly follows occupancy: the trained last layer has a
  participation ratio of 1.2–3.1, close to `Z_sa` (1.7–3.0) in absolute terms and
  far below random networks (10–15), but the *ordering* across environments
  follows the policy table (Spearman 0.83) more than occupancy (0.47).

Scoped claim: in these small goal-conditioned MLP policies trained by
behavioural cloning, learned representations compress predictive future
structure into a geometry organised by whatever action-relevant structure the
training signal exposes: decision equivalence classes for argmax targets, graded
advantage for Boltzmann targets. Occupancy determines the available degrees of
freedom only loosely; the imitation target determines how they are organised.

## Task 1–3: policy-switch environment and training

A 5×13 ring with a short corridor through three hazard cells (each knocks the
agent back to the corridor entrance with probability λ) and a long safe corridor;
32 states, 10 goals, λ from 0 to 0.95 in steps of 0.05 (λ = 1 excluded because it
creates degenerate ties). The entrance-to-far-goal decision switches at λ ≈ 0.35;
56 of the 320 (s, g) pairs have at least one policy boundary in the sweep, with
flips spread over λ ∈ (0, 0.9]. Occupancy vectors change smoothly, most steeply at
small λ (‖ΔZ_sa‖ per step from 0.45 down to 0.08). Behavioural cloning reaches
100% accuracy at every λ. Figures: `task1_ground_truth.png`, `env_switch_lam*.png`.

## Task 4: representational change across the policy boundary

Three measures per layer, all reported relative to a reference: a warm-start chain
(same seed, trained on from λ−Δ; reference = continued training at the same λ),
Procrustes-aligned per-(s,g) distance between independently trained networks
(reference = between-seed distance at the same λ), and RDM change.

Mean per-(s,g) change divided by the reference, pooled over λ steps and seeds:

| targets | layer | flipped (s,g) | same state, other goal | all other | ρ(Δh, Δoccupancy) among unflipped |
|---|---|---|---|---|---|
| argmax | h2 | 2.68 | 1.01 | 0.91 | −0.06 |
| argmax | h3 | 3.50 | 1.10 | 0.97 | −0.04 |
| Boltzmann | h2 | 1.10 | 0.88 | 0.86 | +0.30 |
| Boltzmann | h3 | 1.42 | 0.91 | 0.89 | +0.37 |

With argmax targets, change lands on the flipped pair and nowhere else (the same
state's other goals move no more than the reference), and at λ steps with zero
flips the change is exactly the reference (chain) or exactly zero (Procrustes and
RDM), because the networks are identical. In the pooled regression of per-(s,g)
change on occupancy change, SR change, margin change and a flip indicator, only the
flip indicator and SR change have positive weight (β 0.11–0.17 and 0.05–0.15) and
occupancy change is slightly negative.

With Boltzmann targets the flip indicator's weight drops to 0.03–0.05, change
correlates with occupancy change among unflipped pairs (0.27–0.37), and margin
change (β 0.11–0.23) and SR change (0.08–0.33) take over. The aggregate change per
λ step follows the ‖ΔZ_sa‖ curve (`boltzmann/task4_change_h2.png`).
Figures: `task4_change_h{1,2,3}.png`, `task4_scatter_h2.png`.

## Task 5: quotient pairs

Case A: identical optimal-action tables, occupancy distance above the median of
the Case B distances and in the top quartile of same-table pairs. Case B: tables
differ, occupancy distance in the bottom quartile. Hidden distance divided by the
median pairwise distance, layer h3:

| environment | n_A / n_B | d_occ A / B | d_h A | d_h B | random init A / B | fraction of A pairs closer than B pairs |
|---|---|---|---|---|---|---|
| corridor | 18 / 21 | 1.01 / 0.67 | 0.27 | 1.21 | 1.03 / 0.96 | 1.00 |
| twins | 33 / 382 | 1.89 / 0.46 | 0.38 | 0.96 | 0.96 / 1.02 | 0.91 |
| switch λ=0 | 12 / 46 | 1.85 / 0.72 | 0.45 | 0.98 | 0.98 / 0.99 | 0.72 |
| switch λ=0.2 | 15 / 44 | 1.86 / 0.72 | 0.35 | 1.08 | 1.00 / 0.98 | 0.85 |
| switch λ=0.5 | 6 / 52 | 1.81 / 0.58 | 0.41 | 1.04 | 1.04 / 0.98 | 0.87 |
| switch λ=0.9 | 6 / 52 | 1.86 / 0.49 | 0.36 | 1.03 | 1.03 / 0.98 | 0.87 |

The quotient prediction `d_h(A) < d_h(B)` holds in every environment even though
occupancy predicts the reverse by a factor of 1.5 to 4. The merging develops with
depth (h1 ≈ 0.9, h2 ≈ 0.55, h3 ≈ 0.35) and is absent at initialisation. The base
grid has no Case A pairs (no two non-adjacent states share a full action table).
Figure: `task5_quotient_pairs.png`.

## Task 6: six-way regression

`Q_g(s,a) = ρ(g|s,a)/γ` exactly in this reward structure, so Q geometry is the
occupancy geometry; the regressors are occupancy, SR, policy table, advantage,
action margin and spatial. Standardised β at h2 with argmax targets:

| environment | occupancy | SR | policy | advantage | margin | spatial | R² | R² without policy | partial policy given rest |
|---|---|---|---|---|---|---|---|---|---|
| base | −0.24 | 0.07 | 0.69 | −0.22 | −0.03 | 0.42 | 0.59 | 0.28 | 0.65 |
| twins | −0.02 | 0.03 | 0.76 | 0.01 | 0.02 | 0.04 | 0.60 | 0.29 | 0.66 |
| portal | −0.23 | 0.20 | 0.49 | −0.05 | −0.16 | 0.28 | 0.48 | 0.35 | 0.44 |
| barrier | 0.27 | −0.16 | 0.50 | −0.12 | 0.01 | 0.37 | 0.57 | 0.46 | 0.45 |
| corridor | 0.05 | −0.51 | 0.56 | 0.14 | −0.14 | 0.61 | 0.68 | 0.61 | 0.40 |
| switch λ=0.5 | 0.10 | −0.09 | 0.79 | 0.13 | −0.12 | −0.03 | 0.69 | 0.23 | 0.78 |

Policy remains dominant after adding Q/occupancy, advantage and margin. Advantage
and margin geometry explain nothing beyond discrete policy identity with argmax
targets (partial correlations −0.19 to +0.15). Decision-relevant structure emerges
at h2: β_policy is 0.14–0.40 at h1 and 0.5–0.9 at h2/h3. With Boltzmann targets on
the switch environments, advantage takes the lead (β 0.51–0.61) with policy second
(0.36–0.48) and occupancy third (0.11–0.36). Figure: `task6_regression.png`.

## Task 7: is the interaction component causally necessary?

Two-way decomposition gives 9–11% interaction variance at every hidden layer.

| environment, layer | condition | accuracy | goal sensitivity | steering success |
|---|---|---|---|---|
| base, h3 | baseline | 1.000 | 0.54 | 0.86 |
| base, h3 | subtract `h_sg` exactly | 0.974 | 0.60 | 0.96 |
| base, h3 | subtract random vector, same norm | 1.000 | 0.64 | 0.88 |
| base, h3 | project out top-6 subspace of `h_sg` (holds 97% of additive variance) | 0.356 | 0.29 | 0.38 |
| base, h3 | project out random 6-dim subspace | 1.000 | 0.56 | 0.88 |
| base, h3 | keep `h_s + h_sg`, drop `h_g` | 0.864 | 0.49 | 0.52 |
| switch λ=0.5, h3 | subtract `h_sg` exactly | 0.926 | 0.36 | 0.79 (baseline 0.48) |
| switch λ=0.5, h3 | project out top-3 subspace of `h_sg` (95% of additive variance) | 0.124 | 0.05 | 0.11 |

The additive representation `μ + h_s + h_g` alone yields 93–98% of the optimal
actions from any hidden layer, and removing the interaction makes the mean
goal-difference steering vector *more* effective. The subspace ablation is
uninformative because the interaction's principal subspace is the same low-dimensional
subspace the additive terms occupy. The success criterion "small interaction
components are causally necessary" is not met. Figure: `task7_ablation.png`.

## Task 8: what predicts the steering threshold?

Spearman correlation between the per-state flip threshold α* and each predictor,
with a rank regression on all six:

| environment, layer | linearised boundary crossing | logit margin | m1 | m2 | ‖Δρ‖ | occupancy model | regression R² |
|---|---|---|---|---|---|---|---|
| base, h2 | 0.98 | 0.84 | 0.43 | 0.21 | 0.13 | 0.29 | 0.97 |
| base, h3 | 1.00 | 0.85 | 0.45 | 0.20 | 0.13 | 0.31 | 1.00 |
| switch λ=0.5, h2 | 0.96 | 0.52 | −0.09 | 0.17 | −0.04 | −0.07 | 0.93 |
| switch λ=0.5, h3 | 1.00 | 0.54 | −0.06 | 0.20 | −0.06 | −0.08 | 1.00 |

The linearised crossing is exact at h3 (the readout is linear) and nearly exact at
h2; it takes all the regression weight (β ≈ 1.0, all others ≤ 0.09). In the switch
environment the occupancy quantities are uncorrelated with α*. Steering is direct
movement across an action boundary, not movement through occupancy or value
geometry. Figure: `task8_steering_predictors.png`.

## Task 9: stochastic versus deterministic policies

With argmax targets the h2 geometry tracks the policy table of the λ-optimal
policy at RSA 0.83–0.86 for every λ, while its correlation with the deterministic
(λ = 0) table falls from 0.86 to 0.68; the partial correlation of the λ table given
the λ = 0 table rises to 0.67 and the reverse stays at 0. Occupancy RSA is 0.16–0.20
throughout. With Boltzmann targets the two policy tables are not distinguished
(0.53 versus 0.52) because the geometry has moved to the graded advantage. Given
that the network is trained on π*_λ, the argmax result is expected; the informative
part is that occupancy geometry never enters. Figure: `task9_stochastic_tracking.png`.

## Task 10: dimensionality

| environment | PR Z_sa | PR policy | PR SR | PR h3 trained | PR h3 random |
|---|---|---|---|---|---|
| base | 2.79 | 6.99 | 10.15 | 3.13 | 13.27 |
| twins | 2.57 | 3.73 | 17.73 | 2.98 | 14.80 |
| portal | 2.98 | 7.39 | 9.95 | 2.31 | 13.88 |
| barrier | 1.70 | 5.22 | 9.67 | 1.89 | 14.85 |
| corridor | 1.74 | 1.33 | 5.78 | 1.21 | 10.05 |
| switch λ=0 / 0.5 / 0.9 | 2.01 / 1.73 / 1.72 | 2.37 / 3.01 / 3.20 | 8.78 / 7.60 / 7.31 | 1.44 / 1.96 / 2.17 | 12.42 |

Training collapses the last layer to a participation ratio in the same range as
`Z_sa`, and the number of components for 90% variance matches (2–4 versus 1–3).
Across environments, however, the ordering of hidden PR follows the policy table
(Spearman 0.83) and SR (0.70) more than occupancy (0.47); with nine environments
these correlations are indicative only. The two-NN intrinsic-dimension estimator is
unreliable on these discrete, duplicate-heavy representations and is reported in
the tables with that caveat. Figure: `task10_dimensionality.png`.

## Success criteria

| criterion | outcome |
|---|---|
| Policy-table geometry dominant after controlling for occupancy, SR, Q, advantage, margin, spatial | Met for argmax targets in all nine environments. With Boltzmann targets, advantage geometry leads. |
| Different futures, identical decisions merge | Met: 0.27–0.45 of median at h3, 1.0 at initialisation. |
| Similar futures, different actions stay separated | Met: 0.96–1.21 of median. |
| Change concentrates at policy boundaries | Met for argmax targets (and structurally guaranteed); not met for Boltzmann targets, where change is smooth in occupancy. |
| Interaction component causally necessary | Not met: exact removal costs ≤ 7 points and improves steering. |
| Steering crosses the predicted boundary | Met for the network's own linearised boundary (ρ ≈ 1); occupancy-predicted thresholds add nothing. |
| Reproduces across seeds, stochastic environments, architectures | Seeds and λ: yes. Architectures: only one MLP tested. |

## Interpretation

The hierarchy "dynamics → occupancy/value → decision classes → geometry" is
supported in the specific sense that an argmax-cloned network discards every
distinction in predictive state that does not alter the target action, and does so
increasingly with depth. The Boltzmann condition shows where the quotient comes
from: it is imposed by the imitation target, not chosen by the network. Given
graded targets the same architecture retains graded advantage structure and its
geometry varies smoothly with occupancy. What the network keeps is therefore best
described as "the action-relevant structure exposed by the training signal".

## Next steps

- Repeat with a second architecture (wider or deeper MLP, and a small transformer over
  a token per goal) to address the untested architecture criterion.
- Value-based training (Q-learning or successor features) to test whether the
  advantage-like geometry seen with Boltzmann targets also arises when the network
  must estimate values rather than imitate them.
- Sweep the Boltzmann temperature to map the transition from quotient geometry to
  smooth advantage geometry.
