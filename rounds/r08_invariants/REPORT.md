# Functionally meaningful representation invariants (TASK7)

Run date: 2026-09-18. 64 models in 669 s on 8 CPU workers, plus a follow-up
pass over the saved weights (`rounds/r08_invariants/followup.py`, `rounds/r08_invariants/followup.json`).
Numbers from `rounds/r08_invariants/tables.md`; per-run values, transform ratios, horizon
curves and sensitivity rows in `rounds/r08_invariants/results7.json`; weights in
`rounds/r08_invariants/models/`. Design: `rounds/r08_invariants/DESIGN.md`.
Reproduce with `rounds/r08_invariants/run.py --jobs 8` then `rounds/r08_invariants/followup.py`.

## Setup

TASK4 HMM family, the round 4 GRU, exact-target sequential objective. The
equivalence set at δ=0.4: 10 base seeds; fixed readout gains {0.5, 1, 2, 4, 8};
readout learning-rate ratios {0.1, 0.3, 1, 3, 10}; readout initialisation scales
{0.1, 1, 10}; a 10,000-step horizon run with 14 checkpoints; 3 seeds each unless
stated. Functional-sensitivity control: δ ∈ {0.1, 0.2} × learning-rate ratios
{0.1, 1, 10}. Every model is evaluated on the same 2000 sequences, with fixed
state subsamples so rows correspond across models.

Measures are computed from extracted arrays (`goalgeo/invariants.py`) in four
groups. Raw: mean-state and pairwise separations of the delayed and control
pairs, P_metric, cosine between cue means, hidden norm, Euclidean and cosine
RSA against the belief RDM, participation ratio, top-5 PCA fraction, effective
rank. Information: unregularised OLS decoding (5-fold) of the belief, the
branch at t*−1, and the next-token target; exact rank. Subspace: the sample-space
projection P_H of the centred states and the whitened RSA it induces.
Functional: readout displacement ‖W_cΔh‖ and the x-versus-y contrast
(w_x−w_y)·Δh at t*−1; the linearised future displacement ‖JΔh‖ with
J = ∂z_{t*}/∂h_t computed explicitly at 200 anchors; the Fisher metric D_F built
from J and the model's prediction at t*; and the finite version, the
Jensen–Shannon divergence between the predictions obtained by propagating an
A-branch state and a B-branch state through the same future tokens. The same
quantities are computed for the control pair through its immediate readout.

## Part 1: the equivalence set is tight

61 of 64 models are below 0.003 nats of the Bayes floor; the three excluded are
the gain-0.5 seeds at 0.0031 to 0.0041. Over the 780 pairs of converged δ=0.4
models, the mean pairwise output KL over 300 sequences is 0.00009 nats, the
worst pair's mean is 0.0008 and the worst single position 0.013. These models
compute the same function to within the noise of training.

## Part 2: raw geometry varies, RSA does not

Coefficient of variation (CV) and max/min fold over the 51 converged δ=0.4
models outside the horizon run ("all"), and within each family:

| measure | all CV | all fold | gain CV | lr CV | init CV | seeds CV |
|---|---|---|---|---|---|---|
| ‖Δh‖ at the cue | 0.23 | 2.7 | 0.26 | 0.17 | 0.26 | 0.10 |
| ‖Δh‖ at t*−1 | 0.41 | 6.2 | 0.55 | 0.30 | 0.46 | 0.05 |
| hidden norm | 0.32 | 3.6 | 0.42 | 0.30 | 0.24 | 0.01 |
| participation ratio | 0.30 | 2.4 | 0.28 | 0.36 | 0.09 | 0.03 |
| effective rank | 0.33 | 2.6 | 0.31 | 0.39 | 0.12 | 0.04 |
| readout gain | 0.91 | 9.7 | 0.75 | 0.30 | 0.91 | 0.04 |
| Euclidean RSA vs belief | 0.02 | 1.1 | 0.02 | 0.03 | 0.02 | 0.01 |
| cosine RSA vs belief | 0.02 | 1.1 | 0.02 | 0.02 | 0.02 | 0.02 |
| top-5 PCA fraction | 0.02 | 1.1 | 0.02 | 0.02 | 0.01 | 0.00 |

Separations, norms and dimensionality measures move by 2.5 to 10× across models
with identical outputs, far beyond their seed-to-seed variation. RSA against
the belief geometry is stable (CV 0.02), and so is RSA between equivalent
models directly: 0.94 to 0.97 Spearman between RDMs for every family, equal to
the between-seed value. But RSA is also blind to the function: its F-ratio
across δ ∈ {0.1, 0.2, 0.4} is 0.1. The coarse geometry that RSA sees is set by
the token structure of the HMM and does not register a fivefold change in the
required output gap. The same holds for the top-5 PCA fraction.

## Part 3: exact coordinate transforms

On the base seed-0 model, eight transforms h → Ah with W → WA⁻¹ and J → JA⁻¹
(scalar 0.1 and 10, log-uniform diagonal, orthogonal, full at condition 1, 3,
10, 30). The maximum logit deviation is 2e-14, so the function is unchanged.
Ratio M(AH)/M(H):

| group | measures | worst deviation from 1 |
|---|---|---|
| raw separations, hidden norm | sep_cue, sep_pre, sep_control, D_delay, D_control | 0.1× to 12.6× |
| raw shape | P_metric, cos_pair, PR, effective rank | 0.37, 0.11, 0.36, 0.34 |
| RSA | Euclidean, cosine | 0.012, 0.019 |
| allocation | g_eff, cos θ | 9.0, 0.93 |
| information | belief R², branch accuracy, target R², rank | 2.7e-14, 0, 1.1e-16, 0 |
| subspace | ‖P′ − P‖_F, whitened RSA | 3.5e-11, 4.2e-13 |
| functional | W_cΔh, contrast, ‖JΔh‖, D_F, JS, control versions | ≤ 1.4e-15 |

Every information, subspace and functional candidate is an exact invariant to
machine precision. Rotations leave the raw measures unchanged, scalars scale
them, and diagonal or ill-conditioned maps change even the scale-free ones:
cos θ drops to 0.08 of its value under a diagonal map, the participation ratio
moves by a third.

## Candidate by candidate

**1. Linear decodability (OLS).** Exact, and stable across the equivalence set
(belief R² CV 0.024, branch accuracy CV 0.001). Insensitive to δ (F 0.1): the
branch is 99 % decodable at t*−1 whether the consequence is weak or strong. An
availability invariant, as intended, and nothing more.

**2. Exact rank.** Exact, but trivial here: 63 or 64 in every model. The GRU
uses its whole state space.

**3 and 4. Sample-space projection and whitened geometry.** Exact under every
transform (‖P′ − P‖_F ≈ 1e-11). Because the rank is full, whitening flattens
the representation and the whitened RSA against the belief RDM is noise
(0.01 ± 0.02). The projection itself is informative across models only at
reduced rank. From the follow-up, the mean overlap tr(P_A P_B)/k of the top-k
principal sample subspaces between equivalent models:

| pairs | k=4 | k=8 | k=16 | k=32 | k=64 | chance | RSA between models |
|---|---|---|---|---|---|---|---|
| base seeds (45) | 0.90 | 0.95 | 0.82 | 0.88 | 0.83 | 0.005–0.08 | 0.97 |
| gain family (102) | 0.78 | 0.90 | 0.75 | 0.73 | 0.72 | | 0.95 |
| lr family (102) | 0.85 | 0.84 | 0.72 | 0.74 | 0.74 | | 0.94 |
| init family (33) | 0.76 | 0.92 | 0.76 | 0.78 | 0.76 | | 0.97 |

Equivalent implementations share most of their represented sample subspace,
far above chance, but not all of it, and about as much as independent seeds do.
The subspace is an empirical invariant of the coarse structure, like RSA, and
like RSA it does not encode the delayed pair's scale, which is what the
interventions change.

**5. Readout displacement W_cΔh and the contrast (w_x − w_y)·Δh.** Exact. The
contrast at t*−1 is 4.30 in every converged δ=0.4 model (CV 0.004, fold 1.02)
and separates δ with an F-ratio of 35,000. The norm ‖W_cΔh‖ is slightly looser
(CV 0.06) because the other five readout rows contribute variation that the
target does not pin. Passes all three tests.

**6. Linearised future displacement ‖J_{t→t*}Δh_t‖.** Exact under coordinate
changes, but *not* stable across equivalent models: CV 0.27, fold 3.7, and in
the two-model comparison the slow-readout model gives 1.98 against 3.46 for the
fast-readout model with identical outputs (ratio 0.57). It also drifts over the
horizon (2.8 → 1.5–2.0 between steps 3000 and 10,000) with the function fixed.
The follow-up diagnoses this as linearisation error: the finite displacement,
propagating the actual states through the actual future step and comparing
centred logits, has CV 0.07 across the same models, and using per-pair
differences instead of the mean difference does not help (CV 0.27). The map
from h_t to z_{t*} is a GRU step followed by the readout, and the cue
separations of 2.5 to 6 units are not small relative to its curvature. More
fundamentally, functional equivalence holds for the input-output map, not for
the map from a hidden state to the output: equivalent models have different
h → z maps evaluated at different h. A local metric at h inherits both.

**7. Fisher metric D_F.** Same verdict: exact under coordinates, CV 0.22 across
equivalent models, fold 3.1, drifts from 1.2 to 0.4–0.7 over the long horizon,
and its δ F-ratio of 2.6 is barely above the raw separation's 1.6. The control
version, where the "future" computation is the immediate linear readout and
linearisation is exact, has CV 0.003.

**8. Finite output divergence after propagation.** Exact by construction and
the most stable quantity measured: JS_future is 0.368 in every converged
δ=0.4 model (CV 0.000), 0.082 at δ=0.2 and 0.020 at δ=0.1 (F-ratio 4·10⁶). For
a model at the Bayes floor it equals the divergence between the two branches'
target predictives, so it measures the function directly. Its representational
content is the comparison with availability: a feature can be fully decodable
and produce no divergence, which is exactly the TASK4 case at r=0 or λ=0.

## Part 5: two models, same function

Seed means for the slow-readout (lr ×0.1) and fast-readout (lr ×10) models:

| measure | lr ×0.1 | lr ×10 | ratio |
|---|---|---|---|
| ‖Δh‖ at the cue | 4.24 | 2.73 | 1.55 |
| ‖Δh‖ at t*−1 | 5.89 | 2.51 | 2.35 |
| hidden norm | 7.67 | 2.78 | 2.76 |
| participation ratio | 1.41 | 3.20 | 0.44 |
| readout gain | 0.93 | 2.12 | 0.44 |
| belief R² | 0.94 | 0.99 | 0.95 |
| branch accuracy | 0.988 | 0.988 | 1.00 |
| contrast (w_x−w_y)·Δh | 4.306 | 4.305 | 1.00 |
| ‖JΔh‖ | 1.98 | 3.46 | 0.57 |
| D_F | 0.71 | 1.20 | 0.59 |
| JS after propagation | 0.3680 | 0.3678 | 1.00 |
| KL to targets | 0.0001 | 0.0000 | |

Raw distance differs by 1.5 to 2.8×, the participation ratio by 2.3×, the
readout contrast and the propagated divergence are identical, decodability is
identical, and the linearised functional measures are off by 1.7×.

## Part 7: training horizon

With three 10,000-step runs, the loss reaches the floor by step 100. From then
on g·D·cos θ sits at 4.30 to four digits. The raw separation drifts from 3.7 to
4.1 and then rises to 4.8 between steps 5000 and 10,000 while the gain, having
crept up to 1.35, falls back to 1.24: a late reallocation toward the
representation. cos θ declines slowly. ‖JΔh‖ and D_F wander through this whole
period and fall sharply at the end (D_F 1.2 → 0.4–0.7). Belief R² and RSA are
flat from step 100. The quantities that stabilise with the function are the
readout-projected contrast, the propagated divergence, decodability and RSA;
the ones that keep moving are scale, gain, alignment and the linearised
functional metrics.

## Part 8: seeds

Across 10 base seeds: Euclidean RSA between seeds 0.97, top-64 subspace
overlap 0.83, relative difference in D_F 0.15. Seed CVs are small for
everything (separation 0.10, gain 0.04, contrast 0.006, JS 0.000), so the
optimisation interventions move the raw measures by two to nine times their
seed variation while the exact functional measures do not move at all.

## Functional sensitivity

F-ratio of between-δ variance to within-δ variance, each δ with three
learning-rate ratios × 3 seeds:

| measure | δ=0.1 | δ=0.2 | δ=0.4 | F |
|---|---|---|---|---|
| ‖Δh‖ at the cue | 2.14 | 2.60 | 3.65 | 1.6 |
| Euclidean RSA | 0.458 | 0.461 | 0.466 | 0.1 |
| belief R² | 0.98 | 0.99 | 0.98 | 0.1 |
| contrast (w_x−w_y)·Δh | 0.79 | 1.66 | 4.31 | 35,111 |
| ‖W_cΔh‖ | 0.94 | 1.48 | 3.28 | 9.3 |
| ‖JΔh‖ | 0.85 | 1.31 | 2.73 | 2.1 |
| D_F | 0.37 | 0.70 | 1.00 | 2.6 |
| JS after propagation | 0.020 | 0.082 | 0.368 | 4·10⁶ |

Raw distance overlaps between δ levels (a fast-readout δ=0.4 model sits where a
slow-readout δ=0.1 model does); RSA and decodability do not see δ at all; the
readout contrast and the propagated divergence separate the three levels with
no overlap.

## Verdict against the three tests

| candidate | exact coordinate invariance | stable across equivalent models | sensitive to the function | class |
|---|---|---|---|---|
| ‖Δh‖, hidden norm, PR, effective rank, P_metric, gain, cos θ | no | no (CV 0.15–0.9) | weak | non-invariant descriptor |
| Euclidean / cosine RSA, top-5 PCA fraction | no (1–2 % under diagonal maps) | yes (CV 0.02) | no (F 0.1) | stable but blind |
| OLS decodability, exact rank | yes | yes | no | availability invariant |
| sample-space projection P_H, whitened RSA | yes | 0.7–0.9 overlap at reduced rank; whitened RSA is noise at full rank | no | subspace invariant, informative only below full rank |
| readout contrast (w_x−w_y)·Δh, ‖W_cΔh‖ | yes | yes (CV 0.004 / 0.06) | yes (F 35,000 / 9) | functional invariant |
| ‖J_{t→t*}Δh‖, D_F | yes | no (CV 0.27 / 0.22, drift over horizon) | weak (F 2) | coordinate-invariant descriptor, fails optimisation invariance |
| JS after finite propagation | yes | yes (CV 0.000) | yes (F 4·10⁶) | functional invariant |

Success criteria: 1, 2, 3, 4, 7, 8 and 9 hold. Criterion 5 fails: the
linearised future displacement is no more stable than the raw hidden distance
across the optimisation interventions. Criterion 6 fails for the same reason.
Both are rescued by finite propagation.

## Conceptual outcome

The three-way decomposition survives, with one correction. Information
availability (unregularised decoding) and the represented subspace are exact
invariants that do not see how strongly a feature is used. Functional effect is
an exact invariant that does, provided it is measured by pushing hidden states
through the network's actual downstream computation and reading the output
distribution, not by a local metric at the hidden state. The local versions,
‖JΔh‖ and the Fisher metric, are invariant to coordinates but not to
optimisation history, because two networks that agree on inputs to outputs do
not agree on hidden states to outputs, and the hidden-state differences that
matter are large on the scale of the recurrence's curvature. Euclidean geometry
is neither invariant nor, in RSA form, sensitive: it describes the coarse
structure that every implementation shares and misses the scale that the
function sets. The interpretable object is therefore the pair (what can be
read out, what it does when read out), and the second member has to be
computed by the network itself.
