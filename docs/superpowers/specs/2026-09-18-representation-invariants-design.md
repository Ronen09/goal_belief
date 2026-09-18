# Functionally meaningful representation invariants — design (TASK7)

Date: 2026-09-18. Follows TASK4–6 (`results4/REPORT4.md`, `results5/REPORT5.md`,
`results6/REPORT6.md`): the required logit gap Δℓ = g·D·cos θ is met by different
(gain, separation, alignment) splits depending on readout learning rate, init and
gain, with identical outputs. TASK7 asks which representation measures survive
function-preserving changes of coordinates and optimisation history, and which
still change when the function changes.

## Models (`scripts/run_task7.py`, weights saved to `results7/models/`)

All on the TASK4 HMM family, GRU 64/16, exact-target sequential objective, 3000
steps unless stated, evaluation on 2000 sequences (seed 123) shared by every
model at the same δ.

| family | conditions (δ=0.4, r=1, k=1, λ=1) | seeds |
|---|---|---|
| base | free readout, lr ratio 1, init 1 | 10 |
| gain | fixed-gain readout c ∈ {0.5, 1, 2, 4, 8} | 3 |
| lr | readout lr ratio ∈ {0.1, 0.3, 3, 10} (1 = base) | 3 |
| init | readout init scale ∈ {0.1, 10} (1 = base) | 3 |
| horizon | base trained 10 000 steps, checkpoints {0, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000} | 3 |
| sensitivity | δ ∈ {0.1, 0.2} × lr ratio ∈ {0.1, 1, 10} | 3 |

Equivalence criterion: mean KL to the exact targets < 0.003 nats. Pairwise
output KL (mean and max over 300 sequences × 47 positions) between every pair
of δ=0.4 models is reported. Invariant statistics use converged models only.

## Extracted arrays per model (`goalgeo/invariants.py`, `extract`)

On the shared evaluation set: hidden states at a fixed 800-state subsample
(with beliefs and 1-step predictives), 3000 states for decoding (beliefs,
next-token targets), 400 cue-A / cue-B / U / V states and the pre-relevant
A/B states, mean states of each group, mean hidden norm, readout W and b,
the explicit future Jacobian J = ∂z_{t*}/∂h_t at 200 A-branch anchors
(7 backward passes), the model's predicted distribution at t* for those
anchors, the mean predicted distribution at U positions, and finite
propagation divergences (each A anchor paired with a random B anchor, both run
through the A anchor's future tokens; JS at t*−1; same for U/V).

## Measures (`metrics_from_arrays`)

Raw: sep_cue, sep_pre, sep_control (mean-state separations), D_delay,
D_control (mean pairwise), P_metric, cos_pair (cosine between mean cue
states), hidden_norm, rsa_euclid, rsa_cosine (RDM vs belief RDM), PR, top5
PCA fraction, effective rank, exact rank (relative SVD tolerance 1e-6).
Information (unregularised OLS, 5-fold CV): r2_belief, acc_branch_pre,
r2_target. Subspace: Q (orthonormal column basis of centered H_sub),
rsa_whitened (d_sub² = P_ii + P_jj − 2P_ij vs belief RDM); cross-model
overlap tr(P_A P_B)/rank; transform ‖P' − P‖_F. Functional: D_logit_pre =
‖W_c Δh_pre‖ (row-centered W), I_contrast_pre, D_logit_control, D_future =
mean_i ‖J_i Δh_cue‖, D_F = sqrt(mean_i ΔhᵀJ_iᵀF(p_i)J_iΔh), D_F_control
(J = W_c, p̄_U), JS_future, JS_control.

## Analyses

1. Stability: for each measure and family (gain, lr, init, horizon ≥ 1000
   steps, all combined), CV = std/|mean| and max/min over converged models.
2. Coordinate transforms on the base seed-0 model: scalar (c = 0.1, 10),
   diagonal (log-uniform 0.1–10), orthogonal, full with condition 1/3/10/30;
   ratio M(AH)/M(H) for every measure, ‖P'−P‖_F for the projection, and the
   max logit deviation of W'h' vs Wh.
3. Two-model comparison: lr ratio 0.1 vs 10, seed-averaged, every measure.
4. Horizon: every measure at every checkpoint, 3 seeds.
5. Seeds: CV over 10 base seeds per measure; between-seed RSA agreement
   (Spearman of RDMs) versus subspace overlap.
6. Sensitivity: F-ratio (between-δ variance / within-δ variance) per measure
   over δ ∈ {0.1, 0.2, 0.4} × lr ∈ {0.1, 1, 10}.
7. Classification of each measure as exact invariant (transform ratio = 1 to
   1e-6, provable), empirical invariant (CV < 0.1 across optimisation families
   and F-ratio high), or non-invariant descriptor.

Figures: (1) sep_cue and D_F / D_future vs lr ratio; (2) transform ratios per
measure and family; (3) horizon curves of D, g, cos θ, gDcos θ, D_future, D_F,
decodability; (4) δ sensitivity with optimisation conditions overlaid.
Outputs: `results7/tables7.md`, `results7/results7.json`, `results7/REPORT7.md`.

## Tests (`tests/test_invariants.py`)

Exact invariance of OLS R², rank, P_H, whitened distances, WΔh and JΔh under a
random invertible A (numerical, 1e-6); PR and Euclidean distance change under
diagonal A; the explicit Jacobian matches the JVP-based product; JS finite
propagation is zero for identical states.
