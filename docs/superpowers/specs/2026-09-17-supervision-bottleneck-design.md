# Supervision bottleneck — design spec (TASK3)

Date: 2026-09-17. Requirements: `TASK3.md`. Builds on rounds 1–2.

## 1. Question

Does hidden geometry preserve exactly the distinctions contained in the training
target? Deliverable: `scripts/run_task3.py` writing `results3/` (figures, JSON,
`tables3.md`) and `results3/REPORT3.md` including the theoretical account (Task 12).

## 2. Decisions

| Topic | Decision | Why |
|---|---|---|
| Base dataset | Base 8×8 grid, all 840 (s, g) rows, one-hot inputs, 3×128 MLP, 3000 full-batch Adam steps. | Same as rounds 1–2; one-hot inputs carry no geometry, so hidden geometry can only come from the targets. |
| Q facts | Q ∈ [0, 1], margins ≤ 0.11 (95th pct), 57% of rows have exact top-two ties. | Sets the meaning of τ: τ = 0.02 is already soft (mean max-prob 0.68); τ ≥ 0.5 is near-uniform (0.28). |
| Target kinds (Task 4) | `hard` = uniform over argmax set; `boltzmann(τ)` = softmax(Q/τ); `advantage` = MSE on A(s,·,g) = Q − max Q; `q` = MSE on Q(s,·,g); `occupancy` = MSE on Z_sa(s) (4K-dim positive control). Same MLP, only the output layer's width differs. | In this reward structure Q(s,·,g) = ρ(g|s,·)/γ exactly, so a 4-dim "occupancy profile" target would duplicate Q; the positive control therefore uses the full occupancy vector. |
| Losses (Task 13) | `ce` (soft cross-entropy), `kl` (KL(target‖model), identical gradient to CE: verified by test), `mse_prob` (MSE on probabilities), `mse_logit` (MSE on centred logits versus centred Q/τ). | Separates target information from loss weighting; CE ≡ KL is stated and demonstrated rather than treated as a separate condition. |
| Temperature sweep (Task 1) | τ ∈ {hard, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5}, 5 seeds, checkpoints at {0, 10, 30, 100, 300, 1000, 3000}. | Grid as specified; checkpoints feed Task 7. |
| Target geometry (Task 2) | Pair level: RDM over (s,g) rows of the target vectors (subsampled 1200 rows). State level: RDM over states of the concatenation over goals of the target rows. Both compared with the hidden RDM by RSA and by regression against occupancy, SR, spatial, policy, advantage. | Mirrors the earlier analyses at both granularities. |
| Fixed-argmax counterfactual (Task 3) | (a) Natural: pairs of (s,g) rows with the same argmax set but different Q margins, d_h versus |Δmargin|, hard vs soft. (b) Synthetic: 240 one-hot "states", Q vectors drawn so that 200 background rows have random Q and 40 probe rows share argmax with a reference but have margins m ∈ {0.005 … 0.5}; train hard and each τ; report d_h(probe(m), reference)/median. | (b) removes the environment entirely; (a) shows it in situ. |
| Information bottleneck (Task 5) | For each T: rank and PR of the target matrix, RSA(D_Q, D_T), number of equivalence classes (distinct rows after rounding to 1e-6, or 1e-2 for soft), ridge-CV recoverability of Q, advantage and Z_sa from Y. | As listed in the task. |
| Sufficient statistic (Task 6/7) | Hard classes = argmax sets. Within-class / between-class mean distance on (s,g) rows at every checkpoint, for hard and τ ∈ {0.05, 0.5}; also between-class and within-class separately (theory predicts between grows while within stays). Pairs: identical target / different Z_sa, and different target / similar Z_sa. | Direct test of the collapse prediction. |
| Generalisation (Task 8) | Hold out 30% of (s,g) rows (state still seen with other goals) on base and twins, one-hot; report held-out accuracy and held-out within/between ratio and nearest-class-centroid accuracy. Corridor with coordinate input and 30% of *states* held out: are unseen cells placed with their run? | One-hot embeddings of fully unseen states cannot generalise, so state hold-out uses coordinates. |
| Architectures (Task 9) | `mlp3` (current), `mlp6` (6×128), `resmlp` (3 pre-activation residual blocks of width 128), `transformer` (state and goal tokens, 2 encoder layers, d = 64, 4 heads; state token read out). Conditions: hard, τ = 0.1, τ = 0.5, plus the synthetic counterfactual. 3 seeds. | Minimum replication set from the task. |
| Additive decomposition (Task 10) | The ANOVA decomposition already satisfies the centring constraints (tested). The new quantity is the interaction variance *unique in feature space*: project r off the span of the additive rows {a_s} ∪ {b_g}; report the fraction and ablate only that part, with a norm-matched random control in the same complement. | Addresses the 90–97% overlap found in round 2. |
| Steering across τ (Task 11) | For each τ network at h2/h3: steering success at α = 1 and Spearman of α* with α_lin, logit margin, target-probability margin, Q margins, ‖Δρ‖. | As specified. |
| Theory (Task 12) | Written derivation in the report: loss factors through T(x); gradient on h(x) is a function of (h, T(x)); for CE with hard labels on separable data the logit scale grows while within-class gradients vanish, giving falling within/between ratio (neural-collapse-type argument); for soft targets the optimal readout must reproduce log π_τ up to a per-row constant, forcing h to keep target-logit distinctions; conditions listed. Tested by Task 7 curves. | The task asks for conditions, not a proof of necessity. |
| Final environment (Task 14) | `ring` 5×7 boundary (20 cells) with goals at 4 places; pairs chosen algorithmically: A = same table, top-quartile ‖ΔZ_sa‖; B = different table, bottom-quartile ‖ΔZ_sa‖; C = same table, bottom-half ‖ΔZ_sa‖, top-quartile ‖Δπ_τ‖ at τ = 0.02. Train hard and τ = 0.02, 5 seeds. | Ring geometry gives graded margins with same argmax along each arc. |

## 3. Code

- `goalgeo/targets.py`: target construction, losses, generic trainer with checkpoints.
- `goalgeo/models2.py`: `out_dim` for `PolicyNet`; `DeepMLP`, `ResMLP`, `TinyTransformer`; `SyntheticTask`.
- `goalgeo/supervision.py`: TASK3 analyses.
- `goalgeo/plotting3.py`, `scripts/run_task3.py`.
- Tests: Boltzmann normalisation and τ→0 limit, KL ≡ CE gradient, decomposition constraints and unique-interaction orthogonality, equivalence-class counting, synthetic task construction, ring pairs exist, transformer/resmlp activation shapes.
