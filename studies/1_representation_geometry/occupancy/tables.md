# Auto-generated result tables

## Phase 3: occupancy geometry (base)

- states 57, Z_sa dim 60, participation ratio 2.79, entropy rank 4.14, PCs for 90% var 4
- mean pairwise cosine 0.910
- RSA between hypothesis RDMs: occupancy~occupancy_state 0.99, occupancy~spatial 0.94, occupancy~occupancy_log 0.99, occupancy~policy 0.31, occupancy_state~spatial 0.95, occupancy_state~policy 0.32, geodesic~occupancy 0.95, geodesic~occupancy_state 0.97, geodesic~spatial 0.95, geodesic~occupancy_log 0.97, geodesic~identity 0.00, geodesic~policy 0.34, SR~occupancy 0.77, SR~occupancy_state 0.78, SR~spatial 0.83, SR~geodesic 0.83, SR~occupancy_log 0.80, SR~identity 0.00, SR~policy 0.43, occupancy_log~occupancy_state 0.99, occupancy_log~spatial 0.96, occupancy_log~policy 0.34, identity~occupancy 0.00, identity~occupancy_state 0.00, identity~spatial 0.00, identity~occupancy_log 0.00, identity~policy 0.00, policy~spatial 0.40

## Phase 4 (onehot): accuracy 1.0000, rollout success 1.000, optimal-length fraction 1.000

## Phase 4 (coord): accuracy 1.0000, rollout success 1.000, optimal-length fraction 1.000

## Phase 5 (onehot input), state view = mean over goals; mean over 5 seeds

| metric | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| rsa_occupancy | +0.01 | +0.07 | +0.29 | +0.35 |
| rsa_occupancy_state | +0.01 | +0.07 | +0.30 | +0.36 |
| rsa_occupancy_log | +0.01 | +0.08 | +0.31 | +0.38 |
| rsa_policy | +0.06 | +0.32 | +0.73 | +0.78 |
| rsa_spatial | +0.01 | +0.10 | +0.39 | +0.47 |
| rsa_geodesic | +0.01 | +0.10 | +0.35 | +0.41 |
| rsa_SR | -0.01 | +0.19 | +0.47 | +0.49 |
| rsa_identity | +0.00 | +0.00 | +0.00 | +0.00 |
| partial_occ_given_spatial | -0.00 | -0.07 | -0.25 | -0.30 |
| partial_spatial_given_occ | +0.01 | +0.10 | +0.37 | +0.44 |
| partial_occ_given_geodesic | +0.00 | -0.08 | -0.15 | -0.12 |
| partial_geodesic_given_occ | +0.00 | +0.11 | +0.25 | +0.25 |
| cka_occupancy | +0.15 | +0.22 | +0.40 | +0.44 |
| cka_spatial | +0.13 | +0.21 | +0.50 | +0.56 |
| cka_SR | +0.28 | +0.34 | +0.43 | +0.42 |
| decode_r2_occupancy | -1.29 | -0.19 | +0.16 | +0.20 |
| decode_r2_spatial | -1.03 | +0.22 | +0.34 | +0.52 |
| decode_r2_occupancy_pairs | -1.86 | -0.33 | -0.24 | -0.07 |
| participation_ratio | +20.34 | +19.60 | +7.09 | +3.10 |
| subspace_overlap_k3 | +0.04 | +0.08 | +0.31 | +0.32 |
| subspace_overlap_k5 | +0.09 | +0.16 | +0.31 | +0.31 |
| rsa_occupancy (random init) | -0.00 | -0.00 | +0.00 | +0.01 |
| rsa_spatial (random init) | -0.01 | -0.01 | -0.01 | +0.01 |
| rsa_policy (random init) | +0.01 | -0.00 | -0.01 | +0.01 |
| decode_r2_occupancy (random init) | -2.07 | -0.82 | -0.19 | -0.03 |
| cka_occupancy (random init) | +0.12 | +0.11 | +0.10 | +0.09 |
| participation_ratio (random init) | +20.21 | +19.76 | +16.77 | +13.46 |
| rsa_occupancy (concat-over-goals view) | +0.01 | +0.07 | +0.30 | +0.37 |
| rsa_spatial (concat-over-goals view) | +0.01 | +0.10 | +0.40 | +0.49 |
| rsa_SR (concat-over-goals view) | -0.01 | +0.18 | +0.46 | +0.49 |
| rsa_policy (concat-over-goals view) | +0.06 | +0.31 | +0.73 | +0.79 |
| decode_r2_occupancy (concat-over-goals view) | -1.99 | -0.27 | +0.23 | +0.36 |
| RDM regression β @emb (occ, spatial, geo, SR; R²) | -0.03 | +0.05 | +0.02 | -0.05 | +0.01 |
| RDM regression β @h1 (occ, spatial, geo, SR; R²) | -0.17 | -0.04 | +0.01 | +0.34 | +0.06 |
| RDM regression β @h2 (occ, spatial, geo, SR; R²) | -0.57 | +0.64 | -0.09 | +0.45 | +0.27 |
| RDM regression β @h3 (occ, spatial, geo, SR; R²) | -0.64 | +0.96 | -0.16 | +0.32 | +0.32 |

Pair-level RSA (onehot):
| model | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| rsa_pair_occupancy | +0.05 | +0.07 | +0.13 | +0.13 |
| rsa_state_occupancy_only | +0.05 | +0.07 | +0.12 | +0.13 |
| rsa_pair_Q | +0.01 | -0.01 | +0.03 | +0.04 |
| rsa_pair_spatial | +0.13 | +0.27 | +0.48 | +0.49 |
| rsa_pair_action | +0.08 | +0.20 | +0.48 | +0.56 |
| rsa_goal_identity_only | +0.41 | +0.35 | +0.26 | +0.21 |
| decode_r2_Q | -0.03 | -0.04 | +0.25 | +0.27 |

γ sweep (onehot): RSA of h̄(s) with the occupancy RDM built at γ = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
|   emb | -0.01 | -0.01 | +0.00 | +0.01 | +0.01 | +0.01 | +0.01 |
|   h1 | -0.02 | +0.02 | +0.05 | +0.07 | +0.08 | +0.08 | +0.08 |
|   h2 | +0.06 | +0.17 | +0.25 | +0.29 | +0.30 | +0.31 | +0.31 |
|   h3 | +0.11 | +0.24 | +0.31 | +0.35 | +0.36 | +0.37 | +0.37 |

Emergence at h2 (onehot), steps [0, 10, 30, 100, 300, 1000, 3000]:
| accuracy | +0.41 | +0.77 | +0.92 | +1.00 | +1.00 | +1.00 | +1.00 |
| RSA occupancy | -0.01 | +0.03 | +0.18 | +0.25 | +0.27 | +0.28 | +0.29 |
| RSA spatial | -0.01 | +0.04 | +0.24 | +0.35 | +0.37 | +0.38 | +0.39 |
| decode R² occupancy | -0.14 | -0.08 | +0.03 | +0.11 | +0.13 | +0.14 | +0.16 |

## Phase 5 (coord input), state view = mean over goals; mean over 5 seeds

| metric | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| rsa_occupancy | +0.93 | +0.94 | +0.92 | +0.86 |
| rsa_occupancy_state | +0.94 | +0.95 | +0.94 | +0.88 |
| rsa_occupancy_log | +0.95 | +0.96 | +0.94 | +0.88 |
| rsa_policy | +0.40 | +0.39 | +0.41 | +0.42 |
| rsa_spatial | +0.99 | +0.99 | +0.99 | +0.95 |
| rsa_geodesic | +0.94 | +0.94 | +0.94 | +0.89 |
| rsa_SR | +0.83 | +0.82 | +0.84 | +0.83 |
| rsa_identity | +0.00 | +0.00 | +0.00 | +0.00 |
| partial_occ_given_spatial | -0.04 | +0.27 | -0.02 | -0.27 |
| partial_spatial_given_occ | +0.95 | +0.93 | +0.92 | +0.82 |
| partial_occ_given_geodesic | +0.32 | +0.42 | +0.30 | +0.09 |
| partial_geodesic_given_occ | +0.52 | +0.47 | +0.50 | +0.48 |
| cka_occupancy | +0.94 | +0.96 | +0.94 | +0.89 |
| cka_spatial | +0.99 | +0.99 | +0.99 | +0.95 |
| cka_SR | +0.78 | +0.80 | +0.81 | +0.78 |
| decode_r2_occupancy | +0.78 | +0.90 | +0.93 | +0.94 |
| decode_r2_spatial | +1.00 | +1.00 | +1.00 | +1.00 |
| decode_r2_occupancy_pairs | +0.77 | +0.90 | +0.91 | +0.90 |
| participation_ratio | +1.97 | +2.13 | +2.30 | +2.32 |
| subspace_overlap_k3 | +0.65 | +0.67 | +0.68 | +0.68 |
| subspace_overlap_k5 | +0.41 | +0.92 | +0.92 | +0.87 |
| rsa_occupancy (random init) | +0.93 | +0.92 | +0.91 | +0.92 |
| rsa_spatial (random init) | +0.99 | +0.98 | +0.96 | +0.96 |
| rsa_policy (random init) | +0.38 | +0.38 | +0.37 | +0.37 |
| decode_r2_occupancy (random init) | +0.78 | +0.79 | +0.59 | +0.19 |
| cka_occupancy (random init) | +0.94 | +0.93 | +0.92 | +0.93 |
| participation_ratio (random init) | +1.92 | +1.96 | +2.01 | +2.05 |
| rsa_occupancy (concat-over-goals view) | +0.93 | +0.94 | +0.92 | +0.88 |
| rsa_spatial (concat-over-goals view) | +0.99 | +0.99 | +0.99 | +0.96 |
| rsa_SR (concat-over-goals view) | +0.83 | +0.82 | +0.84 | +0.83 |
| rsa_policy (concat-over-goals view) | +0.40 | +0.39 | +0.41 | +0.42 |
| decode_r2_occupancy (concat-over-goals view) | +0.78 | +0.93 | +0.93 | +0.93 |
| RDM regression β @emb (occ, spatial, geo, SR; R²) | -0.01 | +0.99 | +0.00 | +0.01 | +0.99 |
| RDM regression β @h1 (occ, spatial, geo, SR; R²) | +0.11 | +0.93 | -0.05 | +0.00 | +0.99 |
| RDM regression β @h2 (occ, spatial, geo, SR; R²) | -0.00 | +0.98 | -0.03 | +0.05 | +0.98 |
| RDM regression β @h3 (occ, spatial, geo, SR; R²) | -0.24 | +1.07 | +0.02 | +0.10 | +0.92 |

Pair-level RSA (coord):
| model | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| rsa_pair_occupancy | +0.66 | +0.76 | +0.60 | +0.39 |
| rsa_state_occupancy_only | +0.66 | +0.76 | +0.60 | +0.39 |
| rsa_pair_Q | +0.07 | +0.07 | +0.05 | +0.02 |
| rsa_pair_spatial | +0.98 | +0.91 | +0.78 | +0.62 |
| rsa_pair_action | +0.24 | +0.24 | +0.33 | +0.39 |
| rsa_goal_identity_only | +0.22 | +0.17 | +0.17 | +0.16 |
| decode_r2_Q | +0.03 | +0.89 | +0.92 | +0.94 |

γ sweep (coord): RSA of h̄(s) with the occupancy RDM built at γ = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
|   emb | +0.36 | +0.68 | +0.86 | +0.93 | +0.94 | +0.95 | +0.95 |
|   h1 | +0.37 | +0.69 | +0.87 | +0.94 | +0.95 | +0.96 | +0.96 |
|   h2 | +0.37 | +0.68 | +0.86 | +0.92 | +0.93 | +0.94 | +0.94 |
|   h3 | +0.35 | +0.64 | +0.80 | +0.86 | +0.87 | +0.88 | +0.88 |

Emergence at h2 (coord), steps [0, 10, 30, 100, 300, 1000, 3000]:
| accuracy | +0.41 | +0.77 | +0.85 | +0.90 | +0.96 | +1.00 | +1.00 |
| RSA occupancy | +0.87 | +0.85 | +0.88 | +0.90 | +0.90 | +0.92 | +0.92 |
| RSA spatial | +0.95 | +0.92 | +0.95 | +0.97 | +0.98 | +0.99 | +0.99 |
| decode R² occupancy | +0.56 | +0.74 | +0.81 | +0.87 | +0.91 | +0.93 | +0.93 |

## Phase 6: discriminating environments (mean over seeds)

**base**: RSA occupancy~spatial 0.94, occupancy~geodesic 0.95, occupancy~policy 0.31, spatial~policy 0.40
| encoding | layer | acc | β_occ | β_spatial | β_geo | β_SR | R² | ρ_policy | β_occ (3-way: occ+sp+policy) | β_sp (3-way) | β_policy (3-way) | disagree ρ_occ | disagree ρ_sp | designed pairs | same-spatial ctrl | same-occ ctrl | random-init designed | random-init same-spatial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onehot | emb | +1.00 | -0.03 | +0.06 | +0.06 | -0.10 | +0.01 | +0.05 | +0.04 | -0.05 | +0.06 | +0.01 | +0.01 | +nan | +nan | +nan | +nan | +nan |
| onehot | h1 | +1.00 | -0.07 | -0.01 | -0.06 | +0.29 | +0.04 | +0.31 | +0.04 | -0.06 | +0.32 | +0.07 | +0.11 | +nan | +nan | +nan | +nan | +nan |
| onehot | h2 | +1.00 | -0.53 | +0.63 | -0.12 | +0.43 | +0.26 | +0.72 | -0.29 | +0.40 | +0.65 | +0.21 | +0.39 | +nan | +nan | +nan | +nan | +nan |
| onehot | h3 | +1.00 | -0.62 | +0.94 | -0.16 | +0.31 | +0.31 | +0.78 | -0.39 | +0.56 | +0.67 | +0.26 | +0.46 | +nan | +nan | +nan | +nan | +nan |
| coord | emb | +1.00 | -0.01 | +0.99 | +0.01 | +0.01 | +0.99 | +0.40 | -0.01 | +1.00 | -0.00 | +0.87 | +0.99 | +nan | +nan | +nan | +nan | +nan |
| coord | h1 | +1.00 | +0.12 | +0.93 | -0.05 | +0.00 | +0.99 | +0.39 | +0.10 | +0.90 | +0.00 | +0.89 | +0.99 | +nan | +nan | +nan | +nan | +nan |
| coord | h2 | +1.00 | -0.00 | +0.97 | -0.03 | +0.05 | +0.97 | +0.41 | -0.01 | +0.98 | +0.02 | +0.86 | +0.98 | +nan | +nan | +nan | +nan | +nan |
| coord | h3 | +1.00 | -0.30 | +1.11 | +0.04 | +0.10 | +0.92 | +0.42 | -0.27 | +1.19 | +0.03 | +0.78 | +0.95 | +nan | +nan | +nan | +nan | +nan |

**barrier**: RSA occupancy~spatial 0.70, occupancy~geodesic 0.97, occupancy~policy 0.48, spatial~policy 0.55
| encoding | layer | acc | β_occ | β_spatial | β_geo | β_SR | R² | ρ_policy | β_occ (3-way: occ+sp+policy) | β_sp (3-way) | β_policy (3-way) | disagree ρ_occ | disagree ρ_sp | designed pairs | same-spatial ctrl | same-occ ctrl | random-init designed | random-init same-spatial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onehot | emb | +1.00 | +0.15 | +0.01 | -0.11 | -0.02 | +0.01 | +0.07 | +0.03 | -0.05 | +0.08 | +0.02 | -0.02 | +1.01 | +0.99 | +1.01 | +1.04 | +1.01 |
| onehot | h1 | +1.00 | +0.14 | -0.08 | -0.17 | +0.38 | +0.10 | +0.36 | +0.09 | -0.05 | +0.34 | +0.07 | +0.05 | +1.12 | +0.93 | +1.02 | +1.01 | +1.00 |
| onehot | h2 | +1.00 | -0.02 | +0.44 | +0.25 | +0.04 | +0.44 | +0.67 | +0.14 | +0.30 | +0.44 | +0.24 | +0.49 | +0.96 | +0.69 | +1.09 | +1.00 | +1.01 |
| onehot | h3 | +1.00 | -0.27 | +0.72 | +0.41 | -0.14 | +0.53 | +0.62 | -0.02 | +0.55 | +0.33 | +0.17 | +0.67 | +0.72 | +0.52 | +1.14 | +0.98 | +1.01 |
| coord | emb | +1.00 | +0.10 | +0.89 | +0.10 | -0.08 | +0.96 | +0.57 | +0.15 | +0.85 | +0.03 | +0.30 | +0.93 | +0.28 | +0.28 | +1.32 | +0.25 | +0.25 |
| coord | h1 | +1.00 | +0.26 | +0.70 | +0.21 | -0.16 | +0.89 | +0.59 | +0.36 | +0.60 | +0.09 | +0.53 | +0.75 | +0.36 | +0.34 | +1.38 | +0.25 | +0.27 |
| coord | h2 | +1.00 | +0.34 | +0.52 | +0.29 | -0.19 | +0.82 | +0.58 | +0.49 | +0.41 | +0.12 | +0.65 | +0.56 | +0.44 | +0.35 | +1.51 | +0.27 | +0.28 |
| coord | h3 | +1.00 | +0.22 | +0.61 | +0.31 | -0.15 | +0.87 | +0.57 | +0.42 | +0.54 | +0.07 | +0.58 | +0.69 | +0.40 | +0.34 | +1.40 | +0.26 | +0.29 |

**portal**: RSA occupancy~spatial 0.79, occupancy~geodesic 0.85, occupancy~policy 0.42, spatial~policy 0.47
| encoding | layer | acc | β_occ | β_spatial | β_geo | β_SR | R² | ρ_policy | β_occ (3-way: occ+sp+policy) | β_sp (3-way) | β_policy (3-way) | disagree ρ_occ | disagree ρ_sp | designed pairs | same-spatial ctrl | same-occ ctrl | random-init designed | random-init same-spatial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onehot | emb | +1.00 | +0.06 | +0.02 | +0.02 | -0.08 | +0.01 | +0.06 | +0.01 | -0.01 | +0.06 | +0.01 | +0.00 | +1.00 | +0.98 | +1.00 | +0.99 | +0.99 |
| onehot | h1 | +1.00 | -0.06 | +0.11 | +0.01 | +0.10 | +0.03 | +0.27 | -0.02 | +0.04 | +0.26 | +0.04 | +0.09 | +0.92 | +0.95 | +0.92 | +1.03 | +1.01 |
| onehot | h2 | +1.00 | -0.34 | +0.39 | +0.06 | +0.42 | +0.30 | +0.60 | -0.12 | +0.33 | +0.49 | +0.13 | +0.38 | +0.81 | +1.13 | +0.81 | +1.12 | +1.05 |
| onehot | h3 | +1.00 | -0.28 | +0.40 | -0.06 | +0.58 | +0.40 | +0.67 | -0.02 | +0.29 | +0.54 | +0.23 | +0.41 | +0.35 | +1.24 | +0.35 | +1.13 | +1.08 |
| coord | emb | +1.00 | +0.12 | +0.93 | +0.05 | -0.13 | +0.96 | +0.44 | +0.06 | +0.94 | -0.03 | +0.55 | +0.96 | +2.55 | +2.40 | +2.55 | +2.24 | +2.40 |
| coord | h1 | +1.00 | +0.15 | +0.88 | +0.09 | -0.22 | +0.89 | +0.41 | +0.06 | +0.90 | -0.04 | +0.52 | +0.90 | +2.69 | +2.35 | +2.69 | +2.08 | +2.27 |
| coord | h2 | +1.00 | -0.02 | +1.02 | +0.03 | -0.19 | +0.83 | +0.41 | -0.12 | +1.00 | -0.01 | +0.38 | +0.88 | +2.81 | +2.22 | +2.81 | +2.06 | +2.24 |
| coord | h3 | +1.00 | -0.16 | +1.06 | +0.04 | -0.01 | +0.93 | +0.47 | -0.16 | +1.06 | +0.04 | +0.41 | +0.96 | +2.30 | +2.15 | +2.30 | +1.91 | +2.21 |

**twins**: RSA occupancy~spatial 0.32, occupancy~geodesic 0.34, occupancy~policy 0.06, spatial~policy 0.13
| encoding | layer | acc | β_occ | β_spatial | β_geo | β_SR | R² | ρ_policy | β_occ (3-way: occ+sp+policy) | β_sp (3-way) | β_policy (3-way) | disagree ρ_occ | disagree ρ_sp | designed pairs | same-spatial ctrl | same-occ ctrl | random-init designed | random-init same-spatial |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| onehot | emb | +1.00 | +0.02 | +0.07 | -0.11 | +0.09 | +0.01 | +0.06 | +0.02 | +0.01 | +0.06 | +0.02 | +0.01 | +0.97 | +1.00 | +0.98 | +1.00 | +1.00 |
| onehot | h1 | +1.00 | +0.05 | +0.21 | -0.29 | +0.17 | +0.03 | +0.32 | +0.05 | -0.01 | +0.32 | +0.02 | -0.00 | +0.95 | +0.99 | +0.93 | +1.01 | +1.00 |
| onehot | h2 | +1.00 | +0.03 | +0.53 | -0.48 | +0.10 | +0.05 | +0.77 | -0.00 | +0.05 | +0.76 | -0.07 | +0.13 | +0.89 | +0.98 | +0.81 | +1.02 | +1.00 |
| onehot | h3 | +1.00 | +0.02 | +0.61 | -0.51 | +0.06 | +0.06 | +0.85 | -0.02 | +0.07 | +0.84 | -0.09 | +0.16 | +0.94 | +0.98 | +0.75 | +1.02 | +1.00 |
| coord | emb | +1.00 | -0.06 | +0.95 | +0.06 | -0.01 | +0.98 | +0.12 | -0.06 | +1.00 | -0.00 | -0.31 | +0.98 | +1.21 | +1.11 | +0.80 | +1.24 | +1.14 |
| coord | h1 | +1.00 | -0.06 | +0.94 | +0.08 | -0.02 | +0.97 | +0.13 | -0.06 | +1.00 | +0.00 | -0.33 | +0.98 | +1.20 | +1.10 | +0.82 | +1.23 | +1.14 |
| coord | h2 | +1.00 | -0.15 | +0.81 | +0.12 | +0.04 | +0.85 | +0.13 | -0.14 | +0.95 | +0.02 | -0.44 | +0.90 | +1.36 | +1.15 | +1.01 | +1.17 | +1.13 |
| coord | h3 | +1.00 | -0.20 | +0.61 | +0.20 | +0.08 | +0.66 | +0.12 | -0.18 | +0.84 | +0.03 | -0.52 | +0.77 | +1.46 | +1.19 | +1.15 | +1.16 | +1.13 |

## Phase 7: goal dependence (onehot, base)

| metric | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| state | +0.52 | +0.51 | +0.44 | +0.37 |
| goal | +0.48 | +0.37 | +0.46 | +0.52 |
| interaction | +0.00 | +0.13 | +0.11 | +0.11 |
| state_component_rsa_occupancy | +0.01 | +0.07 | +0.29 | +0.35 |
| state_component_rsa_spatial | +0.01 | +0.10 | +0.39 | +0.47 |
| goal_component_rsa_goal_occupancy | +0.03 | +0.41 | +0.71 | +0.76 |
| goal_component_rsa_goal_spatial | +0.01 | +0.48 | +0.82 | +0.86 |
| interaction_rsa_Q | +0.01 | +0.02 | +0.13 | +0.18 |
| interaction_rsa_action | +0.05 | +0.02 | +0.14 | +0.20 |
| interaction_decode_r2_Q | +0.00 | +0.00 | +0.25 | +0.27 |
| mean_cos_across_goals | +0.51 | +0.65 | +0.81 | +0.73 |
| state (random init) | +0.50 | +0.43 | +0.39 | +0.36 |
| goal (random init) | +0.50 | +0.43 | +0.39 | +0.36 |
| interaction (random init) | +0.00 | +0.13 | +0.22 | +0.27 |
| interaction_rsa_Q (random init) | +0.01 | +0.00 | -0.01 | -0.00 |

## Phase 8: steering (onehot, base; mean over seeds and goal pairs)

| layer | agree g2 α=0 | α=0.5 | α=1 | α=2 | agree g1 α=1 | random dir α=1 | other-goal dir α=1 | flipped by α=1 | flipped by α=2 | corr(flip α, g2 margin) | corr(flip α, predicted α*) | MAE(flip α, predicted α*) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| emb | 0.45 | 0.82 | 1.00 | 0.88 | 0.44 | 0.49 | 0.57 | 1.00 | 1.00 | +0.18 | +0.26 | 0.19 |
| h1 | 0.45 | 0.74 | 0.92 | 0.86 | 0.61 | 0.41 | 0.53 | 0.90 | 0.97 | +0.20 | +0.29 | 0.28 |
| h2 | 0.45 | 0.72 | 0.90 | 0.84 | 0.64 | 0.46 | 0.53 | 0.86 | 0.96 | +0.21 | +0.29 | 0.31 |
| h3 | 0.45 | 0.71 | 0.88 | 0.84 | 0.67 | 0.46 | 0.60 | 0.82 | 0.95 | +0.21 | +0.30 | 0.33 |

## Phase 9: stochastic transitions (slip0.2, onehot)

- RSA between the two hypotheses themselves: 0.999; between the two policy tables 0.548; RSA exact~spatial 0.926; fraction of (goal,state,action) optimal-support entries that changed vs deterministic: 0.144; train accuracy 1.000
| metric | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| rsa_occupancy | -0.00 | +0.06 | +0.27 | +0.30 |
| rsa_shortest_path | -0.00 | +0.06 | +0.27 | +0.31 |
| rsa_spatial | +0.00 | +0.08 | +0.31 | +0.36 |
| rsa_policy_stochastic | +0.04 | +0.21 | +0.74 | +0.79 |
| rsa_policy_shortest | +0.02 | +0.17 | +0.54 | +0.58 |
| partial_stoch_given_sp | -0.03 | -0.06 | -0.12 | -0.15 |
| partial_sp_given_stoch | +0.03 | +0.06 | +0.13 | +0.17 |
| partial_polstoch_given_poldet | +0.04 | +0.14 | +0.63 | +0.70 |
| partial_poldet_given_polstoch | -0.00 | +0.06 | +0.23 | +0.28 |
| decode_r2_stochastic | -1.34 | -0.23 | +0.06 | +0.23 |
| decode_r2_shortest_path | -1.33 | -0.21 | +0.08 | +0.23 |

## Phase 9: stochastic transitions (ice, onehot)

- RSA between the two hypotheses themselves: 0.906; between the two policy tables 0.434; RSA exact~spatial 0.814; fraction of (goal,state,action) optimal-support entries that changed vs deterministic: 0.173; train accuracy 1.000
| metric | emb | h1 | h2 | h3 |
|---|---|---|---|---|
| rsa_occupancy | -0.00 | +0.03 | +0.20 | +0.24 |
| rsa_shortest_path | -0.00 | +0.06 | +0.21 | +0.23 |
| rsa_spatial | -0.00 | +0.10 | +0.24 | +0.27 |
| rsa_policy_stochastic | +0.02 | +0.14 | +0.67 | +0.74 |
| rsa_policy_shortest | +0.01 | +0.11 | +0.37 | +0.40 |
| partial_stoch_given_sp | +0.00 | -0.06 | +0.04 | +0.07 |
| partial_sp_given_stoch | -0.01 | +0.07 | +0.05 | +0.04 |
| partial_polstoch_given_poldet | +0.02 | +0.11 | +0.61 | +0.69 |
| partial_poldet_given_polstoch | +0.00 | +0.05 | +0.12 | +0.12 |
| decode_r2_stochastic | -1.51 | -0.52 | -0.23 | -0.18 |
| decode_r2_shortest_path | -1.46 | -0.46 | -0.23 | -0.27 |
