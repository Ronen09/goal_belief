# TASK2 auto-generated tables

## Task 1-2
- switch env: 32 states, 10 goals, λ ∈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
- (s,g) pairs with at least one policy boundary: 56 of 320
- action flips per λ step: 10, 0, 8, 0, 11, 7, 4, 2, 2, 2, 2, 1, 2, 0, 1, 2, 1, 1, 0
- ‖ΔZ_sa‖ per λ step: 0.45, 0.45, 0.43, 0.42, 0.42, 0.38, 0.34, 0.29, 0.25, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.08

## Task 3
- BC accuracy by λ: 1.000, 1.000, 1.000, 0.999, 1.000, 1.000, 0.998, 0.999, 1.000, 0.999, 1.000, 1.000, 1.000, 0.999, 0.997, 1.000, 0.995, 0.995, 0.999, 0.994

## Task 4: representational change across λ (pooled over λ steps and seeds)

| layer | measure | flipped (s,g) | same state, other goal | other (s,g) | β d_occ | β d_SR | β d_margin | β flip | R² | ρ(d_h, d_occ) unflipped only | mean change flipped / unflipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h1 | chain | 1.47 | 1.27 | 1.11 | -0.01 | +0.33 | +0.11 | +0.05 | 0.17 | +0.27 | 1.61 / 1.12 |
| h1 | procrustes | 0.21 | 0.21 | 0.20 | +0.15 | +0.20 | -0.11 | +0.05 | 0.07 | +0.18 | 0.23 / 0.20 |
| h2 | chain | 1.10 | 0.88 | 0.86 | +0.06 | +0.18 | +0.16 | +0.03 | 0.13 | +0.30 | 1.19 / 0.86 |
| h2 | procrustes | 0.32 | 0.31 | 0.31 | +0.19 | +0.21 | -0.13 | +0.05 | 0.08 | +0.21 | 0.37 / 0.31 |
| h3 | chain | 1.42 | 0.91 | 0.89 | +0.14 | +0.08 | +0.23 | +0.04 | 0.18 | +0.37 | 1.57 / 0.89 |
| h3 | procrustes | 0.42 | 0.37 | 0.36 | +0.21 | +0.22 | -0.08 | +0.06 | 0.11 | +0.28 | 0.53 / 0.36 |

Per-λ-step aggregate (h2): λ midpoint, chain change/floor, Procrustes change/floor, RDM change/floor, ‖ΔZ_sa‖, flips

- λ=0.025: chain 1.11, proc 0.36, rdm 0.08, ΔZ 0.45, flips 10
- λ=0.075: chain 1.09, proc 0.35, rdm 0.08, ΔZ 0.45, flips 0
- λ=0.125: chain 0.83, proc 0.35, rdm 0.08, ΔZ 0.43, flips 8
- λ=0.175: chain 0.96, proc 0.34, rdm 0.09, ΔZ 0.42, flips 0
- λ=0.225: chain 0.89, proc 0.36, rdm 0.10, ΔZ 0.42, flips 11
- λ=0.275: chain 0.88, proc 0.36, rdm 0.09, ΔZ 0.38, flips 7
- λ=0.325: chain 0.94, proc 0.32, rdm 0.06, ΔZ 0.34, flips 4
- λ=0.375: chain 0.91, proc 0.31, rdm 0.05, ΔZ 0.29, flips 2
- λ=0.425: chain 0.80, proc 0.31, rdm 0.05, ΔZ 0.25, flips 2
- λ=0.475: chain 0.93, proc 0.28, rdm 0.04, ΔZ 0.18, flips 2
- λ=0.525: chain 0.81, proc 0.27, rdm 0.04, ΔZ 0.15, flips 2
- λ=0.575: chain 0.83, proc 0.27, rdm 0.04, ΔZ 0.13, flips 1
- λ=0.625: chain 0.77, proc 0.27, rdm 0.05, ΔZ 0.12, flips 2
- λ=0.675: chain 0.66, proc 0.27, rdm 0.05, ΔZ 0.11, flips 0
- λ=0.725: chain 0.76, proc 0.27, rdm 0.05, ΔZ 0.10, flips 1
- λ=0.775: chain 0.80, proc 0.27, rdm 0.04, ΔZ 0.09, flips 2
- λ=0.825: chain 0.72, proc 0.27, rdm 0.05, ΔZ 0.08, flips 1
- λ=0.875: chain 0.79, proc 0.28, rdm 0.04, ΔZ 0.08, flips 1
- λ=0.925: chain 0.89, proc 0.31, rdm 0.05, ΔZ 0.08, flips 0

## Task 5: quotient pairs (hidden distance / median pairwise distance)

| env | n_A | n_B | d_occ A | d_occ B | layer | d_h A | d_h B | d_h A random | d_h B random | frac(A<B) |
|---|---|---|---|---|---|---|---|---|---|---|
| switch0.0 | 12 | 46 | 1.85 | 0.72 | h1 | 0.94 | 1.01 | 1.00 | 0.98 | 0.62 |
| switch0.0 | 12 | 46 | 1.85 | 0.72 | h2 | 0.74 | 0.97 | 1.01 | 0.97 | 0.74 |
| switch0.0 | 12 | 46 | 1.85 | 0.72 | h3 | 0.64 | 0.81 | 0.98 | 0.99 | 0.61 |
| switch0.2 | 15 | 44 | 1.86 | 0.72 | h1 | 0.93 | 1.03 | 0.98 | 0.98 | 0.71 |
| switch0.2 | 15 | 44 | 1.86 | 0.72 | h2 | 0.75 | 0.98 | 0.96 | 0.97 | 0.79 |
| switch0.2 | 15 | 44 | 1.86 | 0.72 | h3 | 0.62 | 0.87 | 1.00 | 0.98 | 0.69 |
| switch0.5 | 6 | 52 | 1.81 | 0.58 | h1 | 0.91 | 1.00 | 0.95 | 0.99 | 0.71 |
| switch0.5 | 6 | 52 | 1.81 | 0.58 | h2 | 0.76 | 0.96 | 0.94 | 0.97 | 0.78 |
| switch0.5 | 6 | 52 | 1.81 | 0.58 | h3 | 0.59 | 0.87 | 1.04 | 0.98 | 0.71 |
| switch0.9 | 6 | 52 | 1.86 | 0.49 | h1 | 0.94 | 0.98 | 0.95 | 0.99 | 0.56 |
| switch0.9 | 6 | 52 | 1.86 | 0.49 | h2 | 0.73 | 0.92 | 0.93 | 0.97 | 0.76 |
| switch0.9 | 6 | 52 | 1.86 | 0.49 | h3 | 0.55 | 0.82 | 1.03 | 0.98 | 0.74 |

## Task 6: six-way RDM regression (standardised β) and partials

| env | layer | β occ | β SR | β policy | β advantage | β margin | β spatial | R² | R² without policy | partial policy|rest | partial occ|rest | partial margin|rest | partial adv|rest | RSA policy | RSA occ | RSA margin | RSA advantage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch0.0 | h1 | -0.18 | -0.03 | -0.10 | +0.70 | -0.26 | +0.16 | 0.22 | 0.22 | -0.06 | -0.05 | -0.11 | +0.27 | +0.34 | +0.15 | +0.21 | +0.41 |
| switch0.0 | h2 | +0.17 | -0.08 | +0.36 | +0.61 | -0.20 | -0.12 | 0.66 | 0.62 | +0.30 | +0.08 | -0.13 | +0.34 | +0.76 | +0.29 | +0.35 | +0.72 |
| switch0.0 | h3 | +0.49 | -0.16 | +0.61 | +0.25 | +0.00 | -0.31 | 0.76 | 0.66 | +0.55 | +0.26 | +0.00 | +0.18 | +0.83 | +0.37 | +0.42 | +0.75 |
| switch0.2 | h1 | +0.04 | -0.19 | +0.09 | +0.42 | -0.24 | +0.16 | 0.18 | 0.17 | +0.07 | +0.02 | -0.13 | +0.21 | +0.31 | +0.13 | +0.14 | +0.35 |
| switch0.2 | h2 | +0.36 | -0.26 | +0.48 | +0.51 | -0.22 | -0.13 | 0.66 | 0.55 | +0.49 | +0.23 | -0.19 | +0.38 | +0.73 | +0.28 | +0.25 | +0.64 |
| switch0.2 | h3 | +0.43 | -0.23 | +0.52 | +0.50 | -0.08 | -0.29 | 0.75 | 0.62 | +0.58 | +0.31 | -0.08 | +0.43 | +0.77 | +0.34 | +0.33 | +0.72 |
| switch0.5 | h1 | +0.15 | -0.17 | +0.16 | +0.37 | -0.18 | +0.03 | 0.19 | 0.17 | +0.15 | +0.08 | -0.11 | +0.25 | +0.28 | +0.16 | +0.15 | +0.36 |
| switch0.5 | h2 | +0.29 | -0.24 | +0.46 | +0.51 | -0.19 | -0.07 | 0.60 | 0.43 | +0.52 | +0.20 | -0.17 | +0.44 | +0.61 | +0.31 | +0.27 | +0.60 |
| switch0.5 | h3 | +0.21 | -0.16 | +0.50 | +0.57 | -0.04 | -0.16 | 0.72 | 0.53 | +0.63 | +0.18 | -0.04 | +0.56 | +0.66 | +0.36 | +0.38 | +0.71 |
| switch0.9 | h1 | +0.04 | -0.09 | +0.15 | +0.49 | -0.23 | +0.07 | 0.24 | 0.22 | +0.15 | +0.02 | -0.15 | +0.32 | +0.30 | +0.21 | +0.15 | +0.41 |
| switch0.9 | h2 | +0.11 | -0.10 | +0.40 | +0.59 | -0.10 | -0.08 | 0.58 | 0.45 | +0.47 | +0.07 | -0.09 | +0.48 | +0.56 | +0.37 | +0.33 | +0.65 |
| switch0.9 | h3 | +0.16 | -0.07 | +0.42 | +0.56 | +0.10 | -0.26 | 0.65 | 0.52 | +0.53 | +0.11 | +0.10 | +0.50 | +0.56 | +0.43 | +0.44 | +0.71 |

Hypothesis inter-correlations (RSA) per env:
- switch0.0: occupancy~policy 0.24, occupancy~value 1.00, occupancy~spatial 0.93, SR~occupancy 0.85, SR~policy -0.07, SR~advantage 0.34, SR~margin 0.68, SR~value 0.87, SR~spatial 0.79, policy~value 0.21, policy~spatial 0.28, advantage~occupancy 0.64, advantage~policy 0.72, advantage~margin 0.77, advantage~value 0.63, advantage~spatial 0.64, margin~occupancy 0.86, margin~policy 0.27, margin~value 0.86, margin~spatial 0.81, spatial~value 0.95
- switch0.2: occupancy~policy 0.24, occupancy~value 0.99, occupancy~spatial 0.90, SR~occupancy 0.83, SR~policy 0.20, SR~advantage 0.36, SR~margin 0.45, SR~value 0.84, SR~spatial 0.79, policy~value 0.22, policy~spatial 0.31, advantage~occupancy 0.57, advantage~policy 0.58, advantage~margin 0.74, advantage~value 0.58, advantage~spatial 0.63, margin~occupancy 0.66, margin~policy 0.15, margin~value 0.69, margin~spatial 0.73, spatial~value 0.93
- switch0.5: occupancy~policy 0.23, occupancy~value 0.99, occupancy~spatial 0.83, SR~occupancy 0.79, SR~policy 0.25, SR~advantage 0.29, SR~margin 0.41, SR~value 0.79, SR~spatial 0.77, policy~value 0.22, policy~spatial 0.32, advantage~occupancy 0.54, advantage~policy 0.36, advantage~margin 0.70, advantage~value 0.56, advantage~spatial 0.54, margin~occupancy 0.66, margin~policy 0.11, margin~value 0.68, margin~spatial 0.67, spatial~value 0.86
- switch0.9: occupancy~policy 0.18, occupancy~value 0.99, occupancy~spatial 0.79, SR~occupancy 0.78, SR~policy 0.24, SR~advantage 0.38, SR~margin 0.50, SR~value 0.78, SR~spatial 0.76, policy~value 0.17, policy~spatial 0.29, advantage~occupancy 0.69, advantage~policy 0.33, advantage~margin 0.69, advantage~value 0.68, advantage~spatial 0.55, margin~occupancy 0.75, margin~policy 0.10, margin~value 0.77, margin~spatial 0.68, spatial~value 0.81

## Task 7: interaction ablation

| env | layer | interaction var | k | additive var inside k-subspace | condition | accuracy | goal sensitivity | steering success |
|---|---|---|---|---|---|---|---|---|

## Task 8: what predicts the steering flip threshold α*?

| env | layer | n | never flipped | ρ α_lin | ρ logit margin | ρ m1 | ρ m2 | ρ ‖Δρ‖ | ρ occ model | MAE α_lin | MAE occ model | regression R² | β (α_lin, logit, m1, m2, dρ, occ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Task 9: tracking π*_λ versus π*_0 across the sweep (h2)

| λ | RSA(policy λ, policy 0) | RSA(occ λ, occ 0) | h2 RSA policy λ | h2 RSA policy 0 | partial λ|0 | partial 0|λ | h2 RSA occ λ | h2 RSA occ 0 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 0.78 | 0.78 | +0.05 | +0.05 | 0.29 | 0.29 |
| 0.05 | 0.99 | 1.00 | 0.77 | 0.77 | +0.06 | +0.11 | 0.29 | 0.29 |
| 0.10 | 0.99 | 0.99 | 0.77 | 0.77 | +0.06 | +0.11 | 0.29 | 0.28 |
| 0.15 | 0.98 | 0.98 | 0.76 | 0.76 | +0.14 | +0.12 | 0.28 | 0.28 |
| 0.20 | 0.98 | 0.97 | 0.74 | 0.74 | +0.15 | +0.10 | 0.28 | 0.28 |
| 0.25 | 0.94 | 0.96 | 0.69 | 0.69 | +0.17 | +0.16 | 0.27 | 0.26 |
| 0.30 | 0.92 | 0.95 | 0.68 | 0.68 | +0.19 | +0.19 | 0.26 | 0.26 |
| 0.35 | 0.90 | 0.94 | 0.66 | 0.66 | +0.20 | +0.20 | 0.26 | 0.26 |
| 0.40 | 0.90 | 0.93 | 0.64 | 0.64 | +0.20 | +0.19 | 0.28 | 0.27 |
| 0.45 | 0.89 | 0.92 | 0.63 | 0.62 | +0.21 | +0.17 | 0.28 | 0.27 |
| 0.50 | 0.87 | 0.91 | 0.62 | 0.61 | +0.24 | +0.18 | 0.30 | 0.28 |
| 0.55 | 0.86 | 0.91 | 0.61 | 0.60 | +0.22 | +0.20 | 0.31 | 0.29 |
| 0.60 | 0.85 | 0.90 | 0.60 | 0.59 | +0.23 | +0.19 | 0.31 | 0.29 |
| 0.65 | 0.84 | 0.90 | 0.60 | 0.60 | +0.22 | +0.22 | 0.33 | 0.30 |
| 0.70 | 0.84 | 0.89 | 0.58 | 0.58 | +0.22 | +0.20 | 0.34 | 0.30 |
| 0.75 | 0.83 | 0.89 | 0.57 | 0.57 | +0.22 | +0.20 | 0.35 | 0.31 |
| 0.80 | 0.83 | 0.89 | 0.58 | 0.57 | +0.23 | +0.20 | 0.36 | 0.31 |
| 0.85 | 0.83 | 0.88 | 0.56 | 0.55 | +0.22 | +0.19 | 0.37 | 0.31 |
| 0.90 | 0.82 | 0.88 | 0.54 | 0.54 | +0.20 | +0.21 | 0.37 | 0.31 |
| 0.95 | 0.82 | 0.88 | 0.53 | 0.52 | +0.20 | +0.19 | 0.37 | 0.30 |

## Task 10: dimensionality

- Spearman across environments of PR(h3) with PR(Z_sa) -1.00, with PR(policy) +1.00, with PR(SR) -1.00

| env | states | PR Z_sa | PR policy | PR SR | PR h1 | PR h2 | PR h3 | PR h3 random | 2NN Z_sa | 2NN policy | 2NN h3 | 2NN h3 random | PCs90 Z_sa | PCs90 h3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| switch0.0 | 32 | 2.01 | 2.37 | 8.78 | 14.68 | 6.83 | 2.54 | 12.42 | 5.09 | 1.88 | 7.14 | 15.91 | 2 | 6 |
| switch0.2 | 32 | 1.86 | 2.52 | 8.12 | 14.67 | 6.88 | 2.69 | 12.42 | 4.03 | 17.61 | 6.49 | 15.91 | 2 | 7 |
| switch0.5 | 32 | 1.73 | 3.01 | 7.60 | 14.67 | 7.80 | 3.17 | 12.42 | 3.80 | 3.26 | 7.16 | 15.91 | 2 | 7 |
| switch0.9 | 32 | 1.72 | 3.20 | 7.31 | 14.44 | 7.54 | 3.32 | 12.42 | 4.06 | 3.12 | 6.93 | 15.91 | 2 | 7 |
