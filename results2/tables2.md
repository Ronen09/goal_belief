# TASK2 auto-generated tables

## Task 1-2
- switch env: 32 states, 10 goals, λ ∈ [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
- (s,g) pairs with at least one policy boundary: 56 of 320
- action flips per λ step: 10, 0, 8, 0, 11, 7, 4, 2, 2, 2, 2, 1, 2, 0, 1, 2, 1, 1, 0
- ‖ΔZ_sa‖ per λ step: 0.45, 0.45, 0.43, 0.42, 0.42, 0.38, 0.34, 0.29, 0.25, 0.18, 0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.08

## Task 3
- BC accuracy by λ: 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000

## Task 4: representational change across λ (pooled over λ steps and seeds)

| layer | measure | flipped (s,g) | same state, other goal | other (s,g) | β d_occ | β d_SR | β d_margin | β flip | R² | ρ(d_h, d_occ) unflipped only | mean change flipped / unflipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| h1 | chain | 2.67 | 1.38 | 1.01 | -0.14 | +0.11 | +0.06 | +0.12 | 0.03 | -0.03 | 2.55 / 1.03 |
| h1 | procrustes | 0.45 | 0.25 | 0.15 | -0.07 | +0.15 | -0.00 | +0.16 | 0.04 | +0.02 | 0.46 / 0.15 |
| h2 | chain | 2.68 | 1.01 | 0.91 | -0.16 | +0.05 | +0.10 | +0.12 | 0.02 | -0.06 | 2.41 / 0.91 |
| h2 | procrustes | 1.30 | 0.34 | 0.19 | -0.09 | +0.14 | +0.01 | +0.17 | 0.04 | -0.00 | 1.31 / 0.20 |
| h3 | chain | 3.50 | 1.10 | 0.97 | -0.16 | +0.05 | +0.12 | +0.11 | 0.02 | -0.04 | 3.10 / 0.97 |
| h3 | procrustes | 2.11 | 0.36 | 0.21 | -0.13 | +0.11 | +0.03 | +0.17 | 0.04 | -0.04 | 2.07 / 0.22 |

Per-λ-step aggregate (h2): λ midpoint, chain change/floor, Procrustes change/floor, RDM change/floor, ‖ΔZ_sa‖, flips

- λ=0.025: chain 1.83, proc 0.32, rdm 0.09, ΔZ 0.45, flips 10
- λ=0.075: chain 1.00, proc 0.00, rdm 0.00, ΔZ 0.45, flips 0
- λ=0.125: chain 0.28, proc 0.43, rdm 0.10, ΔZ 0.43, flips 8
- λ=0.175: chain 1.00, proc 0.00, rdm 0.00, ΔZ 0.42, flips 0
- λ=0.225: chain 0.46, proc 0.57, rdm 0.37, ΔZ 0.42, flips 11
- λ=0.275: chain 1.34, proc 0.38, rdm 0.13, ΔZ 0.38, flips 7
- λ=0.325: chain 0.83, proc 0.30, rdm 0.04, ΔZ 0.34, flips 4
- λ=0.375: chain 0.70, proc 0.21, rdm 0.01, ΔZ 0.29, flips 2
- λ=0.425: chain 0.76, proc 0.24, rdm 0.02, ΔZ 0.25, flips 2
- λ=0.475: chain 0.77, proc 0.21, rdm 0.02, ΔZ 0.18, flips 2
- λ=0.525: chain 0.79, proc 0.21, rdm 0.02, ΔZ 0.15, flips 2
- λ=0.575: chain 0.61, proc 0.16, rdm 0.01, ΔZ 0.13, flips 1
- λ=0.625: chain 0.73, proc 0.24, rdm 0.04, ΔZ 0.12, flips 2
- λ=0.675: chain 1.00, proc 0.00, rdm 0.00, ΔZ 0.11, flips 0
- λ=0.725: chain 0.79, proc 0.17, rdm 0.01, ΔZ 0.10, flips 1
- λ=0.775: chain 0.92, proc 0.22, rdm 0.02, ΔZ 0.09, flips 2
- λ=0.825: chain 0.90, proc 0.15, rdm 0.01, ΔZ 0.08, flips 1
- λ=0.875: chain 0.70, proc 0.20, rdm 0.04, ΔZ 0.08, flips 1
- λ=0.925: chain 1.00, proc 0.00, rdm 0.00, ΔZ 0.08, flips 0

## Task 5: quotient pairs (hidden distance / median pairwise distance)

| env | n_A | n_B | d_occ A | d_occ B | layer | d_h A | d_h B | d_h A random | d_h B random | frac(A<B) |
|---|---|---|---|---|---|---|---|---|---|---|
| corridor | 18 | 21 | 1.01 | 0.67 | h1 | 0.95 | 1.00 | 1.02 | 0.99 | 0.60 |
| corridor | 18 | 21 | 1.01 | 0.67 | h2 | 0.60 | 1.18 | 1.02 | 0.98 | 0.99 |
| corridor | 18 | 21 | 1.01 | 0.67 | h3 | 0.27 | 1.21 | 1.03 | 0.96 | 1.00 |
| twins | 33 | 382 | 1.89 | 0.46 | h1 | 0.95 | 0.98 | 1.00 | 1.01 | 0.54 |
| twins | 33 | 382 | 1.89 | 0.46 | h2 | 0.68 | 0.97 | 0.99 | 1.01 | 0.79 |
| twins | 33 | 382 | 1.89 | 0.46 | h3 | 0.38 | 0.96 | 0.96 | 1.02 | 0.91 |
| switch0.0 | 12 | 46 | 1.85 | 0.72 | h1 | 0.94 | 1.00 | 1.00 | 0.98 | 0.61 |
| switch0.0 | 12 | 46 | 1.85 | 0.72 | h2 | 0.60 | 0.97 | 1.01 | 0.97 | 0.72 |
| switch0.0 | 12 | 46 | 1.85 | 0.72 | h3 | 0.45 | 0.98 | 0.98 | 0.99 | 0.72 |
| switch0.2 | 15 | 44 | 1.86 | 0.72 | h1 | 0.89 | 1.03 | 0.98 | 0.98 | 0.80 |
| switch0.2 | 15 | 44 | 1.86 | 0.72 | h2 | 0.53 | 1.05 | 0.96 | 0.97 | 0.89 |
| switch0.2 | 15 | 44 | 1.86 | 0.72 | h3 | 0.35 | 1.08 | 1.00 | 0.98 | 0.85 |
| switch0.5 | 6 | 52 | 1.81 | 0.58 | h1 | 0.87 | 1.02 | 0.95 | 0.99 | 0.77 |
| switch0.5 | 6 | 52 | 1.81 | 0.58 | h2 | 0.57 | 1.04 | 0.94 | 0.97 | 0.85 |
| switch0.5 | 6 | 52 | 1.81 | 0.58 | h3 | 0.41 | 1.04 | 1.04 | 0.98 | 0.87 |
| switch0.9 | 6 | 52 | 1.86 | 0.49 | h1 | 0.87 | 1.02 | 0.95 | 0.99 | 0.73 |
| switch0.9 | 6 | 52 | 1.86 | 0.49 | h2 | 0.53 | 1.03 | 0.93 | 0.97 | 0.86 |
| switch0.9 | 6 | 52 | 1.86 | 0.49 | h3 | 0.36 | 1.03 | 1.03 | 0.98 | 0.87 |

## Task 6: six-way RDM regression (standardised β) and partials

| env | layer | β occ | β SR | β policy | β advantage | β margin | β spatial | R² | R² without policy | partial policy|rest | partial occ|rest | partial margin|rest | partial adv|rest | RSA policy | RSA occ | RSA margin | RSA advantage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | h1 | +0.07 | +0.07 | +0.33 | -0.12 | -0.07 | -0.07 | 0.13 | 0.06 | +0.27 | +0.03 | -0.05 | -0.08 | +0.31 | +0.08 | -0.17 | +0.01 |
| base | h2 | -0.24 | +0.07 | +0.69 | -0.22 | -0.03 | +0.42 | 0.59 | 0.28 | +0.65 | -0.13 | -0.04 | -0.19 | +0.72 | +0.29 | -0.22 | +0.22 |
| base | h3 | -0.35 | -0.02 | +0.72 | -0.21 | +0.01 | +0.66 | 0.68 | 0.34 | +0.71 | -0.20 | +0.01 | -0.21 | +0.78 | +0.35 | -0.16 | +0.31 |
| twins | h1 | +0.09 | +0.09 | +0.26 | +0.03 | -0.14 | -0.06 | 0.14 | 0.10 | +0.20 | +0.07 | -0.11 | +0.02 | +0.32 | +0.06 | -0.18 | +0.12 |
| twins | h2 | -0.02 | +0.03 | +0.76 | +0.01 | +0.02 | +0.04 | 0.60 | 0.29 | +0.66 | -0.02 | +0.02 | +0.01 | +0.77 | +0.06 | -0.21 | +0.34 |
| twins | h3 | -0.07 | +0.01 | +0.84 | +0.04 | +0.06 | +0.07 | 0.73 | 0.36 | +0.76 | -0.10 | +0.09 | +0.05 | +0.85 | +0.06 | -0.19 | +0.41 |
| portal | h1 | +0.01 | -0.05 | +0.27 | -0.01 | -0.15 | +0.02 | 0.11 | 0.07 | +0.20 | +0.01 | -0.14 | -0.01 | +0.27 | +0.12 | -0.17 | +0.10 |
| portal | h2 | -0.23 | +0.20 | +0.49 | -0.05 | -0.16 | +0.28 | 0.48 | 0.35 | +0.44 | -0.16 | -0.18 | -0.05 | +0.60 | +0.35 | -0.30 | +0.24 |
| portal | h3 | -0.23 | +0.36 | +0.55 | -0.13 | +0.00 | +0.27 | 0.59 | 0.43 | +0.53 | -0.18 | +0.00 | -0.13 | +0.67 | +0.43 | -0.22 | +0.25 |
| barrier | h1 | +0.21 | +0.11 | +0.39 | -0.26 | +0.03 | -0.09 | 0.17 | 0.10 | +0.28 | +0.09 | +0.03 | -0.12 | +0.36 | +0.21 | -0.09 | +0.14 |
| barrier | h2 | +0.27 | -0.16 | +0.50 | -0.12 | +0.01 | +0.37 | 0.57 | 0.46 | +0.45 | +0.17 | +0.01 | -0.08 | +0.67 | +0.55 | +0.01 | +0.53 |
| barrier | h3 | +0.12 | -0.25 | +0.39 | -0.07 | +0.02 | +0.66 | 0.60 | 0.53 | +0.38 | +0.08 | +0.03 | -0.05 | +0.62 | +0.52 | +0.06 | +0.53 |
| corridor | h1 | -0.10 | -0.19 | +0.14 | -0.09 | +0.08 | +0.43 | 0.13 | 0.12 | +0.07 | -0.02 | +0.04 | -0.04 | +0.25 | +0.22 | +0.17 | +0.21 |
| corridor | h2 | +0.05 | -0.51 | +0.56 | +0.14 | -0.14 | +0.61 | 0.68 | 0.61 | +0.40 | +0.01 | -0.10 | +0.09 | +0.78 | +0.48 | +0.32 | +0.65 |
| corridor | h3 | +0.18 | -0.52 | +0.61 | +0.17 | -0.17 | +0.48 | 0.72 | 0.65 | +0.46 | +0.06 | -0.15 | +0.13 | +0.82 | +0.51 | +0.34 | +0.68 |
| switch0.0 | h1 | +0.11 | -0.08 | +0.24 | +0.29 | -0.18 | -0.02 | 0.22 | 0.19 | +0.14 | +0.04 | -0.07 | +0.11 | +0.43 | +0.12 | +0.13 | +0.36 |
| switch0.0 | h2 | +0.10 | +0.04 | +0.86 | +0.01 | +0.01 | -0.16 | 0.74 | 0.53 | +0.66 | +0.05 | +0.01 | +0.01 | +0.85 | +0.20 | +0.23 | +0.62 |
| switch0.0 | h3 | +0.18 | +0.05 | +0.94 | -0.13 | +0.06 | -0.23 | 0.71 | 0.47 | +0.68 | +0.09 | +0.04 | -0.09 | +0.84 | +0.19 | +0.21 | +0.57 |
| switch0.2 | h1 | +0.03 | -0.05 | +0.31 | +0.22 | -0.22 | +0.07 | 0.22 | 0.17 | +0.23 | +0.02 | -0.12 | +0.12 | +0.42 | +0.11 | +0.04 | +0.28 |
| switch0.2 | h2 | +0.08 | -0.07 | +0.77 | +0.16 | -0.16 | +0.01 | 0.74 | 0.46 | +0.72 | +0.06 | -0.16 | +0.15 | +0.85 | +0.19 | +0.09 | +0.51 |
| switch0.2 | h3 | +0.05 | -0.07 | +0.77 | +0.14 | -0.16 | +0.04 | 0.72 | 0.44 | +0.71 | +0.04 | -0.15 | +0.13 | +0.84 | +0.19 | +0.09 | +0.50 |
| switch0.5 | h1 | +0.13 | -0.13 | +0.40 | +0.09 | -0.19 | +0.07 | 0.22 | 0.10 | +0.36 | +0.07 | -0.12 | +0.07 | +0.43 | +0.10 | -0.01 | +0.17 |
| switch0.5 | h2 | +0.10 | -0.09 | +0.79 | +0.13 | -0.12 | -0.03 | 0.69 | 0.23 | +0.78 | +0.08 | -0.12 | +0.14 | +0.81 | +0.18 | +0.06 | +0.34 |
| switch0.5 | h3 | +0.08 | -0.10 | +0.82 | +0.08 | -0.06 | -0.02 | 0.71 | 0.21 | +0.79 | +0.07 | -0.07 | +0.10 | +0.83 | +0.18 | +0.08 | +0.34 |
| switch0.9 | h1 | +0.04 | -0.06 | +0.39 | +0.16 | -0.08 | -0.04 | 0.22 | 0.09 | +0.36 | +0.02 | -0.05 | +0.11 | +0.42 | +0.09 | +0.05 | +0.22 |
| switch0.9 | h2 | +0.07 | -0.03 | +0.81 | +0.12 | -0.04 | -0.10 | 0.70 | 0.21 | +0.79 | +0.05 | -0.04 | +0.13 | +0.82 | +0.16 | +0.09 | +0.34 |
| switch0.9 | h3 | +0.11 | -0.05 | +0.85 | +0.03 | +0.01 | -0.12 | 0.72 | 0.17 | +0.81 | +0.08 | +0.01 | +0.04 | +0.83 | +0.16 | +0.10 | +0.31 |

Hypothesis inter-correlations (RSA) per env:
- base: occupancy~policy 0.31, occupancy~value 0.99, occupancy~spatial 0.94, SR~occupancy 0.77, SR~policy 0.43, SR~advantage 0.38, SR~margin -0.24, SR~value 0.78, SR~spatial 0.83, policy~value 0.32, policy~spatial 0.40, advantage~occupancy 0.61, advantage~policy 0.43, advantage~margin 0.42, advantage~value 0.65, advantage~spatial 0.65, margin~occupancy 0.04, margin~policy -0.11, margin~value 0.07, margin~spatial 0.02, spatial~value 0.95
- twins: occupancy~policy 0.06, occupancy~value 0.99, occupancy~spatial 0.32, SR~occupancy 0.30, SR~policy 0.07, SR~advantage -0.15, SR~margin -0.05, SR~value 0.31, SR~spatial 0.59, policy~value 0.03, policy~spatial 0.13, advantage~occupancy 0.50, advantage~policy 0.44, advantage~margin 0.36, advantage~value 0.47, advantage~spatial 0.07, margin~occupancy 0.46, margin~policy -0.30, margin~value 0.50, margin~spatial 0.12, spatial~value 0.35
- portal: occupancy~policy 0.42, occupancy~value 0.98, occupancy~spatial 0.79, SR~occupancy 0.77, SR~policy 0.44, SR~advantage 0.13, SR~margin -0.32, SR~value 0.78, SR~spatial 0.69, policy~value 0.41, policy~spatial 0.47, advantage~occupancy 0.41, advantage~policy 0.55, advantage~margin 0.29, advantage~value 0.44, advantage~spatial 0.49, margin~occupancy -0.13, margin~policy -0.08, margin~value -0.06, margin~spatial -0.19, spatial~value 0.80
- barrier: occupancy~policy 0.48, occupancy~value 1.00, occupancy~spatial 0.70, SR~occupancy 0.76, SR~policy 0.58, SR~advantage 0.49, SR~margin -0.17, SR~value 0.75, SR~spatial 0.79, policy~value 0.47, policy~spatial 0.55, advantage~occupancy 0.79, advantage~policy 0.57, advantage~margin 0.39, advantage~value 0.80, advantage~spatial 0.61, margin~occupancy 0.08, margin~policy -0.03, margin~value 0.13, margin~spatial 0.04, spatial~value 0.71
- corridor: occupancy~policy 0.67, occupancy~value 0.99, occupancy~spatial 0.93, SR~occupancy 0.96, SR~policy 0.65, SR~advantage 0.53, SR~margin 0.62, SR~value 0.96, SR~spatial 0.91, policy~value 0.66, policy~spatial 0.78, advantage~occupancy 0.57, advantage~policy 0.79, advantage~margin 0.69, advantage~value 0.56, advantage~spatial 0.67, margin~occupancy 0.73, margin~policy 0.47, margin~value 0.70, margin~spatial 0.64, spatial~value 0.95
- switch0.0: occupancy~policy 0.24, occupancy~value 1.00, occupancy~spatial 0.93, SR~occupancy 0.85, SR~policy -0.07, SR~advantage 0.34, SR~margin 0.68, SR~value 0.87, SR~spatial 0.79, policy~value 0.21, policy~spatial 0.28, advantage~occupancy 0.64, advantage~policy 0.72, advantage~margin 0.77, advantage~value 0.63, advantage~spatial 0.64, margin~occupancy 0.86, margin~policy 0.27, margin~value 0.86, margin~spatial 0.81, spatial~value 0.95
- switch0.2: occupancy~policy 0.24, occupancy~value 0.99, occupancy~spatial 0.90, SR~occupancy 0.83, SR~policy 0.20, SR~advantage 0.36, SR~margin 0.45, SR~value 0.84, SR~spatial 0.79, policy~value 0.22, policy~spatial 0.31, advantage~occupancy 0.57, advantage~policy 0.58, advantage~margin 0.74, advantage~value 0.58, advantage~spatial 0.63, margin~occupancy 0.66, margin~policy 0.15, margin~value 0.69, margin~spatial 0.73, spatial~value 0.93
- switch0.5: occupancy~policy 0.23, occupancy~value 0.99, occupancy~spatial 0.83, SR~occupancy 0.79, SR~policy 0.25, SR~advantage 0.29, SR~margin 0.41, SR~value 0.79, SR~spatial 0.77, policy~value 0.22, policy~spatial 0.32, advantage~occupancy 0.54, advantage~policy 0.36, advantage~margin 0.70, advantage~value 0.56, advantage~spatial 0.54, margin~occupancy 0.66, margin~policy 0.11, margin~value 0.68, margin~spatial 0.67, spatial~value 0.86
- switch0.9: occupancy~policy 0.18, occupancy~value 0.99, occupancy~spatial 0.79, SR~occupancy 0.78, SR~policy 0.24, SR~advantage 0.38, SR~margin 0.50, SR~value 0.78, SR~spatial 0.76, policy~value 0.17, policy~spatial 0.29, advantage~occupancy 0.69, advantage~policy 0.33, advantage~margin 0.69, advantage~value 0.68, advantage~spatial 0.55, margin~occupancy 0.75, margin~policy 0.10, margin~value 0.77, margin~spatial 0.68, spatial~value 0.81

## Task 7: interaction ablation

| env | layer | interaction var | k | additive var inside k-subspace | condition | accuracy | goal sensitivity | steering success |
|---|---|---|---|---|---|---|---|---|
| base | h1 | 0.12 | 89 | 0.74 | baseline | 1.000 | 0.543 | 0.919 |
| base | h1 | 0.12 | 89 | 0.74 | remove_component | 0.985 | 0.598 | 0.980 |
| base | h1 | 0.12 | 89 | 0.74 | random_component | 0.999 | 0.637 | 0.899 |
| base | h1 | 0.12 | 89 | 0.74 | remove_subspace | 0.607 | 0.391 | 0.638 |
| base | h1 | 0.12 | 89 | 0.74 | random_subspace | 0.786 | 0.533 | 0.701 |
| base | h1 | 0.12 | 89 | 0.74 | only_interaction_plus_state | 0.800 | 0.405 | 0.652 |
| base | h2 | 0.11 | 57 | 0.93 | baseline | 1.000 | 0.543 | 0.891 |
| base | h2 | 0.11 | 57 | 0.93 | remove_component | 0.975 | 0.597 | 0.965 |
| base | h2 | 0.11 | 57 | 0.93 | random_component | 1.000 | 0.642 | 0.898 |
| base | h2 | 0.11 | 57 | 0.93 | remove_subspace | 0.379 | 0.223 | 0.340 |
| base | h2 | 0.11 | 57 | 0.93 | random_subspace | 0.985 | 0.572 | 0.846 |
| base | h2 | 0.11 | 57 | 0.93 | only_interaction_plus_state | 0.834 | 0.459 | 0.552 |
| base | h3 | 0.11 | 6 | 0.97 | baseline | 1.000 | 0.543 | 0.858 |
| base | h3 | 0.11 | 6 | 0.97 | remove_component | 0.974 | 0.602 | 0.961 |
| base | h3 | 0.11 | 6 | 0.97 | random_component | 1.000 | 0.637 | 0.880 |
| base | h3 | 0.11 | 6 | 0.97 | remove_subspace | 0.356 | 0.292 | 0.379 |
| base | h3 | 0.11 | 6 | 0.97 | random_subspace | 1.000 | 0.555 | 0.882 |
| base | h3 | 0.11 | 6 | 0.97 | only_interaction_plus_state | 0.864 | 0.485 | 0.521 |
| switch0.5 | h1 | 0.10 | 67 | 0.62 | baseline | 1.000 | 0.417 | 0.621 |
| switch0.5 | h1 | 0.10 | 67 | 0.62 | remove_component | 0.951 | 0.394 | 0.861 |
| switch0.5 | h1 | 0.10 | 67 | 0.62 | random_component | 1.000 | 0.424 | 0.605 |
| switch0.5 | h1 | 0.10 | 67 | 0.62 | remove_subspace | 0.624 | 0.254 | 0.551 |
| switch0.5 | h1 | 0.10 | 67 | 0.62 | random_subspace | 0.925 | 0.440 | 0.588 |
| switch0.5 | h1 | 0.10 | 67 | 0.62 | only_interaction_plus_state | 0.820 | 0.163 | 0.404 |
| switch0.5 | h2 | 0.09 | 27 | 0.90 | baseline | 1.000 | 0.417 | 0.524 |
| switch0.5 | h2 | 0.09 | 27 | 0.90 | remove_component | 0.930 | 0.371 | 0.787 |
| switch0.5 | h2 | 0.09 | 27 | 0.90 | random_component | 1.000 | 0.424 | 0.527 |
| switch0.5 | h2 | 0.09 | 27 | 0.90 | remove_subspace | 0.148 | 0.232 | 0.177 |
| switch0.5 | h2 | 0.09 | 27 | 0.90 | random_subspace | 0.998 | 0.432 | 0.553 |
| switch0.5 | h2 | 0.09 | 27 | 0.90 | only_interaction_plus_state | 0.847 | 0.195 | 0.139 |
| switch0.5 | h3 | 0.11 | 3 | 0.95 | baseline | 1.000 | 0.417 | 0.484 |
| switch0.5 | h3 | 0.11 | 3 | 0.95 | remove_component | 0.926 | 0.360 | 0.785 |
| switch0.5 | h3 | 0.11 | 3 | 0.95 | random_component | 1.000 | 0.426 | 0.465 |
| switch0.5 | h3 | 0.11 | 3 | 0.95 | remove_subspace | 0.124 | 0.046 | 0.109 |
| switch0.5 | h3 | 0.11 | 3 | 0.95 | random_subspace | 1.000 | 0.425 | 0.467 |
| switch0.5 | h3 | 0.11 | 3 | 0.95 | only_interaction_plus_state | 0.859 | 0.210 | 0.154 |

## Task 8: what predicts the steering flip threshold α*?

| env | layer | n | never flipped | ρ α_lin | ρ logit margin | ρ m1 | ρ m2 | ρ ‖Δρ‖ | ρ occ model | MAE α_lin | MAE occ model | regression R² | β (α_lin, logit, m1, m2, dρ, occ) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | h1 | 263 | 8 | +0.97 | +0.84 | +0.42 | +0.20 | +0.13 | +0.29 | 0.32 | 0.28 | 0.94 | +0.93, +0.02, +0.01, +0.03, -0.02, +0.03 |
| base | h2 | 259 | 11 | +0.98 | +0.84 | +0.43 | +0.21 | +0.13 | +0.29 | 0.12 | 0.32 | 0.97 | +1.00, -0.04, +0.02, +0.02, -0.01, +0.01 |
| base | h3 | 257 | 14 | +1.00 | +0.85 | +0.45 | +0.20 | +0.13 | +0.31 | 0.02 | 0.34 | 1.00 | +1.00, -0.00, +0.01, +0.00, -0.00, -0.00 |
| switch0.5 | h1 | 147 | 22 | +0.92 | +0.42 | -0.11 | +0.16 | -0.02 | -0.09 | 0.24 | 0.31 | 0.85 | +0.87, +0.09, +0.09, -0.01, -0.01, -0.01 |
| switch0.5 | h2 | 139 | 29 | +0.96 | +0.52 | -0.09 | +0.17 | -0.04 | -0.07 | 0.14 | 0.36 | 0.93 | +0.95, +0.03, +0.07, -0.02, -0.01, -0.01 |
| switch0.5 | h3 | 135 | 34 | +1.00 | +0.54 | -0.06 | +0.20 | -0.06 | -0.08 | 0.03 | 0.39 | 1.00 | +1.00, -0.00, -0.01, +0.01, +0.01, +0.02 |

## Task 9: tracking π*_λ versus π*_0 across the sweep (h2)

| λ | RSA(policy λ, policy 0) | RSA(occ λ, occ 0) | h2 RSA policy λ | h2 RSA policy 0 | partial λ|0 | partial 0|λ | h2 RSA occ λ | h2 RSA occ 0 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.00 | 1.00 | 0.86 | 0.86 | +0.05 | +0.05 | 0.20 | 0.20 |
| 0.05 | 0.99 | 1.00 | 0.85 | 0.85 | +0.19 | +0.03 | 0.20 | 0.20 |
| 0.10 | 0.99 | 0.99 | 0.85 | 0.85 | +0.19 | +0.03 | 0.20 | 0.20 |
| 0.15 | 0.98 | 0.98 | 0.85 | 0.84 | +0.26 | +0.10 | 0.20 | 0.21 |
| 0.20 | 0.98 | 0.97 | 0.85 | 0.84 | +0.26 | +0.10 | 0.20 | 0.21 |
| 0.25 | 0.94 | 0.96 | 0.84 | 0.79 | +0.49 | -0.04 | 0.19 | 0.19 |
| 0.30 | 0.92 | 0.95 | 0.85 | 0.78 | +0.54 | -0.01 | 0.20 | 0.20 |
| 0.35 | 0.90 | 0.94 | 0.84 | 0.76 | +0.54 | +0.02 | 0.19 | 0.19 |
| 0.40 | 0.90 | 0.93 | 0.83 | 0.76 | +0.54 | +0.04 | 0.18 | 0.19 |
| 0.45 | 0.89 | 0.92 | 0.84 | 0.76 | +0.55 | +0.04 | 0.18 | 0.20 |
| 0.50 | 0.87 | 0.91 | 0.83 | 0.74 | +0.56 | +0.10 | 0.18 | 0.20 |
| 0.55 | 0.86 | 0.91 | 0.83 | 0.73 | +0.57 | +0.09 | 0.18 | 0.19 |
| 0.60 | 0.85 | 0.90 | 0.83 | 0.73 | +0.58 | +0.08 | 0.17 | 0.19 |
| 0.65 | 0.84 | 0.90 | 0.83 | 0.72 | +0.61 | +0.06 | 0.17 | 0.19 |
| 0.70 | 0.84 | 0.89 | 0.83 | 0.72 | +0.61 | +0.06 | 0.17 | 0.19 |
| 0.75 | 0.83 | 0.89 | 0.83 | 0.71 | +0.62 | +0.07 | 0.17 | 0.19 |
| 0.80 | 0.83 | 0.89 | 0.83 | 0.70 | +0.63 | +0.05 | 0.16 | 0.19 |
| 0.85 | 0.83 | 0.88 | 0.83 | 0.70 | +0.64 | +0.03 | 0.16 | 0.19 |
| 0.90 | 0.82 | 0.88 | 0.83 | 0.68 | +0.67 | -0.02 | 0.16 | 0.18 |
| 0.95 | 0.82 | 0.88 | 0.83 | 0.68 | +0.67 | -0.02 | 0.16 | 0.18 |

## Task 10: dimensionality

- Spearman across environments of PR(h3) with PR(Z_sa) +0.47, with PR(policy) +0.83, with PR(SR) +0.70

| env | states | PR Z_sa | PR policy | PR SR | PR h1 | PR h2 | PR h3 | PR h3 random | 2NN Z_sa | 2NN policy | 2NN h3 | 2NN h3 random | PCs90 Z_sa | PCs90 h3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 57 | 2.79 | 6.99 | 10.15 | 19.33 | 6.97 | 3.13 | 13.27 | 5.60 | 3.89 | 3.88 | 16.35 | 4 | 3 |
| twins | 69 | 2.57 | 3.73 | 17.73 | 18.64 | 5.27 | 2.98 | 14.80 | 4.53 | 5.65 | 4.09 | 15.31 | 3 | 3 |
| portal | 59 | 2.98 | 7.39 | 9.95 | 19.38 | 8.37 | 2.31 | 13.88 | 11.12 | 4.61 | 4.88 | 16.25 | 5 | 3 |
| barrier | 64 | 1.70 | 5.22 | 9.67 | 18.86 | 4.18 | 1.89 | 14.85 | 8.32 | 4.20 | 4.30 | 16.60 | 3 | 3 |
| corridor | 21 | 1.74 | 1.33 | 5.78 | 10.23 | 2.47 | 1.21 | 10.05 | 4.80 | nan | 4.70 | 16.91 | 2 | 1 |
| switch0.0 | 32 | 2.01 | 2.37 | 8.78 | 12.20 | 2.33 | 1.44 | 12.42 | 5.09 | 1.88 | 6.59 | 15.91 | 2 | 2 |
| switch0.2 | 32 | 1.86 | 2.52 | 8.12 | 12.13 | 2.29 | 1.54 | 12.42 | 4.03 | 17.61 | 5.35 | 15.91 | 2 | 2 |
| switch0.5 | 32 | 1.73 | 3.01 | 7.60 | 12.24 | 2.79 | 1.96 | 12.42 | 3.80 | 3.26 | 3.35 | 15.91 | 2 | 3 |
| switch0.9 | 32 | 1.72 | 3.20 | 7.31 | 12.17 | 3.07 | 2.17 | 12.42 | 4.06 | 3.12 | 3.21 | 15.91 | 2 | 3 |
