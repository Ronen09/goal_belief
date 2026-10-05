# Navigate–commit: worked traces

Model: `studies/3_reward_trained_agents/navigate_commit/runs/ppo_fixed/seed0`, final checkpoint. Decoders: one shared affine decoder per block, fitted on the block's input, middle and output on the probe-fitting layouts. Token indices: 0, 1 are the station records; even indices from 2 are decision tokens, odd ones the events after them.

## 1. One clue (horizontal)

**history.** H station at (4,2) (q = 0.80), V station at (3,4) (q = 0.80)

[2] at (2,2), 12 left · [3] DOWN · [4] at (3,2), 11 left · [5] DOWN · [6] at (4,2), 10 left · [7] QUERY → **RIGHT** · [8] at (4,2), 9 left · [9] RIGHT · [10] at (4,3), 8 left · [11] UP · [12] at (3,3), 7 left

Exact: P(right) = 0.80, P(bottom) = 0.50; optimal: RIGHT. Model: UP 0.02, RIGHT 0.98.

| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |
|---|---|---|---|
| 0 | (0.52, 0.54) → (0.75, 0.55) → (0.80, 0.57) | (+0.23, +0.02) | (+0.05, +0.01) |
| 1 | (0.81, 0.50) → (0.85, 0.56) → (0.90, 0.56) | (+0.04, +0.06) | (+0.05, +0.00) |
| 2 | (0.87, 0.56) → (0.85, 0.57) → (0.89, 0.56) | (-0.01, +0.01) | (+0.04, -0.01) |
| 3 | (0.86, 0.57) → (0.85, 0.52) → (0.83, 0.53) | (-0.02, -0.04) | (-0.01, +0.00) |

Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded displacement of P(right), P(bottom)):

* L0.H2 ← token 7 (RIGHT): attention 0.78, (+0.24, -0.01)

**twin (one report flipped).** H station at (4,2) (q = 0.80), V station at (3,4) (q = 0.80)

[2] at (2,2), 12 left · [3] DOWN · [4] at (3,2), 11 left · [5] DOWN · [6] at (4,2), 10 left · [7] QUERY → **LEFT** · [8] at (4,2), 9 left · [9] RIGHT · [10] at (4,3), 8 left · [11] UP · [12] at (3,3), 7 left

Exact: P(right) = 0.20, P(bottom) = 0.50; optimal: DOWN, LEFT. Model: LEFT 1.00.

| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |
|---|---|---|---|
| 0 | (0.52, 0.54) → (0.16, 0.54) → (0.19, 0.61) | (-0.37, +0.01) | (+0.04, +0.06) |
| 1 | (0.29, 0.56) → (0.21, 0.56) → (0.21, 0.55) | (-0.08, -0.01) | (+0.00, -0.00) |
| 2 | (0.19, 0.57) → (0.18, 0.58) → (0.31, 0.51) | (-0.00, +0.01) | (+0.13, -0.07) |
| 3 | (0.30, 0.51) → (0.34, 0.51) → (0.32, 0.51) | (+0.04, -0.01) | (-0.02, -0.00) |

Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded displacement of P(right), P(bottom)):

* L0.H0 ← token 7 (LEFT): attention 0.51, (-0.08, -0.01)
* L0.H1 ← token 7 (LEFT): attention 0.85, (-0.25, -0.00)

## 2. Two clues, horizontal first

**history.** H station at (1,3) (q = 0.80), V station at (4,3) (q = 0.80)

[2] at (0,3), 12 left · [3] DOWN · [4] at (1,3), 11 left · [5] QUERY → **LEFT** · [6] at (1,3), 10 left · [7] DOWN · [8] at (2,3), 9 left · [9] DOWN · [10] at (3,3), 8 left · [11] DOWN · [12] at (4,3), 7 left · [13] QUERY → **BOTTOM** · [14] at (4,3), 6 left · [15] LEFT · [16] at (4,2), 5 left · [17] LEFT · [18] at (4,1), 4 left · [19] RIGHT · [20] at (4,2), 3 left

Exact: P(right) = 0.20, P(bottom) = 0.80; optimal: LEFT. Model: LEFT 1.00.

| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |
|---|---|---|---|
| 0 | (0.49, 0.50) → (0.29, 0.50) → (0.23, 0.57) | (-0.20, +0.00) | (-0.06, +0.07) |
| 1 | (0.25, 0.55) → (0.22, 0.60) → (0.18, 0.58) | (-0.03, +0.05) | (-0.04, -0.02) |
| 2 | (0.25, 0.60) → (0.23, 0.65) → (0.24, 0.63) | (-0.02, +0.05) | (+0.01, -0.02) |
| 3 | (0.23, 0.63) → (0.25, 0.64) → (0.26, 0.63) | (+0.02, +0.01) | (+0.01, -0.01) |

Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded displacement of P(right), P(bottom)):

* L0.H0 ← token 5 (LEFT): attention 0.17, (-0.03, -0.01)
* L0.H1 ← token 5 (LEFT): attention 0.49, (-0.16, -0.00)

**twin (one report flipped).** H station at (1,3) (q = 0.80), V station at (4,3) (q = 0.80)

[2] at (0,3), 12 left · [3] DOWN · [4] at (1,3), 11 left · [5] QUERY → **RIGHT** · [6] at (1,3), 10 left · [7] DOWN · [8] at (2,3), 9 left · [9] DOWN · [10] at (3,3), 8 left · [11] DOWN · [12] at (4,3), 7 left · [13] QUERY → **BOTTOM** · [14] at (4,3), 6 left · [15] LEFT · [16] at (4,2), 5 left · [17] LEFT · [18] at (4,1), 4 left · [19] RIGHT · [20] at (4,2), 3 left

Exact: P(right) = 0.80, P(bottom) = 0.80; optimal: RIGHT. Model: RIGHT 1.00.

| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |
|---|---|---|---|
| 0 | (0.49, 0.50) → (0.75, 0.50) → (0.78, 0.58) | (+0.25, +0.01) | (+0.03, +0.08) |
| 1 | (0.72, 0.55) → (0.79, 0.65) → (0.78, 0.62) | (+0.07, +0.10) | (-0.02, -0.03) |
| 2 | (0.76, 0.62) → (0.79, 0.66) → (0.79, 0.70) | (+0.03, +0.04) | (+0.01, +0.04) |
| 3 | (0.82, 0.69) → (0.80, 0.70) → (0.80, 0.68) | (-0.02, +0.02) | (+0.00, -0.02) |

Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded displacement of P(right), P(bottom)):

* L0.H2 ← token 5 (RIGHT): attention 0.67, (+0.23, -0.02)

## 3. Two clues, vertical first

**history.** H station at (1,1) (q = 0.80), V station at (0,2) (q = 0.80)

[2] at (3,2), 12 left · [3] UP · [4] at (2,2), 11 left · [5] UP · [6] at (1,2), 10 left · [7] UP · [8] at (0,2), 9 left · [9] QUERY → **TOP** · [10] at (0,2), 8 left · [11] DOWN · [12] at (1,2), 7 left · [13] LEFT · [14] at (1,1), 6 left · [15] QUERY → **RIGHT** · [16] at (1,1), 5 left · [17] UP · [18] at (0,1), 4 left · [19] RIGHT · [20] at (0,2), 3 left

Exact: P(right) = 0.80, P(bottom) = 0.20; optimal: RIGHT. Model: RIGHT 1.00.

| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |
|---|---|---|---|
| 0 | (0.50, 0.47) → (0.77, 0.39) → (0.78, 0.36) | (+0.27, -0.08) | (+0.01, -0.03) |
| 1 | (0.71, 0.36) → (0.76, 0.30) → (0.76, 0.28) | (+0.06, -0.06) | (-0.00, -0.02) |
| 2 | (0.70, 0.31) → (0.73, 0.27) → (0.70, 0.29) | (+0.03, -0.04) | (-0.03, +0.02) |
| 3 | (0.72, 0.29) → (0.74, 0.28) → (0.73, 0.30) | (+0.02, -0.01) | (-0.01, +0.02) |

Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded displacement of P(right), P(bottom)):

* L0.H0 ← token 9 (TOP): attention 0.09, (-0.00, -0.03)
* L0.H1 ← token 15 (RIGHT): attention 0.22, (+0.05, +0.01)
* L0.H2 ← token 15 (RIGHT): attention 0.61, (+0.18, -0.00)

**twin (one report flipped).** H station at (1,1) (q = 0.80), V station at (0,2) (q = 0.80)

[2] at (3,2), 12 left · [3] UP · [4] at (2,2), 11 left · [5] UP · [6] at (1,2), 10 left · [7] UP · [8] at (0,2), 9 left · [9] QUERY → **TOP** · [10] at (0,2), 8 left · [11] DOWN · [12] at (1,2), 7 left · [13] LEFT · [14] at (1,1), 6 left · [15] QUERY → **LEFT** · [16] at (1,1), 5 left · [17] UP · [18] at (0,1), 4 left · [19] RIGHT · [20] at (0,2), 3 left

Exact: P(right) = 0.20, P(bottom) = 0.20; optimal: LEFT. Model: LEFT 1.00.

| block | decoded P(right), P(bottom): in → after attention → out | attention adds | MLP adds |
|---|---|---|---|
| 0 | (0.50, 0.47) → (0.34, 0.41) → (0.32, 0.38) | (-0.17, -0.06) | (-0.02, -0.03) |
| 1 | (0.32, 0.40) → (0.31, 0.35) → (0.30, 0.34) | (-0.02, -0.06) | (-0.01, -0.01) |
| 2 | (0.34, 0.37) → (0.32, 0.35) → (0.33, 0.35) | (-0.02, -0.02) | (+0.02, +0.00) |
| 3 | (0.32, 0.34) → (0.34, 0.33) → (0.33, 0.34) | (+0.02, -0.01) | (-0.00, +0.01) |

Heads that move a decoded marginal by more than 0.02 from a report token (attention weight; decoded displacement of P(right), P(bottom)):

* L0.H0 ← token 15 (LEFT): attention 0.47, (-0.08, -0.02)
* L0.H1 ← token 15 (LEFT): attention 0.20, (-0.04, +0.00)
* L0.H2 ← token 15 (LEFT): attention 0.18, (-0.05, +0.01)
