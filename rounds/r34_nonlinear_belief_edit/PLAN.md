# Round 34: encodings, edits, decision rule and expectations

Written before any nonlinear encoding was fitted on a trained model or any edit of this round applied to one. Code:
`run.py`. It uses round 32's machinery and round 33's site and linear encoding, unchanged. It was smoke-tested on the
untrained initial checkpoint of seed 0 only: the no-edit patch reproduces the natural run (9·10⁻⁸). Brief: `BRIEF.md`.

## Site, pairs, measures

Round 33's: m(h, g), the goal token's state after block 0's attention and before its MLP. Round 27's pairs (main;
goal-dependent 5 653; preserve; one-step agreeing; equivalent recipients 13 165 × 4) and round 32's measures.

## Encodings (no action labels; fit side, all three goals)

| encoding | form | role |
|---|---|---|
| linear | c_{g,L} + E b | round 33's shared encoding (reproduced) |
| **mlp** | c_{g,L} + f(b) | **primary**: f an MLP 14 → 256 → 256 → 128 (GELU), one for every goal; offsets learned jointly; MSE, Adam, 4 000 steps of 8 192, seeded per model |
| mlp_goal | c_{g,L} + f(b, g) | the goal as an input (secondary) |
| **table** | c_{g,L} + μ(b) | **ceiling**: f plus the mean residual for each distinct posterior (619), shrunk toward f by n / (n + 30) rows; the 44 posteriors absent from the fit side fall back to f. The most any function of the posterior alone can give |
| table_goal | c_{g,L} + μ_g(b) | per posterior and goal |

## Edits (goal token only; prefix states kept)

| kind | new state |
|---|---|
| none / whole / pca | as rounds 32–33 (PCA-13 of the donor's state) |
| linear | m(A,g) + E (b_B − b_A) |
| **mlp** | m(A,g) + f(b_B) − f(b_A): the same vector under every goal |
| mlp_goal | m(A,g) + f(b_B, g) − f(b_A, g) |
| **table** | m(A,g) + μ(b_B) − μ(b_A) |
| table_goal | m(A,g) + μ_g(b_B) − μ_g(b_A) |
| rotated | m(A,g) + Q (f(b_B) − f(b_A)), 5 random rotations |
| **residual** | m(B,g) − (μ(b_B) − μ(b_A)): the donor's actual difference without its posterior part |

The table and residual edit vectors sum to the whole difference.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests over the ten, one-sided, p < 0.05. "Goal-dependent rate": the
share of goal-dependent pairs donor-optimal under every change goal.

| | criterion |
|---|---|
| **P** (primary) | goal-dependent rate: mlp ≥ 0.75 × whole; above rotated; above linear |
| N fit | mlp R² within (goal, length) ≥ linear + 0.05 (descriptive) |
| C1 | main change cells: S(mlp) ≥ 0.75, gain ≥ 0.25 |
| C3 | harm(mlp) ≤ 0.05 |
| C4 | one-step agreeing: mlp gain ≥ 0.5 × whole's, above rotated |
| C5b | D(pca) − D(mlp) ≤ 0.05 |
| C6 | equivalent recipients, all four donor-optimal: mlp ≥ 0.75 × whole. The agreement part of rounds 32–33's C6 is reported but not a criterion: those rounds showed any partial edit lowers it mechanically |
| C7 | D(mlp) ≥ D(mlp_goal) − 0.05, and goal-dependent rate mlp ≥ mlp_goal − 0.03 |
| **T** (ceiling) | goal-dependent rate: table ≥ 0.75 × whole |
| R | residual's goal-dependent rate as a share of whole's (descriptive) |

| result | reading |
|---|---|
| P, C1, C3, C7 | **the shortfall was the linear form: a shared, nonlinearly encoded belief variable at block 0's attention output controls even the goal-dependent decisions** |
| P fails, T holds | the posterior suffices, in a form this MLP does not capture |
| P and T fail | on goal-dependent pairs, the route carries information beyond the posterior that the policy uses; R measures how much |
| otherwise | reported criterion by criterion |

## Expectations

| | expectation |
|---|---|
| E1 | N holds (smoke test, untrained: +0.17) |
| E2 | table's R² within ≤ mlp's + 0.03 |
| E3 | P fails |
| E4 | mlp above linear on goal-dependent pairs (p < 0.05) |
| E5 | T fails |
| E6 | R ≥ 0.2 |
| E7 | C1 holds with S(mlp) ≥ 0.9 |
| E8 | C3 holds |
| E9 | C7 holds |
| E10 | C5b holds |

## Limits known in advance

Rounds 32–33's. The MLP and the table are fitted to reproduce states, not behaviour; the table's sparse posteriors
depend on the shrinkage. An encoding fitted this flexibly also fits untrained states well (smoke test), so the fit
itself proves little; the edits carry the result.
