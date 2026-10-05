# MLP depth: per-layer analysis, ablations, decision rule and expectations

Written before any table of this experiment was built from a trained model or any ablation applied to one. Code: `run.py`,
smoke-tested on the untrained initial checkpoint of seed 0 only. Its block 0 numbers equal the MLP-bilinear experiment's for that
checkpoint. Brief: `BRIEF.md`.

## Per MLP (blocks 0–3, goal token)

The MLP-bilinear experiment's analysis, repeated for each block l. u_l = ln2_l(m_l) is the MLP input, a_l its 512 hidden units, y_l its
output. Posterior-level tables (c, S, S_g, main effect M, goal × belief part J) are built from natural fit-side states
under all goals. The analysis set is the posteriors with ≥ 30 fit-side histories, weighted by count, with J_y's
split-half ceiling.

* **Description**: J_y,l ≈ W_g B_l(b) (B_l the input's shared belief code, top 13 principal coordinates); CP ranks 1–8.
* **Mechanism**: the exact interaction the MLP makes from its rebuilt input (ū + B + G) and its second-order term,
  against J_y,l. Also |J_u|²/|M_u|², the goal × belief part the MLP inherits at its input.
* **Hidden units**: units for half and for 80 % of J's energy.

## Ablations (the query-swap experiment's cells; natural tables, fixed)

At the goal token, MLP outputs y_l → y_l − J_y,l(b, g):

| set | layers |
|---|---|
| single | 0; 1; 2; 3 |
| blocks 1–3 | 1, 2, 3 |
| **all** | 0, 1, 2, 3 |
| shuffled-all (control) | all four, subtracting J_y,l(b′, g) for b′ a random other posterior: a removal of the same size from the wrong posterior |

Measured: optimal under g on goal-matters and goal-neutral cells. **Drop** = natural − ablated.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05.

| | criterion |
|---|---|
| **N** necessary | all-four ablation: goal-matters drop ≥ 0.10 (p < 0.05), and ≥ goal-neutral drop + 0.05 (p < 0.05) |
| **R** redundant across depth | the largest single-layer goal-matters drop ≤ 0.5 × the all-four drop |
| **S** specific | all-four drop above shuffled-all drop (p < 0.05) |
| **BLd** bilinear at every depth | for every layer: bilinear_B / ceiling ≥ 0.7 and rank 4 ≥ 0.8 × bilinear_B |

| result | reading |
|---|---|
| N, R | **the goal × belief interaction is necessary but computed redundantly across depth** |
| N, not R | necessary, and concentrated where the single drop is largest |
| not N | the MLP outputs' goal × belief part is not necessary even together; the decision gets it elsewhere |

## Expectations

| | expectation |
|---|---|
| E1 | N holds |
| E2 | R fails: block 3's single ablation alone is ≥ 0.5 × all (late J is the action preference itself, cross-history MLP) |
| E3 | single-layer drops increase with depth |
| E4 | BLd holds |
| E5 | the share each MLP makes itself (exact against J_y) falls with depth (later MLPs inherit more J at their input) |
| E6 | all-four ablation: goal-neutral drop ≤ 0.05 |
| E7 | S holds |

## Limits known in advance

Ablations subtract natural posterior-level tables, fixed: after an earlier ablation, a later MLP's actual J may differ
from its natural table. Goal token only, first decision, reward models only. No untrained reference this experiment (round
39 has block 0's).
