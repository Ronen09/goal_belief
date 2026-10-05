# MLP bilinear: tables, descriptions, mechanism, causal test, decision rule and expectations

Written before any table of this experiment was built from a trained model. Code: `run.py`, smoke-tested on the untrained
initial checkpoint of seed 0 only. Brief: `BRIEF.md`.

## Objects (goal token, block 0)

u = ln2(m) is the MLP input. a = GELU(W₁u + b₁) are its 512 hidden units, and y = W₂a + b₂ is its output. For each, round
38's posterior-level decomposition is built from natural fit-side states under all goals: c[g, L], c̄[L], S(b), S_g(b)
(shrunk toward S), the main effect M(g, L) and the goal × belief part J(b, g) = S_g(b) − S(b).

**Analysis set**: posteriors with ≥ 30 fit-side histories, weighted by their counts. **Noise ceiling**: J_y's split-half
reliability (two random halves of the fit histories), Spearman–Brown corrected. R² is reported raw and over the
ceiling.

## Descriptions of J_y (the brief's form)

* **bilinear_b**: J_y(b, g) ≈ W_g b, linear in the posterior, one map per goal.
* **bilinear_B**: J_y(b, g) ≈ W_g B(b). B is the MLP input's shared belief code S_u(b), in its top 13 principal
  coordinates. This is the brief's Q(B, G) with G the goal.
* **cp_r**: J_y(b, g) ≈ Σ_k (u_kᵀB(b))(v_kᵀG(g)) w_k, with G the centred goal one-hot (2 dimensions). r = 1, 2, 3, 4,
  6, 8; weighted alternating least squares, best of 5 starts.

## Mechanism (no fitted parameters)

For 20 000 fit-side histories, the MLP input is rebuilt from its parts: ū = c̄_u[L], B = S_u(b), G = M_u(g, L).

* **exact**: f(ū + B + G) − f(ū + B) − f(ū + G) + f(ū), the interaction the MLP itself makes from the belief part and
  the goal part.
* **taylor**: Σᵢ W₂ᵢ GELU″(W₁ᵢū + b₁ᵢ)(W₁ᵢ·B)(W₁ᵢ·G). This is its second-order part: a bilinear form with one rank-1
  term per hidden unit.

Their posterior-level J tables are compared with J_y on the same histories. **Hidden units**: J_a per unit; energy
shares |W₂ᵢ|²·|J_aᵢ|²; J_y rebuilt from the top-k units.

## Causal test

The query-swap experiment's cells. The MLP output at the goal token becomes y − J_y(b, g) + X for X in:
* 0 (**ablate**);
* cp_r for r = 1, 2, 4;
* bilinear_B;
* **shuffled**, J_y of a random other posterior.

Measured: optimal under g on goal-matters and goal-neutral cells. **Restored** = (X − ablate) / (natural − ablate).

## Reference

The ten untrained initial checkpoints, all of the above, to separate what the architecture gives from what training
adds.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05.

| | criterion |
|---|---|
| **BL** | bilinear_B: R² / ceiling ≥ 0.7 |
| **LR** | cp_4 ≥ 0.8 × bilinear_B |
| **X** | exact against J_y (same histories): R² ≥ 0.5 |
| **T** | taylor against exact: R² ≥ 0.8 |
| **C** | goal-matters: ablate lowers optimal by ≥ 0.05 (p < 0.05); cp_4 restores ≥ 0.75; shuffled restores ≤ 0.25 |

| result | reading |
|---|---|
| BL, LR, C | **a low-rank bilinear goal × belief interaction is a concrete and causally sufficient description of J** |
| BL, not LR | bilinear, but not low rank |
| X, T | the MLP makes J as its local second-order term: one multiplicative goal × belief product per hidden unit |
| X, not T | the MLP makes J from the two parts, but beyond second order (gating through GELU's curvature over a large goal shift) |
| not X | most of J_y is not made from the posterior-level belief and goal parts of the input |
| C's ablation fails | J_y is not needed for these decisions |

## Expectations

| | expectation |
|---|---|
| E1 | BL holds |
| E2 | LR holds |
| E3 | X holds |
| E4 | T fails (the goal shift is large) |
| E5 | C's ablation lowers optimal by ≥ 0.05 |
| E6 | cp_4 restores ≥ 0.75 |
| E7 | trained bilinear_B within 0.1 of the untrained reference (bilinear form is largely architectural) |
| E8 | half of the unit energy is in ≤ 20 units |

## Limits known in advance

Posterior-level tables (per-history structure beyond the posterior is not described); one MLP; the decomposed input
ignores the input's own small J_u; the first decision; reward models only.
