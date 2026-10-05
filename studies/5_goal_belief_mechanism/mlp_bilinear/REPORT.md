# How block 0's MLP computes the goal × belief part (MLP bilinear)

Run date: 2026-10-02. Brief: `BRIEF.md`. Tables, descriptions, mechanism, causal test and decision rule: `PLAN.md`,
committed (`3547b29`) **before any table was built from a trained model**. Numbers from `tables.md`; data in
`results.json` and `untrained.json`. Reproduce: `reproduce.py mlp_bilinear` (30 min on one GPU, with the untrained reference).

**Setting.** The observation-prediction experiment's ten reward models, frozen, and their ten initial checkpoints as a reference. At the goal token,
block 0's MLP reads u = ln2(m) and writes y = W₂·GELU(W₁u + b₁) + b₂. The self-value-interaction experiment's posterior-level decomposition is built
for u, for the 512 hidden units and for y. J_y(b, g) is the goal × belief part of the MLP output. B(b) = S_u(b) is the
shared belief code at its input, and G = M_u(g) is the goal's main effect there. The analysis set is the 353
posteriors with ≥ 30 fit-side histories, weighted by count. J_y's split-half reliability is 0.999, so the noise ceiling
is 1.00 and R² can be read directly.

## 1. J is low-rank bilinear in the MLP's belief input and the goal

| description of J_y | trained R² | untrained R² |
|---|---|---|
| bilinear in the posterior b: W_g b | 0.79 | 0.52 |
| **bilinear in the MLP's belief input: W_g B(b)** | **0.95** (0.93–0.96) | 0.92 |
| rank 1: (u·B)(v·G) w | 0.55 | 0.25 |
| rank 2 | 0.67 | 0.43 |
| **rank 3** | **0.77** | 0.54 |
| **rank 4** | **0.82** (0.71–0.87) | 0.61 |
| rank 6 / 8 | 0.88 / 0.91 | 0.71 / 0.78 |

* **BL holds**: J_y(b, g) ≈ W_g B(b) explains 0.95 of the goal × belief part (ceiling 1.00).
* **LR holds**: rank 4 reaches 0.87 × the full bilinear form. The smallest rank reaching 0.8 × is 3 in seven models,
  4 in one, 6 in two. So **J(b, g) ≈ Σ_{k=1}^{3–4} (u_kᵀB(b))(v_kᵀG(g)) w_k**, the brief's form, describes it.
* **What training adds.** The untrained MLP already makes its (small) goal × belief part bilinear in B (0.92), so the
  form is largely the architecture's. Training changes three things:
  * the part's size: |J_y|²/|M_y|² 0.53 against 0.08;
  * its rank: rank 4 explains 0.82 against 0.61;
  * how directly it follows the posterior: bilinear in b itself 0.79 against 0.52.

## 2. The MLP makes it as a second-order interaction of its goal and belief inputs

With the MLP applied to its input rebuilt from parts (ū + B + G), and no fitted parameters:

| | trained | untrained |
|---|---|---|
| exact interaction f(ū+B+G) − f(ū+B) − f(ū+G) + f(ū), against J_y: R² | **0.68** (0.55–0.79) | 0.49 |
| its size, / J_y's | 0.84 | 0.47 |
| second-order term Σᵢ W₂ᵢ GELU″(āᵢ)(W₁ᵢ·B)(W₁ᵢ·G), against the exact interaction | **0.88** (0.78–0.90) | 0.99 |
| second-order term against J_y | 0.61 | 0.49 |

* **X holds**: two thirds of J_y is what the MLP itself makes from the shared belief part and the goal part of its
  input. The rest comes through from the input's own goal × belief part (J_u, 0.22 of M_u at the input, from the query;
  the query-swap to self-value-interaction experiments) and from per-history structure.
* **T holds**: that interaction is 0.88 second-order. **Each hidden unit contributes one multiplicative term**:
  GELU″(āᵢ)·(W₁ᵢ·B)·(W₁ᵢ·G)·W₂ᵢ. A unit whose pre-activation sits in GELU's curved region, and whose input weights see
  both the belief code and the goal shift, multiplies them. The 0.12 that is beyond second order is GELU's curvature
  over the goal shift, which is large; in the untrained net it is 0.01.
* **The units are many, not a few**: half of J_y's energy needs 61 units (untrained 90), 80 % needs 186. The top 20
  rebuild J_y with R² 0.44, the top 100 with 0.80. The low rank (3–4) is not a few units: many units contribute along
  a few shared directions.

## 3. Causal test: J_y at this point is not needed for the decision

The query-swap experiment's cells. J_y(b, g) is removed from the MLP output at the goal token, and replaced by its approximations:

| MLP output y − J_y + X | goal-matters: optimal | difference from ablate | goal-neutral: optimal |
|---|---|---|---|
| J_y (natural) | 0.834 | **+0.007** | 0.887 |
| 0 (ablate) | 0.824 | — | 0.875 |
| rank 4 | 0.824 | +0.004 | 0.885 |
| full bilinear | 0.829 | +0.005 | 0.885 |
| another posterior's J_y | 0.816 | −0.009 | 0.872 |

* **C fails at its first step**: removing the whole goal × belief part from block 0's MLP output costs 0.007 of correct
  decisions (consistent, p 0.03, but far below 0.05). With so little to restore, the restoration criterion cannot be
  assessed. The plan's per-model restored fractions have near-zero denominators, so the table shows absolute
  differences instead.
* **Registered reading: J_y is not needed for these decisions.** Blocks 1–3 receive the goal identity (the self value,
  the self-value experiment) and the belief (the prefix interface and the goal token's own residual stream). Their MLPs (goal-swap components) can
  rebuild the combination they need. The nonlinear-belief-edit experiment's goal-conditioned edits mattered on the hardest goal-dependent pairs,
  where a wrong goal-specific code was *inserted*. Here it is only *removed*, and later blocks compensate.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **BL** | bilinear_B / ceiling ≥ 0.7 | 0.95 | **yes** |
| **LR** | rank 4 ≥ 0.8 × bilinear | 0.87 × | **yes** |
| **X** | exact interaction against J_y ≥ 0.5 | 0.68 | **yes** |
| **T** | second-order against exact ≥ 0.8 | 0.88 | **yes** |
| **C** | ablation drop ≥ 0.05; rank 4 restores ≥ 0.75; shuffled ≤ 0.25 | drop 0.007 | no |

**Registered readings:**
* BL and LR: **a low-rank (3–4) bilinear goal × belief interaction describes J**.
* X and T: **the MLP makes it as its local second-order term**, one multiplicative product per hidden unit.
* C's ablation fails: **J at block 0's output is not needed for the first decision**; it is not causally sufficient
  because it is not causally necessary here.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | BL holds | yes | 0.95 |
| E2 | LR holds | yes | 0.87 × |
| E3 | X holds | yes | 0.68 |
| E4 | T fails | **no**: holds | 0.88 |
| E5 | ablation drop ≥ 0.05 | **no** | 0.007 |
| E6 | rank 4 restores ≥ 0.75 | **no** (not assessable) | |
| E7 | trained bilinear within 0.1 of untrained | yes | +0.03 |
| E8 | half the energy in ≤ 20 units | **no** | 61 |

4 of 8.

## Conclusions

1. **The brief's description holds.** J(b, g) ≈ Σ_{k=1}^{3–4} (u_kᵀB(b))(v_kᵀG(g)) w_k, with B the shared belief code
   entering block 0's MLP and G the goal. It explains 0.82 of the goal × belief part at rank 4 and 0.95 without a rank
   limit.
2. **The MLP computes it as a second-order product.** Applied to its own belief and goal inputs, it produces two thirds
   of J, 88 % of that from its second-order term: for each of many hidden units, the GELU curvature at the unit's
   operating point times the unit's projections of belief and goal. Many units share a few directions, which makes the
   rank low.
3. **Training made the interaction large and low-rank.** The architecture already makes a small bilinear one.
4. **But the decision does not depend on this copy.** Removing J from block 0's output costs 0.7 % of correct decisions:
   later blocks rebuild what they need from the goal identity and the belief. Together with the nonlinear-belief-edit experiment (inserting the
   wrong goal's code does harm), J at block 0 is used when present but is redundant: an early, removable computation.

Next, if wanted: the same analysis at blocks 1–3's MLPs, where the goal-swap-components experiment placed the goal switch. Then remove J from
all four MLP outputs together. If the decision then breaks, the goal × belief interaction is necessary but computed
redundantly across depth.

Limits: posterior-level tables; one MLP; the decomposed input omits J_u; the causal test is at the first decision on
the query-swap experiment's cells, not specifically the hardest goal-dependent pairs; reward models only.
