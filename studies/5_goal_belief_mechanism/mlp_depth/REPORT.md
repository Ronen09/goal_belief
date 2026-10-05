# The goal × belief part across the four MLPs, and its removal across depth (MLP depth)

Run date: 2026-10-02. Brief: `BRIEF.md` (the MLP-bilinear experiment's proposed next step). Per-layer analysis, ablations and decision rule:
`PLAN.md`, committed (`7ec83f7`) **before any table was built from a trained model or any ablation applied to one**.
Numbers from `tables.md`; data in `results.json`. Reproduce: `reproduce.py mlp_depth` (55 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen. The MLP-bilinear experiment's analysis is repeated for each of the four MLPs at the goal
token: the posterior-level goal × belief part J_y,l of its output (noise ceiling 1.00 at every layer); its bilinear
description in the MLP input's shared belief code B_l and the goal; and how much of it the MLP makes from its own
rebuilt input. Then J_y,l is removed from one, three or all four MLP outputs at the goal token, on the query-swap experiment's cells.

## 1. Every MLP's goal × belief part is low-rank bilinear; after block 0 it is mostly inherited

| | block 0 | block 1 | block 2 | block 3 |
|---|---|---|---|---|
| \|J_y\|² / \|M_y\|² (goal × belief / goal main effect, output) | 0.53 | 0.92 | 0.99 | 1.13 |
| \|J_u\|² / \|M_u\|² (at the input) | 0.22 | 0.55 | 0.78 | 1.00 |
| bilinear in B: R² | **0.95** | **0.92** | **0.89** | **0.88** |
| rank 4: R² (/ bilinear) | 0.82 (0.87 ×) | 0.83 (0.91 ×) | 0.86 (0.96 ×) | 0.86 (0.98 ×) |
| rank 1: R² | 0.55 | 0.61 | 0.66 | 0.67 |
| made by the MLP from its rebuilt input: R² against J_y | **0.68** | 0.55 | 0.30 | **0.18** |
| its size / J_y's | 0.84 | 0.31 | 0.14 | 0.07 |
| second-order share of that interaction | 0.88 | 0.90 | 0.87 | 0.80 |
| hidden units for half of J's energy | 61 | 72 | 59 | 42 |

* **BLd holds**: at every depth J ≈ Σ_{k≤4} (u_kᵀB)(v_kᵀG) w_k explains 0.82–0.86, and 0.88–0.95 without a rank limit.
  The rank falls with depth: one bilinear term explains 0.55 at block 0 and 0.67 at block 3.
* **Block 0's MLP creates J; later MLPs mostly pass it on.** The interaction an MLP makes from its own belief and goal
  inputs explains 0.68 of its J at block 0, and only 0.18 at block 3 (size 0.84 → 0.07). Meanwhile the J arriving at
  each MLP's input grows (0.22 → 1.00 of the main effect). What the MLP does make is second order at every depth
  (0.80–0.90).
* By block 3 the goal × belief part is as large as the goal's main effect and as the shared belief code.

## 2. Removal: redundant across depth, and specific to the belief

*Correction (2026-10-02, after the interaction-removal experiment): the removal below subtracts the posterior-level J and leaves each history's
deviation from it in place, a state the network never makes. The interaction-removal experiment removed the whole per-history interaction
cleanly: that costs 0.039 at the four MLPs (0.041 at all eight components), against 0.136 here. Removing only the
deviation costs 0.016. The large all-four drop below is therefore mostly an artefact of the partial removal, and the
conclusion that the interaction is necessary does not hold. The redundancy across depth (single removals ≈ 0) stands.*

| J removed from the goal token's MLP outputs | goal-matters: optimal | drop | goal-neutral: optimal | drop |
|---|---|---|---|---|
| nothing | 0.834 | — | 0.887 | — |
| block 0 only | 0.824 | 0.007 | 0.875 | 0.005 |
| block 1 only | 0.821 | 0.006 | 0.881 | 0.007 |
| block 2 only | 0.823 | 0.007 | 0.882 | 0.005 |
| block 3 only | 0.830 | **0.003** | 0.885 | 0.003 |
| blocks 1–3 | 0.747 | 0.071 | 0.818 | 0.052 |
| **all four** | **0.676** | **0.136** (0.048–0.219) | 0.768 | **0.099** |
| control: another posterior's J, all four | 0.811 | 0.018 | 0.858 | 0.027 |

* **R holds, strongly**: no single MLP's J is needed (0.003–0.007). Removed from all four, J costs 0.136, six times the
  sum of the four single drops (0.022). Each MLP's J can be rebuilt by the others. Block 3's alone matters least, since
  the residual stream already carries the earlier MLPs' J.
* **S holds**: removing a same-size J of a wrong posterior costs only 0.018. The harm comes from losing the
  posterior-specific goal × belief structure, not from a disruption of that size.
* **N fails, on its specificity part.** The all-four drop clears 0.10 (p < 0.001). But goal-neutral cells drop nearly as
  much (0.099), so the required margin of 0.05 is not met (0.025).

## 3. Why the goal-neutral cells drop too

The registered reading for "not N" is that the MLP outputs' goal × belief part is not necessary even together. **The data
do not support that reading.** Removing it costs 14 % of correct decisions where the goal matters, specifically
(against the shuffled control). The specificity criterion was mis-specified:

* *Goal-neutral* was defined over a pair of goals: g and g′ have the same optimal set.
* J is defined against the shared code S, the average over all **three** goals. Removing J leaves each goal with the
  three-goal average belief → action map, which is wrong for a goal whenever the third goal disagrees, even if g and g′
  agree.
* So J carries each goal's own belief → action mapping, needed for nearly every decision, not only where two goals
  differ. The goal-neutral drop (0.099) is that.

I report N as failed, as registered, and its reading as not supported.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **N** | all four: drop ≥ 0.10 (p < 0.05); ≥ goal-neutral drop + 0.05 | 0.136 (p < 0.001); difference 0.025 | no |
| **R** | largest single ≤ 0.5 × all four | 0.08 × | **yes** |
| **S** | all four above shuffled | 0.136 against 0.018; p < 0.001 | **yes** |
| **BLd** | every layer bilinear ≥ 0.7, rank 4 ≥ 0.8 × | 0.88–0.95; 0.87–0.98 × | **yes** |

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | N holds | **no** (specificity part) | goal-neutral drop 0.099 |
| E2 | R fails: block 3 alone ≥ 0.5 × all | **no**: block 3 alone matters least | 0.003 |
| E3 | single drops increase with depth | **no** | flat, 0.003–0.007 |
| E4 | BLd holds | yes | |
| E5 | the MLP's own share falls with depth | yes | 0.68 → 0.18 |
| E6 | all four: goal-neutral drop ≤ 0.05 | **no** | 0.099 |
| E7 | S holds | yes | |

3 of 7. I expected block 3's J to be the decision itself and irreplaceable (from the cross-history-MLP experiment's action preference). Instead
it is the most redundant, because the residual stream already holds the earlier MLPs' J.

## Conclusions

1. **The goal × belief interaction is computed once, then carried and refined.** Block 0's MLP creates it as a low-rank
   bilinear, second-order product of the goal and the belief (MLP bilinear). Blocks 1–3's MLPs receive it in the residual
   stream, pass most of it on (they make only 0.55 → 0.18 of their own J), and sharpen it to lower rank. By block 3 it
   is as large as the goal's main effect.
2. **(Corrected by the interaction-removal experiment: the necessity below is an artefact of the partial removal; a clean removal costs 0.04.)
   It is necessary, and redundant across depth.** No single MLP's copy matters (≤ 0.7 %). Removing all four costs 14 %
   of correct decisions where the goal matters (6 × the sum of the singles), specifically (wrong-posterior control
   1.8 %). Each layer's copy can stand in for the others.
3. **It carries each goal's own belief → action map.** Removing it also costs 10 % of decisions where two goals agree,
   because the remaining shared code is the three-goal average.
4. **What remains after removing it all** (0.68 correct where the goal matters) is carried by the per-history structure
   these posterior-level tables do not remove, and by the attention paths.

Next, if wanted: remove J from the attention outputs too, or remove it per history rather than per posterior. That
would show whether the 0.68 that remains is the goal × belief interaction reached by another path or a different
computation.

Limits: posterior-level tables, subtracted fixed (after an earlier removal, a later MLP's actual J may differ); goal
token and first decision only; the goal-neutral contrast was mis-specified, as explained in §3; reward models only.
