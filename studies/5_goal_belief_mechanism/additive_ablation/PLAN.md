# Additive ablation: ablations, decision rule and expectations

Written before `run.py` was run on any trained model (smoke-tested on the untrained checkpoint of seed 0 only). Brief:
`BRIEF.md`.

## Why the fixed-state ablation alone is not enough

The interaction-removal experiment's additive state sets all eight component outputs at the goal token to
xbar_k(h) + M_k(g, L): a goal-averaged history part plus the goal's main effect. Nothing at the goal token is computed
live. Replacing xbar_k(h) by its mean over histories therefore makes the decision a function of (g, L) only, **by
construction**. It must collapse to some history-blind policy. That is reported (`F_hist_mean`), but it cannot show that
the history part is the route the history takes in the working network.

The test that can fail runs **live**. Remove the history's main effect from each component's output and let every later
component recompute. The goal × history interaction I_k(h, g) is left in place, and it is history-dependent too. If the
interaction can carry the history into the decision without the additive part, the decision will not collapse.

## Decomposition (as in the interaction-removal experiment)

At the goal token, components k = attention and MLP output of each block (8). Natural output x_k(h, g).
* xbar_k(h): mean over the three goals of x_k(h, ·), the history part;
* Xbar_k(L): fit-side mean of xbar_k over histories of length L, the history-blind mean;
* M_k(g, L) = c_k[g, L] − Xbar_k(L), the goal's main effect;
* I_k(h, g) = x_k(h, g) − xbar_k(h) − M_k(g, L), the interaction.
Every part comes from natural runs. In the live kinds, the edit is a fixed vector added to each component's output, and
later components recompute.

| kind | edit at the goal token | what it tests |
|---|---|---|
| F_additive | set all 8 to xbar(h) + M(g) | the interaction-removal experiment's all_hist (replication) |
| F_hist_mean | set all 8 to Xbar(L) + M(g) | the additive state without history (collapse by construction) |
| F_goal_mean | set all 8 to xbar(h) + 0 | the additive state without goal (goal-blind by construction) |
| **live_noH** | add −(xbar(h) − Xbar(L)) at all 8 | **history main effect removed; interaction kept live** |
| live_noH_attn | the same at the four attention outputs only (the history's entry points) | |
| live_noI | add −I(h, g) at all 8 | interaction removed, live (dissociation partner) |
| live_noH_rot | live_noH's vectors, each rotated to a random direction (same norm) | same-size control |
| **live_swapH** | add xbar(h′) − xbar(h) at all 8, h′ another held-out history of the same length | does the decision follow the donor history? |
| live_noG | add −M(g, L) at all 8 | goal main effect removed, interaction kept |
| live_swapG | add M(g′, L) − M(g, L) at all 8 | does it follow the other goal? |

## Cells and measures

The query-swap experiment's cells (held-out histories under ordered goal pairs g, g′; goal-matters and goal-neutral),
as in the interaction-removal experiment. Greedy action at the goal token.

* **optimal**: the action is optimal under (h, g), for goal-matters, goal-neutral and all cells.
* **history-blind ceiling** B: per (g, L), the action most often optimal on fit-side histories, scored on the cells.
  It is the best any policy that ignores the history can do.
* **collapse fraction** cf = (natural − kind) / (natural − B), on all cells.
* **history dependence**: 1 − (share of cells taking the modal action of their (g, L) group). It is zero for any
  history-blind policy.
* **swap follow / stay**: on pairs where natural(h, g) ≠ natural(h′, g), the share taking natural(h′, g) or
  natural(h, g).
* **goal dependence**: on goal-matters cells, the share where the action under g differs from the action under g′ with
  the same edit (both computed).
* **goal swap follow**: goal-matters cells, the share taking natural(h, g′).

## Decision rule

Ten models; medians; exact one-sided Wilcoxon signed-rank tests, p < 0.05.

| | criterion |
|---|---|
| **HC** | live_noH collapse fraction ≥ 0.8, above live_noH_rot (history collapses without the additive part) |
| **ID** | (natural − live_noI) ≤ 0.2 × (natural − live_noH), all cells (removing the interaction costs little by comparison) |
| **SW** | live_swapH follow ≥ 0.75, above stay (the history part steers the decision to the donor's) |
| **GC** | live_noG goal dependence ≤ 0.25 × natural's, and live_swapG follow ≥ 0.75 (the goal part is the goal's route) |

| result | reading |
|---|---|
| HC, ID, SW, GC | **the additive code is the policy at the goal token**: its history part is the only route by which the history reaches the decision, its goal part the only route for the goal, and either part, swapped, moves the decision as predicted |
| SW, not HC | the history part is sufficient to steer, but the interaction carries the history redundantly; additive is a description, not the only mechanism |
| HC, not SW | the history part is necessary, but not a self-contained code: a donor's part does not transplant |
| not HC, not SW | the additive decomposition is not how the history acts |

## Expectations

| | expectation |
|---|---|
| E1 | F_additive reproduces all_hist within 0.005 (goal-matters 0.788) |
| E2 | F_hist_mean has history dependence 0 (by construction) and all-cells optimal ≤ B |
| E3 | HC holds; live_noH all-cells optimal within 0.05 of B |
| E4 | ID holds |
| E5 | SW holds, follow 0.8–0.9 |
| E6 | live_noH_attn collapses less than live_noH (the MLPs' history part partly recomputes from the attention's residual) — weak |
| E7 | GC holds |

## Limits known in advance

Fixed vectors from natural runs: after an edit upstream, a later component's actual history part differs from the
natural xbar that is subtracted, so live_noH removes the natural history part, not exactly the recomputed one. The
history reaches the goal token through attention at every block, so live_noH_attn is reported alongside. Goal token and
first decision only; reward models only.
