# Removing the goal × history interaction cleanly (interaction removal)

Run date: 2026-10-02. Brief: `BRIEF.md` (the MLP-depth experiment's proposed next step). Removals and decision rule: `PLAN.md`, committed
(`3a093c7`). **Disclosure**: the plan was written after one preview run of seed 0 on the trained model, which I meant as
a timing check. The rule was designed before it; the expectations were written after seeing it. Every table is given
for all ten models and for seeds 1–9 alone, and the two agree. Numbers from `tables.md`; data in `results.json`.
Reproduce: `reproduce.py interaction_removal` (1 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen. At the goal token, each of the eight components (the attention output
and MLP output of each block) has natural output x_k(h, g). It splits into:
* a goal-free part xbar_k(h), the mean over the three goals for that history;
* the goal's main effect M_k(g, L);
* the **goal × history interaction** I_k(h, g), the rest, per history.

**Per-history removal** replaces the output by xbar_k(h) + M_k(g, L). With all eight replaced, the goal token's final
state is the embedding plus a history part plus a goal part, with no interaction anywhere. The query-swap experiment's cells.

## 1. Without any goal × history interaction, the policy keeps almost all of its decisions

| removed at the goal token | goal-matters: optimal | drop | goal-neutral: drop |
|---|---|---|---|
| nothing | 0.834 | — | — |
| per-history interaction, four attention outputs | 0.814 | 0.012 | 0.003 |
| per-history interaction, four MLP outputs | 0.790 | 0.039 | 0.010 |
| **per-history interaction, all eight: an additive goal + history state** | **0.788** (0.613–0.825) | **0.041** (0.033–0.103) | 0.009 |
| control: another history's interaction subtracted, all eight | 0.656 | 0.135 | 0.130 |

(Seeds 1–9: 0.790; drop 0.042.)

* **F fails**: with no goal × history interaction anywhere at the goal token, the policy still chooses optimally in
  0.79 of the cells where the goal decides the action (natural 0.83). Its final state is a history part plus a goal
  part, read out through the final layer norm. An additive code supports most goal-dependent decisions. The
  interaction adds 0.04.
* **A fails**: the attention outputs' interaction adds almost nothing beyond the MLPs' (0.041 against 0.039).
* **Inserting a wrong interaction is harmful** (control: 0.135, also where the goal does not matter). The network does
  not need its interaction much, but it is sensitive to a wrong one, as the nonlinear-belief-edit experiment found for wrong-goal belief codes.
* **K is not assessable**: with all eight removed, keeping any single component live recovers nothing (0.778–0.789
  against 0.788), but there is only 0.04 to recover. That 0.04 needs more than one component.

## 2. The MLP-depth experiment's removal was not clean

| removed from the four MLP outputs | goal-matters drop |
|---|---|
| the MLP-depth experiment: the posterior-level J(b, g) subtracted | **0.136** |
| only the per-history deviation I − J subtracted (J kept) | 0.016 |
| both: the whole per-history interaction I | 0.039 |

* **H fails, in the opposite direction**: removing the whole interaction costs less than removing its posterior-level
  part alone (p < 0.001, all ten and seeds 1–9).
* Subtracting J while leaving each history's deviation from it produces a state the network never makes: the deviation
  without the mean it deviates from. That state harms decisions (0.136, and 0.099 even where the goal does not matter).
  Removing both, or only the deviation, does not.
* **So the MLP-depth experiment's conclusion that the interaction is necessary was wrong.** Its drop came from the inconsistent partial
  removal, not from losing the goal × belief part. The MLP-depth experiment's report has been corrected (marked there).

## 3. Decision rule

| | criterion | all ten | seeds 1–9 | held |
|---|---|---|---|---|
| **H** | mlp_hist drop ≥ mlp_post drop + 0.05 | 0.039 against 0.136 | 0.041 against 0.141 | no (opposite) |
| **A** | all_hist drop ≥ mlp_hist drop + 0.05 | 0.041 against 0.039 | 0.042 against 0.041 | no |
| **F** | all_hist goal-matters optimal ≤ 0.5 | 0.788 | 0.790 | no |
| **K** | best keep-one recovers ≥ 0.75 of all_hist's drop | 0.09 | 0.08 | no (not assessable) |

**Registered reading (not F): an additive goal + history code at the goal token supports most decisions; the
interaction adds the rest** (4 % of goal-dependent decisions).

## Expectations (written after the seed-0 preview)

| | expectation | held |
|---|---|---|
| E1 | F fails | yes |
| E2 | H fails | yes |
| E3 | mlp_resid harms more than mlp_hist | **no**: 0.016 against 0.039 |
| E4 | A fails | yes |
| E5 | K not assessable | yes |

4 of 5. Their value is limited by the preview.

## What this means for the direct-belief-edit to MLP-depth experiments

* The direct-belief-edit to MLP-depth experiments found a goal × belief part, made by block 0's MLP (the self-value-interaction to MLP-bilinear experiments) and carried through depth (MLP depth).
  It is real, low-rank bilinear and causally readable. Inserting a wrong one hurts (the nonlinear-belief-edit and interaction-removal experiments).
* **But the policy needs it for only about 4 % of goal-dependent decisions.** The rest are made by adding a
  history-dependent action profile and a goal-dependent one at the goal token. The goal's main effect, delivered by
  block 0's self value (self value), shifts an action profile computed from the history. That suffices to choose
  differently under different goals in four fifths of the cells where the optimal actions differ.
* Removal has been small at every step (the MLP-bilinear to interaction-removal experiments). The interaction is used when present and harmful when wrong,
  but it is largely replaceable by an additive code.

Next, if wanted: characterise the additive code. Does the goal-free history part xbar(h) at the goal token contain each
goal's preferred action, so that a goal-specific bias selects among them? Fit logits ≈ H_a(h) + G_a(g) directly and
check which cells need the interaction (expected: the nonlinear-belief-edit experiment's goal-dependent pairs).

Limits: the preview of seed 0; fixed replacements from natural runs; the final layer norm couples the two additive
parts slightly; goal token and first decision only; reward models only.
