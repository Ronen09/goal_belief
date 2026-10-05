# The additive code at the goal token (additive code)

Run date: 2026-10-02. Brief: `BRIEF.md` (the interaction-removal experiment's proposed next step). Additive code, solvability and decision rule:
`PLAN.md`, committed (`e0ca146`) **before `run.py` was run on any trained model**. Its solvability margin was checked
on synthetic data only. Numbers from `tables.md`; data in `results.json`. Reproduce: `reproduce.py additive_code` (20 s on one
GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen. The logits at the goal token are decomposed as l(h, g) ≈ H(h) +
G(g, L). H is the history's action profile, averaged over the three goals. G is a fixed goal bias per prefix length,
estimated on fit-side histories. The additive decision is argmax H + G. Held-out histories under each goal; a cell is
goal-dependent if its optimal set is disjoint from another goal's (102 k of 167 k cells). **Additive solvability**
(solver and G only): could *any* history profile make each goal's optimal action win under all three goals, given the
network's goal bias G? This is a difference-constraint feasibility problem, solved exactly.

## 1. The decision is additive: a history profile plus a fixed goal bias

| | natural | additive H + G |
|---|---|---|
| goal-dependent cells: optimal | 0.832 | **0.793** (0.96 × natural) |
| all cells: optimal | 0.855 | 0.823 |
| goal-dependent cells: the same action | | **0.943** |

* **AD holds**: on cells where the goal decides the action, a history profile plus a goal bias that is the same for
  every history keeps 0.96 of the policy's accuracy, and makes the same choice in 94 % of them.
* Only 0.63 of the logits' goal dependence is additive, but the non-additive rest rarely changes the argmax.
* **Need** (natural optimal, additive not): 4.0 % of goal-dependent cells. **Spoil** (the reverse): 1.1 %.

## 2. What H and G hold

* **U holds**: in 81 % (73–88 %) of histories, H ranks the union of the three goals' optimal actions on top (1.9
  actions on average). **The history profile lists the candidate actions for all goals.**
* **G is a fixed directional preference per goal.** In seed 0 at prefix length 2:

| goal | a0 | a1 | a2 | a3 |
|---|---|---|---|---|
| G1 | 0.02 | −0.78 | 2.87 | −2.11 |
| G2 | −0.07 | 1.12 | 2.91 | −3.96 |
| G3 | 0.05 | −0.35 | −5.78 | 6.07 |

  The goal does not need to know the belief. It pushes towards the moves that lead to its own region of the maze, and
  the history's candidate list does the rest.

## 3. Where additivity is impossible, the network fails too (EX fails)

| | share of goal-dependent cells | natural optimal | additive optimal |
|---|---|---|---|
| posteriors **additively solvable** under G | 0.86 | 0.872 | 0.838 |
| posteriors **not** additively solvable | 0.14 | **0.587** | **0.571** |

* **EX fails**: the cells that need the interaction are *not* concentrated in the unsolvable posteriors (P(need)
  0.026 there against 0.043 elsewhere; 0.58 ×). In one model (seed 5) they are (0.60 of need cells).
* **Instead, the network's errors are concentrated there.** In unsolvable posteriors, where no history profile can
  combine with the goal biases to be right under all goals, the natural policy is optimal in only 0.59 of cells, barely
  above the additive code (0.57). Elsewhere it reaches 0.87. These posteriors are 14 % of goal-dependent cells but hold
  about 35 % of the policy's errors on them (post hoc, from the medians above).
* The interaction the network does use (4 % of cells) is spent in solvable posteriors, where its averaged profile H is
  not quite the right additive one.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **AD** | goal-dependent: additive ≥ 0.9 × natural | 0.962 | **yes** |
| **EX** | P(need \| unsolvable) ≥ 3 × P(need \| solvable) | 0.58 ×; p 0.96 | no |
| **U** | H ranks the goals' optimal actions on top ≥ 0.7 | 0.811 | **yes** |

**Registered reading (AD, U): the history profile lists the candidate actions, and the goal bias picks among them.**
The reading for "not EX" also applies: the interaction is used, but not specifically where additivity is impossible.
There the network does not solve the problem either.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | AD holds | yes | 0.96 |
| E2 | EX holds | **no** | 0.58 × |
| E3 | U holds | yes | 0.81 |
| E4 | additive share of the goal dependence ≥ 0.8 | **no** | 0.63 |
| E5 | most posteriors solvable | yes | 0.89 of histories |
| E6 | unsolvable: natural above additive by ≥ 0.2 | **no** | +0.016 |

3 of 6. I expected the network to use its goal × belief interaction to solve what an additive code cannot. It does not:
it behaves almost additively and fails where additivity fails.

## Conclusions

1. **The goal token's decision is, to a first approximation, additive.** The history computes a profile that ranks the
   actions that are optimal under any of the goals (81 %). The goal adds a fixed directional bias, the same for every
   history. The sum's argmax matches the policy in 94 % of goal-dependent cells.
2. **The network's limits are the additive code's limits.** In the 14 % of goal-dependent cells where no additive code
   can be right, the policy is right only 59 % of the time (87 % elsewhere), no better than the additive code. The
   goal × belief interaction that the direct-belief-edit to MLP-depth experiments characterised does not rescue these cases.
3. **So the goal-route-selection to additive-code experiments fit together.**
   * Block 0's self value delivers the goal as a fixed bias (the query-swap to self-value experiments).
   * The history's evidence becomes a candidate profile.
   * The goal × belief interaction made by block 0's MLP and carried through depth (the self-value-interaction to MLP-depth experiments) exists, but changes
     only about 4 % of decisions (the interaction-removal to additive-code experiments).
   * The policy's residual errors are where its additive strategy cannot work.

Next, if wanted: test whether training longer, or a larger model, closes the unsolvable gap by using the interaction.
Or check whether the nonlinear-belief-edit experiment's hardest goal-dependent pairs are the unsolvable posteriors.

Limits: G shared by all histories of a length; logit-level only; the share of errors in unsolvable posteriors is
derived post hoc from the medians; first decision; reward models only.
