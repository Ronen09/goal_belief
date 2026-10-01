# A goal swap with the history held fixed: which downstream components carry the switch? (round 30)

Run date: 2026-10-01. Brief: `BRIEF.md`. Cells, components, measures and decision rule: `PLAN.md`, committed
(`242b4ce`) **before any component patch was applied to a trained model**. Numbers from `tables.md`; data in
`results.json`, `posthoc.json`. Reproduce: `reproduce.py r30` (4 min on one GPU).

**Setting.** Round 23's ten reward-only models, frozen. A cell is one history under two goals whose exact optimal
action sets are disjoint: the recipient runs it under goal g, the donor under g′. The prefix states are identical in
the two runs, so every patch is at the goal token: one head's output or one MLP's output, from the donor's run into
the recipient's. Components after the two-route interface: the 12 heads and 3 MLPs of blocks 1–3 (block 0 for
reference). Components with discovery patch ≥ 0.3 form the *selected* group. Discovery: 38 030 cells on fit-side
histories; test: 40 084 cells on held-out histories.

The registered transfer, patch = 1 − TV(patched, donor) / TV(recipient, donor), is a mean over cells.

**Registered and post hoc.** Sections 1–2 are registered. The ratio-of-means version in section 3 is post hoc: I
added it because the per-cell mean is dominated by cells where the two runs barely differ, and single-head patches
there give values down to −50. Necessity (restore) turned out not to be a separate measure: the cell set contains
both orders of every goal pair, so restore for g → g′ is patch for g′ → g, and the two are equal by construction.
I did not notice this when registering it.

## 1. Single components (discovery, registered measure)

| | patch | DLA share |
|---|---|---|
| attention heads, blocks 1–3 (each) | −0.14 to 0.01 (median per head) | 0.00–0.01 |
| **L1.mlp** | **0.54** (0.08–0.63) | 0.15 |
| **L2.mlp** | **0.36** (−0.14–0.58) | 0.29 |
| **L3.mlp** | 0.21 (0.11–0.34) | **0.39** |
| block 0, for reference: L0.mlp / heads | 0.83 / −1.18 to −0.12 | 0.07 / 0.00 |
| goal embedding (direct to the output) | | 0.00 |

* **No attention head after the interface carries the switch**: single-head patches transfer nothing, and heads
  write almost nothing of the logit difference directly.
* **The three downstream MLPs carry it.** L1.mlp is selected in 8 models, L2.mlp in 7, L3.mlp in 1. Every model has
  at least one, and no model has a head. The direct logit attribution rises along the stack (0.15 → 0.29 → 0.39),
  while the patch effect falls (0.54 → 0.36 → 0.21). An earlier MLP's change is passed on and amplified by the
  later ones, and the last writes the most directly.
* The goal itself contributes nothing directly to the logits (DLA of the goal embedding 0.00): it acts only through
  the MLPs. Block 0's MLP, before the interface, already carries 0.83 when patched: the goal-specific signal starts
  there and is carried forward.

## 2. Groups on held-out histories, and the decision rule

| group | discovery patch | test patch | test / discovery | test: donor-goal-optimal greedy action |
|---|---|---|---|---|
| none (recipient) | | | | 0.10 |
| **selected** (1–3 MLPs) | 0.73 | **0.71** (0.36–0.94) | 1.00 | 0.69 |
| complement | 0.55 | 0.62 | 1.06 | 0.61 |
| all of blocks 1–3 | 0.95 | **0.94** (0.74–0.97) | 1.00 | 0.83 |
| MLPs of blocks 1–3 | 0.93 | **0.92** | 1.00 | 0.82 |
| attention of blocks 1–3 | −0.20 | −0.11 | | 0.18 |
| donor's own run | | 1 | | 0.83 |

| | criterion | value | held |
|---|---|---|---|
| D1 | a component with discovery patch ≥ 0.3 in ≥ 8 of 10 models | 10 of 10 (all MLPs) | **yes** |
| D2 | selected group: test patch ≥ 0.5 and ≥ 0.8 × discovery | 0.71; 1.00 | **yes** |
| D3 | ≤ 4 selected and complement ≤ 0.25 | 2; **0.62** | **no** |

**By the registered rule: transferable components exist and generalise, but the switch is distributed.** The
selected MLPs transfer the donor-goal decision on held-out histories as well as on the discovery ones, under every
ordered goal pair (0.66–0.90). But the remaining MLPs also carry most of it (0.62). The switch is spread over the
MLP stack, and each part largely suffices.

## 3. Post hoc: ratio-of-means transfer (test cells)

1 − mean TV(patched, donor) / mean TV(recipient, donor):

| | patch |
|---|---|
| each head in blocks 1–3 | 0.00–0.01 (largest single value 0.10) |
| all twelve heads of blocks 1–3 | **0.12** (0.04–0.32) |
| L1.mlp / L2.mlp / L3.mlp | 0.58 / 0.37 / 0.20 |
| all three MLPs | **0.98** (0.95–0.99) |
| all of blocks 1–3 | 0.99 |

Without the near-zero-difference cells the picture is clean. **The downstream switch between goals is computed by
the MLPs at the goal token**: 0.98 of it, against 0.12 for all attention heads together. Nothing is left for the
goal token's residual path to carry past blocks 1–3 (0.99 with every component).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | D1 holds | yes | 10 of 10 |
| E2 | a block-1 or block-2 MLP selected in ≥ 8 models | yes | 10 of 10 |
| E3 | D2 holds | yes | 0.71; 1.00 |
| E4 | D3 fails: distributed | yes | complement 0.62 |
| E5 | all of blocks 1–3 ≥ 0.8 | yes | 0.94 |
| E6 | largest DLA share in blocks 1–3 is an MLP | yes | L3.mlp 0.39 |
| E7 | selected restore ≥ 0.5 | yes, but uninformative | 0.71 (equal to patch by the cells' symmetry) |

7 of 7.

## Conclusions

1. **After the two-route interface, the goal switches the action through the MLPs at the goal token, not through
   attention.** Patching the donor goal's MLP outputs moves the decision 0.92–0.98 of the way to the donor goal's
   action on held-out histories. All attention heads together move it 0.12 (post hoc measure) or nothing
   (registered).
2. **The transformation is serial and distributed across the MLP stack.** The block-1 MLP alone carries 0.58 and
   the block-2 MLP 0.37. Their direct write to the logits grows toward the last MLP, and any large part of the stack
   carries most of the switch. No single component is a goal switch.
3. **This fits round 29.** There, which evidence the policy reads did not depend on the goal; here, the attention
   heads that read that evidence write nearly the same output under both goals. The goal does not steer the
   reading. It changes what the MLPs make of the evidence once it is read. A shared evidence estimate,
   goal-specific MLP transformation.

Next, if wanted: to see how the MLPs combine goal and evidence, patch the MLP inputs. Split the goal token's
residual entering each MLP into its evidence part (from attention) and its goal part (from the direct route), and
ask whether the MLP's goal-specific output is a function of the goal crossed with the same evidence code across
histories.

Limits: single-position patches; heads patched by their output into the residual; the first decision after the
reveal; reward models only; necessity not measured separately (cell symmetry); the clean numbers of section 3 are
post hoc.
