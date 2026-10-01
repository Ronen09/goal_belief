# What the goal token's MLPs carry: a cross-history patch (round 31)

Run date: 2026-10-01. Brief: `BRIEF.md`. Cells, patches, measures and decision rule: `PLAN.md`, committed
(`ad8cbc7`) **before any cross-history patch was applied to a trained model**. Numbers from `tables.md`; data in
`results.json`. Reproduce: `reproduce.py r31` (1 min on one GPU).

**Setting.** Round 23's ten reward-only models, frozen; held-out histories. A **cross-goal cell** has three parts:

* a recipient history B under goal g, where the model correctly chooses a_B;
* under the donor goal g′ the model would correctly choose a_B′ for B;
* a donor history A of the same length under g′, where it correctly chooses a_A′, a third action.

The donor's MLP output at the goal token replaces the recipient's. That is the brief's "A under G2 chooses left, B
under G1 chooses up, B under G2 would choose right" (8 204 cells per model, median). **Same-goal cells** (control)
have the donor A under the recipient's own goal, choosing a_A ≠ a_B (20 000 per model).

## 1. The brief's test: all three MLPs of blocks 1–3

Median (range) over ten models, share of the patched model's greedy choices:

| patched | a_A′ the donor's action ("left") | a_B′ B's own action under the donor goal ("right") | a_B retained ("up") |
|---|---|---|---|
| **all three MLPs, from A under g′** | **0.97** (0.84–0.99) | 0.01 | 0.01 |
| reference: all three MLPs, from B's own run under g′ | 0.00 | **0.99** | 0.01 |
| control: all three MLPs, from A under g (same goal) | **0.97** (a_A) | | 0.02 |

* **The patched model chooses left, the donor's action**, in 97 % of cells, and right (a goal instruction applied to
  B's evidence) in 1 %. The MLP outputs carry no goal instruction that B's evidence could complete. They carry a
  decision that overrides B's evidence.
* **The control agrees**: with the donor under the recipient's own goal, where a goal instruction would change
  nothing, the patch still gives the donor's action (0.97).
* By the registered rule (primary set): **the donor's action**.

## 2. Single MLPs: each is overridden by the other two

| patched (cross-goal) | donor's action | own action under donor goal | retained | within-history reference: own action under donor goal | same-goal: donor's action |
|---|---|---|---|---|---|
| L0.mlp (before the interface) | 0.40 | **0.20** | 0.35 | 0.86 | 0.49 |
| L1.mlp | 0.16 | 0.07 | **0.75** | 0.48 | 0.44 |
| L2.mlp | 0.13 | 0.02 | **0.84** | 0.38 | 0.41 |
| L3.mlp | 0.06 | 0.01 | **0.93** | 0.11 | 0.22 |
| round 30's selected MLPs | 0.52 | 0.04 | 0.44 | 0.72 | 0.65 |

* **Any single MLP from another history is mostly retained over.** The two MLPs computed from the recipient's own
  state win. Within the recipient's own history, a single MLP from the other goal's run moves the choice 0.11–0.48:
  there, the other MLPs read a residual stream that is consistent with it.
* **Block 0's MLP is the only one with a goal-instruction component.** From another history it gives B's own action
  under the donor goal in 0.20 of cells, the donor's action in 0.40, and retained in 0.35. It mixes the goal with the
  donor's evidence. Downstream MLPs read it together with B's evidence.

## 3. Representation: from evidence–goal combination to action preference

Held-out R² of each MLP's goal-token output (every bank history under every goal):

| | goal | goal × chosen action | goal × posterior | both | goal × action / both | reading (registered rule) |
|---|---|---|---|---|---|---|
| L0.mlp | 0.39 | 0.59 | **0.78** | 0.79 | 0.74 | **evidence–goal combination** |
| L1.mlp | 0.37 | 0.71 | 0.79 | 0.84 | 0.84 | action preference; also meets the combination criterion (posterior adds 0.14) |
| L2.mlp | 0.36 | **0.85** | 0.83 | 0.92 | 0.92 | **action preference** |
| L3.mlp | 0.33 | **0.91** | 0.84 | 0.95 | 0.96 | **action preference** |

* Along the stack, what the goal token's MLPs write moves from the goal combined with the posterior (block 0) to the
  goal combined with the chosen action (blocks 2–3, 0.92–0.96 of what both explain). Block 1 is in between. The goal
  alone explains a third at every depth.
* The untrained network's MLPs already give goal × posterior ≈ 0.45 and goal × action ≈ 0.1 (smoke test). Training
  adds the action structure.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | all three MLPs, cross-goal: donor's action ≥ 0.5 | yes | 0.97 |
| E2 | L3.mlp alone: donor's action above own-under-donor-goal | yes | 0.06 against 0.01 |
| E3 | L1.mlp alone: retained most common | yes | 0.75 |
| E4 | same-goal control: donor's action ≥ 0.5 | yes | 0.97 |
| E5 | within-history reference ≥ 0.8 | yes | 0.99 |
| E6 | L3.mlp an action preference; L1.mlp an evidence–goal combination | in part | L3 yes (0.96); L1 meets both criteria, and the rule's order calls it an action preference |

5 of 6, one in part.

## Conclusions

1. **The answer to the brief: left.** Patched with another history's MLP outputs under another goal, the model takes
   that history's action (97 %), not its own action under the donor goal (1 %), and does not keep its own (1 %).
   The goal token's MLPs in blocks 1–3 carry an **action preference**, not a goal instruction.
2. **The preference forms along the MLP stack.** Block 0's MLP writes the goal combined with the evidence, with a
   small instruction-like part. Blocks 2–3 write almost purely the goal × the chosen action. The goal is folded into
   a decision early, and later MLPs pass the decision on.
3. **The decision is a collective of the MLPs.** A single MLP from another history is outvoted by the two computed
   from the recipient's own state (retained 0.75–0.93). Together they decide.
4. **Together with rounds 29–30:** attention gathers a goal-independent evidence estimate. Block 0's MLP combines it
   with the goal, and the MLP stack turns that combination into an action preference, which it carries to the
   output. The goal acts by being combined with the evidence once, early, at the goal token.

Next, if wanted: where exactly the combination happens. Patch block 0's MLP input, split into the goal embedding and
the evidence read by block 0's attention. Ask whether its output is a fixed function of (goal, evidence code) across
histories, which would make the combination itself transplantable.

Limits: cells require the model to be right three or four times, which selects confident cases; single-position
patches; linear representation regressions; reward models only; the first decision after the reveal.
