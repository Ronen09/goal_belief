# Round 31: cells, patches, measures, decision rule and expectations

Written before any cross-history patch was applied to a trained model. Code: `run.py` (smoke-tested on an untrained
network: a self-patch changes nothing, 8·10⁻⁵). Brief: `BRIEF.md`.

## Models and cells

Round 23's ten reward-only models, frozen. Held-out histories of round 22's bank, prefix length 1–4. Donor and
recipient have the same prefix length, so the goal token sits at the same position. The decision is the first after
the reveal. Cells are model-specific (they depend on the model's choices), at most 20 000 of each kind per model,
drawn from 600 000 random tuples (seed 31):

* **Cross-goal cells (the brief's test).** The recipient is history B under goal g, where the model's greedy action
  a_B is optimal. Under the donor goal g′ it would choose a_B′, also optimal, with a_B′ ≠ a_B. The donor is history A
  under g′, where the model's greedy action a_A′ is optimal and differs from both a_B and a_B′.
* **Same-goal cells (control).** The recipient is B under g, choosing a_B (optimal). The donor is A under the same
  goal g, choosing a_A (optimal), with a_A ≠ a_B.

## Patches

The donor run's MLP output at the goal token replaces the recipient's. Sets: L1.mlp, L2.mlp, L3.mlp singly; **all
three MLPs of blocks 1–3** (round 30's carrier of the goal switch); round 30's selected set for that model; L0.mlp
(before the interface, for reference). The same patches from B's own run under g′ form the **within-history
reference** (round 30's design).

## Measures

Per cell, the patched greedy action, classified as:

| patched choice | reading |
|---|---|
| a_B′, B's own action under the donor's goal | **goal instruction**: the output says "g′", and B's evidence does the rest |
| a_A′, the donor's action | **donor's action**: the output carries the donor's action preference, or A's evidence combined with g′ |
| a_B, the recipient's action | **retained**: the rest of the recipient's computation overrides it |
| anything else | other |

Also the probability on each. Same-goal cells: the share choosing a_A (an instruction predicts none, since the goal
is the same) against a_B.

**Representation** (no patch): each MLP's goal-token output for every bank history under every goal, regressed
(affine, fit side → held-out R²) on goal; goal × the model's greedy action; goal × the exact posterior; both. This
separates an action preference (goal × action explains it) from an evidence–goal combination (goal × posterior
explains more).

## Decision rule (medians over the ten models)

Primary set: all three MLPs of blocks 1–3. The same reading is reported for each single MLP.

| cross-goal result | reading |
|---|---|
| a_B′ ≥ 0.5 and above a_A′ | goal instruction |
| a_A′ ≥ 0.5 and above a_B′ | the donor's action (preference or combination) |
| a_B ≥ 0.5 | retained |
| none of these | mixed |

Same-goal control: a_A ≥ 0.5 confirms that the output is not only a goal instruction.

Representation, for an MLP whose patch gives the donor's action:

| | reading |
|---|---|
| R²(goal × action) ≥ 0.8 × R²(both) | an **action preference** |
| R²(goal × posterior) > R²(goal × action) and R²(both) − R²(goal × action) ≥ 0.1 | an **evidence–goal combination** |

The untrained network's MLP outputs already give R²(goal × posterior) ≈ 0.45 (smoke test), so increments, not levels,
are read.

## Expectations

| | expectation |
|---|---|
| E1 | all three MLPs, cross-goal: the donor's action ≥ 0.5 |
| E2 | L3.mlp alone: donor's action above own-under-donor-goal |
| E3 | L1.mlp alone: retained is the most common outcome |
| E4 | same-goal control, all three MLPs: the donor's action ≥ 0.5 |
| E5 | within-history reference, all three MLPs: own-under-donor-goal ≥ 0.8 (round 30: 0.92) |
| E6 | L3.mlp is an action preference by the representation rule; L1.mlp an evidence–goal combination |

## Limits known in advance

Cells exist only where the model is right three or four times, which selects confident cases. Single-position
patches. The representation regressions are linear. Reward models only; the first decision after the reveal.
