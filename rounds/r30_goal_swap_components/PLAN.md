# Round 30: cells, components, measures, decision rule and expectations

Written before any component patch of this round was applied to a trained model. Code: `run.py` (smoke-tested on an
untrained network: a self-patch changes nothing, 2·10⁻⁵; DLA shares sum to 1.04). Brief: `BRIEF.md`.

## Models and cells

Round 23's ten reward-only models, frozen. A cell is (history h, recipient goal g, donor goal g') such that the exact
optimal action sets for h's posterior under g and g' are disjoint. The decision is the first after the reveal.
Histories have prefix length 1–4, from round 22's bank. **Discovery**: 12 000 fit-side histories with every
qualifying ordered goal pair. **Test**: 12 000 held-out histories (step 4 of the brief: other histories where the
donor goal requires a different action).

With the history fixed, the prefix states are identical in the two runs, so the goal token is the only place they
differ; every patch is at the goal token. The interface and the direct route of round 29 both enter block 1. The goal
token's state entering block 1 already differs, because the goal enters there through block 0.

## Components

After the interface: each attention head's output (4 per block) and each MLP's output in blocks 1–3, 15 components.
Block 0's four heads and MLP and the goal embedding are measured for reference only.

## Measures, per component, mean over cells

* **patch** (sufficiency): the donor run's output of the component in the recipient's run; moved =
  1 − TV(patched, donor) / TV(recipient, donor), and the share of cells whose greedy action is optimal for the donor
  goal.
* **restore** (necessity): the recipient's output in the donor's run; the share of the way back to the recipient.
* **DLA**: the component's change between the runs, written through the final layer norm (at the recipient's scale)
  and the policy head, projected on the logit difference. The shares, with the goal embedding's, sum to about 1.
* Groups on both sides: **selected** (components with discovery patch ≥ 0.3, the threshold round 18 used for goal
  heads), its **complement** in blocks 1–3, all of blocks 1–3, attention in blocks 1–3, MLPs in blocks 1–3. The
  remainder of all of blocks 1–3 is what the goal token's own residual path carries from the direct route, past
  blocks 1–3.
* The selected group's patch per ordered goal pair (6).

## Decision rule

Per model, then across the ten models (medians):

| | criterion |
|---|---|
| D1 components exist | at least one component in blocks 1–3 with discovery patch ≥ 0.3, in ≥ 8 of 10 models |
| D2 they generalise | test patch of the selected group: median ≥ 0.5, and ≥ 0.8 × its discovery value |
| D3 they are localised | at most 4 selected components, and the complement's test patch ≤ 0.25 (medians) |

| result | reading |
|---|---|
| D1–D3 | a small set of downstream components carries the switch from one goal's action to another's, on histories it was not found on |
| D1, D2, not D3 | transferable components exist, but the switch is distributed |
| D1 fails, all of blocks 1–3 ≥ 0.5 | no single downstream component carries it; the transformation is spread over blocks 1–3 |
| D1 fails, all of blocks 1–3 < 0.5 | the switch is mostly carried by the goal token's own residual path from block 0, not computed downstream |

Head indices are not aligned across seeds; identities are reported by layer and kind, and MLPs by layer.

## Expectations

| | expectation |
|---|---|
| E1 | D1 holds |
| E2 | a block-1 or block-2 MLP is selected in at least 8 of 10 models |
| E3 | D2 holds |
| E4 | D3 fails: the switch is distributed |
| E5 | all of blocks 1–3 patched: ≥ 0.8; little is carried by the residual path alone |
| E6 | the largest DLA share in blocks 1–3 belongs to an MLP |
| E7 | the selected group's restore ≥ 0.5 on test |

## Limits known in advance

Single-position patches at the goal token; heads patched by their output into the residual stream; the first
decision after the reveal; the 0.3 threshold is round 18's; reward models only.
