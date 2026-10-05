# Pair types: pairs, patches, measures and expectations

Written before any patch of this experiment was run. The pair counts below come from the solver's tables alone; no model
was evaluated to choose them. Code: `studies/3_reward_trained_agents/pair_types/run.py`.

## What is fixed

* **Models**: the maze-belief experiment's six. No training.
* **Interface and patches**: the residual stream at the prefix tokens entering block 1. `whole`: the donor's state.
  `pca`: the donor's component in the top 13 principal directions, from the files the pattern-specificity experiment saved
  (`r21_pattern_specificity/subspaces/`), unchanged. `pca complement`: everything else (reported; not asked for).
  `hybrid`: no patch, the model on the donor's prefix with the same goal — the natural swap of the evidence.
* **Histories**: the held-out side of a new bank of 200 000 histories (seed 220022); pairs drawn with seed 22.
* **Matched in every pair**: prefix length (2, 3 or 4), hence every token position and the position of the goal
  token; the horizon after the reveal is 12 moves for every history; the goal is set by the evaluation (each pair
  under G1, G2, G3). All decisions are at the reveal. Each type has the same number of pairs at each prefix length.

## Pair types

Optimal sets are the exact solver's at the reveal; "same" means identical sets, "disjoint" no shared action.

| type | definition | pairs |
|---|---|---|
| A: same posterior | donor and recipient end their prefixes at the same node of the belief graph (the posterior is identical, L1 = 0) with different token sequences. At most 50 pairs per node, so that no posterior dominates (307 nodes occur; two would otherwise supply 30 % of pairs). | 4 000 per prefix length where available |
| B: same action under one goal | posteriors ≥ 0.25 apart (L1); the optimal sets are the same under goal g₌ and disjoint under goal g≠. Sampled equally over the six ordered (g₌, g≠) and the three prefix lengths. | 700 per cell, 12 600 |
| C: different action | posteriors ≥ 0.25 apart; the optimal sets are disjoint under all three goals. | 4 000 per prefix length |

Every pair is run under all three goals. Each (pair, goal) is a **cell**, classed by the solver as *same*, *disjoint*
or *overlapping*. Type A has only same cells; type C only disjoint cells; type B has at least one of each.

## Conditions, kept separate

* **natural access**: nothing removed.
* **direct route removed**: the goal token's first attention output is set to its mean. Changed from the belief-edit to pattern-specificity experiments:
  the mean is taken per goal *and prefix length*, over a fixed reference set (20 000 held-out histories), the same
  for every pair type, so that the ablation does not depend on the pairs being tested.
  In this condition the whole patch equals the hybrid by construction; `whole` is informative under natural access,
  `pca` in both.

## Measures, per type, condition, goal and class of cell

* **TV(·, base)** for hybrid, whole, pca, pca complement.
* **moved** = 1 − TV(edited, hybrid) / TV(base, hybrid) (ratio of means), where the solver predicts a change.
* **Δ optimal**: the change from base in probability on the solver's optimal set for the donor's evidence. In
  disjoint cells it is the predicted change, reported also as a share of the hybrid's. In same cells the set is the
  recipient's too and the prediction is zero: a negative value is harm.
* **greedy**: how often the greedy action changes, and how often it is in the donor's optimal set.
* For type B, the two classes of cell are compared within the same pairs and the same edited prefix states.

## Expectations

| | expectation |
|---|---|
| E1 | type A, natural access: the hybrid differs from base by TV 0.02–0.06 (history matters a little beyond the posterior), and Δ optimal is within ± 0.01 |
| E2 | type A, direct route removed: TV 0.03–0.14 (the pattern-specificity experiment's closest bin), Δ optimal within ± 0.02 |
| E3 | type A's TV(hybrid, base) is less than a third of type C's, in each condition and every model |
| E4 | type B, same cells: TV(hybrid, base) is less than half of that in the disjoint cells of the same pairs; Δ optimal within ± 0.02; the greedy action changes in < 10 % |
| E5 | type B, disjoint cells, direct route removed: the pca patch gains ≥ 0.85 of the hybrid's Δ optimal; natural access: whole and pca gain 0.1–0.45 of it and agree within 0.03 |
| E6 | type C: as E5; and moved for type C is within 0.05 of moved for type B's disjoint cells |
| E7 | pca complement's share of the whole patch's TV is larger for type A than for type C in at least five models (direct route removed) |

E4–E5 together are the brief's second row: one edit of goal-free states leaves the action where the solver says
the evidence does not matter and changes it where it does.

## What each outcome supports

| result | reading |
|---|---|
| type A: TV(hybrid, base) ≈ 0 | decisions depend on the prefix only through the posterior |
| type A: TV > 0, Δ optimal ≈ 0 | history is carried beyond the posterior and shifts the distribution without cost |
| type A: TV > 0 and comparable to type C's | the state is a history summary; the posterior is not what organises it |
| type B: same cells unchanged, disjoint cells changed by the same edit | the representation keeps evidence beyond what one goal's action needs, and each goal reads what it needs |
| type B: same cells change as much as disjoint cells | the patch perturbs the decision regardless of what the evidence implies |
| type C: patches give the predicted change (E5–E6) | replacing evidence produces the solver's predicted change, to the extent the model on the donor's evidence does |

Limits known in advance: the same six models and one maze; type A's posteriors are the 307 that recur with
different histories, which are those of short, common prefixes; mean ablation is off the training distribution.
