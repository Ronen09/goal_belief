# Block 0's query, swapped (query swap)

Run date: 2026-10-02. Brief: `BRIEF.md` (the block-0-steps experiment's proposed next step). Swaps, measures and decision rule: `PLAN.md`,
committed (`8ad384a`) **before any swap was applied to a trained model**. Numbers from `tables.md`; data in
`results.json` and `untrained.json`. Reproduce: `reproduce.py query_swap` (1 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen, and their ten initial checkpoints as a reference. In block 0 the goal
token's input is its embedding alone, so its query, key and value are functions of goal and prefix length only. For a
history under goal g, block 0's attention output at the goal token is recomputed with the goal token's **query** from
another goal g′, which changes which history tokens are read and how much. Or it is recomputed with the goal token's
**own key and value** from g′, which changes what it sends to itself and how much that attracts. Or with both, which
reproduces the attention output under g′ exactly (difference 0). For behaviour, only that attention output is replaced;
everything else, including the goal embedding in the residual stream, stays g.

## 1. The query carries most of the goal × history part (Q, Qp hold)

Share of the history-dependent part of the goal's effect on block 0's attention output:

| | trained: query | trained: own key and value | untrained: query |
|---|---|---|---|
| goal × history part | **0.64** (0.50–0.82) | 0.36 | 0.79 |
| its posterior-explained part (81 % of it) | **0.61** (0.47–0.79) | 0.39 | 0.75 |

* **Q holds** (query above own key and value, p = 0.002), and so does **Qp**. The block-0-steps experiment's decomposition is confirmed
  causally: the goal × belief part of block 0's attention output comes mostly from reading the history with a
  goal-dependent query.
* Training made the part 50 times larger (squared norm 0.56 against 0.011 untrained). It also shifted it somewhat
  toward the goal token's own key and value (the query's share is 0.16 lower than in the untrained reference, so E6
  fails).
* **The two shares sum to 1 by construction.** Over the six ordered goal pairs, the query swap from g and the
  key-and-value swap from g′ cover the same change from opposite ends. Each share therefore averages a swap's effect
  applied first and applied second. I found this after the run. It makes the attribution symmetric, but the registered
  expectation E4 ("they add up") was empty.

## 2. But the decision hardly depends on it (B fails)

Goal-matters cells: held-out histories under two goals whose optimal sets are disjoint (39 816 cells). Only block 0's
attention output at the goal token is changed:

| block 0's attention at the goal token | optimal under g | same action as natural under g′ | unchanged |
|---|---|---|---|
| natural | 0.834 | 0.193 | 1 |
| **query from g′** | **0.777** | 0.242 | 0.919 |
| **own key and value from g′** | **0.152** | **0.902** | 0.246 |
| both (= attention output under g′) | 0.097 | **0.992** | 0.191 |
| natural under g′ (reference) | 0.097 | 1 | 0.193 |

On goal-neutral cells (identical optimal sets) every swap leaves the optimal rate within 0.02.

* **Swapping the query costs 0.037 of correct decisions where the goal matters.** That is consistent (p < 0.001) and
  larger than on goal-neutral cells (0.004), but below the registered 0.05. **B fails.** The registered reading is: the
  query carries the goal × belief part, but the decision does not depend on it much.
* That does not contradict the nonlinear-belief-edit experiment. There, edits with another goal's belief map cost 0.28 of whole on the 5 653
  hardest pairs, a fraction of all goal-matters cells. Here the swap changes 8 % of decisions across all of them.

## 3. Not registered: the goal reaches the decision through block 0's self-attention

The swaps of the goal token's own key and value were measured as a comparison, without a criterion:

* **With the goal token's own key and value from g′, the model takes g′'s action in 0.90 of goal-matters cells.**
  The goal embedding in the residual stream still says g, the query still says g, and all later blocks run unchanged.
* With block 0's whole attention output from g′ (the embedding still g), it takes g′'s action in 0.99 of cells.
* So **the goal identity the policy acts on is what the goal token sends to itself through block 0's attention**, not
  the embedding it carries in the residual stream. This fits the goal-swap-components experiment: there the goal embedding had no direct logit
  effect, and the goal switch was carried by blocks 1–3's MLPs. Those MLPs read the goal from this self-attention term.
* The goal's identity and the goal × belief part therefore travel by different routes. The identity goes through the
  goal token's own value; the belief's goal-dependence goes through the query's weighting of the history.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **Q** | share(query) ≥ 0.5, above own key and value | 0.637 against 0.363; p 0.002 | **yes** |
| **Qp** | posterior part: share(query) ≥ 0.5 | 0.611 | **yes** |
| **B** | goal-matters drop ≥ 0.05, above goal-neutral | 0.037 (p < 0.001); goal-neutral 0.004 (p 0.003) | no |

**Registered reading (Q, Qp, not B): the query carries the goal × belief part, but the decision depends on it only a
little** (later computation compensates or does not need it).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | Q holds | yes | 0.64 |
| E2 | Qp holds | yes | 0.61 |
| E3 | B holds | **no** | drop 0.037 |
| E4 | shares add within 0.1 of 1 | yes, but **by construction** | |
| E5 | swapping both: g′'s action in < 0.2 | **no** | 0.99 |
| E6 | trained query share within 0.1 of untrained | **no** | −0.16 |

2 of 6 held with content (E4 is empty). E5 was the largest miss: I took the goal-swap-components experiment's "the decision is made in later
MLPs" to mean the goal's identity would be read from the embedding there. Instead it is read from block 0's attention.

## Conclusions

1. **The goal-dependent belief part moves with the query.** Swapping block 0's goal-token query to another goal's
   carries 0.64 of the history-dependent change in the attention output, and 0.61 of its posterior part (Q, Qp).
2. **The decision depends on that part only a little.** Swapping the query costs 3.7 % of correct decisions where the
   goal matters. The nonlinear-belief-edit experiment's edits showed the part matters for the hardest goal-dependent pairs; across all goal-matters
   cells its weight is small.
3. **The goal's identity takes a different route: the goal token's own value in block 0.** Replacing only what the goal
   token sends to itself makes the policy act on the other goal in 90 % of cells, despite the unchanged embedding.
   With the goal-route-selection to block-0-steps experiments: block 0's attention writes the goal (self term) and a mostly goal-free belief, slightly
   goal-weighted (query). Block 0's MLP combines them, and blocks 1–3's MLPs turn that into the action.

Next, if wanted: confirm the identity route by swapping only the goal token's own value (key kept) or only its key
(value kept). Then check which heads' self terms carry the goal identity.

Limits: all heads' queries swapped together; behaviour measured at the first decision only; reward models only; the
section 3 reading rests on comparison swaps without a registered criterion.
