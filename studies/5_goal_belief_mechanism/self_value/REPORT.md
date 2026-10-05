# The goal's identity in block 0: the goal token's own value (self value)

Run date: 2026-10-02. Brief: `BRIEF.md` (the query-swap experiment's proposed next step). Swaps, measures and decision rule: `PLAN.md`,
committed (`9d1df0f`) **before any swap was applied to a trained model**. Numbers from `tables.md`; data in
`results.json`. Reproduce: `reproduce.py self_value` (1 min on one GPU).

**Setting.** The query-swap experiment's cells: held-out histories under two goals g, g′. Where the goal matters (disjoint optimal sets),
there are 39 816 cells. Block 0's attention output at the goal token is recomputed with the goal token's own **value**
from g′, or its own **key** from g′, in all heads or one head. Only that output is replaced. The complement keeps block
0's attention natural and replaces the **goal embedding in the residual stream** with g′'s. The query-swap experiment's key-and-value
swap is reproduced exactly (0.902, identical in every model).

## 1. The value carries the goal's identity; the key and the residual embedding do not

Goal-matters cells; the share taking g′'s action (the natural run under g already does in 0.19, where the actions
coincide by chance):

| swap (from g′) | g′'s action | optimal under g | unchanged |
|---|---|---|---|
| none | 0.19 | 0.834 | 1 |
| own key and value (query swap) | 0.90 | 0.152 | 0.25 |
| **own value only** | **0.83** (0.67–0.95) | 0.174 | 0.27 |
| **own key only** | **0.20** | 0.824 | 0.98 |
| **goal embedding in the residual stream** (block 0's attention natural) | **0.19** | 0.831 | 0.99 |

* **V holds**: what the goal token sends to itself through block 0's attention decides which goal the policy pursues.
  Swapping it alone gives g′'s action in 0.83 of cells (p < 0.001 against the key).
* **The key does nothing** (0.20, unchanged 0.98). How strongly the goal token attends to itself does not carry the
  identity; what it reads from itself does.
* **The embedding the goal token carries in its residual stream does nothing** (0.19, unchanged 0.99; **C holds**).
  The network reads the goal only through block 0's self-attention value, never from the residual copy.
* Goal-neutral cells: every swap leaves the optimal rate within 0.02.

## 2. Per head: usually one dominant head, never quite alone

Effect of one head's own key-and-value swap, as a fraction of all heads' effect (g′'s action above natural):

| | |
|---|---|
| best single head, median | **0.59** (0.18–0.89) |
| models where one head carries ≥ 0.5 | 6 of 10 (0.55, 0.67, 0.64, 0.82, 0.79, 0.89) |
| which head | H3 in four models, H1 in three, H2 in three |
| the best head is the one with the largest self weight and largest self-term norm | **10 of 10** |
| sum of the four single-head effects | 0.65 (0.26–0.89) |
| best head: value-only against key-and-value swap | −0.07 |

* **H fails** at the median (0.59 against 0.75). In six models one head carries most of the identity, and in the other
  four it is spread over two or more heads (seeds 3, 8 and 9 most).
* **The carrier is predictable without any swap**: in every model it is the head that attends most to the goal token
  itself and writes the largest self term. Which head that is was settled differently in each training run.
* **The heads combine more than additively**: the single-head effects sum to 0.65 of the all-heads effect. Switching
  the goal needs several heads' self values to agree, as expected if downstream reads a combined goal signal through
  a threshold.

## 3. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **V** | own value: g′'s action ≥ 0.75, above own key | 0.833 against 0.204; p < 0.001 | **yes** |
| **K** | own key ≥ 0.75, above own value | 0.204 | no |
| **H** | best single head ≥ 0.75 × all heads | 0.59 | no |
| **C** | residual embedding − natural ≤ 0.10 | 0.001 | **yes** |

**Registered reading (V, not H): the goal's identity is the goal token's self value in block 0, spread over heads.**
Section 2 refines it: it is usually concentrated in one head, the most self-attending one, and the heads combine
superadditively.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | selfkv reproduces the query-swap experiment within 0.01 | yes | identical |
| E2 | V holds | yes | 0.83 |
| E3 | own key: g′'s action < 0.3 | yes | 0.20 |
| E4 | H holds, the head differing between models | **no** (in part) | 0.59; the head does differ |
| E5 | C holds | yes | +0.001 |
| E6 | best head's value-only swap within 0.1 of its key-and-value swap | yes | −0.07 |

5 of 6.

## Conclusions

1. **The goal's identity reaches the decision as a value read by the goal token from itself in block 0.** Swap that
   value to another goal's and the policy pursues the other goal (0.83). The key, which sets how much the token
   attends to itself, carries nothing (0.20 = natural). The embedding in the residual stream carries nothing either
   (0.19), although it is identical in information. The network has learned to take the goal from one route only.
2. **That route is usually one head**, the one that attends most to the goal token itself; which head this is differs
   between training runs. The heads' contributions combine more than additively, so in four models the identity is
   spread over two or more heads.
3. **With the goal-route-selection to query-swap experiments, the goal token's computation in block 0 is now:**
   * the self value writes the goal's identity, in a dominant self-attending head;
   * the query reads the history's goal-free values with slightly goal-dependent weights (the small goal × belief part);
   * block 0's MLP combines the two (the direct-belief-edit experiment's goal-specific belief);
   * blocks 1–3's MLPs turn that into the action preference (the goal-swap-components to cross-history-MLP experiments).

Next, if wanted: test whether the self value is a "goal code" read linearly downstream. Interpolate the swapped self
value between g and g′ and trace the decision. Or write each goal's self value into the carrier head of another
model, after aligning the two models' carrier heads.

Limits: per-head swaps one at a time (no pairs); only the goal token's own key and value (functions of goal and length);
the first decision; reward models only.
