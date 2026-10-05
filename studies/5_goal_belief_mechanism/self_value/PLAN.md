# Self value: swaps, measures, decision rule and expectations

Written before any swap of this experiment was applied to a trained model. Code: `run.py`, smoke-tested on the untrained
initial checkpoint of seed 0 only: the recomputed attention gives the natural decision (6·10⁻⁸). Brief: `BRIEF.md`.

## Cells and swaps

The query-swap experiment's cells: 12 000 held-out histories under ordered goal pairs (g, g′). *Goal-matters* cells (39 816) have
disjoint optimal sets under the two goals; *goal-neutral* cells (32 180) have identical ones. Block 0's attention output
at the goal token is recomputed with parts of the goal token's own key and value taken from g′. Only that output is
replaced; the rest runs naturally under g.

| swap | from g′ |
|---|---|
| selfkv | own key and value, all heads (the query-swap experiment's; must reproduce its 0.90) |
| **selfv** | own value, all heads; key kept: what the goal token sends to itself |
| **selfk** | own key, all heads; value kept: how much it attends to itself (and so to the history) |
| selfkv_head k | own key and value in head k only |
| selfv_head k | own value in head k only |
| **embedding_residual** | block 0's attention kept natural; the goal embedding in the residual stream from g′ (m = emb(g′) + a(h, g)): the complement |

Measures, per cell class: the share taking the same action as the natural run under g′ ("g′'s action"), optimal under
g, unchanged. Descriptive: each head's attention weight on the goal token itself, and the norm of its self term.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05. Goal-matters cells.

| | criterion |
|---|---|
| **V** the value carries the identity | g′'s action under selfv ≥ 0.75, and above selfk |
| **K** the key carries it | g′'s action under selfk ≥ 0.75, and above selfv |
| **H** one head carries it | in each model, the best single head's selfkv swap reaches ≥ 0.75 × the all-heads selfkv swap (measured as g′'s action minus natural's); median over models |
| **C** complement | embedding_residual: g′'s action ≤ natural's + 0.10 |

| result | reading |
|---|---|
| V, H | **the goal's identity is one head's self value in block 0** |
| V, not H | it is the goal token's self value, spread over heads |
| K | it is how much the goal token attends to itself rather than to the history |
| neither V nor K | it needs both together (the value and its weight) |

C is a check of the query-swap experiment's reading: the residual copy of the embedding does not carry the identity.

## Expectations

| | expectation |
|---|---|
| E1 | selfkv reproduces the query-swap experiment's 0.90 (within 0.01) |
| E2 | V holds |
| E3 | selfk: g′'s action < 0.3 |
| E4 | H holds; the head differs between models |
| E5 | C holds |
| E6 | the best head's selfv swap is within 0.1 of its selfkv swap |

## Limits known in advance

Swaps of goal-and-length functions only (the goal token's own key and value); per-head swaps one at a time (no pairs);
the first decision; reward models only.
