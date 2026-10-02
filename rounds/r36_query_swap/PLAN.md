# Round 36: swaps, measures, decision rule and expectations

Written before any swap of this round was applied to a trained model. Code: `run.py`, smoke-tested on the untrained
initial checkpoint of seed 0 only. There, swapping both q and the goal token's k, v reproduces the other goal's output
exactly (0), and the recomputed attention gives the natural decision (6·10⁻⁸). Brief: `BRIEF.md`.

## Swaps

In block 0 the goal token's input is its embedding alone, so its query q, key k and value v are functions of (goal,
prefix length) only. For history h under goal g, block 0's attention output at the goal token is recomputed with parts
taken from the same history under another goal g′:

| swap | from g′ | what changes |
|---|---|---|
| **query** | the goal token's q | which history tokens are read, and how much (the prefix term's weights, and the self weight) |
| selfkv | the goal token's own k and v | what the goal token sends to itself, and how much it attracts |
| both | q, k, v | equals the natural output under g′ (check) |

## Measures

**Representation** (held-out histories, all six ordered goal pairs). D_k = a_k(h, g) − a(h, g), with its mean over
histories within (g, g′, L) removed, which leaves the history-dependent part of the goal's effect. The **share** carried
by each swap is Σ⟨D_k, D_both⟩ / Σ|D_both|². The same is computed on the posterior-explained part (means per posterior,
posteriors with ≥ 5 histories).

**Behaviour**: the network runs naturally under g except that block 0's attention output at the goal token is replaced
by the recomputed one. Cells: 12 000 held-out histories × ordered goal pairs, split by the solver into *goal-matters*
(disjoint optimal sets under g and g′) and *goal-neutral* (identical sets). Measured: optimal under g, the same action
as the natural run under g′, and unchanged.

**Reference**: the ten untrained initial checkpoints, representation only (the architecture's own split).

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05.

| | criterion |
|---|---|
| **Q** | share(query) ≥ 0.5, and above share(selfkv) |
| **Qp** | on the posterior-explained part: share(query) ≥ 0.5 |
| **B** | goal-matters cells: optimal(natural) − optimal(query) ≥ 0.05, p < 0.05; and that drop larger than on goal-neutral cells, p < 0.05 |

| result | reading |
|---|---|
| Q, Qp, B | **the goal reaches the belief through block 0's query, and the policy relies on it where the goal matters** |
| Q, Qp, not B | the query carries it, but the decision does not depend on it (later blocks compensate) |
| not Q | the goal token's own key and value carry it |

## Expectations

| | expectation |
|---|---|
| E1 | Q holds |
| E2 | Qp holds |
| E3 | B holds |
| E4 | share(query) + share(selfkv) within 0.1 of 1 |
| E5 | swapping both, goal-matters cells: the same action as under g′ in < 0.2 (the goal's decision is made later, round 30) |
| E6 | the trained share(query) is within 0.1 of the untrained reference |

## Limits known in advance

A swap of all heads' queries at once. The representation shares are projections, not a variance decomposition. Reward
models only; the first decision.
