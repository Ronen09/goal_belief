# Factorial patching: two histories × two goals, every cell into every other, component by component

Written before `run.py` was run on any trained model. Models: the maze10 experiment's six reward-trained
transformers (55-cell aliased maze, four goals, four blocks of four heads, d = 128). Code: `run.py`, `tables.py`; the
H-simplex experiment's `run.py`, `causal.py` and the maze10 experiment's `measure.py` supply the episodes and the
states under every goal. Nothing is fitted: every number is a natural run or a patched run of the frozen network.
Brief: given in conversation (do the patching experiments cover the full 2 × 2 of histories and goals, in every
direction?). They do not: the small-maze experiments patch the donor cell into the recipient only (maze-belief, pair
types), the goal-swap experiments cross goals on one history, and the H-simplex sections 9, 12, 13 and 16 add or swap
in one direction (B's content under A, or differences written under B added under B).

## The question

For a decision made at step t, a history h and a goal g, the network's state and decision are functions of the
cell (h, g). Two histories h1, h2 at the same step and two goals A, B give four cells. The additive code says the
decision logits are L(h, g) ≈ H(h) + G(g, t); the H-simplex sections 8–16 say the state holds the belief in
goal-specific coordinates, made by block 0's gating of the goal token's value and rebuilt by block 1's attention, and
that the MLPs are close to affine. The factorial asks of each component and site, causally and in every direction:

1. **What a component carries.** A component's output taken from a source cell and written into a recipient cell
   moves the recipient's decision to one of three places: the source's own decision (the component carries a
   decision, history and goal combined), the decision for the source's history under the recipient's goal (it
   carries the belief, read under whatever goal the rest of the network holds), or the decision for the recipient's
   history under the source's goal (it carries the goal), or nowhere (the rest of the network overrides it). The
   four natural runs are the references; with four actions, the cells where all four natural decisions differ
   separate the three readings without ambiguity.
2. **Whether a component's causal effect is additive in history and goal.** The source cell (h2, B) differs from the
   recipient (h1, A) in both factors; (h2, A) and (h1, B) in one each. If the component's contribution to the logits
   is additive, the logit change of the two-factor patch is the sum of the two one-factor patches' changes. The
   departure is the component's causal goal × history interaction, measured by patching rather than by regressing
   its output (the interaction-removal experiment did the latter in the small maze).
3. **Symmetry.** Every one of the 4 × 3 directed (source → recipient) patches is run, so a reading does not depend on
   which cell was called the recipient. The symmetric directions must agree on the random pairs (the two histories
   are exchangeable there) and may not on the A-equivalent pairs (where goal A has a role).

## Cells

Test decisions of the H-simplex experiment's test episodes (seed 78 / 79 draw; every model's own episodes), after
the first step (so the decision token has a history to attend to). The four natural runs of each quartet share the
step, so goal × step offsets cancel in every double difference.

* **Random quartets (primary).** 4 096 same-step pairs of decisions per model with posteriors ≥ 0.5 apart in L1 (the
  block-1 experiment's draw), each crossed with the six unordered goal pairs {A, B}: 24 576 quartets per model. The
  pair is drawn without order, so h1 and h2 are exchangeable.
* **A-equivalent quartets (secondary).** The equivalence experiment's pairs (section 9: QMDP profiles within 0.5
  effective step, the same optimal set and goal A worth ≥ 0.4 under A; ≥ 1 step apart with disjoint optimal sets
  under B), about 5 300 per model, each one quartet with its own ordered (A, B). Here h1 and h2 are exchangeable
  but A and B are not.

Natural decisions d(h, g) = argmax of the logits in each cell; natural logits L(h, g), action-centred. Per quartet the
pattern of the four decisions is recorded: **4-distinct** (all pairwise different), and which pairs coincide.

## Patches

A patch takes one quantity at the decision token p (or at a set of tokens) from the source cell's natural run and
writes it into the recipient cell's run, everything else natural; the logits are read at p under the recipient's goal
token. Sources: each of the three other cells. Recipients: each of the four cells. So 12 directed patches per quantity
per quartet, grouped by type: **same-goal** (the other history), **same-history** (the other goal), **both** (the
diagonal cell).

| quantity | sites | what it tests |
|---|---|---|
| `dec@resid l` the decision token's residual | l = 1, 2, 3, 4 | everything the token holds entering block l (l = 4: the state the readout sees; a sanity check, must give the source's decision) |
| `hist@resid l` every history token's residual (positions 2 … p − 1) | l = 1, 2, 3 | what the later blocks read from the history tokens; for a same-history source this is the goal content written into the history tokens |
| `goaltok@resid l` the goal token's residual (position 1) | l = 1, 2, 3 | the goal as blocks ≥ l see it (the goal token attends only to positions 0–1, so its state is history-free); for a same-goal source an identity |
| `attn l`, `mlp l` the component's output at p | l = 0 … 3 | each sublayer's contribution alone |
| `head l.k` one head's output at p | 16 heads | the goal head(s) and the belief heads |
| `random` | at `dec@resid 2` and at every `attn l`, `mlp l` | a random vector of the natural change's norm added at the recipient (floor) |

The goal token's own state at `goaltok@resid 1` from a same-history source is block 0's output at position 1 under
the other goal plus the other goal's embedding: patching it changes the goal for blocks 1–3 and leaves block 0's
reading of the goal in place; `goaltok@resid 0` would be the natural cell under the other goal and is not run.
`dec@resid 4` reproduces the source's logits exactly and is kept only as the check on the plumbing.

## Measures

For a directed patch into recipient r = (h, g) from source s, with the patched decision d̂ and logits L̂, and the
other two cells q_h = (h', g) (the source's history, the recipient's goal) and q_g = (h, g') (the recipient's history,
the source's goal):

**A. Where the decision goes.** On quartets where the target cell's decision differs from the recipient's:
* **to source** P(d̂ = d_s | d_s ≠ d_r); for the same-goal and same-history types the source is also q_h or q_g, so
  this is the belief carried or the goal carried;
* **to q_h** and **to q_g** (the both type only): P(d̂ = d_{q_h} | d_{q_h} ≠ d_r) and the same for q_g;
* **kept** P(d̂ = d_r), **elsewhere** the rest;
* **unique attribution**: on the 4-distinct quartets, the four shares (source / q_h / q_g / kept) sum to one and
  are read directly. Their count per model is reported; they are expected to be a small minority.
* **logit transfer** mean over quartets of ⟨L̂ − L_r, L_s − L_r⟩ / ‖L_s − L_r‖² (and the same toward q_h, q_g).

**B. Additivity.** For a quantity and a recipient, the logit changes of the three patches Δ_h (same-goal source),
Δ_g (same-history source), Δ_both (the diagonal source). The **causal interaction** is
ι = 1 − R²(Δ_both ; Δ_h + Δ_g) over the quartets (the residual's variance over Δ_both's), and its signed
counterpart: the mean cosine between Δ_both − Δ_h − Δ_g and Δ_both. The natural reference is the same statistic on
the natural logits: ι_nat = 1 − R²(L(h2, B) − L(h1, A) ; [L(h2, A) − L(h1, A)] + [L(h1, B) − L(h1, A)]), which is
the additive code's failure at the logits on these quartets. `dec@resid l` reproduces the natural cells and gives
ι_nat at every site by construction (another check). The eight component outputs and the sixteen heads give the
per-component ι. Also per component the variance share of the goal × history interaction *in its output* at p,
computed from the four natural runs (the double difference's share), for the comparison in E4.

**C. Symmetry.** For each quantity and type, the share "to source" with (h1, A) as recipient against (h2, A) and
against (h1, B), (h2, B): on the random quartets the four recipients must agree within sampling error for the
same-goal type pooled over goals; the both and same-history types compared between the two goal orders.

**D. Decision-level readouts.** The share of quartets where the decision depends on the history (d(h1, g) ≠ d(h2, g)
for the goal g), on the goal (d(h, A) ≠ d(h, B)), on both; and the share of 4-distinct quartets.

All shares are means over quartets per model and goal pair; reported as medians (min–max) over the six models of the
mean over the six goal pairs (random quartets) or the twelve ordered goal pairs (A-equivalent quartets).

## Decision rule

Six models; exact one-sided Wilcoxon signed-rank tests across models where a comparison is registered, p < 0.05;
otherwise the median with its range.

| | criterion |
|---|---|
| **WHOLE** | `dec@resid 1`, both type: to source ≥ 0.8 (section 6's 0.96 of H was the whole token after block 1 on the goal-averaged state) |
| **BELIEF-L1** | `hist@resid 1`, same-goal type: to source ≥ 0.5 (section 13: the history tokens' values carry 0.63 of block 1's attention change) |
| **GOAL-TOK** | `goaltok@resid 1`, same-history type: to source ≥ 0.5 (the goal for blocks 1–3 without block 0's reading) |
| **NO-SINGLE** | no single `attn l` or `mlp l`, same-history type, reaches to source ≥ 0.5 in ≥ 5 of 6 models |
| **ADD** | for every component, ι < ι_nat (a component's contribution less interactive than the whole) |
| **RANK** | across the eight components, Spearman between ι and the interaction share of the component's own output ≥ 0.5 in ≥ 5 of 6 models |
| **SYM** | random quartets, same-goal type, `dec@resid 2`: the four recipients' to-source shares within 0.05 of each other in every model |

## Expectations

| | expectation |
|---|---|
| E1 | WHOLE holds; `dec@resid 2` and `3` likewise ≥ 0.8; `dec@resid 4` = 1 |
| E2 | BELIEF-L1 holds, and falls across sites (sections 6 and 13: after block 1 the decision token holds the belief itself; `hist@resid 2` < `hist@resid 1`) |
| E3 | GOAL-TOK holds at site 1 and falls across sites (the goal is read at every block: `goaltok@resid 3` < 0.3) |
| E4 | in the both type, `dec@resid 1` goes to the source, not to q_h or q_g (to source ≥ 0.8, to q_h and to q_g each ≤ 0.2 where distinct); `hist@resid 1` from the both source goes to q_h more than to q_g (belief in, goal from the recipient's own token) |
| E5 | NO-SINGLE holds; `attn 0` from a same-history source is the largest single carrier of the goal (section 12: the goal token's value route switches 0.24 of goal-dependent decisions) |
| E6 | ADD holds; RANK holds; the least additive components are `attn 0` and `attn 1` (sections 15 and 16) with `mlp 1` close behind (its output's interaction share is the largest, 0.15) |
| E7 | SYM holds on the random quartets; on the A-equivalent quartets the same-goal patch into (·, A) switches fewer decisions than into (·, B) (by the pairs' construction, d(h1, A) = d(h2, A) in 0.60 of them, so the A recipients have fewer informative quartets and lower to-source shares where the decisions do differ) |
| E8 | 4-distinct quartets are ≤ 0.1 of the random quartets; on them the unique attribution agrees with the pooled shares within 0.1 for `dec@resid 1` |

## Limits known in advance

Patches are at the decision token (and whole token sets), never finer than a head; a head's output from another
cell is off the recipient's manifold, so the attribution reads what the rest of the network does with it, not what
the head encodes. The both-type one-factor decomposition is defined with respect to the recipient's cell, so ι is a
recipient-relative quantity (the four recipients give four values; their agreement is part of SYM). No fitting means
no held-out concern; the A-equivalent quartets are a selected set, as in section 9. The same models and episodes as
sections 9–16, so the result is not an independent replication of those.
