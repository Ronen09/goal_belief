# The goal × belief part, step by step from block 0's attention output to its MLP input (block-0 steps)

Run date: 2026-10-02. Brief: `BRIEF.md`. Steps, measures and decision rule: `PLAN.md`, committed (`113eacf`) **before any
measure of this experiment was taken on a trained model**. Numbers from `tables.md`; data in `results.json` and
`identity_check.json`. Reproduce: `reproduce.py block0_steps` (2 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen; the goal token in block 0. Every state between the attention output
and the MLP, exactly:

* **a**, block 0's attention output. It splits into the goal token's attention to itself (*self*: its value is a function
  of goal and prefix length only) and the prefix tokens' terms (*prefix*: goal-free values, weighted through a query that
  contains the goal). It also splits into four heads.
* **m** = emb(g, L) + a: the attention-belief-edit to nonlinear-belief-edit experiments's site. The embedding is a function of goal and length only.
* **u** = ln2(m): the MLP's actual input. The layer norm centres m, divides by its scale σ(h, g), and applies a gain and
  bias.

At each step: the goal × history interaction share (attention belief edit), and held-out fits of shared and goal-conditioned
encodings of the posterior. At a, m and u: the nonlinear-belief-edit experiment's table edits on the goal-dependent pairs with a shared map, the
recipient's goal's map, and another goal's map.

## 1. Nothing but the embedding lies between the attention output and m

* m − emb − a = 1·10⁻⁷ at most; self and prefix terms sum to a within 5·10⁻⁷.
* **Edits with the same vectors at a and at m give the same outputs** (largest |Δp| 6·10⁻⁶, `identity_check.py`).
* The registered form of this check failed: edits built from each step's own fit differed by up to 2·10⁻³. The cause
  is in the fit, not the network. `Affine` is a ridge regression that also shrinks the (goal, length) offsets. Those
  offsets differ between a and m by the embedding, so the two fitted maps differ slightly. I found this after the run;
  `identity_check.py` is the corrected check, and the registered number is reported alongside.

## 2. The attention output is already goal-conditioned (A holds)

| at the attention output | value |
|---|---|
| goal × history interaction share | **0.045** (0.028–0.070); ≥ 0.02 in every model |
| shared table R² (within goal, length) | 0.758 |
| goal-conditioned table R² | 0.784; **gain 0.025** |
| goal-dependent pairs, table edit / whole: shared / recipient's goal / another goal | 0.64 / **0.78** / **0.50** |
| right − wrong | 0.26 (0.20–0.37); right above wrong in all ten models, p < 0.001 |

The attention output and m have identical interaction and fits, as they must. The edits at the two points are the
same edits. The attention-belief-edit to nonlinear-belief-edit experiments's goal × belief part is fully present in block 0's attention output.

**Where in attention** (descriptive):

| | interaction share | interaction size | goal × posterior gain |
|---|---|---|---|
| self term (the goal token reading itself) | 0.51 | 0.041 | 0.37 |
| prefix term (the history, read with the goal's query) | 0.03 | **0.130** | 0.014 |
| attention output | 0.045 | 0.186 | 0.025 |

* **The prefix term carries most of the interaction's size**: 3 × the self term, in all ten models. The goal changes
  how the goal token weights the history's goal-free values.
* The self term is small but half goal × history. Its value is the goal's own embedding; how much the token attends to
  itself depends on the history, so the goal vector enters scaled by the history.
* **No single head**: the largest head holds 0.43 (0.31–0.65) of the per-head interaction, and which one differs
  between models (H2 in five, H3 in three, H1 in two).

## 3. The layer norm before the MLP adds a little (LN holds, by the representation criterion only)

| | m (and a) | u = ln2(m), the MLP input | u with σ fixed across goals |
|---|---|---|---|
| interaction share | 0.045 | **0.070** | 0.062 |
| goal × posterior gain (R²) | 0.025 | 0.043 | |
| goal-dependent pairs, edits / that step's whole: shared / right goal / wrong goal | 0.64 / 0.78 / 0.50 | 0.64 / 0.84 / 0.50 | |
| right − wrong | 0.26 | 0.35 | |

* **The interaction rises by 0.025 through the layer norm**, which meets the registered criterion. Most of the rise
  does not come from the goal: with σ held at its mean over the three goals, it is still 0.062. The rest comes from
  normalising by a history-dependent scale, which multiplies the goal's offset by a history-dependent number, and from
  the per-dimension gain. σ varies across goals for a fixed history by 8 % (CV 0.079).
* **Behaviourally the layer norm adds nothing significant.** The shared edit's ratio is unchanged (0.64; p 0.42).
  The right-minus-wrong gap grows by 0.07 (p 0.065; registered 0.05 with p < 0.05).
* Note: edits at u change only what block 0's MLP reads. The attention output itself still flows on to blocks 1–3 in
  the residual stream, unedited. So whole replacement at u reaches a lower goal-dependent rate (0.224) than at m
  (0.276). Relative to no edit (0.038), **0.78 of the route's effect on these pairs passes through block 0's MLP**; the
  rest passes directly to later blocks.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **A** | attention output: interaction ≥ 0.02 every model; goal × posterior gain ≥ 0.01; right goal above wrong | min 0.028; 0.025; p < 0.001 | **yes** |
| **N** | m − emb − a ≤ 10⁻⁵; edits identical within 10⁻⁴ | 1·10⁻⁷; 6·10⁻⁶ with identical vectors (registered form: 2·10⁻³, a ridge-fit artefact) | **yes** (corrected check) |
| **LN** | any: interaction +0.02; shared ratio −0.05; right − wrong +0.05 | +0.025; −0.02 (p 0.42); +0.07 (p 0.065) | **yes** (first only) |

**Registered reading (A, N, LN): the goal × belief part is in the attention output, and the layer norm's
goal-dependent scale adds to it.** Taken with §3, the addition is small: its goal-dependent part is about 0.008 of
interaction share, and its effect on behaviour is not significant.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | A holds | yes | |
| E2 | N holds | in part | the identities hold; the registered edit comparison failed through the ridge fit, not the network |
| E3 | LN does not hold; σ CV < 0.05 | **no** | LN holds on interaction (+0.025); CV 0.079 |
| E4 | prefix term's interaction size exceeds the self term's | yes | 0.130 against 0.041, 10 of 10 |
| E5 | one head carries over half | **no** | 0.43, and the head varies by model |
| E6 | mid reproduces the nonlinear-belief-edit experiment's tables within 0.05 | yes | 0.64 against 0.65; 0.78 against 0.81 |

3 of 6, one in part.

## Conclusions

1. **Yes, block 0's attention output already holds the goal-conditioned belief part.** Everything the attention-belief-edit to nonlinear-belief-edit experiments found at
   the MLP's residual input (interaction 4.5 %, goal × posterior gain 0.025, right-goal edits 0.78 against wrong-goal
   0.50 on goal-dependent pairs) is there, identically. Only the embedding, a function of goal and length, is added
   between the two. Edits at the two points give the same outputs.
2. **It comes mainly from reading the history with a goal-dependent query.** The goal changes how the goal token
   weights the prefix tokens' goal-free values. That term carries three times the interaction of the goal token's own
   value, and it is spread over heads differently in each model.
3. **The layer norm in front of the MLP adds a little goal × history structure, mostly not through the goal.** Its
   goal-dependent scale adds about 0.008 of interaction share. Normalisation and gain add the rest of the 0.025.
   Behaviour on the goal-dependent pairs does not change significantly.
4. **The MLP is where most of it is used.** 0.78 of the route's effect on goal-dependent pairs passes through block 0's
   MLP input; the attention output's direct path to later blocks carries the rest.

So the goal enters the belief in two places. Block 0's attention reads the evidence with a goal-dependent query (small,
but read by the policy). Block 0's MLP then combines the belief with the goal much more strongly (the direct-belief-edit experiment: goal-specific
maps 0.11 R² better after it).

Next, if wanted: test the query directly. Recompute block 0's attention at the goal token with the query from another
goal's run (values and keys kept), and the reverse. If the goal-dependent belief part moves with the query, the
mechanism is confirmed causally rather than by decomposition.

Limits: interaction shares are not invariant to the layer norm's per-dimension gain, so comparisons across that step
compare different metrics; behavioural comparisons are each relative to that step's own whole. Linear and table
encodings; main pairs only; reward models only; the first decision.
