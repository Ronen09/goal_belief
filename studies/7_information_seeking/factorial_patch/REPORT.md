# Factorial patching: two histories × two goals, every cell into every other, component by component

Run date: 2026-10-13. Cells, patches, measures and decision rule: `PLAN.md`, committed (`2a9aaa0`) **before any
patch of this experiment was applied to a trained model**. Numbers from `tables.md`; data in `results.json`.
Reproduce: `reproduce.py factorial_patch` (5 min on one GPU).

**Disclosures.** The code was smoke-tested on seed 0's untrained checkpoint (`_smoke/`). The first run on a trained
model stopped at a sanity assertion of my own (the natural logits recomputed on full-length sequences against the
stored ones, absolute tolerance 1e-3): the difference was 0.011 on logits of scale 9, median 0.0002, argmax agreeing
on 0.9999 of decisions; the tolerance was made relative (1e-2 of the logit scale) and nothing else was changed after
trained-model numbers were seen. Two wordings of the plan were imprecise and are corrected here: the history tokens
are positions 0 and 2 … p − 1 (position 0 is the first observation, which the plan's "2 … p − 1" left out), and the
goal token attends to position 0, so a same-goal goal-token patch is not exactly an identity (it moved 0.00–0.01 of
decisions).

**Setting.** The maze10 experiment's six transformers (four blocks of four heads, d = 128), frozen; their own test
episodes. A quartet is two same-step decisions after the first step, h1 and h2, under two goals A and B; the four
natural runs are the references. **Random quartets** (primary): 4 096 pairs with posteriors ≥ 0.5 apart × 6 goal
pairs, 24 576 per model; the decision depends on the history in 0.48 (under A) and 0.40 (under B) of them, on the goal
in 0.64, and all four natural decisions differ in 0.04 (about 1 020 quartets per model). **A-equivalent quartets**
(secondary): the equivalence experiment's pairs, 5 161 per model. A patch writes one quantity of a source cell into a
recipient cell's run, all 12 directions; "to source" is the share of the recipient's decisions switched to the
source's where the two differ, q_h the cell with the source's history and the recipient's goal, q_g the recipient's
history and the source's goal. Random floors (a random vector of the natural change's norm) switch 0.01–0.07, except
block 0's attention output (0.12 for the same-goal type: the natural change there is large). Medians (min–max) over
six models.

## 1. The decision is assembled in the decision token during block 1

| the decision token's residual, from the source cell | same goal, other history | same history, other goal | both differ: to source | both: unique attribution on four-distinct quartets, source / q_h / q_g / kept |
|---|---|---|---|---|
| entering block 1 | 0.72 (0.66–0.77) | 0.35 (0.31–0.48) | 0.42 (0.37–0.52) | 0.27 / **0.51** / 0.06 / 0.14 |
| entering block 2 | 0.89 (0.87–0.94) | 0.86 (0.73–0.93) | **0.85** (0.74–0.92) | **0.79** / 0.12 / 0.05 / 0.02 |
| entering block 3 | 0.97 | 0.95 | **0.95** (0.91–0.97) | 0.94 / 0.03 / 0.02 / 0.00 |
| at the readout (check) | 1.00 | 1.00 | 1.00 | 1.00 / 0 / 0 / 0 |

* **At the input of block 1 the token does not hold the decision; it holds the belief.** Patched from the diagonal
  cell, it switches the recipient to the source's decision in 0.42 of the differing quartets, but where the four
  natural decisions can be told apart it goes to q_h in 0.51 and to the source in 0.27: the source's *history* is
  carried in, and read under the goal the rest of the network still holds. **WHOLE failed** (registered ≥ 0.8).
* **One block later the token holds nearly everything** (0.85 to the source, 0.79 unambiguously), and after block 2
  0.95: blocks 2 and 3 read the decision token itself. Consistently, the history tokens' states entering block 2 carry
  0.05–0.07 and entering block 3 0.02, and the attention outputs of blocks 2 and 3 at the decision token carry
  0.02–0.08 (section 3).

## 2. Entering block 1: the token holds the belief, the history tokens hold the goal, the goal token is spent

| residual entering block 1, from the source | same goal, other history (the belief) | same history, other goal (the goal) | both: unique attribution, source / q_h / q_g / kept |
|---|---|---|---|
| the decision token | **0.72** | 0.35 | 0.27 / **0.51** / 0.06 / 0.14 |
| the history tokens (positions 0, 2 … p − 1) | 0.19 (0.16–0.24) | **0.41** (0.30–0.43) | 0.11 / 0.08 / **0.35** / 0.46 |
| the goal token | 0.00 | 0.10 (0.09–0.12) | 0.01 / 0.02 / 0.13 / 0.84 |
| the goal token entering block 2 / block 3 | 0.00 / 0.00 | 0.01 / 0.00 | |

* **The belief reaches the decision through the decision token's own block-0 state**, not through the history
  tokens: swapping every history token's state for another history's moves 0.19 of decisions, swapping the decision
  token's 0.72. Block 0's attention at the decision token computes the belief from the raw tokens (the predictive
  pretraining study's "recomputes the evidence from the raw tokens", and the block-1 experiment's 0.70 of the
  belief-dependent part in the token's own state). **BELIEF-L1 failed** (registered ≥ 0.5 for the history tokens).
* **The goal reaches blocks 1–3 through the history tokens and the decision token's own state, not through the goal
  token.** The goal token's residual entering block 1, swapped to the other goal, moves 0.10 of decisions, entering
  block 2 0.01, entering block 3 0.00; the history tokens' states 0.41, the decision token's own 0.35 (the three
  together are the natural switch, 1.00). Every token attended to the goal token in block 0 and carries goal
  content from then on; the goal token itself is read once. **GOAL-TOK failed** (registered ≥ 0.5).
* The asymmetry is clean in the unique attribution: the decision token's block-0 state carries the belief (0.51 to
  q_h, 0.06 to q_g), the history tokens' block-0 states carry the goal (0.35 to q_g, 0.08 to q_h).

## 3. Components: block 0's attention carries the belief, block 1's the goal, the MLPs the combined decision; no head carries anything

| sublayer output at the decision token | same goal (belief) | same history (goal) | both: to source | both: unique, source / q_h / q_g / kept |
|---|---|---|---|---|
| attn 0 | **0.71** (0.65–0.76) | 0.35 | 0.42 | 0.26 / **0.50** / 0.06 / 0.15 |
| mlp 0 | 0.62 | 0.24 | 0.30 | 0.16 / **0.46** / 0.07 / 0.31 |
| attn 1 | 0.37 (0.28–0.48) | **0.56** (0.34–0.68) | 0.45 | **0.35** / 0.11 / **0.26** / 0.27 |
| mlp 1 | 0.55 | 0.43 | 0.41 | 0.31 / 0.14 / 0.06 / 0.49 |
| attn 2 | 0.08 | 0.07 | 0.05 | 0.02 / 0.06 / 0.09 / 0.79 |
| mlp 2 | 0.49 | 0.43 | 0.40 | 0.34 / 0.05 / 0.04 / 0.55 |
| attn 3 | 0.03 | 0.03 | 0.02 | 0.01 / 0.02 / 0.05 / 0.93 |
| mlp 3 | 0.33 | 0.27 | 0.24 | 0.19 / 0.03 / 0.04 / 0.73 |
| the largest single head (head 0.0) | 0.22 | 0.07 | 0.10 | 0.10 (both type; all 16 heads ≤ 0.12 for the goal) |

* **Block 0's attention output is the belief carrier** (0.71 for the other history; patched from the diagonal cell it
  goes to q_h, 0.50) and is nearly the whole of the token's block-0 state (0.72). It carries the goal at 0.35, the
  goal token's gated value of the block-0 experiment.
* **Block 1's attention output is the largest single carrier of the goal** (0.56; ≥ 0.5 in five of six models, so
  **NO-SINGLE failed**) and the first component that carries the combined decision (0.35 unambiguously, with 0.26 to
  the goal alone). It is what the history tokens' goal-laden values bring (section 2's 0.41, the block-1 experiment's
  0.63): block 1 aggregates the goal from the history tokens into the decision token.
* **The MLPs of blocks 1–3 carry the combined decision and are overridden about half the time** (kept 0.49–0.73):
  each is one of several copies. Block 0's MLP carries the belief (0.46 to q_h). Blocks 2 and 3's attention carry
  nothing (0.02–0.08, at the random floor), and **no single head carries anything** (≤ 0.22 for the belief, ≤ 0.12
  for the goal): unlike the small maze's one goal head per model, the goal here is spread over heads and blocks.

## 4. Additivity: the MLP outputs are less additive than the logits, the attention outputs more

ι = Σ‖Δ_both − Δ_h − Δ_g‖² / Σ‖Δ_both‖² over quartets, the two-factor patch's logit change against the sum of the two
one-factor changes; for the natural logits (and for the decision token's residual, which reproduces the natural cells)
ι_nat is the additive code's failure on these quartets. Output share: the goal × history interaction's variance share
in the quantity's own natural value across the four cells.

| | ι | cos(residual, Δ_both) | ‖residual‖ / ‖Δ_both‖ | output share | ι < ι_nat (Wilcoxon p) |
|---|---|---|---|---|---|
| **natural logits (ι_nat)** | **0.52** (0.49–0.59) | 0.00 | 0.87 | 0.11 | |
| attn 0 | 0.48 (0.38–0.54) | 0.02 | 0.88 | 0.04 | 0.031 |
| attn 1 | **0.40** (0.30–0.47) | 0.01 | 0.76 | 0.05 | 0.016 |
| attn 2 | 0.53 | −0.01 | 0.88 | 0.08 | 0.72 |
| attn 3 | 0.41 (0.30–0.75) | 0.03 | 0.72 | 0.07 | 0.078 |
| mlp 0 | 0.50 | 0.01 | 0.92 | 0.06 | 0.031 |
| mlp 1 | **0.78** (0.75–0.85) | 0.07 | 1.16 | 0.11 | 1.0 |
| mlp 2 | **0.85** (0.79–0.94) | 0.11 | 1.14 | 0.13 | 1.0 |
| mlp 3 | **0.86** (0.81–0.95) | 0.11 | 1.09 | 0.13 | 1.0 |
| the history tokens entering block 1 | 0.32 | 0.00 | 0.59 | | |
| the goal token entering block 1 | 0.05 | 0.02 | 0.17 | | |
| heads | 0.16–0.54 | ≈ 0 | 0.46–0.86 | 0.03–0.08 | |

* **The natural logits are far from additive on these quartets** (ι_nat 0.52: the interaction residual is 0.87 of
  the diagonal change's norm), as the additive code's 0.81 of the return and the goal-belief sections already said.
* **The attention outputs of blocks 0 and 1 and block 0's MLP are more additive than the logits** (0.40–0.50,
  p ≤ 0.031); **the MLP outputs of blocks 1–3 are less** (0.78–0.86, p = 1), with a positive cosine (0.07–0.11) and a
  residual longer than the change itself: the two one-factor MLP patches fall short of the two-factor one along its
  own direction, the interaction in an MLP's output reinforcing its main effects. **ADD failed**; the least additive
  components are mlp 3, mlp 2, mlp 1 in every model, not attn 0 and attn 1 as expected.
* **The order follows the components' own interaction content** (Spearman between ι and the output share over the
  eight sublayers 0.64–0.93, **RANK held**), and this is close to an identity rather than a finding: with a linear
  readout of the patched quantity, ι is four times the squared norm of the quantity's own double difference over
  its diagonal difference, so ι ranks the components by their relative interaction content, and the logits sit
  between the attention outputs (0.04–0.08) and the late MLPs (0.11–0.13). The causal statement that survives is the
  comparison itself: what the late MLPs write is more interactive than what the policy reads out, what attention
  writes is less. This agrees with the gate experiments' picture of the MLPs amplifying an interaction that attention
  assembles (block 1's MLP doubling its share), and says that the amplification goes past the logits' own level.

## 5. Symmetry, and the A-equivalent quartets

* **The four recipients agree on the random quartets** (dec@resid2, same goal: spread across the four roles
  0.017–0.038 within a model, **SYM held**; dec@resid1 0.054, largest 0.108, where the A and B roles are the lower-
  and higher-indexed goal and differ by goal identity, not by which history is called the recipient).
* **A-equivalent quartets.** The same pattern, with the natural logits more interactive by construction (ι_nat 0.88)
  and every late component with it (mlp 1–3 1.2–1.4, attn 1–3 0.65–0.73). The same-goal patch of the decision
  token's block-0 state switches fewer decisions into the A recipients than into the B ones (0.60 against 0.65) and
  has fewer informative quartets there (2 066 against 3 124), as the pairs' construction predicts: the beliefs are
  A-equivalent, so under A the two histories' decisions agree more often (0.48 differ against 0.67 under B).

## Scorecard

| | criterion | numbers | held |
|---|---|---|---|
| WHOLE | the decision token entering block 1, both: to source ≥ 0.8 | 0.42 (0.37–0.52); entering block 2: 0.85 | **no** |
| BELIEF-L1 | the history tokens entering block 1, same goal: ≥ 0.5 | 0.19 (0.16–0.24) | **no** |
| GOAL-TOK | the goal token entering block 1, same history: ≥ 0.5 | 0.10 (0.09–0.12) | **no** |
| NO-SINGLE | no sublayer output carries the goal ≥ 0.5 in ≥ 5 of 6 models | attn 1: 0.56, five models ≥ 0.5 | **no** |
| ADD | every sublayer ι < ι_nat | attn 0, attn 1, mlp 0 yes (p ≤ 0.031); attn 2, attn 3 not significant; mlp 1–3 the reverse | **no** |
| RANK | Spearman(ι, output share) ≥ 0.5 in ≥ 5 of 6 | 0.64–0.93 in all six | yes |
| SYM | four recipients within 0.05, dec@resid2 same goal | spread ≤ 0.038 | yes |

| | expectation | held |
|---|---|---|
| E1 | the decision token holds the decision from the input of block 1 on | **in part**: entering block 2 (0.85) and block 3 (0.95) yes, entering block 1 no (0.42, and 0.27 unambiguously) |
| E2 | the history tokens entering block 1 carry the belief ≥ 0.5, falling across sites | **no** (0.19), though falling (0.07, 0.02) |
| E3 | the goal token entering block 1 carries the goal ≥ 0.5, falling across sites | **no** (0.10), though falling (0.01, 0.00) |
| E4 | the token's block-0 state goes to the source, not q_h or q_g; the history tokens go to q_h more than q_g | **no**, both reversed: the token goes to q_h (0.52; unique 0.51 against 0.27), the history tokens to q_g (0.36 against 0.18) |
| E5 | NO-SINGLE; attn 0 the largest single carrier of the goal | **no**: attn 1 (0.56) in five models, attn 0 0.35 |
| E6 | ADD and RANK; attn 0 and attn 1 the least additive | **in part**: RANK yes; ADD no; the least additive are mlp 3, 2, 1 in every model |
| E7 | SYM; A-equivalent set: fewer switches into the A recipients | yes (0.60 against 0.65; 2 066 against 3 124 informative) |
| E8 | four-distinct ≤ 0.1 of quartets; unique attribution within 0.1 of the pooled share | **in part**: 0.04 yes; 0.27 against 0.42 no (the pooled share counts quartets where the source's decision coincides with q_h's) |

Two of seven registered criteria held; of the expectations, one held, three in part, four failed.

## Reading

Patching in every direction gives the maze10 policy's routing in one table. Block 0 reads the raw history at the
decision token and writes the belief there; it reads the goal token once and writes goal content into every token.
Block 1 brings the goal from the history tokens into the decision token and combines it with the belief; from its
output on, the decision token holds the decision (0.85, then 0.95) and the rest of the sequence, the goal token
included, is no longer consulted (≤ 0.07). No head carries any of it alone. The goal × history interaction is written
most strongly by the MLPs of blocks 1–3, whose outputs are less additive than the logits the policy reads out, while
the attention outputs are more additive than the logits. My expectations about *where* things are at the input of
block 1 were wrong in a consistent direction: I expected the decision token, the history tokens and the goal token to
hold the decision, the belief and the goal respectively; they hold the belief, the goal and nothing.

## Limits

Patches are at the decision token or whole token sets, never finer than a head, and a value from another cell is off
the recipient's manifold; the attribution reads what the rest of the network does with it. The both-type one-factor
decomposition is relative to the recipient's cell (SYM shows the four recipients agree). ι for a component is, to the
extent the downstream is linear, a normalised measure of that component's own interaction content, so RANK is nearly
an identity and only the comparison with ι_nat is a causal statement. The four-distinct quartets are 0.04 of the
random set; the unique attribution rests on about 1 000 quartets per model. The same models and episodes as the
H-simplex sections 9–16.
