# Round 35: steps, measures, decision rule and expectations

Written before any measure of this round was taken on a trained model. Code: `run.py`, smoke-tested on the untrained
initial checkpoint of seed 0 only. There, the three identities below hold to 6·10⁻⁸, 2·10⁻⁷ and 8·10⁻⁶. Brief: `BRIEF.md`.

## The steps, exactly (goal token, position p = 1 + L)

| step | state | what can depend on the goal |
|---|---|---|
| **attn** | a(h,g), block 0's attention output | |
| — self | the goal token's attention to itself, plus the output bias | its value is a function of (g, L) only; its weight depends on h and g |
| — prefix | the prefix tokens' terms | goal-free values; weights from a query that contains the goal |
| — head k | one head's self and prefix terms | |
| **mid** | m = emb(g, L) + a | rounds 33–34's site. The embedding is a function of (g, L) only, so this step is exactly additive |
| **ln** | u = ln2(m) = γ (m − mean m) / σ(m) + β | **the MLP's input**. The division by σ(h, g) is the only nonlinear step between attn and the MLP |

Checks: m − emb − a = 0; the self and prefix terms sum to a; edits at attn and at mid give identical outputs.

## Measures

**Representation**, on every step (held-out histories, all three goals):

* **Interaction share**: round 33's goal × history interaction, with its absolute size (squared norm per history).
  Shares at attn and mid must be equal (additive step).
* **Goal × posterior fit**: held-out R² within (goal, length) of four encodings of the posterior. Linear shared
  c_{g,L} + E b; linear goal-specific; table shared (the linear fit plus the mean residual for each posterior, shrunk
  by n / (n + 30)); table goal-conditioned (per posterior and goal). **Goal × posterior gain** = R²(table_goal) −
  R²(table).
* Layer norm: the coefficient of variation of σ across the three goals for a fixed history. Also the interaction share
  of u recomputed with σ replaced by its mean over the three goals (**ln_sigma_fixed**). The difference ln −
  ln_sigma_fixed is what the goal-dependent scale adds.

**Behaviour**, at attn, mid and ln, main pairs (round 27's; goal-dependent pairs 5 653; preserve cells). Edits at the
goal token with the step's own tables: none, whole, table (shared), table_goal (recipient's goal), table_wrong (another
goal's, both averaged). Goal-dependent rate / whole as in round 34; main-pair S; harm.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests over the ten, one-sided, p < 0.05.

| | question | criterion |
|---|---|---|
| **A** | does block 0's attention output already hold a goal-conditioned belief part? | at attn: interaction share ≥ 0.02 in every model; goal × posterior gain ≥ 0.01; goal-dependent rate table_goal above table_wrong (p < 0.05) |
| **N** | does anything but the embedding lie between the attention output and the MLP's residual input? | identities hold (m − emb − a ≤ 10⁻⁵; edits at attn and mid identical within 10⁻⁴) |
| **LN** | does the layer norm before the MLP add goal conditioning? | any of: interaction share at ln ≥ mid's + 0.02; goal-dependent table ratio at ln ≤ mid's − 0.05 (p < 0.05); right-minus-wrong gap (table_goal − table_wrong, / whole) at ln ≥ mid's + 0.05 (p < 0.05) |

| result | reading |
|---|---|
| A, N, not LN | **the goal × belief part is already in block 0's attention output, and nothing between it and the MLP changes it** |
| A, N, LN | it is in the attention output, and the layer norm's goal-dependent scale adds to it |
| not A | the attention output is goal-free beyond an offset; the part rounds 33–34 found comes from elsewhere (only possible if N or LN fails) |

**Descriptive: where in attention.** The interaction's absolute size in the self term and in the prefix term (their sum
can differ from attn's through their cross term), and per head.

## Expectations

| | expectation |
|---|---|
| E1 | A holds |
| E2 | N holds |
| E3 | LN does not hold: σ varies little across goals (CV < 0.05) |
| E4 | the prefix term's interaction size exceeds the self term's |
| E5 | one head carries over half of the summed per-head interaction size (round 30: L0.H1 was the most goal-sensitive) |
| E6 | at mid, the goal-dependent ratios reproduce round 34's table results within 0.05 (table 0.65, table_goal 0.81) |

## Limits known in advance

Representation measures are linear and table encodings of the posterior; the interaction share includes non-posterior
history information. Main pairs only (no one-step or equivalent-recipient sets). Reward models only; the first decision.
