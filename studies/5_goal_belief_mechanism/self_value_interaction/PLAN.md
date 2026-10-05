# Self-value interaction: decomposition, swaps, measures, decision rule and expectations

Written before any decomposition was fitted on a trained model or any swap of this experiment applied to one. Code: `run.py`,
smoke-tested on the untrained initial checkpoint of seed 0 only. Brief: `BRIEF.md`.

## Sites

At the goal token:
* **u = ln2(m)**: block 0's MLP input, what the MLP reads (primary; the block-0-steps experiment).
* m: the residual stream before that layer norm.
* z: the residual stream after block 0's MLP. This is the direct-belief-edit experiment's site, where the goal × belief interaction is large; it
  tests whether the MLP creates the interaction from the goal identity.

## Decomposition (natural states, fit-side histories, all three goals)

* c[g, L]: the mean state per goal and length; c̄[L] its mean over goals.
* S(b): the shared code, the mean of (state − c[g, L]) per posterior (619), shrunk by n / (n + 30).
* S_g(b): the same per posterior and goal, shrunk toward S(b).
* f(b, g) = c[g, L] + S_g(b) and f(b) = c̄[L] + S(b). So I(b, g) = f(b, g) − f(b) = **M(g, L)** + **J(b, g)**:
  * **M(g, L)** = c[g, L] − c̄[L], the goal's main effect;
  * **J(b, g)** = S_g(b) − S(b), the goal × belief part.

## Swaps

Held-out histories × the six ordered goal pairs g₁ → g₂. Block 0's attention output at the goal token is recomputed;
everything else runs naturally under g₁.

| swap | |
|---|---|
| **selfv** | the goal token's own value from g₂, all heads (the self-value experiment: the goal's identity) — primary |
| query | the goal token's query from g₂ (the query-swap experiment: most of the goal × history part of the attention output) — comparison |
| goal | the natural run under g₂ — the full change, the reference |

## Measures, per swap and site

* The state change D = x_swap − x_nat, regressed (pooled over histories and pairs) on dM = M(g₂, L) − M(g₁, L) and
  dJ = J(b, g₂) − J(b, g₁): coefficients α_M, α_J. A posterior-level table cannot capture each history's own change, so
  even the full change gives α_J < 1 (smoke test, untrained: 0.72–0.87). The registered measures are therefore relative
  to it: **ρ_J = α_J(swap) / α_J(goal)** and ρ_M likewise.
* **Shared belief fixed**: a goal-free affine decoder of b from the state (fit side, all goals). Measured: the median L1
  change of the decoded posterior under the swap, against the natural goal change (same history, so the belief is
  unchanged) and against the decoded difference between the belief-encoding-edit experiment's main pairs (the scale of a real belief change). Also
  the swap's squared change inside the shared code's subspace (top 13 principal directions of S(b)), against the
  natural goal change's.
* Sizes |dM|², |dJ|² per site.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05.

| | criterion |
|---|---|
| **J** (the brief's question, at u) | ρ_J(selfv) ≥ 0.5 |
| **M** | ρ_M(selfv) at u ≥ 0.75 |
| **F** shared belief fixed | at u: decoded change(selfv) − decoded change(goal) ≤ 0.1 × the decoded pair difference |
| **Z** after the MLP | ρ_J(selfv) at z ≥ 0.5 |
| Q (comparison) | ρ_J(query) at u, and whether it exceeds selfv's (p < 0.05) |

| result | reading |
|---|---|
| J, F | **swapping the self value moves the goal × belief interaction at the MLP input to the other goal's, and leaves the shared belief in place** |
| not J; Z; F | at the MLP input the self value moves the goal's main effect; block 0's MLP turns it into the goal × belief interaction. The input's own (small) interaction follows the query (Q) |
| not J, not Z | the interaction does not follow the self value |
| F fails | the swap also moves the shared belief component |

## Expectations

| | expectation |
|---|---|
| E1 | M holds |
| E2 | J fails: ρ_J(selfv) at u between 0.2 and 0.5 (the query-swap experiment: the own key and value carried 0.36 of the attention output's goal × history part) |
| E3 | Q: ρ_J(query) at u ≥ 0.5, above selfv |
| E4 | Z holds |
| E5 | F holds |
| E6 | |dJ|² / |dM|² at z is at least twice that at u |

## Limits known in advance

M and J are posterior-level tables; per-history goal × history structure that is not a function of the posterior is in
the residual. The linear decoder is one reading of "the shared belief"; the subspace share is a second. The first
decision; reward models only.
