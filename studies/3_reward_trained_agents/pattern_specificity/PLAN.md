# Pattern specificity: interventions, controls, measures and expectations

Written before any edit of this experiment was run. The belief-edit experiment's `pattern` and `pattern shift` edits were post hoc; this
round fixes them and tests them against controls. Code: `goalgeo/mazeedit.py`, `studies/3_reward_trained_agents/pattern_specificity/run.py`.

## What is frozen

* **Models**: the maze-belief experiment's six (reward-trained seeds 1, 3, 4; supervised; seeds 0, 2). No training.
* **Interface**: the residual stream at the prefix tokens entering block 1. The belief-edit experiment: entering block 2 is too late.
* **Construction**: per model, the decoder W, the covariance Σ, the pattern span (columns of ΣW, rank k) and the
  shift map ΣW (WᵀΣW)⁺ Wᵀ, exactly as in the belief-edit experiment, fitted on the maze-belief experiment's fitting histories. Every control subspace
  below is fitted on the same histories. All are written to `subspaces/` before any evaluation pair is drawn.
* **Evaluation histories**: the held-out side (by the prefix hash) of a new bank of 80 000 histories (seed 210021),
  pairs drawn with seed 21. No history of this bank was used in the belief-edit experiment; because prefixes are short, some prefix
  token sequences recur. Pair definitions (mode, uncertainty, similar), the hybrid reference, `moved`, solver gain,
  the arbitrary-behaviour measure and the raw-route ablation are the belief-edit experiment's, unchanged. Decisions at the reveal only.
* Every measure is reported with the raw route intact and removed. The removed condition is where edits are
  comparable (the whole patch is 1.00 there); the intact condition is reported beside it.

## Step 0: replication

`whole`, `pattern`, `pattern shift`, `pattern complement`, `belief` on the new pairs. The belief-edit experiment, raw route removed:
pattern 0.76–0.93, shift 0.60–0.81, complement 0.03–0.12, belief 0.00–0.04.

## Controls

For every edit, beside `moved`: its **rank**, its **magnitude** ‖edit − x_A‖ / ‖x_B − x_A‖, and its **captured
share** of ‖x_B − x_A‖². Random edits are averaged over 5 draws, with the range.

| control | edit | what it answers |
|---|---|---|
| PCA, equal rank | the donor's component in the top-k principal directions of Σ | does replacing the high-variance part of the history state do as well? |
| PCA outside pattern | the top-k principal directions, projected off the pattern span | is there an effect in high-variance directions that do not co-vary with the posterior? |
| pattern outside PCA | the pattern span, projected off the top-k principal directions | does the pattern keep an effect once the dominant variance is removed? |
| random, equal rank | k random orthonormal directions | the rank-matched baseline (expected share k / 128) |
| random, equal share | a random subspace of rank round(128 · s), s the pattern's captured share for that model in the belief-edit experiment (0.90–0.92) | is "most of the state" enough, whichever part? |
| random, equal magnitude | x_A + a random vector of the pattern edit's norm | does a displacement of that size do anything by itself? |
| rank sweep | pattern (leading directions of ΣW by singular value), PCA and random at ranks 1, 2, 4, 7, k | `moved` against captured share, one curve per family |

The **specificity comparison** is the rank sweep: at equal captured share, is the pattern's curve above the PCA
curve and the random curve? Equal rank alone does not match the edits; the table above gives rank, magnitude and
share for each so the reader can see what is and is not matched.

## Posterior-matched histories

Pairs sampled on purpose, not as a by-product: donors binned by the L1 distance between the two posteriors
(< 0.05, 0.05–0.1, 0.1–0.25, 0.25–0.5, 0.5–1, > 1), different token histories, ≥ 1 000 pairs per bin where the bank
allows. Per bin and edit: TV(edited, base), and the same as a share of TV(whole patch, base).

* A posterior-specific edit has an effect that goes to zero with the posterior distance, and in the closest bins
  the whole patch's remaining effect is carried by the pattern complement.
* `pattern shift` moves the state only as far as the decoded posterior differs, so it is near zero in the closest
  bins by construction; it is reported, and the informative edit there is `pattern` (a projection).

## The same edit across goals

The belief-edit experiment counted whether the edited greedy action equals the hybrid's. Here the reference is the solver. For each
goal g, with the same edited prefix states: **gain_g** = the gain in probability on the actions optimal for
(B, g) and not for (A, g), as a share of the hybrid's gain; **changed_g** = TV(edited, base).
On pairs where the solver's optimal sets for B under the two goals are disjoint, and disjoint from A's under each:

* **appropriate under both**: the edited greedy action is in the solver's optimal set for (B, g) under both goals;
* **changed but not appropriate**: the greedy action changed under both goals and is outside B's optimal set under
  at least one.

Reported for whole, pattern, pattern shift, PCA equal rank, random equal share. Only goal pairs with ≥ 200 such
pairs per model are reported.

## Uncertainty against mode pairs

The belief-edit experiment's sets have disjoint optimal action sets, which says nothing about how much is at stake. Two sizes of the
change, per pair: **s_solver** = the regret, under B's posterior, of the action optimal for A (from the exact Q);
**s_model** = TV(base, hybrid). Mode and uncertainty pairs are compared within quintile bins of s_solver
(quintiles of the pooled pairs), and again within bins of s_model; `moved` for the whole patch and the pattern edit,
raw route intact, per bin, and the bin-weighted difference with a bootstrap interval over pairs.
The reversal stands only if uncertainty pairs respond more within bins.

## Expectations

| | expectation |
|---|---|
| E1 | replication, raw route removed: pattern ≥ 0.7, shift ≥ 0.5, complement ≤ 0.15, in all six models |
| E2 | PCA of equal rank captures ≥ 0.8 of the donor difference and moves the decision ≥ 0.8 of what the pattern edit does |
| E3 | at equal captured share the pattern curve is above the PCA curve at ranks ≤ 7 by at least 0.1, in at least four models |
| E4 | pattern outside PCA and PCA outside pattern: both ≤ 0.2 |
| E5 | random, equal rank ≤ 0.15; random, equal share ≥ 0.6; random, equal magnitude ≤ 0.05 |
| E6 | posterior-matched (L1 < 0.1): the pattern edit carries more than half of the whole patch's effect |
| E7 | the whole patch's effect falls with posterior distance but does not reach zero in the closest bin |
| E8 | across goals, pattern: appropriate under both ≥ 0.5 and above changed-but-not-appropriate; PCA of equal rank within 0.1 of it |
| E9 | within bins of s_solver, uncertainty pairs respond more than mode pairs by less than half the unmatched difference |

I expect E2 to hold: a span holding nine tenths of the difference between two histories is probably close to the
top principal span, and then equal-rank PCA cannot separate them. E3 is the test that can, and I hold it weakly.
E6 is an expectation *against* specificity: if the pattern span is most of the evidence-dependent state, it will
carry what same-posterior histories differ in as well.

## What each outcome supports

| result | reading |
|---|---|
| pattern above PCA and random at equal share (E3), inert on posterior-matched pairs (E6 fails), appropriate across goals (E8) | the posterior-associated directions explain behaviour more specifically than the dominant history representation |
| pattern ≈ PCA at equal share; random of equal share well below both | the effect belongs to the dominant history subspace; ΣW finds it and does not single out the posterior within it |
| random of equal share ≈ pattern | the effect is that of replacing most of the state; no subspace claim is supported |
| pattern edit changes actions on posterior-matched pairs as the whole patch does | the pattern span carries history information beyond the posterior; "belief-associated" is the most that can be said |
| replication fails (E1) | the belief-edit experiment's post hoc result was fitted to its pairs |

**If E3 is borderline** (the pattern curve above the PCA curve by 0.05–0.1, or by ≥ 0.1 in only three or four of
the six models): five fresh seeds are trained with the maze-belief experiment's configuration and this analysis is rerun on them
unchanged.

Limits known in advance: the same six models and one maze (new models would need the maze-belief experiment's training, 1.5 h GPU);
one interface; mean ablation of the raw route is off the training distribution.
