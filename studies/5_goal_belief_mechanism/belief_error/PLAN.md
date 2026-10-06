# Belief error: subspaces, arms, decision rule and expectations

Written before `run.py` was run on any trained model. Brief: `BRIEF.md`.

## The test, and why not just regress r on the decoding error

As in the history-mediator and residual-trace experiments, at the goal token's four attention outputs (concatenated):
d(h) = x_b(h) + r(h), x_b the fit-side table of d over (posterior, L). A linear decoder b̂ = W d has error
b̂ − b ≈ W r + (W x_b − b). So "the decoding error predicts r" is largely true by construction for whatever part of r
lies in W's row space. The hypothesis says something stronger: **r's effect on the decision is a misplaced belief.**
Then the part of r that moves the state along the directions in which the state encodes the posterior should carry
r's effect, and the part orthogonal to them should not.

## Belief subspace and decoder

* **S**: the span in which the state varies with the exact posterior. It is the top principal components of x_b(h)
  over fit-side histories (after removing the per-length means), k = the number reaching 95 % of x_b's variance. A
  fixed k = 13 (the posterior's degrees of freedom) is reported alongside.
* r_∥ = projection of r onto S; r_⊥ = r − r_∥. Their shares of r's variance are reported.
* **Decoder**: ridge regression from d to the exact posterior (14 cells), fit-side histories, ridge chosen on a
  10 % fit-side hold-out; held-out R². Decoded error e(h) = b̂(h) − b(h); its size is total variation ½‖e‖₁.

## Arms (informative pairs, random same-length donor; new draw, fixed seed)

| arm | added at the four attention outputs |
|---|---|
| real_r | r(h′) − r(h) (the residual-trace experiment's real arm) |
| real_par | its projection onto S |
| real_perp | its projection off S |
| rot_r, rot_par, rot_perp | each rotated to a random direction of the same size: rot_par inside S, rot_perp inside S's complement, rot_r in the whole space |

**Excess flip** of an arm = its flip rate − its rotated control's (flip: the action differs from the recipient's
natural action).

## Predictive part

The residual-trace experiment's joint fit (per-pair excess shift of the real_r arm against rot_r, on margin,
unsolvable, interaction used, goal, entropy, H's deviation, residual size), refit **with and without** the decoded
error's change between the pair, ‖e(h′) − e(h)‖ in total variation, standardized. Reported: its coefficient and the
entropy coefficient in both fits.

## Decision rule

Ten models; medians; exact one-sided Wilcoxon signed-rank tests, p < 0.05.

| | criterion |
|---|---|
| **BE** | excess flip of real_par ≥ 0.75 × real_r's, and real_perp's ≤ 0.25 × real_r's (k at 95 %) |
| **MED** | decoded-error coefficient > 0 (Wilcoxon), and the entropy coefficient falls by ≥ 50 % when it is added |

| result | reading |
|---|---|
| BE, MED | **r is belief error**: a misplaced belief in the belief code, largest where the posterior is uncertain |
| BE, not MED | r acts through the belief code, but its effect is not indexed by the decoded error's size |
| not BE | r's effect lies outside the belief code: history information that is not a belief estimate |

## Expectations

| | expectation |
|---|---|
| E1 | decoder held-out R² ≥ 0.75 (the maze-belief experiment decoded the posterior at R² 0.77 from the goal token's residual) |
| E2 | r_∥ holds ≤ 0.4 of r's variance |
| E3 | BE fails: real_par carries 0.4–0.7 of the excess, real_perp more than 0.25 |
| E4 | the decoded-error coefficient is positive; the entropy coefficient falls by less than 50 % (MED fails) |

## Limits known in advance

A linear subspace and a linear decoder; S defined from the table's variation over 619 posteriors; one draw of donors;
goal token's first decision only.
