# Belief-encoding edit: edits, pairs, measures, decision rule and expectations

Written before any edited state was passed through a trained head. Code: `heads.py`, `run.py`. Pair counts:
`counts.json` (solver tables only). Brief: `BRIEF.md`.

## Models (frozen)

* **Backbones**: the predictive-transfer experiment's, ten seeds each. Primary: *predict1* (next-symbol prediction on goal-free walks).
  Secondary: *predict2*, *reward*, *random*.
* **Heads**: the predictive-transfer experiment's 16-unit goal-conditioned heads, three head seeds per backbone, at the predictive-transfer experiment's selected site
  (the residual stream at the last prefix token, site 4; site 1 for random). Primary: the heads trained on 100 000
  examples; secondary: 1 000. The predictive-transfer experiment did not keep its heads. `heads.py` rebuilt them with the predictive-transfer experiment's code, seeds and
  layout. Every one of the 120 heads per (site, N) reproduces the predictive-transfer experiment's held-out regret **exactly** (largest
  difference 0.0; `heads.json`). The head applies a parameter-free layer norm to the state it is given, adds the goal,
  and is never trained here.

## The encoding model and the edits

Fitted per backbone on the fit side of the pair-types experiment's bank (144 306 histories, the side the heads were trained on). All
pairs are drawn from the held-out side. b is the exact goal-free posterior over the 14 cells at the last prefix
token; h is the state the head reads.

* **Encoding**: ridge regression h ≈ c + E b (`navprobe.Affine`, λ 10⁻³). Its held-out R² (share of the state's
  variance) is reported. Belief differences sum to zero, so E acts through 13 dimensions.
* **Edits** of the recipient's state h_A toward the donor B, each passed through the frozen head under every goal:

| edit | state given to the head | uses |
|---|---|---|
| none | h_A | |
| **whole** | h_B | the donor's state (reference) |
| **encoding** | h_A + E (b_B − b_A) | **the donor's posterior only** (the candidate) |
| rotated | h_A + Q E (b_B − b_A), Q a random rotation of the state space (5 draws) | the same edit vectors, generic directions: matched generic control |
| random patch | h_A + P_R (h_B − h_A), P_R a random rank-13 projection (5 draws) | the donor's state in a generic subspace of the same rank: matched generic control |
| encoding patch | h_A + P_E (h_B − h_A), P_E the encoding's 13-dimensional column space | the donor's state in the belief-encoding subspace |
| decoder patch | h_A + P_D (h_B − h_A), P_D the posterior decoder's 13-dimensional subspace | the belief-edit experiment's decoder-direction edit |
| PCA patch | h_A + P_PCA (h_B − h_A), top 13 principal components (fit side) | the pattern-specificity experiment's control |
| one-step | h_A + F (p_B − p_A), F an encoding of the exact one-step prediction p | what the prediction objective alone requires; zero on one-step-agreeing pairs |
| wrong | h_A + E (b_C − b_A), a third posterior C | the edit should give C's action, not B's |

## Pairs (`counts.json`)

Held-out histories, prefix length 2–4, recipient and donor of the same length; pair seed 27.

* **Main**: 30 000 random pairs with posteriors at least 0.25 apart (L1). Each pair under each goal is a cell:
  *change* cells (the optimal sets for b_A and b_B are disjoint: 47 861), *preserve* cells (identical optimal sets:
  42 042), overlapping (97, not scored). 17 914 pairs have a change cell under one goal and a preserve cell under
  another. For *wrong*: a third history C of the same length, at least 0.25 from both, scored on the 30 984 cells
  where C's optimal set is disjoint from both A's and B's.
* **One-step agreeing**: 20 000 pairs whose exact one-step predictions are identical for all four moves, with
  different posteriors and disjoint optimal sets under at least one goal (52 one-step classes). 34 799 change cells.
* **Shared belief**: the pair-types experiment's type A pairs (two held-out histories with the same posterior and different tokens)
  as two recipients of the same donor: 6 959 triples, 10 986 change cells. The encoding edit adds the same vector to
  both recipients.

## Measures, per cell, averaged over the three heads of a backbone

* **donor-optimal**: the head's greedy action on the edited state is in the donor's optimal set. On change cells this
  is "produces the donor-belief action"; on preserve cells, where the two sets are equal, it is "keeps an appropriate
  action".
* Also: recipient-optimal; unchanged (same greedy action as with no edit); as whole (same greedy action as on h_B);
  regret evaluated at the donor's posterior; *moved*, 1 − TV(edited, whole) / TV(none, whole) on cells where the
  latter exceeds 0.1.
* **Selective**: on pairs with both kinds of cell, the share where the same edit is donor-optimal under every change
  goal *and* every preserve goal.
* **Shared belief**: both recipients donor-optimal; agreement of their two greedy actions, against the same agreement
  with no edit.
* Edit statistics: the edit's norm and cosine with the actual difference h_B − h_A; the share of that difference
  inside each subspace.

## Decision rule (primary: predict1, 100 000-example heads)

Per backbone the mean over its three heads; medians over the ten backbones; paired comparisons by exact one-sided
Wilcoxon signed-rank tests over the ten backbones, p < 0.05.

| | criterion | measure |
|---|---|---|
| S1 | produces the donor-belief action when required | change cells: encoding's donor-optimal ≥ 0.75 × whole's |
| S2 | preserves the recipient action when appropriate | preserve cells: encoding's donor-optimal ≥ none's − 0.05 |
| S3 | the effect is specific to belief-associated directions | change cells: encoding above rotated and above random patch (both p < 0.05), and rotated's median ≤ half of encoding's |
| S4 | works across histories sharing the same belief | shared triples, change cells: both recipients donor-optimal ≥ 0.75 × whole's, and agreement ≥ none's − 0.05 |
| S5 | uses belief information beyond the one-step prediction | one-step agreeing change cells: encoding's donor-optimal ≥ 0.5, and above rotated (p < 0.05) |

| result | reading |
|---|---|
| S1–S4 | **a selective causal belief edit**: the transferred controller uses belief-associated information, learned without goals, combined with the goal |
| and S5 | including information the prediction objective did not require |
| S1 fails, whole and PCA patch succeed | the decision information is in the state, but not in the form h ≈ c + E b |
| S1 holds, S3 fails | the edit works, but generic perturbations of the same size work too: no specific claim |
| S2 fails | the edit is not selective: it moves decisions that should stay |

The same measures are reported for predict2, reward and random and for the 1 000-example heads, without criteria.

## Expectations

| | expectation |
|---|---|
| E1 | the encoding explains more of the state in predict1 than in reward and random (held-out R²) |
| E2 | S1 holds |
| E3 | S2 holds |
| E4 | S3 holds: rotated and random-patch edits move few change cells (below 0.2) |
| E5 | S4 holds |
| E6 | S5 holds |
| E7 | PCA patch ≥ 0.9 × whole on change cells (as in the pattern-specificity experiment) |
| E8 | decoder patch below encoding on change cells (the belief-edit experiment: the decoder's subspace holds little of the difference) |
| E9 | on main change cells, the one-step edit is below the encoding edit |
| E10 | for reward, encoding / whole on change cells is lower than for predict1: its states are less belief-consistent (predictive transfer) |
| E11 | wrong: donor-optimal for C at least 0.75 × the encoding's rate on main change cells, and for B below 0.1 |

I hold E6 least firmly: on one-step-agreeing pairs the belief difference lies in the directions the objective did not
require, where the encoding may be weakest.

## Limits known in advance

One site, one head width, one decision (the first after the reveal). The heads read a state passed through layer norm,
so an edit's effective size depends on the state's norm. The encoding is linear; a nonlinear dependence of the state
on the belief would show up as an S1 failure with whole and PCA patches intact. Success is causal use by the
transferred head, not by any policy trained with reward.
