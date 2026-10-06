# Is the residual a misplaced belief? (belief error)

Run date: 2026-10-06. Brief: `BRIEF.md` (the residual-trace experiment's proposed next step). Subspaces, arms and decision
rule: `PLAN.md`, committed (`e422235`) **before `run.py` was run on any trained model**. Numbers from `tables.md`; data
in `results.json`. Reproduce: `reproduce.py belief_error` (1 min on one GPU).

**Setting.** As in the residual-trace experiment: at the goal token's four attention outputs, d(h) = x_b(h) + r(h),
x_b the posterior's table. The residual-trace experiment found that r matters most where the posterior is uncertain.
The hypothesis: r is the network's error in estimating the posterior. Regressing r on a decoder's error would be
circular, because a linear decoder's error is mostly r projected onto it. Instead, r is split into its part **inside
S**, the subspace in which the state varies with the exact posterior (top principal components of x_b: k = 10–12 at
95 %), and its part **outside S**. Each part is swapped between informative pairs, against a rotated control of the same
size in the same space. If r is a misplaced belief, the part inside S carries its effect.

Decoder (ridge, d → exact posterior): held-out R² 0.917. The posterior is well encoded here.

## 1. r's effect lies mostly outside the belief subspace

k at 95 % (k = 13 agrees):

| swapped | share of r's variance | flips | flips, rotated (same space, same size) | excess | share of r's excess |
|---|---|---|---|---|---|
| whole r | 1 | 0.116 | 0.045 | **0.078** | 1 |
| r inside S | 0.62 | 0.127 | 0.114 | **0.021** | 0.29 (−0.26–0.47) |
| r outside S | 0.38 | 0.065 | 0.021 | **0.044** | 0.53 (0.26–1.08) |

* **Inside S, r is no more effective than a random direction of the same size.** Directions in S move the decision
  strongly (rotated flips 0.114), as expected for the belief code. The part of r that lies there, 62 % of its
  variance, adds only 0.021 beyond random.
* **Outside S, 38 % of r's variance carries 0.53 of its specific effect**, with small logit shifts (0.27) that are
  directed. Rotated, the same size flips 0.021. Outside S above inside S: p 0.002 (not registered).
* The two parts' excesses (0.021 + 0.044) fall short of the whole (0.078): their effects combine more than additively.

## 2. The decoded error does not explain where r matters

| | entropy coefficient | decoded-error coefficient |
|---|---|---|
| joint fit without the decoded error | 0.517 | — |
| with the change in decoded error between the pair | 0.496 | −0.035 (−0.31–0.21; p 0.75) |

* Adding the decoded posterior error leaves the entropy effect intact (−1 %), and the error's own coefficient is zero.
* The decoded error is *smaller* where the posterior is uncertain (correlation with entropy −0.33). So "r matters
  where the posterior is uncertain" is not "r matters where the network's belief is wrong".

## 3. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **BE** | r inside S carries ≥ 0.75 of r's excess, outside S ≤ 0.25 | 0.29 ×; 0.53 × | no |
| **MED** | decoded-error coefficient > 0, and entropy coefficient falls ≥ 50 % | −0.035 (p 0.75); 1 % | no |

**Registered reading: r's effect lies outside the belief code.** It is history information that is not a belief
estimate.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | decoder R² ≥ 0.75 | yes | 0.917 |
| E2 | r_∥ holds ≤ 0.4 of r's variance | **no** | 0.62 |
| E3 | BE fails, inside S carries 0.4–0.7 | in part: fails, but inside S carries less (0.29) | 0.29 |
| E4 | decoded-error coefficient positive; entropy falls < 50 % | in part: entropy unchanged; coefficient zero | −0.035 |

1 of 4, two in part.

## Conclusions

1. **The residual beyond the posterior is not a belief error.** Most of r lies in the belief subspace, but that part
   acts like random noise of its size. Its specific effect comes from a smaller component outside the subspace in which
   the state encodes the posterior. The decoded error neither predicts where r matters nor absorbs the entropy effect.
2. So the history reaches the decision through the posterior-level code (0.94–0.96 of the transfer, history-mediator
   experiment), plus a small, directed signal outside the belief code. That signal matters more where the posterior is
   uncertain, which is where the decision depends least on the belief and other features of the history can tip it.
   This reading is untested.
3. My suggestion in the residual-trace experiment (a misplaced belief) is rejected.

Next, if wanted: decode history features from r outside S (the prefix actions, the last symbol, prefix length,
whether the landmark was seen) and test which one, swapped alone, carries its effect.

Limits: linear subspace and decoder; the ridge penalty chose the smallest value in the grid in every model; S from the
table's variation over 619 posteriors; one draw of donors; goal token's first decision only.
