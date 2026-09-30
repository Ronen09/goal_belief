# Is the ΣW edit specific? Replication with controls (round 21)

Run date: 2026-09-30. Brief: `BRIEF.md`. Construction, controls, measures and expectations: `PLAN.md`, committed
(`7cec98b`) **before anything was run**. No model was trained: round 18's six models. Numbers from `tables.md`; data
in `results.json`. Reproduce: `reproduce.py r21` (2 min on one GPU).

**Frozen and new.** Every subspace was fitted on round 18's fitting histories and saved before the evaluation
histories existed. The pairs are new: the held-out side of a new bank, 40 036 pairs (13 465 mode, 7 032 uncertainty),
and 17 403 pairs binned by the distance between the two posteriors. One number in section 5 is post hoc and is
marked. Ranges are over the six models; "removed" and "intact" refer to the decision token's direct route to the raw
prefix tokens.

## 1. The replication holds

Moved to the hybrid, raw route removed (mode / uncertainty pairs):

| edit | captured share of ‖x_B − x_A‖² | moved | round 20 |
|---|---|---|---|
| pattern (rank 13) | 0.89–0.91 | 0.83–0.93 / 0.79–0.91 | 0.84–0.93 / 0.76–0.91 |
| pattern shift | 0.53–0.65 | 0.65–0.80 / 0.60–0.75 | 0.67–0.81 / 0.60–0.76 |
| pattern complement | 0.09–0.11 | 0.03–0.07 / 0.04–0.11 | 0.03–0.07 / 0.04–0.12 |
| belief (decoder's row space) | 0.02–0.03 | 0.00–0.04 / 0.00–0.01 | 0.00–0.04 |

Round 20's post hoc result was not fitted to its pairs (E1 held). With the raw route intact the pattern edit moves
0.11–0.24 / 0.16–0.44, the whole patch 0.12–0.25 / 0.18–0.45.

## 2. It is not specific: the effect follows the captured share of the donor difference, whichever subspace

Raw route removed, mode pairs:

| edit | rank | magnitude | captured share | moved | solver gain |
|---|---|---|---|---|---|
| pattern | 13 | 0.95–0.96 | 0.89–0.91 | 0.83–0.93 | 0.87–0.98 |
| PCA, equal rank | 13 | 0.97–0.98 | 0.94–0.96 | 0.89–0.96 | 0.95–1.00 |
| random, equal share | 115–117 | 0.95–0.96 | 0.90–0.92 | 0.85–0.93 | 0.86–0.97 |
| random, equal rank | 13 | 0.30–0.32 | 0.09–0.10 | 0.03–0.09 | 0.04–0.11 |
| PCA outside pattern | 13 | 0.25–0.29 | 0.06–0.08 | 0.03–0.07 | 0.01–0.10 |
| pattern outside PCA | 13 | 0.16–0.19 | 0.03–0.04 | 0.01–0.03 | −0.01–0.03 |
| random, equal magnitude | — | 0.95–0.96 | none | 0.10–0.30 | 0.06–0.24 |

* **The top 13 principal components do as well as the pattern edit or better**, in every model. They hold 0.96–0.97
  of the variance of the prefix states; the pattern span holds 0.93–0.94, and the two spans overlap 0.74–0.78.
* **A random subspace that captures the same share does as well as the pattern edit.** It needs rank 115–117 to do
  it. Thirteen random directions capture a tenth of the difference and move the decision a tenth of the way.
* **What either span has that the other lacks does nothing** (≤ 0.07, E4 held).
* **Rank sweep.** At ranks 1, 2, 4, 7 the pattern edit captures 0.14–0.27, 0.37–0.49, 0.60–0.74, 0.77–0.84 and moves
  0.12–0.37, 0.17–0.43, 0.54–0.75, 0.75–0.85; PCA captures 0.04–0.19, 0.34–0.43, 0.67–0.76, 0.84–0.88 and moves
  0.01–0.21, 0.17–0.41, 0.61–0.76, 0.80–0.90. At equal captured share the pattern curve is above the PCA curve by
  −0.07 to 0.05 (mean over ranks ≤ 7, per model; section 2b of the tables). E3 failed. No model reaches 0.1 on the
  mean; three models reach it at a single rank, and two are below the PCA curve.
* With the raw route intact the three families agree to 0.02 at every rank.
* A random displacement of the pattern edit's size moves the decision 0.10–0.30 of the way with the raw route
  removed (0.01–0.04 intact): without the raw route, destroying the prefix state takes the decision part of the way
  to any other history's. E5 held for the two random subspaces and failed here.

So `moved` is close to the captured share for the pattern, PCA and random families alike. The evidence-dependent
part of the prefix state is low-dimensional, which PCA finds without reference to the posterior, and ΣW finds the
same part. Nothing here singles out the posterior within it.

## 3. Posterior-matched histories: the pattern edit still changes actions

TV(edited, base), and in brackets the share of the whole patch's, raw route removed:

| posterior L1 | whole | pattern | pattern complement | PCA | random, equal share |
|---|---|---|---|---|---|
| < 0.05 | 0.03–0.14 | 0.03–0.13 (0.88–0.96) | (0.19–0.52) | (0.84–1.00) | (0.84–0.98) |
| 0.05–0.1 | 0.01–0.28 | 0.01–0.18 (0.60–1.27) | (0.30–0.79) | (0.80–1.04) | (0.76–1.00) |
| 0.1–0.25 | 0.03–0.12 | (0.83–1.01) | (0.14–0.43) | (0.96–1.03) | (0.89–0.99) |
| 0.25–0.5 | 0.05–0.24 | (0.91–0.99) | (0.13–0.33) | (0.95–1.00) | (0.92–0.98) |
| 0.5–1 | 0.09–0.33 | (0.87–0.98) | (0.10–0.26) | (0.95–0.98) | (0.91–0.97) |
| > 1 | 0.17–0.42 | (0.89–1.01) | (0.10–0.21) | (0.96–0.99) | (0.87–0.99) |

* When the two posteriors are within 0.05 the whole patch still changes the action distribution by 0.03–0.14
  (0.03–0.06 with the raw route intact), and **the pattern edit carries 0.88–0.96 of that** (E6 held). What
  same-posterior histories differ in is inside the pattern span, not outside it.
* The pattern complement's share is largest where the posteriors are closest (0.19–0.79 below 0.1, 0.10–0.21
  above 1), so a little more of the history-specific effect lies outside the span. It exceeds half in three cells
  (0.52, 0.58, 0.79), all below 0.1.
* The effect of the whole patch grows with posterior distance from 0.25 upwards in every model and does not reach
  zero in the closest bin; below 0.25 it is not monotone (E7 in part).

## 4. Across goals: appropriate as far as the model is, and no more for the pattern edit than for the controls

Per goal, the gain in probability on the solver's optimal actions for the donor's evidence, as a share of the
hybrid's, raw route removed: pattern 0.84–1.00 (G1), 0.84–1.11 (G2); PCA 0.93–1.00, 0.85–1.00; random of equal share
0.87–1.00, 0.77–1.00; pattern complement ≤ 0.14. Raw route intact: 0.15–0.32 and 0.09–0.42, equal to the whole
patch's for all three.

Pairs where the solver's optimal sets for the donor are disjoint under two goals and from the recipient's under
each (appropriate under both / changed under both and not appropriate), raw route removed:

| goals | hybrid (the ceiling) | pattern | PCA | random, equal share | pattern complement |
|---|---|---|---|---|---|
| G1, G2 | 0.02–0.33 / 0.07–0.29 | 0.02–0.27 / 0.07–0.29 | 0.02–0.27 / 0.07–0.29 | 0.02–0.27 / 0.06–0.28 | 0.01–0.07 / ≤ 0.05 |
| G1, G3 | 0.33–0.68 / ≤ 0.05 | 0.30–0.68 / ≤ 0.03 | 0.32–0.67 / ≤ 0.05 | 0.23–0.70 / ≤ 0.04 | 0.04–0.60 / ≤ 0.03 |
| G2, G3 | 0.03–0.53 / ≤ 0.22 | 0.04–0.53 / ≤ 0.22 | 0.04–0.53 / ≤ 0.22 | 0.04–0.53 / ≤ 0.20 | 0.01–0.45 / ≤ 0.01 |

* Scored against the solver, the ceiling is low: with the raw route removed the model on the hybrid sequence is
  itself appropriate under both goals in 0.02–0.68 of these pairs. Round 20 scored against the hybrid's own action
  and so did not show this.
* The pattern edit reaches the hybrid's rate to within 0.06, and its rate of changed-but-not-appropriate is the
  hybrid's. It moves actions towards the donor's goal-specific optimal set as far as the model does.
* PCA and the random subspace of equal share do the same (within 0.1 of the pattern edit in every cell).
* With the raw route intact the edits are appropriate under both goals in 0.01–0.22 of pairs (hybrid 0.04–0.95).
* E8: the rate of 0.5 was reached for G1, G3 in five models and for G2, G3 in two, nowhere for G1, G2; PCA within
  0.1 held.

## 5. Uncertainty pairs respond more than mode pairs at matched size

Whole patch, raw route intact; difference in `moved` (uncertainty − mode):

| model | unmatched [95 %] | within quintiles of s_solver | within quintiles of s_model | s_model, *post hoc*: bins with a bounded ratio |
|---|---|---|---|---|
| reward, seed 1 | 0.06 [0.05, 0.07] | 0.06 [0.05, 0.07] | 0.04 [0.01, 0.06] | 0.04 |
| reward, seed 3 | 0.13 [0.12, 0.14] | 0.13 [0.12, 0.14] | 0.08 [−0.08, 0.21] | 0.11 |
| reward, seed 4 | 0.09 [0.08, 0.11] | 0.09 [0.08, 0.11] | 0.03 [−0.00, 0.06] | 0.03 |
| supervised | 0.09 [0.08, 0.10] | 0.10 [0.09, 0.11] | 0.05 [0.04, 0.06] | 0.05 |
| reward, seed 0 | 0.20 [0.18, 0.22] | 0.21 [0.19, 0.23] | not usable | 0.19 |
| reward, seed 2 | 0.21 [0.19, 0.23] | 0.22 [0.21, 0.24] | not usable | 0.20 |

* **The solver's stake does not differ between the sets** (s_solver, the regret under the donor's posterior of the
  recipient's optimal action: 0.064–0.072 for uncertainty pairs, 0.070–0.074 for mode pairs), and matching on it
  changes nothing. The reversal is not explained by the size of the optimal change.
* **The model's own change is smaller for uncertainty pairs** (TV(base, hybrid) 0.51–0.95 against 0.69–0.98).
  Matching on it removes up to two thirds of the difference in the four models that learned G2 and leaves it
  positive (0.03–0.11).
* The registered s_model statistic fails in seeds 0 and 2: their lowest quintile has TV(base, hybrid) near zero and
  the ratio is unbounded (−0.7 and −114). The last column drops bins where the ratio leaves [−1, 1]; it was added
  after seeing this.
* The pattern edit gives the same differences to 0.02.
* E9 (matching removes more than half) failed for s_solver in all six models and held for s_model in one.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | replication: pattern ≥ 0.7, shift ≥ 0.5, complement ≤ 0.15 | yes | 0.79–0.93; 0.60–0.80; 0.03–0.11 |
| E2 | PCA of equal rank captures ≥ 0.8 and moves ≥ 0.8 of the pattern edit's | yes | captures 0.94–0.96; moves more than the pattern edit |
| E3 | pattern above PCA at equal share by ≥ 0.1, in ≥ 4 models | **no** | −0.07 to 0.05 |
| E4 | pattern outside PCA, PCA outside pattern ≤ 0.2 | yes | ≤ 0.07 |
| E5 | random: equal rank ≤ 0.15; equal share ≥ 0.6; equal magnitude ≤ 0.05 | in part | 0.03–0.09; 0.85–0.93; 0.10–0.30 (removed), 0.01–0.04 (intact) |
| E6 | posterior-matched: pattern carries > half of the whole patch's effect | yes | 0.60–1.27 |
| E7 | whole patch falls with posterior distance, not to zero | in part | 0.03–0.14 in the closest bin; not monotone below 0.25 |
| E8 | across goals: pattern appropriate under both ≥ 0.5; PCA within 0.1 | in part | ≥ 0.5 for G1, G3 in five models, G2, G3 in two; the hybrid's own rate is 0.02–0.68; PCA within 0.1 |
| E9 | matching on s_solver removes more than half of the uncertainty–mode difference | **no** | unchanged |

## Conclusions, by the plan's table

1. **The answer to the brief's question is no.** The posterior-associated intervention does not explain behaviour
   more specifically than a generic replacement of the dominant history representation. PCA of equal rank and a
   random subspace of equal captured share match it on every measure: size of effect, solver gain, arbitrary
   behaviour, posterior-matched pairs, and appropriateness across goals.
2. **Plan's row 3, with one qualification.** "The effect is that of replacing most of the state; no subspace claim
   is supported" holds for the posterior. One claim about a subspace survives: 13 directions carry nine tenths of
   what distinguishes two histories and of the effect, and 13 random directions carry a tenth. The evidence the
   decision reads from the prefix states is low-dimensional. It is the dominant variance of those states, found
   equally by PCA.
3. **Plan's row 4.** The pattern edit changes actions between histories with the same posterior as much as the
   whole patch does. Round 20 called the span "belief-associated"; "evidence-dependent" is what the data support.
4. **Round 20's result replicates** and its wording should be narrowed: ΣW is a valid way to move the state where
   the decoder's row space is not, because it moves along the state's variance, and for that purpose the top
   principal components serve as well.
5. **The uncertainty–mode reversal stands** under matching on the solver's stake, and is reduced, not removed,
   under matching on the model's own change.

**The borderline clause** (train fresh seeds if the pattern curve is above PCA by 0.05–0.1, or by ≥ 0.1 in three or
four models): the per-model mean gap is −0.07 to 0.05, with two models at 0.05. I read it as not triggered; no model
was trained.

Limits: the same six models and one maze; one interface; mean ablation of the raw route is off the training
distribution, and section 4 shows the ablated model is often wrong by the solver's standard; posterior distance and
evidence are not separable in this task beyond the posterior-matched bins, where few distinct posteriors recur.
What would separate a belief from an evidence summary is a task in which different evidence yields the same
posterior often and the same evidence-summary yields different posteriors; this maze offers the first only thinly.
