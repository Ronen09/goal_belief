# Selected-action against balanced candidate-action supervision of the prediction head (balanced prediction)

Run date: 2026-09-30. Brief: `BRIEF.md`. Arms, measures and decision rule: `PLAN.md`, committed (`95ff710`) **before
any model of this experiment was trained**. Numbers from `tables.md` and `matched.md`; data in `results.json`,
`matched.json`. Reproduce: `reproduce.py balanced_prediction` (1.5 h on one GPU; needs the observation-prediction experiment's runs).

**Arms**, ten seeds each, the same initialisations, PPO settings, 1 500 updates and auxiliary coefficient 1:
*selected* (the symbol after the move taken; the observation-prediction experiment's models), *balanced, all* (a simulator-sampled symbol for
each of the four candidate moves), *balanced, one* (the same for one candidate move drawn per token), and round
23's reward-only models for reference. No posterior, cell or exact distribution is given to any arm.

**Registered and post hoc.** Sections 1–2 are the registered measures and rule (median below the selected arm's and
one-sided exact rank-sum p < 0.05). Section 3, the comparison at equal regret, is post hoc: I added it after
seeing that the balanced arms also had somewhat lower regret.

## 1. Question 1: prediction error and identical-posterior inconsistency — both reduced

Goal token, total variation over the three symbols; median (range):

| | selected | balanced, all | p | balanced, one | p |
|---|---|---|---|---|---|
| **P1** error against the exact predictive distribution, all four moves | 0.043 (0.038–0.059) | 0.017 (0.016–0.019) | < 0.001 | 0.019 (0.018–0.022) | < 0.001 |
| **P2** inconsistency between identical-posterior histories, all four moves | 0.026 (0.022–0.040) | 0.013 (0.013–0.014) | < 0.001 | 0.013 (0.012–0.015) | < 0.001 |
| P1, the greedy move | 0.020 | 0.020 | 0.52 | 0.022 | 0.97 |
| P2, the greedy move | 0.014 | 0.014 | 0.16 | 0.013 | 0.18 |
| P1, the other three moves | 0.051 | 0.016 | < 0.001 | 0.019 | < 0.001 |
| P2, the other three moves | 0.031 | 0.014 | < 0.001 | 0.014 | < 0.001 |
| P1 / P2, last prefix token | 0.015 / 0.012 | 0.013 / 0.010 | < 0.001 | 0.015 / 0.010 | 0.37 / 0.04 |
| head's inconsistency relative to its response to another posterior | 0.058 | 0.029 | < 0.001 | 0.029 | < 0.001 |

* **Yes to question 1.** Error falls by 60 % and identical-posterior inconsistency by half, in every seed (the
  ranges do not overlap). E1 held.
* **The whole gain is in the moves the policy does not take.** On the greedy move neither error nor inconsistency
  changes (0.020 and 0.014 in both arms). Balanced supervision brings the untaken moves down to the level of the
  taken one and no further. E2 held.
* **Coverage, not the number of labels.** One random candidate per token does as well as four on inconsistency
  (0.013) and nearly as well on error (0.019 against 0.017). E5 held.
* A floor remains. Identical-posterior histories still get predictions 0.013 apart on every move, at the goal
  token and (0.010) at prefix tokens, in every arm.

## 2. Question 2: policy inconsistency — reduced by the rule, by a tenth

Natural access, no patch; median (range):

| | selected | balanced, all | p | balanced, one | p | reward only |
|---|---|---|---|---|---|---|
| **Q1** identical-posterior histories: TV between action distributions | 0.072 (0.057–0.091) | 0.063 (0.052–0.080) | 0.018 | 0.060 (0.034–0.072) | 0.004 | 0.066 (0.035–0.116) |
| relative to the response to another posterior | 0.078 | 0.068 | 0.012 | 0.068 | 0.001 | 0.075 |
| greedy action changes | 0.071 | 0.062 | 0.06 | 0.060 | 0.004 | 0.063 |
| greedy changes where the optimal set is the same (O2) | 0.190 | 0.200 | 0.89 | 0.206 | 0.84 | 0.277 |
| greedy regret | 0.0042 | 0.0034 | 0.07 | 0.0032 | 0.06 | 0.0063 |
| seeds failing a goal | 0 | 0 | | 1 (seed 3) | | 2 |

* **By the registered rule Q1 is reduced** in both balanced arms (−11 % and −16 % at the median), and regret is not
  worse. By the plan's table this is the second row: prediction and policy both improve — a connection, with
  mediation not established. E3, the dissociation, failed as registered.
* **The size is a fifth of the head's.** The head's inconsistency halves; the policy's falls by a tenth, to the
  reward-only level (0.063 against 0.066). The selected arm was slightly above reward only in the observation-prediction experiment (0.072
  against 0.066, not significant); the balanced arms are back where reward only is.
* **Across seeds the two inconsistencies go together**: within *balanced, all* the rank correlation between P2 and
  Q1 is 0.85 (p 0.002); 0.54 and 0.59 in the other two arms; 0.71 over the thirty models. As in the head-consistency experiment, the
  head's inconsistency is larger in the cells where the greedy action differs between the two histories (0.020
  against 0.013).
* In *balanced, one* the lowest Q1 (0.034) is the seed that failed G2; without it the median is 0.061.

## 3. Post hoc: at equal regret the policy's reduction is not there

Q1 at every checkpoint from update 200, binned by greedy regret (mean over checkpoints, number of checkpoints):

| arm | < 0.004 | 0.004–0.006 | 0.006–0.01 | 0.01–0.02 | 0.02–0.05 | > 0.05 | difference from selected within bins [95 %] |
|---|---|---|---|---|---|---|---|
| selected | 0.072 (7) | 0.079 (25) | 0.096 (33) | 0.114 (12) | 0.129 (15) | 0.128 (8) | |
| balanced, all | 0.072 (20) | 0.074 (25) | 0.096 (24) | 0.132 (13) | 0.133 (10) | 0.122 (8) | 0.001 [−0.008, 0.011] |
| balanced, one | 0.068 (18) | 0.072 (19) | 0.091 (29) | 0.135 (9) | 0.095 (13) | 0.121 (12) | −0.007 [−0.019, 0.006] |
| reward only | — | 0.056 (5) | 0.085 (30) | 0.122 (13) | 0.109 (27) | 0.117 (25) | −0.012 [−0.024, −0.001] |

* Q1 falls with regret in every arm, and **at equal regret *balanced, all* equals *selected*** (0.001, interval
  −0.008 to 0.011). The balanced arms spend more checkpoints at low regret (20 and 18 below 0.004, against 7), and
  that accounts for their lower final Q1.
* The reward-only arm is lower than the selected arm at equal regret (−0.012), as the observation-prediction experiment found.
* O2 at equal regret does not differ between arms either.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | *balanced, all*: P1 below 0.025, P2 below 0.018 | yes | 0.017; 0.013 |
| E2 | the reduction is in untaken moves; the greedy move changes by less than a quarter | yes | greedy move 0.020 → 0.020, 0.014 → 0.014 |
| E3 | Q1 not reduced | **no** by the rule | 0.063 against 0.072, p 0.018; at equal regret 0.001 [−0.008, 0.011] (post hoc) |
| E4 | regret not worse | yes | 0.0034 against 0.0042 |
| E5 | *balanced, one* gives at least half the reduction | yes | P1 0.019, P2 0.013 |

## Conclusions, by the brief's two questions

1. **Balanced supervision reduces held-out prediction error and identical-posterior inconsistency**, by 60 % and
   50 %, entirely on the moves the policy does not take. What was wrong with selected-action supervision was
   coverage.
2. **Policy inconsistency decreases by the registered rule, and the decrease is small and explained by
   competence.** It is a tenth, against a half for the head; it brings the policy to the reward-only level and not
   below it; and at equal regret it is zero to within ± 0.01. The registered verdict is "both improve"; the
   evidence for a connection through the representation is weak.
3. **Neither of the brief's two outcomes cleanly.** It is not the clean dissociation, because the policy measure
   did move and the two inconsistencies are correlated across seeds (ρ 0.85) and across cells. It is not good
   support for a connection either, because where the head became more consistent — untaken moves — is not where
   the policy reads, and on the move the policy takes the head did not change. The part of the state that serves
   the policy's own move was already as consistent as this training makes it.
4. **A floor of about 0.013 in the head and 0.06–0.07 in the policy remains in every arm.** Identical-posterior
   histories are told apart by a state that three kinds of auxiliary supervision leave unchanged in this respect.

What would decide between a shared cause and mediation: an intervention on consistency itself. A loss that pulls
together the states (or the predictions) of histories known to share a posterior would change the head's
inconsistency on the taken move, which balanced targets did not, and the policy's inconsistency could then be read
at equal regret.

Limits: one maze, one architecture, ten seeds, one coefficient; the selected arm was trained in the observation-prediction experiment (same
code path, earlier run); the equal-regret comparison is post hoc and bins checkpoints of different ages together;
counterfactual samples carry label noise.
