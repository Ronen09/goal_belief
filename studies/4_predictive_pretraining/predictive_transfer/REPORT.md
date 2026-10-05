# Prediction-only, reward-only and random backbones under a frozen, goal-conditioned head (predictive transfer)

Run date: 2026-10-01. Brief: `BRIEF.md`. Checks, backbones, measures and decision rule: `checks.md`, `PLAN.md`,
committed (`fd5f5d3`) **before any backbone of this experiment was trained**. Numbers from `tables.md`; data in
`results.json`; figure `fig1_curves.png`. Reproduce: `reproduce.py predictive_transfer` (30 min on one GPU; needs the pair-types and observation-prediction experiments).

**Backbones**, ten seeds each, the same initialisation per seed in every condition (asserted): *predict1*, trained
only to predict the next symbol after each of the four supplied moves, on goal-free random walks; *reward*, round
23's reward-only PPO models; *random*, the same initialisations untrained; and, secondary, *predict2*, which
predicts the two symbols after each of the sixteen supplied move pairs. **Readout**: the residual stream at the last
prefix token (goal-free in every backbone), at each of five sites; each condition is scored at its own best site,
chosen on a separate half of the held-out histories. **Head**: layer norm of that state plus a one-hot goal → 16
ReLU units → 4 actions (primary), or linear (secondary). It is trained on the solver's optimal actions at the reveal,
with the same examples, initialisations and 4 000 steps for every backbone, at N = 30 … 100 000.

**Registered and post hoc.** Everything in sections 1–3 is registered. The comparison of predict1 and reward at each
N separately (section 3) and the comparison with the reward model's own policy are post hoc.

## 0. The precaution: one-step prediction does not determine the decision

Before training (`checks.md`): one-step predictions carry 5 of the belief's 13 free dimensions. On held-out
decisions, the one-step class contains posteriors with another optimal action in 85 / 68 / 43 % of cases (G1 / G2 /
G3). The best policy that knows only the class loses 0.0050 on average, against 0.0192 for knowing nothing. Two-step
predictions carry 11 dimensions and leave 5–20 % undetermined, at a loss of 0.0009. A predictor's shortfall can
therefore be what its objective allowed. The *exact one-step prediction* reference measures this directly: the same
head, trained on the exact one-step prediction, plateaus at **0.0065**.

## 1. What prediction training learned (stage 1, supporting)

At the last prefix token, held-out histories; median over ten backbones. Each condition's best posterior R² across
sites, and the consistency ratio at the final site:

| | predict1 | reward | random | predict2 |
|---|---|---|---|---|
| posterior R², best site (affine) | **0.931** | 0.870 | 0.731 | 0.957 |
| one-step prediction R², final site | 0.998 | 0.937 | 0.897 | 0.997 |
| identical-posterior consistency, final site: squared distance A / C (0 = identical) | **0.015** | 0.122 | 0.287 | 0.016 |
| own head's one-step error (TV, four moves) | 0.0078 | — | — | 0.0064 |
| own head's D_same / D_diff (type A / type C) | 0.0057 / 0.456 | — | — | 0.0050 / 0.457 |

* **The predictor learned its objective, and more.** The one-step error is 0.008, and the one-step prediction
  decodes at R² 0.998. The posterior decodes at 0.93, where the exact one-step prediction gives 0.45. What the
  objective requires is about half of what the representation holds.
* **Identical posteriors get nearly identical states.** At the final site, the state distance for same-posterior
  histories is 1.5 % of that for different posteriors: eight times closer than reward's 12 % and nineteen times
  closer than random's 29 %. Consistency improves with depth in the predictors and with reward training, but gets
  worse with depth in the random backbone.
* The decoded posteriors are less consistent than the states (A / C 0.19 for predict1, 0.25–0.27 for the others):
  the affine decoder's own errors dominate that measure.

## 2. Transfer to a frozen, goal-conditioned head (stage 2, primary)

Greedy regret at the reveal on the report half (83 481 decisions); median over backbones of the mean over three head
seeds. Sites selected on the selection half: 4, 4, 1, 4.

| | N = 30 | 100 | 300 | 1 000 | 3 000 | 10 000 | 30 000 | 100 000 | AUC | N50 | N90 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **predict1** | 0.0148 | 0.0071 | 0.0046 | 0.0029 | 0.0022 | 0.0020 | 0.0020 | 0.0020 | **0.0047** | 65 | 4 299 |
| **reward** | 0.0232 | 0.0105 | 0.0060 | 0.0038 | 0.0022 | 0.0016 | 0.0015 | 0.0012 | 0.0061 | 121 | 3 468 |
| **random** | 0.0198 | 0.0111 | 0.0081 | 0.0058 | 0.0049 | 0.0042 | 0.0042 | 0.0040 | 0.0079 | 162 | > 100 000 |
| predict2 | 0.0144 | 0.0063 | 0.0042 | 0.0022 | 0.0018 | 0.0014 | 0.0014 | 0.0014 | 0.0042 | 60 | 1 351 |
| *goal only* | 0.0224 | 0.0220 | 0.0192 | 0.0192 | 0.0192 | 0.0192 | 0.0192 | 0.0192 | 0.0200 | | |
| *exact posterior* | 0.0130 | 0.0068 | 0.0028 | 0.0008 | 0.0004 | 0.0003 | 0.0002 | 0.0002 | 0.0031 | 57 | 452 |
| *exact one-step prediction* | 0.0133 | 0.0082 | 0.0073 | 0.0068 | 0.0065 | 0.0066 | 0.0066 | 0.0065 | 0.0077 | 69 | > 100 000 |
| *raw prefix tokens* | 0.0299 | 0.0169 | 0.0109 | 0.0074 | 0.0048 | 0.0036 | 0.0033 | 0.0034 | 0.0100 | 451 | > 100 000 |

| | comparison (AUC, exact rank-sum) | p | medians |
|---|---|---|---|
| C1 | predict1 below random | < 0.001 | 0.0047 / 0.0079 |
| C2 | predict1 against reward | < 0.001 | 0.0047 / 0.0061, predict1 lower |
| C3 | reward below random | < 0.001 | 0.0061 / 0.0079 |
| C4 | predict2 against predict1 | < 0.001 | 0.0042 / 0.0047, predict2 lower |

The ten AUC ranges do not overlap between predict1 and random (0.0043–0.0050 against 0.0074–0.0082). Excluding the
two reward seeds that never learned G2 changes nothing (reward 0.0061). The linear head gives the same order at every
test (0.0059 / 0.0069 / 0.0088 / 0.0054, all p < 0.001). With the linear head predict1 equals the exact-posterior
reference (0.0057), which the linear head itself caps.

* **Prediction-trained states support decisions with few labels.** With 100 examples predict1's regret is 0.0071,
  equal to the exact posterior's (0.0068); with 300 it has closed 77 % of the gap from goal-only, and its N50 (65) is
  the exact posterior's (57). Random needs about 2.5 times as many labels for half the gap and never reaches 90 %;
  reward needs about twice as many for half the gap.
* **But predict1 plateaus.** From 3 000 examples on it stays at 0.0020, ten times the exact posterior's floor,
  while the exact posterior keeps improving to 0.0002.

## 3. Where the predictor falls short, and the crossing with reward

**On decisions its objective does not determine.** At N = 100 000, predict1 loses 0.0008 where the one-step class
fixes the optimal set and 0.0026 where it does not. The exact one-step prediction loses 0.0045 and 0.0075 there.
So the predictor is not limited to what its objective requires: it does three times better than the
objective-sufficient reference on undetermined decisions. But its remaining error sits there. predict2, whose
objective determines most of those decisions, lowers it to 0.0017 and the plateau to 0.0014.

**Post hoc: predict1 against reward at each N** (two-sided exact rank-sum, final site):

| N | 30 | 100 | 300 | 1 000 | 3 000 | 10 000 | 30 000 | 100 000 |
|---|---|---|---|---|---|---|---|---|
| predict1 / reward | 0.0148 / 0.0232 | 0.0071 / 0.0105 | 0.0046 / 0.0060 | 0.0029 / 0.0038 | 0.0022 / 0.0022 | 0.0020 / 0.0016 | 0.0020 / 0.0015 | 0.0020 / 0.0012 |
| p | < 0.001 | < 0.001 | < 0.001 | < 0.001 | 0.91 | 0.029 | 0.023 | 0.005 |

* **The curves cross at about 3 000 examples.** Below that, predict1 is ahead at every N. Above it, reward is ahead
  (0.0012 against 0.0020 at 100 000). By goal, reward's advantage at N = 100 000 is G1 (0.0018 against 0.0034); by
  prefix length, it is length 1 (0.0017 against 0.0049). The registered AUC weights the eight N equally and so
  favours the region where predict1 is ahead.
* With 100 000 examples, the head on the reward backbone's last prefix state (0.0012) beats the reward model's own
  policy on the same decisions (0.0030; 0.0029 without the two failing seeds; post hoc). The head cannot see the
  raw tokens, so the decision-relevant information is in that state; the PPO policy does not exploit it fully.

**The head's own identical-posterior inconsistency** (TV between its action distributions, type A / type C,
N = 100 000): predict1 0.084, reward 0.113, random 0.168, predict2 0.058; raw tokens 0.151. A more consistent state
gives a more consistent head, in the same order as stage 1.

**How much the head infers itself.** Given raw tokens, the 16-unit head reaches 0.0034 at N = 100 000 but needs 451
examples for half the gap. The random backbone is better than raw tokens at small N (AUC 0.0079 against 0.0100):
random transformer features are easier for the head to use than one-hot tokens. At large N the raw tokens are
better (0.0034 against 0.0040).

## Readings, by the brief's four outcomes

| outcome | registered as | held |
|---|---|---|
| prediction supports accurate decisions with little training | C1, and predict1's N50 ≤ 1 000 and below raw tokens' | **yes**: p < 0.001; N50 65 against 451 |
| belief decodes well, but the head struggles | posterior R² ≥ 0.9, and C1 fails or regret at 100 000 > 2 × exact posterior's + 0.001 | **yes, by the second clause**: R² 0.93; 0.0020 against a threshold of 0.0014 |
| reward transfers better | C2 with reward lower | **no** by AUC: predict1 lower, p < 0.001. Post hoc: reward lower from 10 000 examples |
| random performs similarly | C1 and C3 both fail | **no**: both hold, p < 0.001 |

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | predict1's one-step error < 0.01; predict2's implied one-step error < 0.015 | yes | 0.0078; 0.0064 |
| E2 | posterior R² at the selected site: predict1 ≥ random + 0.05 and above 0.45 | yes | 0.929 against 0.731 |
| E3 | consistency ratio at site 4: predict1 below random and reward | yes | 0.015 against 0.287 and 0.122 |
| E4 | C1 holds | yes | p < 0.001 |
| E5 | C2: no significant difference | **no** | predict1 lower by AUC (p < 0.001); reward lower at large N (post hoc) |
| E6 | predict1 at N = 100 000 below the exact one-step reference | yes | 0.0020 against 0.0065 |
| E7 | C4: predict2 below predict1 | yes | 0.0042 against 0.0047, p < 0.001 |
| E8 | random worse than raw tokens (AUC) | **no** | 0.0079 against 0.0100 (worse only at large N) |

6 of 8 held.

## Conclusions

1. **Yes: learning to predict the maze produces a representation from which different goals are solved with few
   action labels.** Through a 16-unit head that cannot see the history, the prediction-only state matches the exact
   posterior's regret up to about 100 examples. It closes half the gap from goal-only with 65 labelled examples,
   against 121 for the reward-trained state, 162 for the random one and 451 for the raw tokens. No reward, goal or
   posterior was ever given to the backbone.
2. **What prediction learned exceeds what its objective requires.** The one-step objective needs only a
   5-dimensional projection of the belief; the trained state decodes the full posterior at 0.93. It is a much better
   decision substrate than the exact one-step prediction (0.0020 against 0.0065 at 100 000 examples). It is also
   the most belief-consistent of the three: identical posteriors give states 1.5 % as far apart as different ones.
3. **It is not a complete substitute for the posterior.** Predict1 plateaus at ten times the exact posterior's
   regret, mostly on decisions the one-step objective leaves undetermined. A two-step objective lowers the plateau
   (0.0014) and the labels needed for 90 % (1 351 against 4 299).
4. **Reward training gives the better asymptote, prediction the better start.** With 10 000 or more examples the
   reward-trained state is ahead (0.0012 against 0.0020, post hoc), and at that size its frozen state beats its own
   policy. The two objectives make decision information accessible in different amounts: prediction makes more of
   it available with few labels, and reward makes the information for the goals it was trained on more precise.
5. **Not an artefact of the head.** The random backbone and the raw tokens are both clearly worse. The same order
   holds under a linear head, which on the predictor's state does as well as it does on the exact posterior.

Next, if wanted: the plateau and the crossing point to whether a multi-step predictor reaches the reward asymptote,
and whether pretraining on prediction before reward (rather than either alone) gives both the start and the
asymptote.

Limits: one maze, one architecture, one decision (the first after the reveal), one hidden width plus a linear head,
ten backbones per condition. The predictors and the reward models had different training budgets (4 000
supervised updates on random walks; 1 500 PPO updates). The readout is one token, the last prefix token, which
favours a backbone that summarises the history there (the predictors are trained at every token, the reward models
are not). The site is chosen per condition on held-out histories. The registered AUC weights small and large N
equally, so the predict1–reward comparison depends on the regime; the per-N comparison is post hoc.
