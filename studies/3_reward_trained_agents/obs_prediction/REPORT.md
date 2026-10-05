# Reward only against reward plus observation prediction (observation prediction)

Run date: 2026-09-30. Brief: `BRIEF.md`. Arms, fixed evaluation, decision rule and expectations: `PLAN.md`, committed
(`d9a947f`) **before any model was trained**. Numbers from `tables.md`; data in `results.json`; training logs in
`runs/`. Reproduce: `reproduce.py obs_prediction` (2.5 h on one GPU).

**What was trained.** The maze-belief experiment's task, transformer and PPO settings. Three arms of ten seeds (0–9; the same seed
has the same initialisation in every arm), 1 500 updates, 3.7 × 10⁷ interactions each: reward only; reward plus
next-symbol prediction with coefficient 1 (primary); the same with coefficient 0.1. The head is present in every
arm. **Evaluation**: the pair-types experiment's pairs and code, unchanged.

**Registered and post hoc.** The decision rule (median below reward only and one-sided exact rank-sum p < 0.05, ten
seeds) and all measures are as planned. The table that excludes seeds which failed a goal is post hoc. One run
note: the weak arm ran out of GPU memory at update 326 beside another arm and was rerun from scratch alone
(`runs/aux01_oom.log`); nothing from the interrupted run is used.

## 1. The three outcomes

Natural access, no patch, final checkpoint; median (range) over ten seeds:

| outcome | reward only | + prediction, c = 1 | p | + prediction, c = 0.1 | p |
|---|---|---|---|---|---|
| **O1** identical-posterior histories: TV between their action distributions | 0.066 (0.035–0.116) | 0.072 (0.057–0.091) | 0.86 | 0.071 (0.042–0.086) | 0.74 |
| **O2** evidence changes, optimal action does not: greedy action changes | 0.277 (0.232–0.309) | 0.190 (0.146–0.244) | < 0.001 | 0.245 (0.208–0.351) | 0.14 |
| **O3** greedy regret | 0.0063 (0.0047–0.0426) | 0.0042 (0.0027–0.0052) | < 0.001 | 0.0058 (0.0038–0.0426) | 0.24 |
| G1 | 0.0040 | 0.0031 | 0.003 | 0.0033 | 0.02 |
| G2 | 0.0120 (0.0092–0.1234) | 0.0079 (0.0041–0.0098) | < 0.001 | 0.0121 (0.0071–0.1247) | 0.40 |
| G3 | 0.0022 | 0.0017 | 0.16 | 0.0017 | 0.04 |
| seeds that failed a goal (regret > 0.05) | 2 of 10 | 0 of 10 | | 2 of 10 | |

* **O1 is not reduced.** Two histories with the identical posterior are treated as differently with the prediction
  objective as without it (0.072 against 0.066). The spread over seeds narrows (0.057–0.091 against 0.035–0.116).
  E3 failed.
* **O2 is reduced at c = 1**, by 31 % at the median: the greedy action changes in 19 % of the cells where the
  solver says it should not, against 28 %. Every secondary measure agrees (TV 0.200 against 0.279; optimal to
  non-optimal 0.091 against 0.119). E4 held.
* **O3 is reduced at c = 1.** Overall regret falls by a third, G2's by a third, G1's by a quarter; G3's is not
  reduced by the rule (p 0.16). No seed fails G2, against two of ten with reward only (the maze-belief experiment: two of five).
  E2 held.
* **Learning is faster at c = 1**: greedy regret falls below 0.01 at update 326–505 in every seed, against 505–1 206
  in the eight reward-only seeds that get there.
* **c = 0.1 does little.** The head learns as well (E1: excess cross-entropy 0.002 nats, against 0.001 at c = 1 and
  0.415 untrained), two seeds still fail G2, and neither O2 nor O3 is reduced by the rule. It lies between the two
  other arms on O2 and O3 (E7 in part). Excluding the seeds that failed (post hoc), O2 is 0.240 against 0.266
  (p 0.04).

## 2. How much of the reduction in O2 is competence

Checkpoints from update 200, all seeds, binned by greedy regret; O2 (mean over checkpoints, number of checkpoints):

| arm | < 0.004 | 0.004–0.006 | 0.006–0.01 | 0.01–0.02 | 0.02–0.05 | > 0.05 |
|---|---|---|---|---|---|---|
| reward only | — | 0.243 (5) | 0.307 (30) | 0.380 (13) | 0.362 (27) | 0.402 (25) |
| + prediction, c = 1 | 0.199 (7) | 0.217 (25) | 0.276 (33) | 0.318 (12) | 0.388 (15) | 0.405 (8) |
| + prediction, c = 0.1 | 0.244 (1) | 0.247 (13) | 0.285 (24) | 0.378 (15) | 0.360 (22) | 0.406 (25) |

* O2 falls with regret in every arm: incorrect action changes are mostly the policy's errors.
* **At equal regret the prediction arm is lower by 0.016 [−0.039, 0.007]**, against 0.087 unmatched. About four
  fifths of the reduction is that the policy is better; the rest is within the bootstrap interval of zero. E5 held.
* **O1 at equal regret is higher with prediction**, by 0.012 [0.001, 0.024]. O1 also falls with regret in every arm
  (0.12 at regret 0.01–0.02, 0.06–0.08 below 0.006); the prediction arm reaches lower regret and ends no lower on
  O1.
* The models are on the optimal set more often with prediction (type A cells 0.884 against 0.837; type B same
  cells 0.889 against 0.841).

## 3. Patches, kept separate

Whole and rank-13 PCA patches of the prefix states; median (range); c = 1 against reward only:

| | reward only | + prediction, c = 1 | |
|---|---|---|---|
| natural access, type C: moved, whole patch | 0.13 (0.08–0.20) | 0.15 (0.10–0.17) | no difference (p 0.32) |
| natural access, type B disjoint cells: moved | 0.24 (0.17–0.34) | 0.24 (0.19–0.33) | no difference |
| direct route removed, type C: pca's share of the hybrid's gain | 0.95 (0.87–0.98) | 0.94 (0.84–0.98) | no difference |
| condition 1 — the hybrid's greedy action in the donor's optimal set (type C) | 0.86 (0.79–0.89) | 0.90 (0.87–0.92) | higher (p < 0.001) |
| condition 2 — type B same cells, optimal to non-optimal under the whole patch | 0.10 (0.07–0.14) | 0.13 (0.09–0.15) | higher (p 0.003) |
| the same under the natural swap, no patch | 0.12 (0.11–0.15) | 0.09 (0.07–0.12) | lower (p < 0.001) |
| condition 3 — ablated model, greedy action in the optimal set (type C, hybrid) | 0.43 (0.39–0.51) | 0.45 (0.36–0.55) | no difference |
| direct route removed, type A: TV(hybrid, base) | 0.08 (0.04–0.16) | 0.14 (0.09–0.16) | higher (p 0.004) |
| direct route removed, type B same cells: greedy changed | 0.19 (0.06–0.35) | 0.33 (0.22–0.39) | higher (p 0.006) |

* **Patch transfer did not change.** The share of the evidence that the decision reads from the prefix states is
  the same in both arms (0.13 and 0.15 of the way for type C; 0.24 for type B), and with the direct route removed
  the PCA patch carries the same share of the model's response. E6 failed. There is no stronger transfer to
  interpret, so the three conditions are reported and not needed.
* **The patch does more harm where the solver predicts no change** with prediction (0.13 against 0.10), while the
  natural swap does less (0.09 against 0.12). The better the model follows its raw tokens, the more a patch that
  contradicts them costs.
* **With the direct route removed the prediction arm depends more on the history, not less**: same-posterior
  histories differ by 0.14 against 0.08, and the greedy action changes in same cells in 33 % against 19 %. The
  ablated models are equally competent (0.45 and 0.43; on the optimal set 0.67 and 0.68). The prefix states of the
  prediction arm carry more that distinguishes histories, including histories with the same posterior.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | the head learns: excess cross-entropy ≤ 0.01 (c = 1), ≤ 0.03 (c = 0.1) | yes | 0.001; 0.002–0.003 |
| E2 | prediction does not raise regret; fewer seeds fail a goal | yes | 0.0042 against 0.0063; 0 against 2 of 10 |
| E3 | O1 reduced at c = 1 by at least a quarter | **no** | 0.072 against 0.066 |
| E4 | O2 reduced at c = 1 by at least a quarter | yes | 0.190 against 0.277 (−31 %) |
| E5 | within bins of regret the differences are less than half the unmatched ones | yes for O2 | −0.016 against −0.087; O1 was not reduced |
| E6 | natural access, type C: moved is higher at c = 1 | **no** | 0.15 against 0.13, p 0.32 |
| E7 | c = 0.1 lies between the other arms on O1, O2 and E6's measure | in part | O2 0.245, regret 0.0058: between; O1 and moved: no ordering |

## Conclusions, by the brief's three questions

1. **Behavioural differences between identical-posterior histories: not reduced.** Predicting the next symbol
   requires the posterior and does not penalise carrying more, and the models carry as much as before under
   natural access and more in the prefix states themselves (section 3).
2. **Incorrect action changes when the optimal action does not change: reduced by a third at c = 1.** Most of that
   is a better policy. At equal regret the difference is 0.016 and not distinguishable from zero.
3. **Regret: reduced at c = 1 for G1 and G2 and overall; no seed fails G2; learning is about twice as fast.** G3,
   where the action hardly depends on the evidence, is not reduced by the rule.
4. **The causal reading is unchanged from the belief-edit to pair-types experiments.** The prediction objective makes the policy better and
   does not make the decision read more of its evidence from the prefix states, nor make those states more like a
   posterior by the same-posterior test.
5. **The coefficient matters.** At 0.1 the head learns equally well and the policy gains little. The benefit is not
   in having learned to predict; it follows how strongly prediction shapes the shared representation during
   training (or, not separable here, the larger effective gradient under the shared clip).

Limits: one maze, one architecture, ten seeds per arm, two coefficients; the auxiliary gradient shares the
gradient clip with the policy's; O1 and O2 are measured at the reveal only; the matched comparison bins checkpoints
of different ages together.
