# Round 23: training arms, measures, decision rule and expectations

Written before any model of this round was trained. Code: `goalgeo/mazeaux.py`, `rounds/r23_obs_prediction/train.py`,
`measure.py`.

## Training

Task, tokens, transformer (4 layers, width 128, 4 heads) and PPO settings are round 18's, unchanged: 4 096 episodes
per update, 3 epochs × 4 minibatches, Adam 10⁻⁴, clip 0.2, GAE λ 0.95, γ 0.9, entropy 0.03 → 0.003, gradient clip
0.5, 1 500 updates, the same checkpoints.

**The auxiliary objective.** A linear head on the final residual stream gives, at every token, a distribution over
the next symbol for each of the four moves. Its loss is the cross-entropy of the symbol in the next event token,
for the move that token holds, at every token whose successor is an event token: prefix tokens (the move was
imposed), the goal token and the event tokens after it (the move was the model's own). The last prefix token is
followed by the goal token and has no target. The loss is computed on the PPO batch in the same forward passes and
added to the PPO loss with coefficient c. The head exists in every arm, so the architecture is the same; with c = 0
it receives no gradient.

| arm | c | seeds |
|---|---|---|
| reward only | 0 | 0–9 |
| reward + prediction | 1.0 (primary) | 0–9 |
| reward + prediction, weak | 0.1 | 0–9 |

The same seed gives the same initialisation in every arm. All arms are trained fresh with the same code; round 18's
five reward-only models are not reused (they are a check: two of five had not learned G2).
One coefficient is primary and fixed here; no coefficient is tuned on the measures below.

**Reference for the auxiliary loss.** The exact predictive distribution of the next symbol from the belief graph
(after the reveal: conditional on the episode continuing). Reported: the head's cross-entropy minus the exact one,
on the fixed evaluation episodes.

## Evaluation: fixed

Round 22's pairs, unchanged (bank seed 220022, pair seed 22; they depend on the solver's tables only): type A, same
posterior and different histories; type B, the same optimal set under one goal and disjoint sets under another;
type C, disjoint under all three goals. Every pair under all three goals, at the reveal. Round 22's code.

**The three outcomes of the brief** are behavioural and need no patch. Natural access, final checkpoint:

| | measure |
|---|---|
| O1 identical-posterior histories | type A: TV(hybrid, base), all goals pooled. Secondary: probability crossing the boundary of the optimal set; greedy changes |
| O2 incorrect action changes | type B, cells with the same optimal set: the share in which the greedy action changes between recipient's and donor's evidence. Secondary: TV; the share that goes from optimal to non-optimal |
| O3 regret | greedy regret on the 16 384 fixed evaluation episodes, overall and for G1, G2, G3; the number of seeds with regret above 0.05 on any goal |

**Decision rule**, fixed here: an outcome is *reduced* by an arm if its median over the ten seeds is below the
reward-only median and a one-sided exact rank-sum test gives p < 0.05. Medians, ranges and p are reported for both
coefficients; the primary comparison is c = 1.0.

**Is a reduction more than competence?** A better policy makes fewer errors of every kind. O1 and O2 are computed
at every checkpoint from update 200 on, for every seed and arm, and shown against greedy regret. Statistic: the
difference between arms in O1 and O2 within bins of regret (edges 0.004, 0.006, 0.01, 0.02, 0.05), weighted by the
number of checkpoints, with a bootstrap interval over seeds. Reported, with no threshold.

## The causal claim, kept separate

Whole and rank-13 PCA patches of the prefix states entering block 1 (PCA fitted per model on round 18's fitting
histories, as in round 21), natural access and direct route removed, on types B and C. Reported per arm: moved and
the share of the hybrid's gain on the donor's optimal set.

A rise in patch transfer is reported as a change in how much of the evidence runs through the prefix states. It is
read as better use of a belief only if all three hold in the same arm:

1. **competence**: O3 is not worse, and the model on the donor's evidence (the hybrid) puts its greedy action in
   the donor's optimal set at least as often;
2. **conflicting routes**: under natural access the patch does not do more harm where the solver predicts no
   change (type B same cells: the share of greedy actions going from optimal to non-optimal under the whole patch);
3. **without the conflict**: with the direct route removed, the ablated model's own competence (greedy action in
   the optimal set, no patch) is reported beside the transfer, and the transfer is not read where that competence
   is below the reward-only arm's.

## Expectations

| | expectation |
|---|---|
| E1 | the head learns: excess cross-entropy ≤ 0.01 nats at c = 1, ≤ 0.03 at c = 0.1 |
| E2 | O3: prediction does not raise regret (median at c = 1 ≤ reward only), and fewer seeds fail a goal (reward only: 2–5 of 10 fail G2) |
| E3 | O1 is reduced at c = 1, the median by at least a quarter |
| E4 | O2 is reduced at c = 1, the median by at least a quarter |
| E5 | within bins of regret the differences in O1 and O2 are less than half the unmatched ones: most of the reduction is competence |
| E6 | natural access, type C, whole patch: moved is higher at c = 1 (more of the evidence is read from the prefix states, which now have to predict) |
| E7 | c = 0.1 lies between reward only and c = 1 on O1, O2 and E6's measure |

I hold E3, E4 and E6 weakly. The prediction loss asks for the posterior and does not penalise carrying more.

Limits known in advance: one maze, one architecture, one primary coefficient; ten seeds per arm; the auxiliary
gradient shares the gradient clip with the policy's, so c = 1 also changes the effective step of the policy.
