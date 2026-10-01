# Round 26: backbones, measures, decision rule and expectations

Written before any backbone of this round was trained. Code: `goalgeo/mazepred.py`, `train_pred.py`, `measure.py`;
solver-only checks in `checks.py` / `checks.md`. Brief: `BRIEF.md`.

## The precaution, checked first (`checks.md`)

Two goal-free posteriors are *k-step equivalent* if they predict the same distribution of the next k symbols for
every sequence of k moves. At a token, a k-step prediction objective requires no more than the class. On the
held-out decisions of round 22's bank, at the reveal:

| prediction | rank of the predictions (of 13) | decisions whose class holds another optimal action set, G1 / G2 / G3 | regret of the best class-only policy, G1 / G2 / G3 | posterior R² from the exact prediction (affine) |
|---|---|---|---|---|
| 1-step | 5 | 0.85 / 0.68 / 0.43 | 0.0100 / 0.0038 / 0.0011 | 0.45 |
| 2-step | 11 | 0.17 / 0.20 / 0.05 | 0.0009 / 0.0009 / 0.0009 | 0.88 |

Goal only: 0.0325 / 0.0224 / 0.0026; uniform 0.054 / 0.048 / 0.063. **Yes, different beliefs predict the same next
symbol for all four moves yet require different actions, in most decisions.** A one-step prediction says only how
much belief lands on a-cells, b-cells and the landmark after each move. A one-step-sufficient representation closes
about three quarters of the gap between goal-only and optimal decisions; a two-step-sufficient one about 95 %. Hence the fourth
backbone (agreed in conversation) and the split of every regret by one-step-determined and undetermined decisions.

## Backbones

Round 18's transformer (4 layers, d 128, 4 heads, context 18 tokens), maze, noise (0.4) and start cells. Seeds 0–9;
a seed gives the same initialisation in every condition (checked: `measure.py` asserts it).

| condition | training | models |
|---|---|---|
| **predict1** | next symbol after each of the four supplied moves, at every token of goal-free random walks of 17 moves (the full context; no goal token, no reward) | trained here |
| **reward** | round 23's reward-only PPO (1 500 updates) | round 23's `reward`, final checkpoint |
| **random** | none | round 23's initial checkpoint (u000000) |
| predict2 (secondary) | the joint pair of symbols after each of the sixteen supplied move pairs, same walks | trained here |

* **Prediction targets** are sampled by the simulator from the true cell for every supplied move (sequence), at every
  token: round 25's balanced targets, which removed the coverage problem of selected-action supervision. The
  posterior, the cell and the exact predictive distribution are never given. Random walks start uniformly on the
  four start cells; moves are uniform.
* **Training**: 4 000 updates of 1 024 walks per model, Adam (lr 3·10⁻⁴, 50 warm-up updates), gradient norm clipped
  at 1 per model, mean cross-entropy over all targets. The policy and value heads are never trained. Smoke runs
  showed nothing about representations; no pilot backbone was trained.
* Positions 0–4 are the only ones measured. Attention is causal, so their states do not depend on anything later in
  the walk.

## Where the representation is read

**The last prefix token**, the token just before the goal is shown: goal-free in every backbone by construction, the
same token position for the same history in all four, and the last point at which the whole history has been seen.
The residual stream is read entering each of the four blocks and after the last (sites 0–4).

The site matters more for some backbones than others, and I know it in advance: round 18 found that a reward-trained
decision reads its evidence from raw tokens and prefix states of blocks 0–1, and nothing from prefix states after
block 2. So the reward backbone's last-block prefix state is never read by anything trained, while the predictor's
is its training output. Rather than fix one site, **every site is run, and each condition is scored at its own best
site**, chosen on the selection half of the held-out histories and reported on the other half (below). Site 4 (the
final one) is reported for every condition as well.

## Stage 1: representations (supporting)

Round 22's bank (200 000 histories from round 22's seed). Decoders are fitted on the fit side (144 306 histories) and
scored on the held-out side (55 694), at the last prefix token, per site:

* **R1 posterior decodability**: affine (ridge) R² of the exact posterior over the 14 cells. Also R² of the exact
  one-step (12 values) and two-step (144) predictions.
* **R2 identical-posterior consistency**: round 22's type A pairs (same node of the belief graph, different tokens,
  same prefix length 2–4; 7 243 pairs). Mean squared distance between the two layer-normalised states, divided by
  the same for type C pairs (posteriors ≥ 0.25 apart in L1, different optimal actions under every goal; 12 000 pairs). 0 = identical
  states for identical posteriors. Also the same ratio for the decoded posteriors (L1).
* **R3 prediction error** (predictors only): TV between the predictor's head and the exact prediction at the last
  prefix token, averaged over the candidate moves (the four moves; for predict2 also its 16 pairs, and the one-step
  prediction its joint distribution implies). Also its D_same on type A pairs.
* Reward models only: their own greedy regret at the reveal (their own policy, goal token) on the same decisions.

## Stage 2: transfer to a frozen, goal-conditioned head (primary)

* **Input**: layer norm (no parameters) of the backbone's state at one site of the last prefix token, concatenated
  with a one-hot goal. Nothing else; no raw history.
* **Head**: one hidden layer of 16 ReLU units → 4 action logits (primary); a linear map (secondary). The width was
  chosen before any backbone of this round existed, on exact-feature references only (pilot, scratch): with 16 units
  the exact posterior gives regret 0.0008 at 1 000 examples and 0.0002 at 100 000, while the raw tokens give 0.0074
  and 0.0034. At 64 units the raw tokens reach 0.0017 at 100 000, i.e. the head learns most of the inference itself.
* **Loss**: −log of the probability the head gives to the solver's optimal action set.
* **Examples**: (history, goal, optimal set) at the reveal, histories from the fit side in a fixed random order, each
  with a uniformly drawn goal; the first N are used, so sets are nested in N. N ∈ {30, 100, 300, 1 000, 3 000,
  10 000, 30 000, 100 000}. Three head seeds, each with its own example order and initialisation. **Every backbone
  gets the same examples and the same initialisation for a given head seed and N.**
* **Budget**: 4 000 Adam steps (lr 10⁻³), minibatches of min(N, 512), for every N and every backbone. Fixed, without
  early stopping (a validation set would cost labels). The pilot found 4 000 and 16 000 steps within 0.0004 of each
  other on the references.
* **Evaluation**: every held-out history under all three goals (167 082 decisions), split by history into a
  selection half and a report half (fixed random split). Greedy regret V*(b) − Q*(b, argmax), discounted return
  units, 12 moves left. Also: expected regret under the head's softmax, probability of an optimal greedy action, by
  goal, by prefix length, on one-step-determined and undetermined decisions, and the head's identical-posterior
  inconsistency (TV between its action distributions on type A pairs, against type C).
* **References**, through the same head, examples and budget: goal only (no features); the exact posterior; the exact
  one-step and two-step predictions; the raw prefix tokens (one-hot symbols and moves at positions 0–4 and the prefix
  length). They bound the curve and show how much the head can infer by itself.

### Primary measures

Per backbone, the regret curve R(N) on the report half, averaged over the three head seeds:

* **AUC**: mean of R(N) over the eight N (area under the curve in log N). Lower is better.
* **N50 / N90**: the number of examples at which R(N) first falls to R_post + 0.5 (resp. 0.1) × (R_goal − R_post),
  interpolated linearly in log N; R_goal and R_post are the goal-only and exact-posterior references at N = 100 000.
  Not reached: recorded as > 100 000 (ranked last).
* The site of each condition: the one with the lowest median AUC on the selection half, separately for the primary
  and the linear head.

## Decision rule

Exact rank-sum tests on the per-backbone AUC at each condition's selected site, ten against ten, p < 0.05.

| | comparison | test |
|---|---|---|
| C1 | predict1 below random | one-sided |
| C2 | predict1 against reward | two-sided |
| C3 | reward below random | one-sided |
| C4 (secondary) | predict2 against predict1 | two-sided |

Readings, by the brief's four outcomes (not exclusive):

| outcome | registered as |
|---|---|
| prediction supports accurate decisions with little training | C1 holds. "Little training" in addition if predict1's median N50 is at most 1 000 examples and below the raw-token reference's N50 |
| belief decodes well, but the head struggles | predict1's posterior R² ≥ 0.9 at its selected site, and C1 fails or its median regret at N = 100 000 is more than twice the exact-posterior reference's plus 0.001 |
| reward transfers better | C2 with reward lower |
| random performs similarly | C1 and C3 both fail |

**The precaution, in the reading.** Where predict1 falls short, the shortfall is read against the one-step class:
regret on one-step-undetermined decisions is reported apart, and against the one-step-prediction reference (the same
head on the exact one-step prediction). If predict1 is no worse than that reference, its failure is what its
objective allowed, not a failure to learn the objective; predict2 then shows whether a two-step objective, which
determines 80–95 % of decisions, transfers.

## Expectations

| | expectation |
|---|---|
| E1 | predict1's one-step error at the last prefix token is below 0.01 (TV, all moves); predict2's implied one-step error below 0.015 |
| E2 | posterior R² at the selected site: predict1 above random by at least 0.05, and above the 0.45 its objective requires |
| E3 | identical-posterior consistency ratio at site 4: predict1 below random and below reward |
| E4 | C1 holds: predict1 transfers better than random |
| E5 | C2: no significant difference between predict1 and reward |
| E6 | predict1's regret at N = 100 000 is below the one-step-prediction reference's: the representation holds more than its objective requires |
| E7 | C4: predict2 below predict1 |
| E8 | the random backbone is worse than the raw-token reference (AUC): a random transformer loses information the tokens hold |

I hold E5 and E6 weakly. For E5, reward-trained prefix states at blocks 1–2 are read by the decision, but they are
read together with the raw tokens, which the head here cannot see.

## Limits known in advance

One maze, one architecture, one head width (and a linear head), ten seeds per condition, one decision (the first
after the reveal). The reward backbones' training and budget are round 23's and differ from the predictors' (4 000
supervised updates on 1 024 walks against 1 500 PPO updates on 4 096 episodes); the brief compares objectives, not
budgets. Two of the ten reward seeds never learned G2 in round 23; they are kept and also reported apart. The reward
backbone was trained on sequences containing a goal token; at the last prefix token it has not seen the goal, as in
its own training. The site is selected per condition, which favours every condition equally (five candidates each)
and is done on a separate half of the held-out histories.
