# A belief edit at the goal token's state after block 0 (round 32)

Run date: 2026-10-02. Brief: `BRIEF.md`. Site, encoding, edits, pairs, measures and decision rule: `PLAN.md`,
committed (`4c86f3d`) **before any encoding was fitted on a trained model or any edit applied to one**. Numbers from
`tables.md`; data in `results.json`. Reproduce: `reproduce.py r32` (2 min on one GPU).

**Setting.** Round 23's ten reward-only models, frozen. The site is z(h, g), the goal token's residual state entering
block 1. Block 0 computes it from the raw tokens and the goal: round 29's direct route, which carries about 0.7 of the
history-dependent behavioural difference. The encoding is fitted without action labels on 144 306 fit-side histories
under all three goals. Each edit replaces only the goal token's state; the recipient's prefix states are kept and the
network runs on. Pairs come from round 27, are selected by the solver alone, and are all held out. Rates are pooled over
each model's cells, then the median (range) over ten models is reported.

## 1. The encoding fits; a goal-specific map fits better

| encoding (held-out) | R² | R² within (goal, prefix length) |
|---|---|---|
| **shared**: c_{g,L} + E b | 0.813 | **0.555** (0.533–0.579) |
| brief's literal form: c_g + E b | 0.701 | 0.309 |
| goal-specific: c_{g,L} + E_g b | 0.862 | **0.674** (0.657–0.686) |
| offsets c_{g,L} only | 0.568 | 0 |

* **The prerequisite F holds** (0.555 ≥ 0.5). One map from the posterior explains a little over half of the
  within-(goal, length) variation.
* **A different map per goal explains 0.12 more**, in every model. The goal-specific maps differ from the shared one by
  13–28 % of its squared norm (largest for G3). Round 31 had already found that block 0's MLP writes goal × posterior.
* The brief's offset form leaves the goal token's positional term to E (within R² 0.31). Recipient and donor share
  their length, so the term cancels in the edits. Its edits match the primary's to within 0.01 everywhere.

## 2. Main pairs: the shared edit transfers two thirds of what the route carries

Change cells (the two posteriors' optimal sets are disjoint under that goal; 47 861 per model):

| edit | donor-optimal | as hybrid | T (of hybrid) | **S (of whole)** | pooled logit projection |
|---|---|---|---|---|---|
| none | 0.108 | 0.213 | 0 | 0 | 0 |
| whole direct-route replacement | 0.595 | 0.739 | 0.69 | 1 | 0.77 |
| **encoding, shared E** | **0.424** (0.353–0.525) | 0.570 | 0.48 | **0.66** (0.58–0.83) | 0.51 |
| encoding, goal-specific E_g | 0.563 (0.451–0.649) | 0.704 | 0.66 | **0.94** (0.89–1.00) | 0.69 |
| rotated encoding (same norms) | 0.109 | 0.213 | 0.00 | 0.00 | 0.06 |
| PCA-13 patch (donor's state) | 0.585 | 0.728 | 0.68 | 0.98 | 0.76 |
| random rank-13 patch | 0.126 | 0.232 | 0.02 | 0.02 | 0.04 |
| hybrid (B under g, natural) | 0.812 | 1 | 1 | | 1 |

* **The whole direct route carries 0.69 of the hybrid's transfer.** This is the effect available at this site, the
  same as round 29's a_direct.
* **The shared belief edit carries two thirds of it** (S 0.66). It raises donor-optimal actions from 0.11 to 0.42 using
  only the donor's posterior, and the same vector serves all three goals. That is below the registered 0.75.
* **It is direction-specific.** Rotated into generic directions with identical norms, the same edit vectors do nothing
  (gain 0.000). A random rank-13 patch of the donor's actual state gives 0.013. All ten models, p < 0.001.
* **Generic replacement of equal rank does better.** The PCA-13 patch of the donor's actual state reaches 0.98 of whole,
  0.15 above the shared edit (C5b fails). The 13 principal components hold 95 % of the route's history difference. The
  shared encoding's direction matches less of it: cosine 0.69, and 47 % of the squared difference left over.
* **The goal-specific encoding closes the gap** (0.94 of whole, within 0.02 of PCA). It too uses only the donor's
  posterior, with one map per goal: cosine 0.76, 34 % left over.
* By goal, the shared edit's donor-optimal rate is 0.39 / 0.42 / 0.59 for G1 / G2 / G3, against 0.56 / 0.54 / 0.69
  for the goal-specific edit.

## 3. The decisive cases

**Goal-dependent pairs** (5 653): the belief change matters under two goals that need different actions for B ("left
under G1, right under G2"):

| edit | donor-optimal under every change goal | right under two goals, with different actions |
|---|---|---|
| none | 0.038 | 0.052 |
| whole | 0.276 | 0.356 |
| **encoding, shared** | **0.105** | 0.137 |
| encoding, goal-specific | 0.234 | 0.301 |
| rotated / PCA-13 | 0.040 / 0.272 | 0.055 / 0.354 |
| hybrid | 0.518 | 0.566 |

* **This is where the shared edit falls shortest**: 0.36 of whole (registered 0.75), against 0.85 for the
  goal-specific edit. A single E (b_B − b_A) seldom makes the network produce the right, different action under two
  goals at once. The goal-specific vectors do.
* The direct route alone also reaches only about half the hybrid on these pairs (0.28 against 0.52). Here the
  prefix interface matters more.

**Preserve cells** (42 042; the posteriors differ, the optimal sets are the same):

* Harm is negligible for every edit: shared 0.001, goal-specific 0.004, whole 0.003 (still optimal 0.92). **C3 holds.**
  Unlike round 27's head, the policy does not lose correct decisions when its belief state is moved within an
  optimal region.

**One-step agreeing pairs** (34 799 change cells; identical one-step predictions, different optimal actions):

| | donor-optimal | gain | S (of whole) |
|---|---|---|---|
| none | 0.179 | | |
| whole | 0.451 | 0.267 | 1 |
| **encoding, shared** | 0.394 | 0.217 | **0.85** |
| encoding, goal-specific | 0.450 | 0.274 | 1.02 |
| rotated | 0.178 | −0.001 | 0.00 |

* **The edit transfers information beyond the next observation** (C4 holds). Where one-step predictions are identical,
  the shared edit carries 0.85 of what the route carries, more than on main pairs. The route carries less here (0.27
  against 0.50 on main pairs), as round 29 found.

**Equivalent recipients** (13 165 pairs; the same edit on four histories with the recipient's posterior and different
tokens):

| edit | all four donor-optimal | mean | pairwise agreement | variance between belief changes |
|---|---|---|---|---|
| none | 0.052 | 0.086 | 0.926 | 0.81 |
| whole | 0.473 | 0.576 | 0.896 | 0.87 |
| **encoding, shared** | **0.347** | 0.449 | 0.828 | 0.80 |
| encoding, goal-specific | 0.486 | 0.586 | 0.862 | 0.84 |
| hybrid | 0.801 | 0.801 | 1 | 1 |

* **The edit's effect is mostly a property of the belief change, not of the token sequence.** Four fifths of the
  variance of donor-optimality is between belief changes (0.80), as with no edit (0.81). The shared edit makes all four
  recipients donor-optimal in 0.35 of cells, 0.71 of whole (registered 0.75).
* **Agreement among the four falls by 0.09** (registered: at most 0.05), so C6 fails. Part of this is mechanical. The
  no-edit baseline agrees because all four keep the recipient's action; a partial edit that moves some recipients and
  not others must lower it. Whole replacement puts the identical state into all four goal tokens, and agreement still
  falls by 0.03.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| F | shared R² within (goal, length) ≥ 0.5 | 0.555 | **yes** |
| C1 transfer | S(encoding) ≥ 0.75 and gain ≥ 0.25 | S 0.66; gain 0.33 | no |
| C2 goal-dependent | every change goal right: ≥ 0.75 × whole, above rotated | 0.36 ×; p < 0.001 | no |
| C3 preservation | harm ≤ 0.05 | 0.001 | **yes** |
| C4 beyond one-step | gain ≥ 0.5 × whole's, above rotated | 0.85 ×; p < 0.001 | **yes** |
| C5a specificity | above rotated and random; rotated ≤ half | p < 0.001, < 0.001; rotated gain 0.000 | **yes** |
| C5b generic replacement no better | D(pca) − D(encoding) ≤ 0.05 | 0.154 | no |
| C6 equivalent histories | all four ≥ 0.75 × whole; agreement ≥ none − 0.05 | 0.71 ×; −0.091 | no |
| C7 shared map suffices | D(encoding) ≥ D(enc_goal) − 0.05 | −0.122 | no |
| reading: enc_goal meets C1 | S(enc_goal) ≥ 0.75, gain ≥ 0.25 | S 0.94; gain 0.47 | **yes** |

**By the registered rule: C1 fails while the goal-specific encoding meets it.** That reading is: **belief is encoded at
the site, but in a goal-specific way; the goal is already mixed in after block 0.** The strongest result the brief
describes, a shared edit that does it all, is not obtained.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | F holds | yes | 0.555 |
| E2 | goal-specific R² exceeds shared by ≥ 0.03 | yes | +0.05 overall, +0.12 within |
| E3 | whole: T between 0.5 and 0.9 | yes | 0.69 |
| E4 | S(encoding) ≥ 0.6 | yes | 0.66 |
| E5 | C3 holds | yes | 0.001 |
| E6 | rotated and random T ≤ 0.15 | yes | 0.00, 0.02 |
| E7 | D(pca) ≥ D(encoding) | yes | +0.15 |
| E8 | one-step agreeing: whole's gain smaller than on main pairs | yes | 0.27 against 0.50 |
| E9 | C6 holds | **no** | 0.71 ×; agreement −0.09 |
| E10 | C7 holds | **no** | −0.12 |

8 of 10. The two that failed are the ones that test whether one map serves every goal and every history.

## Conclusions

1. **The direct route holds a belief-like variable that the native policy uses.** Moving the goal token's state along
   the posterior-encoding directions, using only the donor's posterior, moves the policy toward the donor's correct
   action. A goal-specific map does this with 0.94 of the effect of replacing the whole state. Rotated directions with
   the same norms do nothing. Correct decisions that should stay are not disturbed (harm ≤ 0.004). The edit works where
   one-step predictions are identical, and its effect depends on the belief change, not on the recipient's token
   sequence (between-change variance 0.80).
2. **That variable is not shared across goals.** One map E for all goals fits 0.12 less of the state and transfers two
   thirds as much. It is weakest exactly where the goal must turn one belief change into different actions (0.36 of
   whole, against 0.85 for goal-specific maps). After block 0, the goal token does not hold "belief, plus a goal
   offset". It holds belief in a goal-dependent embedding, the evidence–goal combination round 31 found in block 0's
   MLP.
3. **So the manipulable variable at the dominant interface is goal-conditioned belief.** The belief-to-state map differs
   by 13–28 % between goals. That is enough to cost a third of the shared edit's effect, and nearly two thirds on the
   goal-dependent pairs.
4. **The generic PCA patch is not an advantage over belief.** It uses the donor's whole state in 13 dimensions. The
   goal-specific belief edit reaches within 0.02 of it from the 13 numbers of the donor's posterior.

Next, if wanted: fit the encoding at block 0's attention output at the goal token, before its MLP. Rounds 29–31 put a
goal-independent evidence estimate there. If a shared E fits and transfers at that point, the goal-specificity found
here is created by block 0's MLP alone, and the belief edit would be shared upstream of it.

Limits: one site, so the edit reaches at most the direct route's share (the prefix interface is unedited). Linear
encodings. The first decision after the reveal; reward models only. The equivalent-recipient test covers posteriors with
at least four distinct held-out token sequences (13 165 of 30 000 pairs). C6's agreement criterion is mechanically
lowered by any partial edit.
