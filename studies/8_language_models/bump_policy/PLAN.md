# Bump policy: what function does the fine-tuned language model compute?

Written before any of the measurements below were run. **Disclosure:** the design probe (`design.md`: the space of
reachable belief sets, its coverage by the optimum's episodes and by the training data, readout ceilings, the pool of
edit pairs) was run first; the single-goal and four-goal experiments (`../bump_maze`, `../bump_goals`) are complete and
their models are reused, no new training. Brief: given in conversation ("let's just focus on finding the policy",
then "B": characterise the function the trained model computes, with the optimum as the reference it was trained
toward).

## The object

The fine-tuned Qwen2.5-0.5B of the single-goal bump maze at 100 000 episodes (`../bump_maze/runs/llm_100000`, the
probed model; exactly optimal on the optimum's decisions, the exact consistent set a linear readout of layers 16–20
at 0.97), with the 10 000-episode model as a second point and the from-scratch transformer at 100 000 as the contrast
(parts A and B). The exact optimum is a function of the belief set alone: π*(S) = argmax Q*(S, ·) over the 17 285
reachable sets, stationary, with ties in 0.081 of sets. The model is a function of the transcript. "Finding the
policy" means establishing (A) whether the model's policy is a function of the set and what it does on every set,
(B) what the decision is read from and in what form, (C) whether the represented set is what the decision is read
from, causally, and (D) where the set is computed.

Design probe (`design.md`): 17 285 reachable sets (all within the horizon, max depth 24; sizes 1 / 2–4 / 5–16 / > 16:
128 / 5 850 / 10 776 / 531). The optimum's 4 096 episodes visit **247** of them; the training data (ε = 0.2 random
actions) visits 3 894 at 10 000 episodes and **6 826** at 100 000; every decision of the optimum's is on a set the
training data visits. The optimal action is linear in the 140-dimensional membership vector at 0.997 on the optimum's
distribution, 0.975 on the training distribution and **0.787 uniformly over sets** (an MLP 1.000); readouts fitted on
the training-visited sets transfer to the 10 459 never-visited ones at **0.682 (linear) and 0.811 (MLP)**. For 9 503
sets a reachable set at Hamming distance 1 has disjoint optimal actions (13 997 within distance 2).

## Part A. The policy as a function on belief sets

**A1, set-sufficiency.** Up to four distinct histories per set (the shortest, from `sets.pt`, and others collected
from 200 000 episodes of random-action and ε = 0.5 policies), for every set with ≥ 2 histories among the training-
visited and the never-visited sets (reported separately). The share of sets on which the model's action is the same
for every history; the baseline is the same share for pairs of different sets of the same size. The share of the
logits' variance explained by set identity (between-set / total, histories pooled).

**A2, the table.** The model's action on every reachable set from its shortest history, against the optimal set:
the in-optimal-set share and the mean relative regret (V*(S) − Q*(S, a)) / V*(S), on the sets the optimum visits, on
the training-visited sets, on the never-visited sets, and by size; the same for the 10k model and the from-scratch
transformer. On the never-visited sets, the model's agreement with the optimum against its agreement with the
transfer readouts of the design (linear 0.682, MLP 0.811 optimal there), both refitted here.

**A3, the exact value of the policy.** The table as a policy over the belief MDP (the solver's transitions): policy
evaluation gives V^π(S) for every set. V^π(start) / V*(start); the mean of V^π / V* over all sets and over the
never-visited sets; the share of sets from which V^π < 0.5 V* (the policy loses its way).

## Part B. What the decision is read from

**B1, probes by layer**, the 100k model, on 8 192 decisions drawn uniformly over reachable sets (shortest histories),
held out by set: at the decision token, linear probes for the set (exact recovery, as before), Q*(S, ·) (ridge, R²),
V*(S), the optimal action (in-optimal-set accuracy), P(bump | forward, S), the cell marginal (35) and the heading
marginal (4). The best layer of each and the profile.

**B2, the form of the readout.** The model's centred logits on all 17 285 sets as a function of the exact membership
vector: a linear fit and an MLP (fit on half the sets, scored on the other half) — R² and agreement with the model's
own decision — against the design's ceilings for the optimum (linear 0.787 over sets).

## Part C. Is the represented set what the decision is read from?

**C1, same-set swaps.** 1 024 pairs of histories with the same set (from A1) and 1 024 pairs with different sets of
the same size: the decision token's residual entering layers 8, 12, 16, 18, 20, 22 replaced by the other history's;
the share of decisions switched (to the donor's, where the two natural decisions differ, for different sets; away
from the natural decision for the same set).

**C2, membership edits.** 2 048 pairs (S, S′) at Hamming distance ≤ 2 with disjoint optimal actions. At the best set
layer (B1) and two layers above and below it, the decision token's residual edited by the minimum-norm change that
moves the linear set probe's prediction for the differing states from S's membership to S′'s (the probe fitted on
other sets); measured as the share of decisions switched to π*(S′), against the natural swap (S′'s own history's
residual, the ceiling), a random direction of the same norm, and a same-norm change along the probe rows of states
that do not differ (the control for norm along the probe's subspace).

## Part D. Where the set is computed

**D1, step tokens.** On the B1 decisions, at the newline token ending step i for every i, a linear probe for the set
after i steps (exact recovery, by layer): does each transcript position hold the set so far?

**D2, prefix patch (prior vs recompute).** 1 024 pairs of histories A, B with prefixes of the same length p ∈ [4, 12]
whose sets after the prefix differ, followed by the same three-step suffix, with π*(A's final set) ≠ π*(B's final
set). In A's run, at layer l ∈ {8, 12, 16, 18, 20}, the residual of the newline token ending A's prefix is replaced by
B's; the decision's switch to π*(B's final set), against replacing all of A's prefix tokens' residuals (the ceiling)
and a middle prefix token's alone (the floor).

## Decision rule

Bootstrap 95 % intervals over sets (A, B2), decisions (B1, D1) or pairs (C, D2); a criterion holds if its interval
excludes the threshold. The 100k model is the registered object.

| | criterion |
|---|---|
| **SUF** | the model's action is the same on every history of a set for ≥ 0.95 of sets, ≥ 0.10 above the different-set baseline |
| **TABLE-IN** | in-optimal-set share on training-visited sets ≥ 0.95 |
| **TABLE-OUT** | in-optimal-set share on never-visited sets ≥ 0.85 (the transfer MLP's 0.81) |
| **VALUE** | V^π(start) ≥ 0.98 V*(start) and the mean of V^π / V* over all sets ≥ 0.90 |
| **READ** | the best layer for Q*(S, ·) lies ≥ 2 layers above the best layer for the set, with R² ≥ 0.90 there |
| **LIN** | the linear fit of the model's logits from membership agrees with the model's own decisions on held-out sets in ≤ 0.90 (the MLP ≥ 0.97) |
| **SAME** | same-set swaps switch ≤ 0.10 of decisions at every layer tested; different-set swaps ≥ 0.70 at the best set layer or above |
| **EDIT** | the membership edit at the best set layer switches ≥ 0.50 of decisions to π*(S′); random and the same-subspace control ≤ 0.15 |
| **STEP** | the set after i steps is exactly recovered from the step-i newline token at its best layer in ≥ 0.90 of decisions (i ≥ 4) |
| **PRIOR** | replacing the last prefix token's residual at the best set layer switches ≥ 0.50 of decisions to π*(B's set) |

## Expectations

| | expectation |
|---|---|
| E1 | SUF holds: the policy is a function of the set (set-sufficiency ≥ 0.98 on training-visited sets, lower on never-visited ones) |
| E2 | TABLE-IN holds; TABLE-OUT holds and the model's never-visited agreement with the optimum exceeds the transfer MLP's 0.81 by ≥ 0.05: the model generalises from the map, not from a set → action lookup; errors concentrate at near-ties (margin < 0.01) and on large sets |
| E3 | VALUE holds |
| E4 | READ holds: the set at layers 16–20 (0.97 exact), Q* and the optimal action at 20–23; P(bump) and the marginals follow the set's profile; the decision is read from the set in the last layers (the four-goal model read the goal at layers 20–21) |
| E5 | LIN holds: the readout is not linear in membership (agreement ≈ 0.8, like the optimum's 0.787), the MLP ≥ 0.97 |
| E6 | SAME holds. EDIT holds, at 0.5–0.8 of the natural swap: the first belief-directed edit in the project that steers a policy, because the set code is linear and exact; the same-subspace control ≤ 0.15 |
| E7 | STEP holds: each step token carries the set so far (incremental filtering). PRIOR holds in part (0.3–0.7): the decision token reads the carried set and the raw tokens both |

## Limits known in advance

One maze, one goal, one model per size; the shortest history stands for a set in A2, A3 and B2 (A1 says how much that
matters); A3's policy evaluation treats the table as the policy, exact only if A1 holds; probes read a 140-dimensional
set from an 896-dimensional state with 7 000 fit decisions; C2's edit is linear along one probe's rows and a failure
there does not exclude other encodings of the set; D2 patches one token's residual at one layer and cannot separate
a carried set from a carried summary that the later tokens complete.
