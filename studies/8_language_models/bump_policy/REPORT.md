# Bump policy: the fine-tuned model's policy is not a function of the belief set, and its "set code" was a lookup over the optimum's few sets

Run date: 2026-10-14. Object, measures and decision rule: `PLAN.md`, committed (`a861c7e`) **before any measurement
was run**; the models are the single-goal experiment's (`../bump_maze/runs/llm_100000`, `llm_10000`, `tfm_100000`), no
new training. Numbers from `tables.md`; data in `design.json`, `sets.pt`, `table.json`, `table_*.pt`, `histories.pt`,
`probes.json`, `probe_weights.pt`, `swaps.json`, `prefix.json`, `posthoc_histories.json`, `posthoc_probe.json`.
Reproduce: `reproduce.py bump_policy` (about 1.5 h on one GPU, given the single-goal models).

**Disclosures.** (1) Two post-hoc analyses were added after part A and B1 were seen: the same training-visited sets
reached by three kinds of history (`posthoc_histories.py`), and the single-goal study's set probe rerun with the test
split by set instead of by episode (`posthoc_probe.py`). (2) The swaps script failed at the edit stage on a dtype
error after C1 had completed and was rerun whole; C1's numbers were identical. (3) D1's fit/test split is by the final
set of each decision, not by the set after i steps, so early-step sets recur between fit and test; D1 is therefore a
by-episode-style probe and is read as such below. (4) One model per size.

**The object.** The exact optimum is a function of the belief set: π*(S) = argmax Q*(S, ·) over the 17 285 reachable
sets. The fine-tuned 0.5B is exactly optimal on the optimum's own decisions and matches its return. Design probe
(`design.md`): the optimum's 4 096 episodes visit **247** of the 17 285 sets, the 100k training data (ε = 0.2) 6 826,
and every decision of the optimum's is on a training-visited set; the optimal action is linear in membership at 0.997
on the optimum's distribution and 0.787 over all sets; readouts fitted on the training-visited sets transfer to the
10 459 never-visited ones at 0.682 (linear) and 0.811 (MLP).

## 1. The policy on every reachable set: optimal where the optimum goes, not elsewhere

Every reachable set reached by its shortest history; the model's action against the optimal set.

| model | all sets | optimum-visited (247) | training-visited, not by the optimum (6 579) | never visited (10 459) | never visited: agreement with the linear / MLP transfer readout |
|---|---|---|---|---|---|
| **fine-tuned 0.5B, 100k** | **0.741** | **0.968** (0.94–0.99) | **0.879** (0.87–0.89) | **0.648** (0.64–0.66) | 0.61 / 0.62 (the readouts themselves optimal 0.68 / 0.81) |
| fine-tuned 0.5B, 10k | 0.617 | 0.915 | 0.767 (its own 3 647) | 0.571 (13 391) | 0.56 / 0.56 |
| transformer from scratch, 100k | 0.585 | 0.822 | 0.660 | 0.533 | 0.54 / 0.51 |

* **TABLE-IN and TABLE-OUT fail.** Across all training-visited sets the model is optimal in 0.88, on never-visited
  sets in 0.65, below both transfer readouts (0.68, 0.81): off the training distribution it does worse than a lookup
  from membership to action fitted on the same sets, and it does not agree with those lookups either (0.61–0.62).
  Whatever the model generalises from, it is not the set's membership and not the map. The errors are spread: 0.63 /
  0.65 / 0.78 by size on never-visited sets, 0.53 at near-ties against 0.73 on clear decisions. The 10k model and the
  from-scratch transformer have the same profile, lower.
* **Set-sufficiency fails (SUF).** With up to four histories per set, the model gives the same action on every history
  for **0.59** of sets (baseline for two different sets of the same size: 0.43): 0.90 of optimum-visited sets, 0.70
  of other training-visited sets, **0.50 of never-visited ones**, at the baseline. The set accounts for 0.89 of the
  logits' variance over (set, history) pairs; the remaining 0.11 decides a quarter of same-set pairs.
* **The table is not the policy (VALUE fails).** Evaluated exactly over the belief MDP, the shortest-history table is
  worth 0.946 of V* from the start and 0.825 on average over sets (the majority action over histories 0.727 and 0.602),
  and it loses its way from 0.10 of sets — while the model's own closed-loop return is 1.00 of the optimum's. The
  policy the model runs is not the one its actions on shortest histories define; it depends on the history it has
  actually walked. The from-scratch transformer's table is worth 0.24 from the start and 0.03 on average (0.887 success
  online): its decisions on non-training histories are nearly unrelated to the task.

### Post hoc: the same sets by three kinds of history (`posthoc_histories.json`)

6 826 training-visited sets, each reached by a history from the training data itself, by the shortest history, and by
a random walk.

| history | optimal | optimum-visited sets | other training-visited | mean length |
|---|---|---|---|---|
| **from the training data** | **0.935** | 0.996 | 0.933 | 11.6 |
| shortest | 0.883 | 0.968 | 0.879 | 10.5 |
| random walk | 0.838 | 0.964 | 0.833 | 15.0 |

* On the same set the training-like history gets the optimal action 0.05 more often than the shortest and 0.10 more
  than a random walk; the actions agree in 0.89 / 0.84 / 0.86 of sets between kinds. Disagreement between the
  training-like and the shortest history grows with their difference in length (0.03 at equal length, 0.19 at 1–2
  steps, 0.26–0.36 beyond) and with a different last action (0.26 against 0.08), and **when they disagree the
  training-like history's action is the optimal one in 0.69 of cases**, the shortest's in 0.20. The policy is tuned
  to how the training trajectories reach a set, including their length and last move.

## 2. What the decision token holds: not the set; "will forward bump", weakly Q*

Linear probes at the decision token on 8 192 decisions **uniform over reachable sets** (shortest histories), **held out
by set**.

| layer | set: exact / Jaccard | Q*: R² (argmax optimal) | V*: R² | optimal action: in-set accuracy | P(bump \| forward): R² | cell marginal: R² | heading marginal: R² |
|---|---|---|---|---|---|---|---|
| 8 | 0.000 / 0.04 | 0.19 (0.64) | 0.14 | 0.64 | 0.69 | 0.19 | 0.22 |
| 13 | 0.024 / 0.34 | 0.42 (0.70) | 0.42 | 0.69 | **0.84** | 0.31 | 0.31 |
| 16 | 0.027 / 0.41 | 0.40 (0.67) | 0.39 | 0.68 | 0.81 | 0.26 | 0.32 |
| **18** | **0.035** / 0.39 | 0.38 (0.69) | 0.38 | 0.68 | 0.81 | 0.25 | 0.29 |
| 22 | 0.019 / 0.39 | 0.40 (0.74) | 0.40 | 0.76 | 0.80 | 0.23 | 0.27 |
| 24 | 0.022 / 0.38 | 0.41 (0.75) | 0.41 | 0.75 | 0.78 | 0.24 | 0.23 |
| the model's own decision on these | | | | 0.74 | | | |

* **The set is not there.** On sets the probe has not seen, exact recovery peaks at 0.035 (layer 18), Jaccard 0.41;
  Q* and V* are linear at R² ≤ 0.42, the best layer for Q* (15) is below the best for the set (18), so **READ fails**
  as posed. What the state does hold, from layer 10 on, is whether a forward move will bump (R² 0.84 at layer 13), a
  one-dimensional summary of the set that the next decision needs; the cell and heading marginals are at R² ≤ 0.35.
* **The readout is not a function of membership (LIN fails as posed).** Fitted from the exact 140-dimensional
  membership vector to the model's centred logits over all sets (half fit, half test), a linear map agrees with the
  model's own decision in 0.66 and MLPs in 0.66–0.67 (R² 0.18–0.31); the linear part of LIN holds trivially, the MLP
  part fails. The model's logits are not any function of the set, consistent with section 1.

### Post hoc: the single-goal study's set code was a lookup over the optimum's sets (`posthoc_probe.json`)

The same probe as the single-goal study (its 8 192 decisions from the optimum's episodes), split by episode as there
and split by set; and decisions from the training data's own episodes, split both ways.

| decisions | split | sets | layer 12: exact / Jaccard / AUC | 16 | 18 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| optimum decisions | by episode | 246 | 0.781 / 0.83 / 0.998 | 0.953 / 0.96 / 1.000 | 0.959 / 0.97 / 1.000 | 0.965 / 0.97 / 1.000 | 0.956 / 0.97 / 1.000 |
| optimum decisions | by set | 246 | 0.006 / 0.26 / 0.861 | 0.016 / 0.23 / 0.824 | 0.027 / 0.24 / 0.800 | 0.007 / 0.21 / 0.848 | 0.018 / 0.22 / 0.775 |
| training-data decisions | by set | 983 | 0.032 / 0.18 / 0.932 | 0.019 / 0.18 / 0.934 | 0.025 / 0.18 / 0.916 | 0.018 / 0.15 / 0.908 | 0.012 / 0.15 / 0.892 |
| training-data decisions | by episode | 983 | 0.498 / 0.59 / 0.980 | 0.688 / 0.77 / 0.985 | 0.679 / 0.76 / 0.985 | 0.676 / 0.75 / 0.983 | 0.632 / 0.72 / 0.978 |

* **Held-out decisions on seen sets: 0.96 exact. Held-out sets: 0.01–0.03 exact, Jaccard 0.21–0.26, AUC 0.78–0.86.**
  The optimum's episodes visit 247 sets (246 in this sample), and a linear probe with 896 inputs fitted on 7 000 decisions over them
  recovers each set's 140 bits as a lookup from a state that identifies the trajectory. The single-goal report's
  reading, that the model "holds the full set of 140 state hypotheses as a linear readout of its middle layers", is
  withdrawn: the state identifies which of the optimum's few situations the model is in, and the set is a function of
  that. On decisions from the training data's own episodes (983 distinct sets in the 8 192 sampled decisions) the lookup is weaker (by episode: 0.69 exact at layer 16) and held-out sets again give 0.019 exact, Jaccard 0.18, AUC 0.93: an AUC in the 0.9s with a Jaccard below 0.2 is what a state that knows roughly where the agent is, without the set, looks like.

## 3. Causal: the decision token carries the history, not the set

**Same-set and different-set swaps** (1 024 pairs each; the decision token's residual entering layer l replaced by
the other history's; natural decisions differ in 0.24 of same-set pairs and 0.57 of different-set pairs):

| layer | same set: decision changed | same set, where the two histories' decisions differ: to the donor's | different sets, same size: to the donor's | logit transfer |
|---|---|---|---|---|
| 8 | 0.015 | 0.03 | 0.03 | 0.00 |
| 12 | 0.047 | 0.07 | 0.05 | 0.06 |
| 16 | 0.120 | 0.25 | 0.34 | 0.37 |
| 18 | 0.134 | 0.32 | 0.41 | 0.44 |
| 20 | 0.160 | 0.38 | 0.53 | 0.58 |
| 22 | **0.205** | **0.65** | **0.79** | 0.82 |

* **SAME fails.** Swapping the residual between two histories of the same set changes the decision in up to 0.21 of
  pairs, and where the model's natural decisions for the two histories differ it follows the donor's in 0.65 at layer
  22: the token carries whatever history-specific information makes the two decisions differ. Different-set swaps
  reach 0.79 at layer 22 (the four-goal model's history swaps on the optimum's distribution: 0.97), with 0.21 of the
  decision still coming from elsewhere (the transcript tokens) at the second-to-last layer.

**Membership edits** (2 048 pairs S, S′ at Hamming distance ≤ 2 with disjoint optimal actions; the set probe at the
edited layer fitted on other sets):

| layer (probe exact there) | installed state | to the model's decision on S′ | into π*(S′) | norm / natural change |
|---|---|---|---|---|
| 16 (0.04) | **membership edit** (probe logits of the differing states mirrored) | 0.018 | 0.073 | 0.10 |
| 16 (0.04) | the edit × 2 | 0.022 | 0.075 | 0.21 |
| 16 (0.04) | the edit × 4 | 0.035 | 0.085 | 0.41 |
| 16 (0.04) | same norm along the probe rows of other states | 0.021 | 0.074 | 0.10 |
| 16 (0.04) | random direction, same norm | 0.021 | 0.074 | 0.10 |
| 16 (0.04) | **natural swap** (S′'s history's residual) | 0.331 | 0.371 | 1.00 |
| 18 (0.04) | **membership edit** (probe logits of the differing states mirrored) | 0.020 | 0.073 | 0.10 |
| 18 (0.04) | the edit × 2 | 0.022 | 0.074 | 0.20 |
| 18 (0.04) | the edit × 4 | 0.033 | 0.081 | 0.41 |
| 18 (0.04) | same norm along the probe rows of other states | 0.020 | 0.074 | 0.10 |
| 18 (0.04) | random direction, same norm | 0.019 | 0.073 | 0.10 |
| 18 (0.04) | **natural swap** (S′'s history's residual) | 0.430 | 0.447 | 1.00 |
| 20 (0.03) | **membership edit** (probe logits of the differing states mirrored) | 0.021 | 0.075 | 0.10 |
| 20 (0.03) | the edit × 2 | 0.023 | 0.079 | 0.20 |
| 20 (0.03) | the edit × 4 | 0.038 | 0.090 | 0.40 |
| 20 (0.03) | same norm along the probe rows of other states | 0.018 | 0.073 | 0.10 |
| 20 (0.03) | random direction, same norm | 0.020 | 0.074 | 0.10 |
| 20 (0.03) | **natural swap** (S′'s history's residual) | 0.577 | 0.563 | 1.00 |
| 22 (0.03) | **membership edit** (probe logits of the differing states mirrored) | 0.019 | 0.074 | 0.09 |
| 22 (0.03) | the edit × 2 | 0.021 | 0.075 | 0.19 |
| 22 (0.03) | the edit × 4 | 0.032 | 0.083 | 0.38 |
| 22 (0.03) | same norm along the probe rows of other states | 0.018 | 0.073 | 0.10 |
| 22 (0.03) | random direction, same norm | 0.020 | 0.074 | 0.09 |
| 22 (0.03) | **natural swap** (S′'s history's residual) | 0.797 | 0.728 | 1.00 |

* **EDIT fails, and the test is moot.** At the best set layer (18) the edit along the probe moves 0.020 of decisions to the model's S′ decision, as random (0.019) and the same-subspace control (0.020) do, at four times its norm 0.033; the natural swap 0.430 (0.797 at layer 22). The probe whose rows define the edit recovers the set in 0.03–0.04 of these decisions, so there is no set code to edit; the edit is a small move (0.10 of the natural change) in a direction that means nothing to the network, as the belief-directed edits of study 3 were.

## 4. Where the computation is: in the early layers, from the raw tokens, not through a carried set

**Step tokens (D1).** At the newline ending step i, the set after i steps is exactly recoverable at layer ≥ 14 in
1.00 of decisions for i ≤ 4, 0.86 at i = 8, 0.55 at i = 12, 0.48 at i = 15 (mean over i ≥ 4 at the best layer: 0.74;
**STEP fails**). The decline tracks the number of distinct sets the step can hold (a handful after 4 steps, thousands
after 12), and the split here is not by set-so-far (disclosure 3): this is the same lookup as above, exact where the
family is small.

**The prefix patch (D2).** 1 024 pairs A, B with same-length prefixes of different sets and the same 3-step suffix,
disjoint optimal actions at the end (the model's natural decisions differ in 0.80).

| layer | last prefix token ← B: to B's decision | all prefix tokens ← B | middle prefix token ← B |
|---|---|---|---|
| 4 | 0.007 | 0.887 | 0.024 |
| 8 | 0.023 | 0.646 | 0.016 |
| 12 | 0.017 | 0.387 | 0.006 |
| 16 | 0.010 | 0.139 | 0.001 |
| 18 | 0.002 | 0.077 | 0.001 |
| 20 | 0.002 | 0.043 | 0.002 |
| 22 | 0.002 | 0.022 | 0.002 |

* **PRIOR fails: nothing passes through the step boundary.** Replacing the last prefix token's residual with B's moves
  0.2–2 % of decisions at any layer, the middle token's less. Replacing every prefix token transfers 0.65 at layer 8
  and 0.08 by layer 18: the later positions have taken what they need from the prefix tokens in the first eight
  layers, and from all of them, not from a summary at the boundary. The transcript is read as raw tokens by early
  attention; there is no incremental filter whose state the next step inherits.

## Scorecard

| | criterion | numbers | held |
|---|---|---|---|
| SUF | same action on every history of a set ≥ 0.95, ≥ 0.10 above baseline | 0.591 (0.58–0.60); baseline 0.431 | **no** |
| TABLE-IN | optimal on training-visited sets ≥ 0.95 | 0.882 (optimum-visited 0.968, other 0.879) | **no** |
| TABLE-OUT | optimal on never-visited sets ≥ 0.85 | 0.648 (0.64–0.66) | **no** |
| VALUE | V^π(start) ≥ 0.98 V*, mean V^π / V* ≥ 0.90 | 0.946; 0.825 | **no** |
| READ | best Q* layer ≥ 2 above the best set layer, R² ≥ 0.90 | set 18, Q* 15 (R² 0.42) | **no** |
| LIN | linear fit agrees ≤ 0.90, MLP ≥ 0.97 | 0.664; MLP 0.674 | **no** (the MLP part) |
| SAME | same-set swaps ≤ 0.10 at every layer; different-set ≥ 0.70 from the best set layer | 0.205 at layer 22; 0.41–0.79 | **no** |
| EDIT | membership edit switches ≥ 0.50 to π*(S′); controls ≤ 0.15 | edit 0.073 into π*(S′) (0.020 to the S′ decision); random 0.073, control 0.074; natural swap 0.447 | **no** |
| STEP | set so far from the step token ≥ 0.90 (steps ≥ 4) | 0.74 | **no** |
| PRIOR | last prefix token patched switches ≥ 0.50 | 0.002 (all prefix tokens 0.08) | **no** |

| | expectation | held |
|---|---|---|
| E1 | the policy is a function of the set | no: 0.59 of sets, 0.50 on never-visited ones |
| E2 | optimal on training-visited sets, ≥ 0.85 on never-visited, better than the transfer MLP | no: 0.88, 0.65, below both transfer readouts |
| E3 | the table's value ≥ 0.98 at the start | no: 0.946, and 0.83 on average |
| E4 | the set at 16–20, Q* and the action read from it at 20–23 | no: the set is not decodable on held-out sets (0.035); P(bump) is, from layer 10 |
| E5 | the readout nonlinear in membership, an MLP ≥ 0.97 | no: not a function of membership at all (MLP 0.67) |
| E6 | SAME holds, EDIT at 0.5–0.8 of the natural swap | no: same-set swaps change up to 0.21; the edit is at the random floor (0.020 against the natural swap's 0.430) |
| E7 | incremental filtering at the step tokens; the decision token reads both | no: no set beyond small families at the step tokens; nothing passes through the step boundary (0.002); the prefix is read from its raw tokens in layers < 8 |

None of the ten registered criteria held, and none of the seven expectations: every one presupposed that the model computes the belief set, and the measurements say it does not.

## Reading

The fine-tuned model does not compute the belief set and read the action from it. Its policy is a function of the
transcript tuned to the trajectories it was trained on: exactly optimal where the optimum's own episodes go (247 sets,
reached the optimum's way), 0.93 optimal on the 6 800 sets the ε-perturbed training data reaches by training-like
histories, 0.88 on the same sets by a shortest history, 0.84 by a random walk, 0.65 on the 10 000 reachable sets the
training never visits, and different on a quarter of same-set history pairs, following the history's length and last
move. Its state carries the history's identity, from which a probe fitted on familiar situations can look up the set
(0.96) but a probe asked about an unseen set cannot (0.03); what it carries about the belief in general is the
one-dimensional thing the next step needs, whether forward will bump. The transcript is consumed as raw tokens in the
first eight layers; nothing is carried through the step boundaries. The single-goal study's conclusion that the model
"holds the full set of consistent states as a linear readout" is withdrawn: it held an index of the optimum's few
situations. With the four-goal results this completes the picture of these fine-tuned models: they are exact imitators
of the optimum's trajectory distribution, not implementations of the filter-and-plan structure the optimum has, and
their behaviour off that distribution, on histories and sets the optimum never produces, is the behaviour of an
interpolator.

## Limits

One maze, one goal, one model per size; the shortest history and random walks are out-of-distribution histories, so
the never-visited sets are confounded with unfamiliar histories (the post hoc separates the two only on
training-visited sets); the set probes are linear with 896 inputs and 7 000 fit decisions, so a nonlinear or
distributed set code would be missed, although an MLP readout from the exact set cannot reproduce the model's
decisions either (B2), which no code of the set could survive; the prefix patch tests one token and all tokens, not
the attention routes; D1's split is by final set; the exact value of the table assumes the table is the policy, which
section 1 shows it is not, so A3 measures the gap rather than the policy's value.
