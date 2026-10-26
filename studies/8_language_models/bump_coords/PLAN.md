# Bump coordinates: are the fine-tuned model's goal-specific states one linear coordinate system, how many directions carry the goal, and does the goal-free belief survive the change of goal?

Written before any of the measurements below were run. **Disclosure:** the model (`../bump_goals/runs/llm_40000`),
its goal-swap data (`../bump_goals/factorial.json`: the single linear map at layer 23, 0.883 of the natural swap's
0.985; the mean offset at the random floor) and the policy study's probe pool (`../bump_policy/sets.pt`,
`histories.pt`) exist; the maze10 composition result (`../../7_information_seeking/H_simplex`, section 11: composed
maps agree with the direct one at R² 0.99 and switch 0.85 against 0.86) is the benchmark. Brief: three tests given
in conversation (transformation composition, low-rank structure, shared belief preservation), the decoder choice
("the soft set probe with the bump probe as control") made there.

## The question

The bump goals study found that at the top of the fine-tuned four-goal model the goal's effect on the decision
token's state is nearly a fixed linear map (0.90 of the natural swap at layer 23) with no constant part, read in
over layers 20–21 of 24, after the history's belief arrives at layers 13–19. The maze10 study found the same kind of
maps form one consistent, invertible system over a shared nonlinear code of the posterior. Three questions on the
language model: (A) do its pairwise goal maps compose and invert, as one coordinate system would; (B) how many
directions of the state does the goal's effect occupy, in reconstruction and in what the decision uses; (C) does the
goal-free belief, as far as it is linearly decodable (the policy study: the exact set is not, a soft set at Jaccard
0.41 and whether forward will bump at R² 0.84 are), sit in coordinates the goal map carries, so that a decoder fitted
under one goal is restored under another by the inverse map?

## The model, sites and data

The 40 000-episode fine-tuned Qwen2.5-0.5B of the bump goals study (exactly optimal under the four corner goals;
the model of its swap section). The decision token's residual entering layers **15, 19, 21, 23** (the swap study's
sites: the history swap works from 13 and reaches 0.78 at 19 while the goal swap is at 0.56; the goal swap reaches
0.84 at 20, 0.93 at 21, 0.98 at 23). The best layer for every criterion is the one with the largest natural goal
swap among the four (23 by the swap study; 21 reported alongside).

**Swap decisions** (parts A and B), as in the swap study and rebuilt with its seeds: decisions
after the first step from the optimum's episodes under every goal (`decisions_with_all_goals(worlds, sols, 256, 700)`),
same-step pairs with different goal-free sets; **6 144 fit decisions** and **1 024 test pairs**, every history run under
all four goals, the residual at every site recorded. Maps are fitted on the fit decisions only and every number below
is on the test pairs. Causal measures follow the swap study's convention exactly: the history's prompt is run **under the source goal
A**, the decision token's residual at the site is replaced by the candidate state for B (the `Patched` hook of
`factorial_llm.py`), the logits are read; **switch** is the share of goal-dependent decisions (the natural decisions
under A and B differ) that move to B's natural decision. The natural state under B is the ceiling (0.985 at layer 23),
a random vector of the natural change's norm the floor; the unpatched run gives A's decision by construction. (Running
the prompt under the target goal instead would make the ceiling 1.0 and every site look perfect; it is not used.)

**Probe decisions** (part C): 8 192 decisions uniform over the policy study's 17 285 goal-free reachable sets by their
shortest histories (`sets.pt`, `histories.pt`; the goal-free filter is `Filter(w, n, remove_goal=False)`), held out
**by set** (half the sets fit, half test, as in the policy study), each run under all four goals. Labels are the
goal-free set (140 states) and P(bump | forward) from it. The four-goal map is the single-goal map; a history that
would have ended under a goal (the set within that goal's states) is dropped.

**Maps.** For each ordered goal pair (A, B), M(A→B): ridge from the state under A and the one-hot step to the state
under B, fitted on the fit decisions, λ = 10⁻² as before (λ tuned on a held-out quarter of the fit decisions is
reported too; the registered numbers use the tuned λ). The inverse map M(B→A) is fitted the same way, not inverted.

## Part A. Composition

For every ordered triple (A, B, C) of distinct goals (24) and site: on the test histories under A, the direct prediction
M(A→C) x_A and the composed one M(B→C) M(A→B) x_A.
* **Representation**: R² of the composed prediction against the direct one (the B-independent part), and of each
  against the actual state under C; the composed prediction's norm over the direct's.
* **Causal**: each prediction installed in the prompt under A (the source), switch to C's natural decision among
  decisions where C's and A's natural decisions differ; the direct map's switch is the reference, the natural state
  under C the ceiling.
* **Round trips**: M(B→A) M(A→B) x_A against x_A (R²; installed in the prompt under A, the share of goal-dependent
  decisions that keep A's natural decision, against x_A itself re-installed and against x_B, which switches them);
  three-cycles M(C→A) M(B→C) M(A→B) x_A.

## Part B. Rank

At each site and ordered goal pair, the goal difference Δ_{A→B}(h) = x(h, B) − x(h, A) over the fit decisions, with
the per-step mean removed (the goal × step offset; its own norm share is reported, G-OFF put it at the floor).
* **Reconstruction rank**: the singular spectrum of the centred differences across histories; the number of
  directions for 50, 90 and 99 % of their variance (k50, k90, k99); the same for the pooled differences of all
  twelve ordered pairs, and the overlap between pairs' top-k subspaces (mean squared cosine of principal angles).
* **Causal rank**: the rank-k map x_A + P_k (M(A→B) x_A − x_A), P_k the projector onto the top-k singular directions
  of Δ_{A→B} on the fit decisions, k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 896}, installed in the prompt under A
  on the test pairs; switch against the full map's. The **effective causal rank** k_c(A→B) is, per ordered pair and
  site, the smallest k in the list at which the rank-k map switches ≥ 0.9 × the full map; the median and range over
  the twelve pairs are reported, and every later use of k_c (the shared-subspace sweep, the probe overlap, part C's
  step 5) takes the pair's own k_c. Controls at every k: the bottom-k directions, and k random directions of the
  state space (three draws); the full map and the natural state as references.
* **Shared subspace**: the same sweep with P_k from the pooled differences of all twelve pairs, and with P_k from the
  singular directions of M(A→B) − I itself (the map's own change, rather than the differences').
* **The probe's overlap** (for part C): the fraction of the set probe's row space (its 140 weight rows under A, part
  C, mapped back through the probe's standardisation) lying in the pair's top-k_c causal subspace, against random
  subspaces of the same dimension.

## Part C. Shared belief preservation

At each site, under each goal g, a linear **soft set probe** (140 logistic outputs, as `probes.py`; Jaccard between the
thresholded prediction and the set, and exact recovery) and a linear **bump probe** (ridge, R²) fitted on the fit sets'
decisions under g. Baselines first: each probe's held-out score under its own goal (the four-goal model has no probe
numbers yet; the single-goal model's were Jaccard 0.41, exact 0.035, bump 0.84), and a chance floor from probes fitted
on set labels shuffled across decisions.

Then, for each ordered goal pair (A, B), the probe fitted under A scored on held-out sets' decisions:
1. under A (j_A, the baseline);
2. under B, the states unchanged (j_B, the drop);
3. under B, the states pulled back through M(B→A) (j_back, the restoration);
4. under B, the per-step mean offset B→A removed only (j_off, the control for a constant shift);
5. under B, pulled back through a rank-k_c M(B→A) (part B's causal rank; whether the directions that move the
   decision are the ones that carry the decodable belief);
6. the probe fitted under B scored under B (j_BB, the ceiling).
The same six cells for the bump probe. **Restoration** is (j_back − j_B) / (j_A − j_B), defined where the drop
j_A − j_B exceeds its bootstrap interval. **A causal check of the pull-back**, on the held-out half of the probe
decisions: the full and the rank-k_c pulled-back states installed in the prompt under B, the share of goal-dependent
decisions (B's and A's natural decisions differ) that move to A's natural decision, against the natural state under A;
a pull-back that the probe reads as an A-state should also decide as one, and the two pull-backs' switches are
reported next to their restorations. The probe's row-space overlap with the causal goal subspace (part B) is read
with these: if the set probe does not drop under B and its rows lie outside the goal subspace, the belief's coordinates
and the goal's are separate; if it drops and the map restores it, the goal transforms the coordinates the belief is
read from.

## Decision rule

One model; maps fitted on the fit decisions, every number on the held-out test pairs or held-out sets; bootstrap 95 %
intervals over test pairs (A, B) or held-out decisions (C); a criterion holds if its interval excludes the threshold.
Criteria are evaluated at the best layer (the largest natural goal swap among the four sites, expected 23); the other
sites are reported.

| | criterion |
|---|---|
| **COMP-R** | the composed prediction agrees with the direct map's at R² ≥ 0.95 (mean over the 24 triples; maze10: 0.99) |
| **COMP-S** | the composed map's switch is ≥ 0.90 × the direct map's (maze10: 0.85 / 0.86) |
| **ROUND** | the round trip M(B→A) M(A→B) x_A patched under A keeps A's natural decision in ≥ 0.90 of goal-dependent decisions (x_A re-patched: the ceiling) |
| **RANK-C** | the rank-32 map (of 896 dimensions) switches ≥ 0.90 × the full map, mean over the twelve ordered pairs (equivalently k_c ≤ 32 on the aggregate; k_c per pair reported) |
| **RANK-SH** | at each pair's own k_c, the pooled-subspace rank-k_c map switches ≥ 0.90 × the full map, mean over pairs (one goal subspace shared by all pairs) |
| **PRES-D** | the set probe fitted under A loses ≥ 0.05 Jaccard under B (mean over the 12 ordered pairs) |
| **PRES-R** | where PRES-D holds, the pulled-back states restore ≥ 0.80 of the drop, and ≥ 0.10 of it more than the offset control |
| **BUMP-SH** | the bump probe fitted under A loses ≤ 0.05 R² under B unmapped (the control decoder transfers without a map) |

## Expectations

| | expectation |
|---|---|
| E1 | COMP-R and COMP-S hold at layers 21 and 23, as in maze10: the four goal coordinate systems are one linear system; at 15 and 19 the maps barely move the decision (the goal is not yet in the token) and composition is reported but uninformative |
| E2 | ROUND holds: the maps invert, with the ridge's shrinkage as the only loss |
| E3 | RANK-C holds with the median k_c between 8 and 32, while k90 of the differences exceeds 100: the map's 0.67 state R² is spread over many directions, the decision uses few. The bottom-k and random-k controls stay at the floor up to k_c |
| E4 | RANK-SH holds: the twelve pairs' causal subspaces coincide (principal-angle overlap ≥ 0.7), consistent with E1 |
| E5 | the four-goal model's own probes look like the single-goal model's: exact ≤ 0.05, Jaccard 0.3–0.5 at the best layer, bump R² ≥ 0.7; the floor is well below (Jaccard ≤ 0.2) |
| E6 | PRES-D holds at 21 and 23 and not at 15 (the goal has not arrived there: the token's state is the same under every goal and the probe transfers trivially); PRES-R holds at 23: the goal rewrites the coordinates the soft set is read from, and the map carries them back; the rank-k_c pull-back restores less than the full one (the decodable belief is spread beyond the causal subspace) |
| E7 | BUMP-SH holds at every site: the bump feature is goal-independent and read from the same coordinates under every goal, so it transfers without a map; its restoration is then undefined, which is the control's point |

The alternative E6 would reject: the set probe does not drop under B at any site, and its rows lie outside the causal
goal subspace. Then the goal acts in coordinates complementary to those the belief is read from, and "a change of
coordinates" is the wrong picture: the goal adds a direction, it does not transform.

## Limits known in advance

One model, one seed, four goals, one maze; the maps are linear in 896 dimensions fitted on 6 144 decisions and
the single map's state R² was 0.67, so composition is tested on an imperfect map (the maze10 maps had R² 0.95);
the probe decisions' shortest histories are out of distribution under every goal, equally; the soft set probe's
0.41 is weak and PRES-R measures a fraction of a small drop, hence PRES-D's floor of 0.05; the rank sweep's directions
come from the differences' SVD, one basis among many, with the map's own SVD as the one alternative; the causal
measures read the decision token only; the exact set is not decodable, so part C cannot test the belief itself, only
its linearly decodable shadow.

## Compute

About 2–3 h on one GPU: 7 168 decisions × 4 goals recorded once (every site in one pass), 8 192 probe decisions × 4
goals once; patched runs of 1 024 prompts each: part A 24 triples × 2 predictions + 12 round trips + 24 cycles per
site, part B 11 ranks × 4 variants × 12 pairs per site, part C's causal check 12 pairs × 2 pull-backs × 4 096 probe
decisions per site; about 5 000 patched batches of 1 024 at 4 sites, part C's check about the same again. Runs alongside the reward study's transformer training (2.5 GB per process).
