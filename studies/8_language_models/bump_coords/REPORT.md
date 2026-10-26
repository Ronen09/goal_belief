# Bump coordinates: on histories the maps never saw, the goal maps still compose and the decision reads four of their directions, but the maps lose a third of the state, invert to 0.90, and are maps of the optimum's trajectories only; a belief probe does not survive the change of goal

Run date: 2026-10-26. Plan, criteria and expectations: `PLAN.md`, committed (`6b3a209`) **before any measurement was run**.
Numbers from `tables.md`; data in `compose.json`, `rank.json`, `preserve.json`, `posthoc_pool.json`, `collect.json` (the
recorded states, the maps and the singular directions in `cache/`, not committed). Reproduce: `reproduce.py bump_coords`
(`run_all.sh`: `collect.py`, `compose.py`, `rank.py`, `preserve.py`, `posthoc_pool.py`, `tables.py`; 53 min on one GPU).

**Disclosures.** (0) **The first full run was withdrawn and the study rerun with a different pool construction** (its
files are in commit `0d6fc32`; nothing from it is used). The plan's swap pool ("6 144 fit decisions and 1 024 test pairs,
as the goal-swap study builds them") draws decisions along the optimum's episodes, and the optimum is deterministic from
140 start states: 14 643 decisions held 2 349 distinct histories, the 6 144 fit decisions 1 418, and 834 of the 1 024
"test" histories were among the fit histories. The review of the first run found it; on the 190 test histories the fit
had not seen, the layer-23 maps predicted the state under C at R² −0.40 (in-fit 0.96; layer 21: −0.04 against 0.97;
19: 0.47 against 0.98; 15: 0.88 against 1.00), so that run's parts A and B were mostly in-sample, its λ-tuning
quarter was contaminated the same way, and its "held-out R² 0.89" was a lookup. The rerun deduplicates decisions by
history, draws 1 024 episodes per goal (58 449 decisions, 2 639 distinct histories), splits them by history (a
held-out quarter, 660 histories, is the test set, 1 979 the fit set; no history in both), clusters every bootstrap by
history (the same test histories enter every goal pair and triple), extends the λ grid to 100, and takes the map's
own singular directions from the uncentred matrix. The pairs' second histories, which no script used, are no longer
recorded. **The goal-swap study's single linear map at layer 23 (0.883 of the natural swap) rests on the same
construction; on held-out histories here the same map gives 0.81 (section 1). That study is not corrected here.**
(1) Before any run, the plan's description of the causal measure was corrected at review: a parenthesis said the
prediction is installed in the prompt under the target goal; every causal number follows the goal-swap study's
convention (the prompt under the source goal, the candidate state for the target installed, the share of
goal-dependent decisions moving to the target's decision), which the plan's text now states. (2) The rank sweep's
controls (bottom-k, three random-k draws, the map's own singular directions) were run only at the two sites with the
largest natural goal swap (23 and 21), and all patched passes of part B run on each pair's goal-dependent decisions
only (the switch is defined on those alone, so no number changes); the plan said "at each site". (3) The policy
study's probe fitter could not be imported (its module imports a `common` of its own, colliding with this study's);
`preserve.py` carries the same recipe (standardise on the fit rows, LBFGS with strong-Wolfe line search, 800
iterations, L2 10⁻⁴). (4) The maps' ridge penalty was tuned on a held-out quarter of the fit histories, as the plan
says; the chosen λ is 1–10 at layer 15 and 100, the top of the grid, at layers 19–23. (5) Every script was smoke-tested
in quick mode before each full run (outputs under `_smoke/`, unused); the quick run exposed one defect fixed before the
first full run (an incomplete SVD basis with fewer fit rows than dimensions, irrelevant at the full sizes). (6)
**After the registered part C was seen**, one post hoc analysis was added (`posthoc_pool.py`, section 3's C3): the maps
refitted on the probe pool, because the diagnostic in section 3 shows the registered maps are not maps on that pool.
(7) One model, one seed; the probe pool dropped 87 of 8 192 histories as ended under some goal (8 105 kept, 4 053 fit /
4 052 test by set).

**The question.** The bump goals study found the goal's effect on the fine-tuned four-goal model's decision token to be
nearly a fixed linear map at the top (0.90 of the natural swap at layer 23) with no constant part, read in over layers
20–21 after the belief arrives at 13–19; the maze10 study found such maps form one consistent, invertible system over a
shared code of the posterior. Here: (A) do the language model's pairwise goal maps compose and invert; (B) how many
directions carry the goal's effect, in reconstruction and in what the decision uses; (C) does the goal-free belief's
linearly decodable shadow (the soft set probe; the bump probe as control) sit in coordinates the goal map carries?

Sites: the decision token's residual entering layers 15, 19, 21, 23; maps on 1 979 fit histories, every number on the
660 held-out histories (229 goal-dependent per pair). The best site is 23 (natural goal swap 0.978; 21: 0.881, 19:
0.478, 15: 0.370; the goal-swap study's 0.985 / 0.934 / 0.573 / 0.399 on its own, partly in-sample, pairs).

## 1. Composition: the maps compose causally within 0.04 of the direct map at every layer; by R² only where the maps are good; they invert to 0.90 at the top

| site | maps: held-out state R² | R² composed vs direct (24 triples) | R² direct / composed vs x_C | switch: direct | composed | **composed / direct** | natural x_C | random | round trip: R² vs x_A | **keeps A's decision** | x_B keeps | three-cycle keeps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 0.952 | 0.991 | 0.954 / 0.946 | 0.371 | 0.373 | **1.005** (0.992–1.017) | 0.370 | 0.074 | 0.977 | **0.984** | 0.597 | 0.979 |
| 19 | 0.819 | 0.968 | 0.822 / 0.798 | 0.418 | 0.401 | **0.960** (0.944–0.975) | 0.478 | 0.074 | 0.906 | **0.956** | 0.487 | 0.943 |
| 21 | 0.703 | 0.941 | 0.712 / 0.700 | 0.737 | 0.710 | **0.962** (0.949–0.974) | 0.881 | 0.057 | 0.890 | **0.906** (0.891–0.921) | 0.096 | 0.872 |
| **23** | 0.611 | **0.918** (0.914–0.922) | 0.629 / 0.627 | 0.791 | 0.767 | **0.969** (0.957–0.981) | 0.978 | 0.051 | 0.882 | **0.897** (0.882–0.911) | 0.015 | 0.853 |

* **COMP-S holds at every site; COMP-R and ROUND hold at 15 and 19 and not at the top.** Going A → B → C switches C's
  decision in 0.767 of goal-dependent decisions at layer 23 against the direct map's 0.791 (ratio 0.969; at 21 0.962,
  at 19 0.960, at 15 1.005), with the natural state at 0.978 and random vectors at 0.05: whatever the maps carry, the
  composition carries it. By R² the composed vector agrees with the direct one at 0.918 at 23 (0.941 / 0.968 / 0.991
  lower down), tracking the maps' own quality: a map that predicts x_C at 0.63 applied twice agrees with one applied
  once at 0.92, and where the maps predict at 0.95 the agreement is 0.99. A round trip A → B → A keeps A's decision in
  0.897 of the decisions B would switch at 23 (0.882–0.911, just under the registered 0.90; 0.906 at 21, 0.956 and
  0.984 lower down) and returns x_A at R² 0.88–0.98; a three-cycle 0.85 at the top.
* **The single map on held-out histories is 0.81 of the natural swap at the top, not 0.90.** M(A→B) x_A switches 0.791
  against the natural 0.978 at layer 23 and 0.737 against 0.881 at 21; the maps lose 0.3–0.4 of the state's held-out
  variance there (R² 0.61 / 0.70; 0.95 at 15). The goal-swap study's 0.883 / 0.985 and the first run's 0.885 were
  measured on histories the maps had been fitted on (disclosure 0).

## 2. Rank: the decision reads four directions of a change that spans two hundred

| site | k50 / k90 / k99 of the centred difference (median pair; pooled over pairs) | goal × step offset: share of the difference | switch: full map / natural | **k_c median (range)** | rank-32 / full | pooled subspace at k_c / full | overlap of pairs' top-k_c subspaces (random) |
|---|---|---|---|---|---|---|---|
| 15 | 5 / 101 / 424 (9 / 171 / 579) | 0.587 | 0.372 / 0.368 | 128 (8–512) | 0.861 | 0.873 | 0.407 (0.079) |
| 19 | 18 / 224 / 592 (34 / 348 / 745) | 0.234 | 0.417 / 0.476 | 64 (16–128) | 0.865 | 0.895 | 0.409 (0.061) |
| 21 | 13 / 214 / 586 (19 / 319 / 731) | 0.139 | 0.740 / 0.879 | **6 (4–32)** | **0.965** (0.953–0.976) | 0.915 (0.898–0.933) | 0.656 (0.005) |
| **23** | 11 / 208 / 584 (13 / 290 / 712) | 0.106 | 0.791 / 0.978 | **4 (4–16)** | **0.978** (0.968–0.988) | **0.935** (0.919–0.951) | 0.764 (0.003) |

The rank-k map at site 23, switch over the 12 ordered pairs (full map 0.791):

| directions | k = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 896 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **top-k of the difference** | 0.35 | 0.56 | 0.73 | 0.76 | 0.77 | 0.77 | 0.78 | 0.78 | 0.79 | 0.79 | 0.79 |
| pooled over pairs | 0.35 | 0.55 | 0.73 | 0.75 | 0.76 | 0.77 | 0.78 | 0.78 | 0.78 | 0.79 | 0.79 |
| the map's own singular directions | 0.35 | 0.53 | 0.69 | 0.72 | 0.74 | 0.74 | 0.75 | 0.76 | 0.77 | 0.78 | 0.79 |
| bottom-k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.79 |
| random-k (three draws) | 0.00 | 0.00 | 0.00 | 0.01 | 0.01–0.02 | 0.02–0.03 | 0.03–0.05 | 0.07–0.08 | 0.13–0.18 | 0.43–0.46 | 0.79 |

* **RANK-C holds at 21 and 23, RANK-SH at 23 (at 21 by 0.002 not).** The first singular direction of the per-history
  goal difference gives 0.35 of what the full map gives, two 0.56, four 0.73 of 0.79; the effective causal rank is 4
  in ten of the twelve pairs (8 and 16 in the other two; median 6 at layer 21), and the bottom directions and k random
  directions do nothing until k is half the space. Reconstructing the difference needs 200–220 directions for 90 % of
  its variance (290 pooled), 580 for 99 %: the decision reads one direction in fifty. One pooled subspace serves every
  pair (0.935 of the full map at each pair's rank; 0.915 at 21, interval 0.898–0.933), and the pairs' subspaces
  overlap at 0.76 against 0.003 random (0.66 at 21). The map's own singular directions are a slightly worse basis.
* **Lower down the goal's effect is a constant and diffuse.** At layer 15, 0.59 of the difference's norm is a goal ×
  step offset (0.11 at 23), the remainder's causal rank 128 and the rank-32 map 0.86 of a full map that switches 0.37;
  at 19, 64 and 0.87 of 0.42. The low-rank readout appears where the goal is read into the decision, layers 20–21, as
  the goal-swap study placed it.

## 3. Shared belief preservation: the set probe drops under the other goal, the registered maps make it worse, and the maps are not maps off the optimum's trajectories

**Probes by goal** (held-out sets; the four goals within 0.02 of each other): the soft set probe's Jaccard 0.32–0.34 at
every site (exact 0.01–0.016; floor from shuffled labels 0.000–0.001); the bump probe's R² 0.80 at layer 15 falling to
0.74–0.76 at 23 (floor −0.33 to −0.41). The single-goal model's were 0.41 and 0.84: **E5 holds**.

| site | set: drop j_A − j_B | restoration by the inverse map | over the offset control | by the rank-k_c map | bump: R²_A − R²_B unmapped | causal check: the pull-back installed under B switches to A's (natural x_A) | probe rows in the causal subspace (random) | the maps' R² on the probe pool (identity) |
|---|---|---|---|---|---|---|---|---|
| 15 | **0.073** (0.069–0.077) | **−0.98** | −1.19 | −0.01 | **0.89** | 0.20 (0.21) | 0.145 (0.156) | 0.89 (0.45) |
| 19 | **0.127** | **−0.99** | −0.83 | −0.06 | **0.31** | 0.22 (0.27) | 0.138 (0.156) | 0.64 (0.31) |
| 21 | **0.148** | **−0.80** | −0.66 | −0.04 | **0.26** | 0.47 (0.74) | 0.075 (0.156) | 0.24 (0.23) |
| **23** | **0.157** (0.152–0.161) | **−0.72** (−0.76–−0.69) | −0.61 | −0.02 | **0.21** (0.20–0.21) | 0.48 (0.94) | 0.058 (0.155) | −0.16 (0.19) |

* **PRES-D holds everywhere, PRES-R and BUMP-SH fail everywhere.** A set probe fitted under A loses half its Jaccard
  under B (0.33 → 0.12–0.21 at layer 23), already at layer 15 where the goal barely moves the decision; the bump
  probe loses 0.21 of R² at 23 and 0.89 at 15. The pull-back through M(B→A) restores neither: it takes the set probe
  to 0.05–0.06 (restoration −0.72) and the bump probe's R² to 0.26–0.54; removing the per-step mean offset alone is
  better than the map and still below the unmapped state. The rank-k_c pull-back, which moves the state only in the
  four directions the decision reads, leaves both probes where the unmapped state left them (−0.02): **the probes'
  rows lie outside the causal goal subspace**, below the random overlap (0.06 against 0.16 at 23).
* **The maps are not maps on the probe pool.** The last column decides how to read this. The maps, fitted on
  histories along the optimum's trajectories, reach held-out R² 0.61–0.95 there; on the probe pool (every reachable
  set by its shortest history, out of distribution under every goal) they predict the state under A from the state
  under B at 0.89 at layer 15, 0.64 at 19, 0.24 at 21 and −0.16 at 23, at or below leaving the state as it is at the
  top (0.19–0.23). The causal check agrees: the pulled-back state installed under B switches 0.48 of goal-dependent
  decisions to A's where the natural x_A switches 0.94. On the shortest histories the goal's transformation of the
  state is not the one fitted on the optimum's histories; the restoration test as registered ran through the wrong
  map, and its failure cannot say whether the belief's coordinates are carried.

### Post hoc C3. The maps refitted on the probe pool (`posthoc_pool.py`)

The maps M(B→A) refitted on the probe pool's fit half (same ridge recipe, λ tuned on a quarter; λ = 100 at every pair
at 23) and the A-probes' cells re-scored on its test half (4 052 decisions):

| site | map R² on the probe pool: refitted (registered map; identity) | set: drop | **restoration by the refitted inverse map** | over the offset control | bump: R² lost unmapped | **after the pull-back** | causal check: the pull-back installed under B switches to A's (natural x_A) |
|---|---|---|---|---|---|---|---|
| 15 | 0.958 (0.89; 0.45) | 0.073 | **0.53** (0.50–0.56) | 0.26 | 0.89 | **−0.02** | 0.20 (0.21) |
| 19 | 0.830 (0.64; 0.31) | 0.127 | **0.30** (0.27–0.32) | 0.36 | 0.31 | **−0.03** | 0.22 (0.27) |
| 21 | 0.732 (0.24; 0.23) | 0.148 | **0.35** (0.33–0.37) | 0.41 | 0.26 | **−0.02** | 0.50 (0.74) |
| **23** | 0.655 (−0.16; 0.19) | 0.157 | **0.34** (0.33–0.36) | 0.39 | 0.21 | **−0.02** | 0.58 (0.94) |

* **The bump feature is carried whole, the soft set by a third.** With the inverse map fitted where it is tested, the
  bump probe fitted under A reads the pulled-back states under B at its own R² (0.76–0.77 at layer 23 in every pair,
  within 0.03 of its value under A, at every site), and the set probe recovers 0.34 of its drop at 23 (0.33 →
  0.12–0.21 → 0.21–0.23; pairs 0.11–0.46) and 0.53 at 15. The per-step offset alone recovers nothing (the offset-only
  state is at or below the unmapped one; the map is 0.39 of the drop above it). A third of the set probe's
  goal-sensitivity is a linear change of coordinates between the goals' states; the rest no linear map between them
  reproduces. The share depends on the ridge penalty: the first run's post hoc, with the grid capped at 10, restored
  0.42 at 23 and 0.53 at 19 through maps of slightly lower state R²; the stronger shrinkage chosen by state R² predicts
  the state better and the probe worse.
* **On this pool the maps are poorer and causally weaker.** R² 0.66 at layer 23 (0.61 on the swap pool's held-out
  histories), and the pull-back installed under B switches 0.58 of goal-dependent decisions to A's where x_A switches
  0.94 (the swap-pool map on its own held-out histories: 0.79 of 0.98). The transformation the goal applies to the
  state is approximately linear on each pool and different between them; each pool's map fails on the other. PRES-R's
  registered threshold (0.80 of the drop) is not reached by any map on this pool.

## Scorecard (site 23; 21 in brackets)

| | criterion | value (95 %, cluster bootstrap by history) | holds |
|---|---|---|---|
| COMP-R | composed vs direct R² ≥ 0.95 | 0.918 (0.914–0.922) [0.941] | **no** [no] |
| COMP-S | composed switch ≥ 0.90 × direct | 0.969 (0.957–0.981) [0.962] | yes [yes] |
| ROUND | round trip keeps ≥ 0.90 | 0.897 (0.882–0.911) [0.906 (0.891–0.921)] | **no** [no] |
| RANK-C | rank-32 map ≥ 0.90 × full | 0.978 (0.968–0.988) [0.965] | yes [yes] |
| RANK-SH | pooled subspace at k_c ≥ 0.90 × full | 0.935 (0.919–0.951) [0.915 (0.898–0.933)] | yes [no, by 0.002] |
| PRES-D | the A set probe drops ≥ 0.05 under B | 0.157 (0.152–0.161) [0.148] | yes [yes] |
| PRES-R | the inverse map restores ≥ 0.80 of the drop and ≥ 0.10 over the offset control | −0.721 (−0.756–−0.690); over the offset −0.609 [−0.802] | **no** [no] |
| BUMP-SH | the bump probe loses ≤ 0.05 R² under B unmapped | 0.206 (0.198–0.214) [0.264] | **no** [no] |

At layers 15 and 19 COMP-R, COMP-S and ROUND hold (0.99 / 1.00 / 0.98 and 0.97 / 0.96 / 0.96), RANK-C and RANK-SH
fail (0.86–0.87, 0.87–0.90), and part C is as at 23.

| | expectation | held |
|---|---|---|
| E1 | COMP-R and COMP-S at layers 21 and 23, as in maze10; at 15 and 19 the maps barely move the decision | in part: COMP-S at every site; COMP-R at 15 and 19 only, where the maps are good, and 0.92–0.94 at the top where they lose a third of the state; the maps at 15 and 19 switch 0.37 and 0.42, the natural swap's own values there |
| E2 | ROUND: the maps invert, with the ridge's shrinkage as the only loss | at 15 and 19 (0.98, 0.96); at the top 0.90–0.91, just under, with the maps' held-out error the loss |
| E3 | RANK-C with the median k_c between 8 and 32 while k90 exceeds 100; bottom-k and random-k at the floor up to k_c | yes, and lower: k_c 4 at 23, 6 at 21, k90 210; the controls at 0.00–0.01 to k = 16 |
| E4 | RANK-SH: the twelve pairs' causal subspaces coincide (overlap ≥ 0.7) | yes at 23 (0.935; overlap 0.76); at 21 the pooled ratio's interval reaches 0.898 and the overlap is 0.66 |
| E5 | the four-goal model's probes like the single-goal model's: exact ≤ 0.05, Jaccard 0.3–0.5, bump R² ≥ 0.7, floors well below | yes (0.01; 0.32–0.34; 0.74–0.80; floors 0.00 and < 0) |
| E6 | PRES-D at 21 and 23 and not at 15; PRES-R at 23; the rank-k_c pull-back restores less than the full one | no: the drop is at every site including 15; the full map restores nothing and worsens; the rank-k_c map leaves the probe where it was, and the probe's rows lie outside the causal subspace. The maps are not maps on the probe pool (section 3), so the test is reread post hoc in C3: a third restored |
| E7 | BUMP-SH at every site | no: the bump probe loses 0.21–0.89 of R² under the other goal (and is restored whole by a map fitted on its own pool) |

4 of 8 registered criteria held at the best site (3 of 8 at layer 21; 4 of 8 at 15 and 19, a different four): the
composition's causal half, the two rank criteria and the probe's drop; the composition's R² half and the round trip
miss at the top by the maps' held-out error, and the two restoration criteria fail through the maps' domain.

## Reading

Three answers, all on histories the maps were not fitted on. **(A)** The goal's effect on the decision token is one
system of linear transformations as far as the decision is concerned: the twelve pairwise maps compose causally at
every layer (A → B → C switches the decision 0.96–1.00 as often as A → C) and invert to 0.90 at the top (0.96–0.98
lower down). As transformations of the state they are only partly linear: on held-out histories a map loses 0.4 of
the state's variance at layer 23 (0.05 at 15), composition by R² degrades accordingly (0.92 at the top, 0.99 at 15),
and on the histories it was fitted on the same map looked exact, which is what the goal-swap study and this study's
first run measured. **(B)** What the decision reads of the transformation is four directions of 896: the first
singular direction of the goal difference gives 0.35 of the full map's effect, two 0.56, four 0.73 of 0.79, bottom and
random directions nothing, and the twelve pairs share the subspace (one pooled subspace 0.94 of the full effect; the
pairs' subspaces overlap at 0.76 against 0.003). Reconstructing the goal's change in the state takes 200 directions:
the goal rewrites much of the state and the decision uses a sliver, which appears at layers 20–21, where the goal-swap
study found the goal read into the decision; below, the difference is mostly a goal × step constant and what remains
is diffuse. **(C)** The maps are maps of a distribution. Fitted on the optimum's trajectories they predict little on the
same model's states for shortest histories (R² −0.16 at the top, 0.89 at 15), and fitted there they are as weak (0.66)
and causally weaker (0.58 of the natural swap). On that pool a linear probe for the goal-free belief fitted under one
goal loses half its Jaccard under another at every layer, including layer 15 where the goal does not yet move the
decision: the goal rewrites coordinates the probe reads. The probe's rows lie outside the four causal directions, so
moving the state only along them leaves the probe untouched; the inverse map fitted in place gives back a third of
what the set probe lost and all of what the bump probe lost. For the study's question, how the model holds its belief
while what it wants changes: the belief's decodable shadow sits in coordinates the goal transforms, a third of it
linearly, while the goal's effect on the decision is a four-direction readout beside those coordinates, not a
transformation of them. "A shared belief code in goal-specific coordinates" holds for the bump feature, a third holds
for the soft set, and the maze10 picture of one global linear change of coordinates is, in the language model, exact
only on the histories a map was fitted on and approximate elsewhere.

## Limits

One model, one seed, four goals, one maze; the optimum's deterministic episodes yield 2 639 distinct histories, so the
maps are ridge fits of 937 inputs on 1 979 rows with λ at the top of the grid at layers 19–23, and the maps' held-out
error (0.4 of the state at 23) is the loss every composition and round-trip number carries; the probe pool's shortest
histories are out of distribution under every goal, equally, and section 3 shows the goal maps do not hold there, so
part C as registered measures the maps' domain, not the belief's coordinates, and C3 is post hoc and depends on the
ridge penalty (0.34–0.42 restored at 23 across the two penalty grids); the soft set probe's Jaccard is a weak shadow
of the belief (the exact set is not decodable); the rank sweep's directions come from the differences' SVD, with the
map's own SVD as the one alternative basis; k_c is read off a discrete list of ranks; the causal measures read the
decision token only; the controls of part B were run at two sites; the three-cycle's keep is scored on the (A, B)
goal-dependent decisions; the probe-overlap column is the share of the k_c causal directions inside the probe's
140-dimensional row space (random 140 / 896); the goal-swap study's in-sample map is flagged, not re-measured.
