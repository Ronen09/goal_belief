# Bump coordinates: the goal maps compose and invert and the decision reads six of their directions, but they are maps of the optimum's trajectories only, and a belief probe does not survive the change of goal

Run date: 2026-10-26. Plan, criteria and expectations: `PLAN.md`, committed (`6b3a209`) **before any measurement was run**.
Numbers from `tables.md`; data in `compose.json`, `rank.json`, `preserve.json`, `posthoc_pool.json`, `collect.json` (the
recorded states, the maps and the singular directions in `cache/`, not committed). Reproduce: `reproduce.py bump_coords`
(`collect.py`, `compose.py`, `rank.py`, `preserve.py`, `posthoc_pool.py`, `tables.py`; about 2.5 h on one GPU alone,
the run here took 3.5 h on a GPU shared with the reward study's trainings).

**Disclosures.** (1) Before any run, the plan's description of the causal measure was corrected at review: a
parenthesis said the prediction is installed in the prompt under the target goal; every causal number follows the
goal-swap study's convention (the prompt under the source goal, the candidate state for the target installed, the
share of goal-dependent decisions moving to the target's decision), which the plan's text now states. (2) The rank
sweep's controls (bottom-k, three random-k draws, the map's own singular directions) were run only at the two sites
with the largest natural goal swap (23 and 21), and all patched passes of part B run on each pair's goal-dependent
decisions only (the switch is defined on those alone, so no number changes); the plan said "at each site". The GPU
was shared with five PPO trainings, and the measured throughput put the plan's full sweep over seven hours. (3) The
policy study's probe fitter could not be imported (its module imports a `common` of its own, colliding with this
study's); `preserve.py` carries the same recipe (standardise on the fit rows, LBFGS with strong-Wolfe line search,
800 iterations, L2 10⁻⁴). (4) The maps' ridge penalty was tuned on a held-out quarter of the fit decisions, as the
plan says; the chosen λ is 1 at layer 15 and 10 at layers 19–23 (the goal-swap study used 10⁻²), which lifts the
held-out state R² of the single map at layer 23 from 0.67 to 0.89. (5) Every script was smoke-tested in quick mode on
64 fit decisions before the full run (outputs under `_smoke/`, unused). The quick run exposed one defect fixed before
the full run: with fewer fit decisions than dimensions the SVD basis was incomplete and the rank-896 projector was not
the identity; the basis is now completed (irrelevant at the full sizes, 6 144 > 896). (6) **After the registered part C
was seen**, one post hoc analysis was added (`posthoc_pool.py`, section 3's C3): the maps refitted on the probe pool,
because the diagnostic in section 3 showed the registered maps are not maps on that pool. (7) One model, one seed; the
probe pool dropped 87 of 8 192 histories as ended under some goal (8 105 kept, 4 053 fit / 4 052 test by set).

**The question.** The bump goals study found the goal's effect on the fine-tuned four-goal model's decision token to be
nearly a fixed linear map at the top (0.90 of the natural swap at layer 23) with no constant part, read in over layers
20–21 after the belief arrives at 13–19; the maze10 study found such maps form one consistent, invertible system over a
shared code of the posterior. Here: (A) do the language model's pairwise goal maps compose and invert; (B) how many
directions carry the goal's effect, in reconstruction and in what the decision uses; (C) does the goal-free belief's
linearly decodable shadow (the soft set probe; the bump probe as control) sit in coordinates the goal map carries?

Sites: the decision token's residual entering layers 15, 19, 21, 23; 1 024 test pairs, maps on 6 144 fit decisions
(the goal-swap study's pools); the best site is 23 (natural goal swap 0.985, as in that study; 21: 0.934, 19: 0.573,
15: 0.399, all within 0.01 of its numbers, which checks the harness).

## 1. Composition: the maps compose and invert at every layer, causally within 0.014 of the direct map

| site | R² composed vs direct (24 triples) | R² direct / composed vs x_C | switch: direct | composed | **composed / direct** | natural x_C | random | round trip: R² vs x_A | **keeps A's decision** | x_B keeps | three-cycle keeps |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 0.997 | 0.979 / 0.979 | 0.396 | 0.396 | **0.999** (0.992–1.006) | 0.399 | 0.052 | 0.995 | **0.992** | 0.577 | 0.991 |
| 19 | 0.982 | 0.897 / 0.896 | 0.551 | 0.543 | **0.986** (0.979–0.993) | 0.573 | 0.041 | 0.976 | **0.988** | 0.405 | 0.985 |
| 21 | 0.964 | 0.796 / 0.799 | 0.857 | 0.846 | **0.987** (0.982–0.993) | 0.934 | 0.036 | 0.958 | **0.962** | 0.055 | 0.944 |
| **23** | **0.949** (0.947–0.951) | 0.723 / 0.726 | 0.885 | 0.873 | **0.986** (0.981–0.992) | 0.985 | 0.038 | 0.943 | **0.945** (0.939–0.951) | 0.010 | 0.929 |

* **COMP-S and ROUND hold at every site; COMP-R by 0.001 not at 23.** Going A → B → C switches C's decision in 0.873
  of goal-dependent decisions at layer 23 against the direct map's 0.885 (ratio 0.986, every triple 0.828–0.891
  against 0.853–0.907), and the composed vector agrees with the direct one at R² 0.949 (0.947–0.951; triples
  0.939–0.956), against the registered 0.95. The agreement is 0.964 at 21, 0.982 at 19 and 0.997 at 15, where the maps
  are better (the direct map predicts x_C at 0.98 at layer 15 and 0.72 at 23): what the composition loses is the maps'
  own error twice over, not a B-dependence; the composed and direct predictions' norms agree to 0.02. A round trip
  A → B → A returns x_A at R² 0.943 and keeps A's natural decision in 0.945 of the decisions that B would switch (x_A
  re-installed 1.000, x_B 0.010); a three-cycle 0.929. The maze10 numbers were 0.99 / 0.85 against 0.86 / 0.98 on
  maps with R² 0.95; the language model's maps are worse by state R² and as good by what the decision does with them.

## 2. Rank: the decision reads two to eight directions of a change that spans a hundred and sixty

| site | k50 / k90 / k99 of the centred difference (median pair; pooled over pairs) | goal × step offset: share of the difference | switch: full map / natural | **k_c median (range)** | rank-32 / full | pooled subspace at k_c / full | overlap of pairs' top-k_c subspaces (random) |
|---|---|---|---|---|---|---|---|
| 15 | 5 / 63 / 301 (10 / 125 / 482) | 0.613 | 0.393 / 0.399 | 128 (16–512) | 0.830 | 0.872 | 0.420 (0.097) |
| 19 | 16 / 159 / 474 (31 / 282 / 685) | 0.305 | 0.550 / 0.574 | 96 (16–512) | 0.847 | 0.895 | 0.399 (0.080) |
| 21 | 13 / 164 / 486 (24 / 282 / 686) | 0.214 | 0.857 / 0.935 | **8 (4–16)** | **0.965** (0.957–0.973) | **0.928** (0.918–0.939) | 0.557 (0.008) |
| **23** | 13 / 168 / 493 (21 / 271 / 677) | 0.181 | 0.886 / 0.985 | **6 (2–8)** | **0.980** (0.972–0.987) | **0.926** (0.915–0.936) | 0.690 (0.005) |

The rank-k map at site 23, switch over the 12 ordered pairs (full map 0.886):

| directions | k = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 896 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **top-k of the difference** | 0.31 | 0.66 | 0.81 | 0.84 | 0.86 | 0.87 | 0.87 | 0.88 | 0.88 | 0.88 | 0.89 |
| pooled over pairs | 0.31 | 0.68 | 0.82 | 0.84 | 0.86 | 0.87 | 0.87 | 0.88 | 0.88 | 0.88 | 0.89 |
| the map's own singular directions | 0.32 | 0.62 | 0.79 | 0.82 | 0.83 | 0.84 | 0.85 | 0.86 | 0.86 | 0.88 | 0.89 |
| bottom-k | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.89 |
| random-k (three draws) | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01–0.02 | 0.02–0.03 | 0.06–0.07 | 0.10–0.15 | 0.54–0.59 | 0.89 |

* **RANK-C and RANK-SH hold at 21 and 23.** At the top the goal's effect on the decision lives in a handful of
  directions: the first singular direction of the per-history difference switches 0.31 of what the full map switches,
  the first two 0.66, four 0.81, eight 0.84, and the effective causal rank is 2–8 per pair (median 6; 4–16 at layer
  21). The bottom directions do nothing until the last few, and k random directions do nothing until k is half the
  space. Reconstructing the difference is another matter: its 90 % needs 160–170 directions (270 pooled), 99 % about
  490, so the decision uses one direction in thirty of what the goal changes in the state. The map's own singular
  directions are a slightly worse basis (k_c 4–32), the pooled subspace of all twelve pairs as good as each pair's own
  (0.926 of the full map at each pair's k_c): one subspace serves every goal pair, and the pairs' top-k_c subspaces
  overlap at 0.69 against a random 0.005 (0.56 at layer 21).
* **Lower down the goal's effect is a constant and diffuse.** At layer 15, 0.61 of the difference's norm is a goal ×
  step offset (0.18 at 23), the remainder's causal rank is 128 and the rank-32 map gives 0.83 of a full map that itself
  switches 0.39; at 19, 96 and 0.85 of 0.55. The low-rank readout appears where the goal is read into the decision,
  layers 20–21, as the goal-swap study placed it.

## 3. Shared belief preservation: the set probe drops under the other goal, the registered maps make it worse, and the maps are not maps off the optimum's trajectories

**Probes by goal** (held-out sets; the four goals within 0.02 of each other): the soft set probe's Jaccard 0.32–0.34 at
every site (exact 0.01–0.015; floor from shuffled labels 0.000–0.001); the bump probe's R² 0.80 at layer 15 falling to
0.74–0.76 at 23 (floor −0.34 to −0.41). The single-goal model's were 0.41 and 0.84: **E5 holds**.

| site | set: drop j_A − j_B | restoration by the inverse map | over the offset control | by the rank-k_c map | bump: R²_A − R²_B unmapped | causal check: the pull-back installed under B switches to A's (natural x_A) | probe rows in the causal subspace (random) | map R² on the probe pool (on its own pool; identity) |
|---|---|---|---|---|---|---|---|---|
| 15 | **0.073** (0.071–0.075) | **−0.40** | −0.62 | 0.09 | **0.82** | 0.20 (0.21) | 0.148 (0.156) | 0.78 (0.99; 0.46) |
| 19 | **0.127** | **−0.51** | −0.36 | −0.06 | **0.31** | 0.25 (0.27) | 0.143 (0.156) | 0.26 (0.96; 0.31) |
| 21 | **0.147** | **−0.51** | −0.37 | 0.00 | **0.27** | 0.46 (0.74) | 0.097 (0.156) | −0.50 (0.92; 0.22) |
| **23** | **0.156** (0.154–0.158) | **−0.47** (−0.49–−0.45) | −0.35 | 0.03 | **0.21** (0.20–0.21) | 0.47 (0.94) | 0.074 (0.155) | −1.18 (0.89; 0.19) |

* **PRES-D holds everywhere, PRES-R and BUMP-SH fail everywhere.** A set probe fitted under A loses half its Jaccard
  under B (0.33 → 0.12–0.21 at layer 23), already at layer 15 where the goal barely moves the decision; the bump
  probe loses 0.21 of R² at 23 and 0.82 at 15. The pull-back through M(B→A) does not restore either: it lowers the set
  probe further (0.08–0.11; restoration −0.47) and takes the bump probe's R² to about zero; removing the per-step mean
  offset alone is better than the map and still below the unmapped state. The rank-k_c pull-back, which moves the state
  only in the six directions the decision reads, leaves the probes exactly where the unmapped state left them
  (restoration 0.03): **the probes' rows lie outside the causal goal subspace**, at or below the random overlap
  (0.07 against 0.16 at 23).
* **The maps are not maps on the probe pool.** The diagnostic in the last column decides how to read this. The
  registered maps, fitted on decisions along the optimum's trajectories, reach held-out R² 0.89–0.99 there; on the
  probe pool (every reachable set by its shortest history, out of distribution under every goal) they predict the state
  under A from the state under B at R² 0.78 at layer 15 and below zero at 21 and 23, worse than leaving the state as it
  is (0.19–0.46). The causal check agrees: the pulled-back state installed under B switches 0.47 of goal-dependent
  decisions to A's where the natural x_A switches 0.94. On the shortest histories the goal's transformation of the
  state is not the one fitted on the optimum's histories; the restoration test as registered ran through the wrong
  map, and its failure cannot say whether the belief's coordinates are carried.

### Post hoc C3. The maps refitted on the probe pool (`posthoc_pool.py`)

The maps M(B→A) refitted on the probe pool's fit half (same ridge recipe, λ tuned on a quarter; λ = 1–10) and the
A-probes' cells re-scored on its test half (4 052 decisions):

| site | map R² on the probe pool: refitted (registered map; identity) | set: drop | **restoration by the refitted inverse map** | over the offset control | bump: R² lost unmapped | **after the pull-back** | causal check: the pull-back installed under B switches to A's (natural x_A) |
|---|---|---|---|---|---|---|---|
| 15 | 0.958 (0.78; 0.45) | 0.073 | **0.53** (0.51–0.54) | 0.25 | 0.82 | **−0.02** | 0.20 (0.21) |
| 19 | 0.812 (0.26; 0.31) | 0.127 | **0.53** (0.52–0.54) | 0.59 | 0.31 | **−0.02** | 0.23 (0.27) |
| 21 | 0.705 (−0.50; 0.23) | 0.147 | **0.49** (0.48–0.50) | 0.55 | 0.27 | **−0.01** | 0.52 (0.74) |
| **23** | 0.633 (−1.18; 0.19) | 0.156 | **0.42** (0.41–0.43) | 0.46 | 0.21 | **−0.01** | 0.59 (0.94) |

* **The bump feature is carried whole, the soft set by half.** With the inverse map fitted where it is tested, the bump
  probe fitted under A reads the pulled-back states under B at its own R² (0.75–0.77 at layer 23 in every pair, within
  0.02 of its value under A, at every site), and the set probe recovers 0.42 of its drop at 23 (0.33 → 0.12–0.21 →
  0.22–0.24; pairs 0.21–0.52) and 0.53 at 15 and 19. The per-step offset alone recovers nothing (the offset-only state
  is at or below the unmapped one; the map is 0.46 of the drop above it). Half of the set probe's goal-sensitivity is a
  linear change of coordinates between the goals' states; the other half no linear map between them reproduces.
* **On this pool the maps are poorer and causally weaker.** R² 0.63 at layer 23 against 0.89 on the swap pool, and the
  pull-back installed under B switches 0.59 of goal-dependent decisions to A's where x_A switches 0.94 (the registered
  map on its own pool: 0.885 of 0.985). The transformation the goal applies to the state is linear and invertible on
  the optimum's histories, and a different, less linear one on histories the optimum never produces; each pool's map
  fails on the other. PRES-R's registered threshold (0.80 of the drop) is not reached by any map on this pool.

## Scorecard (site 23; 21 in brackets)

| | criterion | value (95 %) | holds |
|---|---|---|---|
| COMP-R | composed vs direct R² ≥ 0.95 | 0.949 (0.947–0.951) [0.964] | **no**, by 0.001 [yes] |
| COMP-S | composed switch ≥ 0.90 × direct | 0.986 (0.981–0.992) [0.987] | yes |
| ROUND | round trip keeps ≥ 0.90 | 0.945 (0.939–0.951) [0.962] | yes |
| RANK-C | rank-32 map ≥ 0.90 × full | 0.980 (0.972–0.987) [0.965] | yes |
| RANK-SH | pooled subspace at k_c ≥ 0.90 × full | 0.926 (0.915–0.936) [0.928] | yes |
| PRES-D | the A set probe drops ≥ 0.05 under B | 0.156 (0.154–0.158) [0.147] | yes |
| PRES-R | the inverse map restores ≥ 0.80 of the drop and ≥ 0.10 over the offset control | −0.470 (−0.485–−0.453); over the offset −0.349 [−0.514] | **no** |
| BUMP-SH | the bump probe loses ≤ 0.05 R² under B unmapped | 0.208 (0.202–0.214) [0.268] | **no** |

At layers 15 and 19 RANK-C and RANK-SH fail (0.83–0.85, 0.87–0.90) and the rest is as at 23 (COMP-R holds there).

| | expectation | held |
|---|---|---|
| E1 | COMP-R and COMP-S at layers 21 and 23, as in maze10; at 15 and 19 the maps barely move the decision | in part: COMP-S at every site, COMP-R at 21 and by 0.001 not at 23; the maps at 15 and 19 switch 0.40 and 0.55, the natural swap's own values there |
| E2 | ROUND: the maps invert, with the ridge's shrinkage as the only loss | yes (0.945–0.992; the round trip's norm within 0.02) |
| E3 | RANK-C with the median k_c between 8 and 32 while k90 exceeds 100; bottom-k and random-k at the floor up to k_c | yes, and lower: k_c 6 at 23, 8 at 21, k90 160–170; the controls at 0.00–0.01 to k = 64 |
| E4 | RANK-SH: the twelve pairs' causal subspaces coincide (overlap ≥ 0.7) | yes for RANK-SH (0.926); the overlap 0.69 at 23 and 0.56 at 21 against 0.005 random, just under the 0.7 |
| E5 | the four-goal model's probes like the single-goal model's: exact ≤ 0.05, Jaccard 0.3–0.5, bump R² ≥ 0.7, floors well below | yes (0.01; 0.32–0.34; 0.74–0.80; floors 0.00 and < 0) |
| E6 | PRES-D at 21 and 23 and not at 15; PRES-R at 23; the rank-k_c pull-back restores less than the full one | no: the drop is at every site including 15; the full map restores nothing and worsens; the rank-k_c map leaves the probe where it was, and the probe's rows lie outside the causal subspace. The registered maps are not maps on the probe pool (section 3), so the test is reread post hoc in C3 |
| E7 | BUMP-SH at every site | no: the bump probe loses 0.21–0.82 of R² under the other goal |

5 of 8 registered criteria held at the best site (6 of 8 at layer 21); the three misses are COMP-R by 0.001 and the two
of part C, whose registered form was confounded.

## Reading

Three answers. **(A)** The goal's effect on the decision token at the top of the fine-tuned model is one linear system
of transformations: the twelve pairwise maps compose (A → B → C against A → C at R² 0.95–1.00, switching the decision
0.99 as often) and invert (round trips keep 0.95–0.99 of decisions), at every layer from 15 to 23, as the maze10
maps did; what the composition loses is the maps' own error twice over. **(B)** What the decision reads of that
transformation is two to eight directions of 896: the first two singular directions of the goal difference carry two
thirds of the full map's effect, eight carry 0.84 of 0.89, bottom and random directions nothing, and the twelve pairs
share the subspace (one pooled subspace gives 0.93 of the full effect; the pairs' subspaces overlap at 0.69 against
0.005). Reconstructing the goal's change in the state takes 160–170 directions: the goal rewrites much of the state and
the decision uses a sliver, which appears at layers 20–21, where the goal-swap study found the goal read into the
decision; below, the difference is mostly a goal × step constant and what remains is diffuse (k_c 96–128). **(C)** The
maps are maps of a distribution. Fitted on the optimum's trajectories they predict nothing on the same model's states
for shortest histories (R² below zero at the top), and fitted there they are weaker (0.63) and causally half as
effective. On that pool a linear probe for the goal-free belief fitted under one goal loses half its Jaccard under
another at every layer, including layer 15 where the goal does not yet move the decision: the goal rewrites coordinates
the probe reads. The probe's rows lie outside the six causal directions, so moving the state only along them leaves the
probe untouched, and the inverse map fitted in place gives back half of what the set probe lost and all of what the
bump probe lost. For the study's question, how the model holds its belief while what it wants changes: the belief's
decodable shadow sits in coordinates the goal transforms, partly linearly, while the goal's effect on the decision is a
six-direction readout beside those coordinates, not a transformation of them. "A shared belief code in goal-specific
coordinates" holds for the bump feature, half holds for the soft set, and holds only within the trajectory distribution
a map was fitted on; the maze10 picture of one global change of coordinates is, in the language model, local to the
optimum's histories.

## Limits

One model, one seed, four goals, one maze; the maps are linear in 896 dimensions on 6 144 decisions, with held-out
state R² 0.89 at the top, so composition is tested on maps that lose a tenth of the state each hop; the probe pool's
shortest histories are out of distribution under every goal, equally, and section 3 shows the goal maps do not hold
there, so part C as registered measures the maps' domain, not the belief's coordinates, and C3 is post hoc; the soft
set probe's Jaccard is a weak shadow of the belief (the exact set is not decodable); the rank sweep's directions come
from the differences' SVD, with the map's own SVD as the one alternative basis; the causal measures read the decision
token only; the controls of part B were run at two sites; k_c is read off a discrete list of ranks.
