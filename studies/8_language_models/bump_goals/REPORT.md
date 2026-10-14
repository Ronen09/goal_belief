# Bump goals: a fine-tuned language model on the bump maze with four goals — additivity in goal and history, and goal swaps in the residual stream

Run date: 2026-10-13/14. Task, models, measures and decision rule: `PLAN.md`, committed (`a9fa5cd`) **before any model
was trained on the four-goal task**. Numbers from `tables.md`; data in `design.json`, `runs/*/results.json`,
`runs/llm_*/additive.json`, `runs/llm_40000/factorial.json`, `posthoc_ceiling.json`. Reproduce: `reproduce.py
bump_goals` (about 5 h on one GPU; the fine-tuned models are not committed, 1.9 GB each, rebuilt by `train_goals.py`
in 50–150 min).

**Disclosures.** (1) The factorial run (`factorial_llm.py`) was interrupted after site 17 and resumed with
`--layers 18 … 24` into a second file, merged afterwards; the pairs, natural runs and seeds are the same. The
additive decomposition of the 120 000-episode model was interrupted after its online model run and rerun whole.
(2) The plan's unique attribution on four-distinct quartets (E7) is undefined in this task: with three actions no
quartet has four different decisions. The both-swap's "to q_h / to q_g" shares stand in. (3) Site 24, the residual
after the last layer, was recorded with the 24 layer entrances; it fixes the logits and is a check, not a candidate for
the best layer, which is chosen among sites 0–23. (4) After the registered ADD-ON result was seen, one post-hoc
analysis was added (`posthoc_ceiling.py`): the exact additive ceiling rule run online. (5) One training seed per model.

**The task** (`design.md`). The single-goal bump maze (9 × 9, 35 free cells, 140 states, one bit of observation per
step) with four goals in the corners of the interior, placed by farthest-point sampling; one goal per episode, named
in the prompt and marked on the map. The exact optimum per goal returns 0.512–0.616 (mean 0.570) with success
0.987–1.000; QMDP 0 under every goal; random 0.04–0.08. On the optimum's decisions scored under every goal the optimal
action depends on the goal in 0.567; the best additive rule argmax h[goal-free set] + g[goal, step] is optimal in
**0.881** of decision-goal pairs, the best goal-blind rule h[set] in 0.820, the best history-blind rule g[goal, step]
in 0.598 (maze10's exact ceiling: 0.963).

## 1. Training: the fine-tuned 0.5B is exactly optimal under four goals at 40 000 episodes; the models from scratch are not at 120 000

Supervision from the exact optimum under the episode's goal (ε = 0.2 random actions in the data), 2 epochs; closed
loop on 2 048 test starts per goal; teacher-forced optimality on 2 048 decisions per goal.

| model, episodes | success (per goal) | return | decision optimality | by set size 1 / 2–4 / 5–16 / > 16 | minutes |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B fine-tuned, 40 000** | **0.995** (1.00 0.98 1.00 1.00) | **0.569** | **1.000** | 1.00 / 1.00 / 1.00 / 1.00 | 50 |
| same, 120 000 | 0.995 (1.00 0.98 1.00 1.00) | 0.569 | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 | 150 |
| transformer from scratch, 40 000 | 0.404 (0.26 0.55 0.53 0.27) | 0.282 | 0.884 | 0.79 / 0.87 / 1.00 / 1.00 | 1 |
| same, 120 000 | 0.792 (0.64 0.76 0.97 0.79) | 0.476 | 0.978 | 0.95 / 0.99 / 1.00 / 1.00 | 1 |
| GRU, 40 000 | 0.471 (0.31 0.64 0.71 0.22) | 0.310 | 0.888 | 0.84 / 0.82 / 0.96 / 0.99 | 1 |
| same, 120 000 | 0.874 (0.74 0.93 0.96 0.86) | 0.503 | 0.977 | 0.97 / 0.96 / 1.00 / 1.00 | 1 |
| exact optimum | 0.995 | 0.570 | 1 | | |

* **L-OPT and SCRATCH hold.** At 10 000 episodes per goal the fine-tuned model makes the optimal decision in every one
  of 8 192 teacher-forced decisions and matches the optimum's return and success under every goal; the two fine-tuned
  models' closed-loop numbers are identical to the last digit because both are exactly optimal and break the
  optimum's ties the same way (their logits differ: the decompositions below are not identical). The transformer from
  scratch succeeds in 0.40 at 40 000 and 0.79 at 120 000, the GRU in 0.47 and 0.87: as in the single-goal experiment,
  the from-scratch models' errors are at known states (0.79–0.84 optimality at set size 1 against 1.00 on large sets),
  where the route to the named goal has to be learned per state and goal.

## 2. The additive decomposition: a fifth of the logits is goal × history interaction, and the additive part cannot run the task

4 096 decisions from the optimum's trajectories under every goal; the fine-tuned model's centred logits under each of
the four goals with the history fixed, L(h, g) = H(h) + G(g, step) + I(h, g); the additive and goal-blind policies
scored offline against the exact optimal sets under every goal and run online on the test starts under every goal.

| | 40 000 | 120 000 |
|---|---|---|
| variance shares H / G / I | 0.754 / 0.043 / **0.203** | 0.737 / 0.034 / 0.229 |
| the interaction's share by goal-free set size 1 / 2–4 / 5–16 / > 16 | 0.456 / 0.337 / 0.227 / 0.083 | 0.423 / 0.363 / 0.276 / 0.078 |
| goal-dependent decisions: model / optimum | 0.447 / 0.570 | 0.471 / 0.570 |
| offline optimal: model / **additive** / goal-blind (ceilings 0.881 / 0.820) | 0.897 / **0.806** / 0.788 | 0.920 / 0.809 / 0.788 |
| offline by set size 1 / 2–4 / 5–16 / > 16: model | 0.65 / 0.84 / 0.97 / 1.00 | 0.70 / 0.88 / 0.99 / 1.00 |
| same: additive | 0.53 / 0.70 / 0.86 / 0.98 | 0.53 / 0.71 / 0.86 / 0.99 |
| same: goal-blind | 0.52 / 0.70 / 0.85 / 0.94 | 0.52 / 0.70 / 0.84 / 0.94 |
| online return (success): model / additive / goal-blind | 0.570 (0.996) / **0.316** (0.502) / 0.274 (0.449) | 0.570 (0.996) / 0.299 (0.469) / 0.268 (0.433) |
| **recovery of the goal-directed return by the additive policy** (maze10: 0.81) | **0.142** | 0.101 |

* **INT holds**: the interaction is 0.20 of the logits' variance (maze10's natural logits on random quartets: 0.11),
  and it sits where E4 put it: 0.46 of the variance at set size 1, 0.08 on sets above 16. During localisation the
  goal barely enters the decision; once the state is known the goal decides the route, and it does so in interaction
  with the history, not as an added bias.
* **ADD-OFF holds, by 0.006.** The additive policy is optimal in 0.806 of decision-goal pairs, above 0.80 but 0.075
  below the ceiling of 0.881, so E2's second half fails. Its gain over the goal-blind policy is 0.018 where the
  ceiling's is 0.061, and by set size the additive policy is the goal-blind one: 0.53 against 0.52 at known states,
  0.70 against 0.70 on sets of 2–4. H(h) + G(g, step) carries no usable goal information at the decisions that need
  it; the model's goal-dependence, 0.447 of decisions against the optimum's 0.570, lives in I.
* **ADD-ON fails.** Online, the additive policy returns 0.316 with success 0.50, the goal-blind policy 0.274 with
  0.45, the model 0.570 with 0.996: the additive part recovers **0.14** of the goal-directed return (maze10: 0.81).
  The teacher-forced 0.806 does not survive closed loop: the optimum's trajectories, on which the offline score is
  taken, reach each cell at steps that depend on the goal, so a step-indexed goal bias looks informative there and
  is not once the policy's own early errors shift its timing.

### Post hoc: the exact additive ceiling rule online recovers 0.14 fitted on the optimum's decisions and about half refitted on its own (`posthoc_ceiling.md`)

The exact additive ceiling rule argmax h[goal-free set] + g[goal, step], fitted on the optimum's decisions, run in the
environment under every goal; then refitted on its own online decisions as well, so that sets the optimum never meets
are covered (an unseen set falls back to argmax g[goal, step]).

| fit | offline optimal on the optimum's decisions | by set size 1 / 2–4 / 5–16 / > 16 | online return | success | unseen sets | exact goal-blind rule online: return / success | recovery |
|---|---|---|---|---|---|---|---|
| the optimum's decisions (the ceiling) | 0.881 | 0.87 / 0.77 / 0.90 / 1.00 | 0.216 | 0.318 | 0.27 | 0.161 / 0.189 | 0.135 |
| + own decisions, round 1 | 0.865 | 0.82 / 0.77 / 0.90 / 1.00 | 0.235 | 0.353 | 0.06 | 0.163 / 0.191 | 0.177 |
| + own decisions, round 2 | 0.857 | 0.79 / 0.77 / 0.90 / 1.00 | 0.284 | 0.437 | 0.03 | 0.163 / 0.192 | 0.296 |
| + own decisions, round 3 | 0.849 | 0.77 / 0.77 / 0.90 / 1.00 | 0.279 | 0.421 | 0.01 | 0.163 / 0.192 | 0.285 |
| + own decisions, round 5 | 0.841 | 0.75 / 0.76 / 0.90 / 1.00 | 0.288 | 0.428 | 0.01 | 0.164 / 0.194 | 0.306 |
| + own decisions, round 8 | 0.834 | 0.74 / 0.75 / 0.89 / 1.00 | 0.329 | 0.513 | 0.01 | 0.164 / 0.194 | 0.408 |
| + own decisions, round 12 | 0.827 | 0.72 / 0.75 / 0.89 / 1.00 | 0.357 | 0.565 | 0.00 | 0.153 / 0.179 | 0.489 |
| + own decisions, round 16 | 0.813 | 0.68 / 0.73 / 0.89 / 1.00 | 0.343 | 0.533 | 0.00 | 0.161 / 0.190 | 0.445 |
| + own decisions, round 20 | 0.823 | 0.71 / 0.74 / 0.89 / 1.00 | 0.362 | 0.577 | 0.00 | 0.161 / 0.190 | 0.493 |
| rounds 12–20, mean | | | 0.356 | 0.565 | | | 0.477 |
| the exact history-blind rule g[goal, step] | 0.598 | | 0.048 | 0.059 | | | |
| the fine-tuned model's additive policy (section 2) | 0.806 | 0.53 / 0.70 / 0.86 / 0.98 | 0.316 | 0.502 | | 0.274 / 0.449 | 0.142 (its own goal-blind as the base) |

(All 21 rounds in `tables.md`; recovery here is (additive − goal-blind) / (optimum − goal-blind), both rules exact.)

* **The ceiling does not transfer to closed loop.** The rule that is optimal in 0.881 of the optimum's decision-goal
  pairs, 0.87 of them at known states, returns 0.216 online with success 0.318, below the fine-tuned
  model's additive policy (0.316); the exact goal-blind rule returns 0.161. Its recovery of the goal-directed
  return, 0.135, is the model's additive part's 0.142: fitted to the optimum's distribution, as the model's G(g, step)
  is, the additive form is not a policy, whoever holds it.
* **Refitted on its own episodes, the best additive rule recovers about half.** Unseen sets are gone by round 3; the
  return rises to 0.28 by round 2 and, slowly, to 0.33–0.37 over rounds 8–20 (success 0.51–0.60; mean of rounds
  12–20: 0.356 and 0.565, recovery 0.48, still drifting upward by about 0.01 per five rounds). As it does, its optimality
  on the optimum's decisions falls from 0.881 to 0.82, at known states from 0.87 to 0.71: the rule that runs is not the
  rule that scores. The best self-consistent additive rule in (goal-free set, goal, step) found here recovers about
  half of the goal-directed return; the model, exactly optimal, recovers all of it with its interaction.
* **Reading.** The additive ceiling of 0.881 is a property of the optimum's trajectory distribution: along the optimum
  the step is a proxy for position under each goal, and g[goal, step] can encode a route. On the policy's own episodes
  the proxy fails. So the plan's comparison was mis-posed on both sides: offline, the model's additive part is 0.075
  short of a ceiling that no policy attains; online, 0.14 is what the ceiling rule itself achieves when fitted the
  same way, and about 0.5 is the most any additive rule of this form reached. The maze10 result, 0.81 recovered by the
  additive part, was found in a task whose exact ceiling (0.963) held in closed loop; here four corner goals force the
  interaction, and the model, trained to the optimum, computes it.

## 3. Goal swaps in the residual stream: the decision token holds the belief from the middle layers and reads the goal in the last five

1 024 pairs of same-step decisions with different goal-free sets, each under every pair of the four goals (six goal
pairs, both directions); the decision token's residual entering each layer replaced by the residual from another
cell of the quartet, scored among the quartets whose natural decisions differ (random floor 0.03–0.04).

| residual entering layer | goal swap (same history): switched to the source's decision | logit transfer | history swap (same goal) | both swapped: to the source | to q_h (the source's history under the recipient's goal) | to q_g |
|---|---|---|---|---|---|---|
| 0–5 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6–12 | 0.03 | 0.02–0.03 | 0.00–0.07 | 0.01–0.05 | 0.01–0.07 | 0.03–0.07 |
| 13 | 0.04 | 0.04 | 0.16 | 0.13 | 0.16 | 0.11 |
| 14 | 0.04 | 0.05 | 0.29 | 0.24 | 0.29 | 0.17 |
| 15 | 0.39 | 0.30 | 0.58 | 0.57 | 0.51 | 0.25 |
| 16 | 0.47 | 0.34 | 0.65 | 0.63 | 0.55 | 0.26 |
| 17 | 0.51 | 0.38 | 0.74 | 0.73 | 0.61 | 0.27 |
| 18 | 0.52 | 0.39 | 0.74 | 0.73 | 0.61 | 0.28 |
| 19 | 0.56 | 0.41 | 0.78 | 0.77 | 0.62 | 0.28 |
| 20 | **0.84** | 0.74 | 0.90 | 0.90 | 0.64 | 0.34 |
| 21 | **0.93** | 0.81 | 0.96 | 0.96 | 0.65 | 0.35 |
| 22 | 0.95 | 0.87 | 0.97 | 0.97 | 0.65 | 0.35 |
| **23** | **0.98** | 0.98 | 0.99 | 0.99 | 0.65 | 0.36 |
| after layer 24 (check: fixes the logits) | 1.00 | 1.00 | 1.00 | 1.00 | 0.65 | 0.36 |

(The to-q_h and to-q_g shares count coincidences with the source's decision, so from layer 20 on they measure only
how often q_h's and q_g's decisions agree with the source's: 0.65 and 0.36. The informative comparison is below
layer 20.)

* **SWAP holds, at the top, not in the middle.** The same-history goal swap switches 0.85 of goal-dependent decisions
  only from the residual entering layer 21 (0.93; layer 20: 0.84; layer 23: 0.98). Up to layer 19 it switches at most
  0.56. E5 placed the swap at layers 12–20, where the single-goal probes found the belief set; the belief is there,
  the goal's effect on the decision is not yet.
* **The history arrives before the goal.** The same-goal history swap works from layer 13 (0.16) and reaches 0.58 at
  layer 15 and 0.78 at layer 19, where the goal swap is at 0.39 and 0.56; the both-swap follows the history swap at
  every layer (0.57 at 15 against 0.58), and below layer 20 it goes to the source's history under the recipient's goal
  more than to the recipient's history under the source's goal (0.51 against 0.25 at layer 15, 0.62 against 0.28 at
  layer 19): a decision token patched at layers 15–19 carries its own history's belief but has not yet absorbed the
  goal's consequence for the decision, which the layers above still compute from the prefix. From layer 20 the token
  carries both (both-swap 0.90–0.99), and the unique attribution E7 asked for cannot be run here (disclosure 2); the
  ordering stands in for it. In the maze10 models the token's state entering block 1 (of 4) carried the belief and the
  history tokens the goal; here the same order holds, stretched over layers 13–21 of 24, with the goal read in last.
* **The goal's effect on the state is not an offset; it is a linear change of coordinates, nearly.** At layer 23 the
  natural swap switches 0.985; a linear map from the state under A (plus the step) to the state under B, fitted on
  other episodes, 0.883 (logit transfer 0.90; it explains 0.67 of the state under B); the mean goal offset per step
  0.080, at the random floor (0.036). **G-OFF holds** (0.08 of the whole swap against ≤ 0.75) and **MAP misses by
  0.004** (0.883 / 0.985 = 0.896 against ≥ 0.90), without an interval that excludes the threshold. E6 is half wrong:
  in maze10 the mean offset was about two thirds of the swap and the linear map the rest; here the goal does nothing
  constant to the token's state — consistent with G's 0.04 share of the logits — and what it does is still mostly a
  fixed linear function of the state under the other goal. The same holds at the final residual (linear map 0.874,
  offset 0.081, random 0.066).

## Scorecard

| | criterion | numbers | held |
|---|---|---|---|
| L-OPT | fine-tuned at 40 000: success ≥ 0.98 and decision optimality ≥ 0.99 | 0.995 (lowest goal's lower CI 0.98), 1.000 | yes |
| SCRATCH | transformer from scratch at 40 000: success < 0.5 | 0.404 (upper CI 0.42) | yes |
| ADD-OFF | additive policy offline optimal ≥ 0.80 | 0.806 (ceiling 0.881) | yes, by 0.006 |
| ADD-ON | online recovery between 0.5 and 0.85 | 0.142 | **no** |
| INT | interaction share > 0.15 | 0.203 (120 000: 0.229) | yes |
| SWAP | best-layer goal swap switches ≥ 0.85 | layer 23: 0.985 (0.85 reached from layer 21) | yes |
| G-OFF | mean offset ≤ 0.75 × the whole swap | 0.080 / 0.985 = 0.08 | yes |
| MAP | linear map ≥ 0.90 × the whole swap | 0.883 / 0.985 = 0.896 | **no**, by 0.004 |

| | expectation | held |
|---|---|---|
| E1 | L-OPT and SCRATCH | yes |
| E2 | ADD-OFF, and the additive policy within 0.03 of the ceiling | in part: 0.806 holds, but 0.075 below the ceiling; by set size the additive policy equals the goal-blind one |
| E3 | ADD-ON, recovery about 0.7 | no: 0.14; post hoc, the exact ceiling rule recovers 0.13 fitted on the optimum's decisions and about 0.5 refitted on its own |
| E4 | INT; interaction concentrated after localisation | yes: 0.46 of the variance at set size 1, 0.08 above 16 |
| E5 | SWAP at a middle layer (12–20); history swap at the same layers | no: the goal swap reaches 0.85 at layer 21; the history swap leads it by 4–6 layers |
| E6 | G-OFF and MAP: the mean offset about two thirds, a linear map the rest | in part: the offset is at the random floor (0.08), the linear map 0.90 of the swap |
| E7 | unique attribution: the token holds history and goal from the middle layers | not evaluable (three actions); the ordering shows the history first (layers 13–19) and both from layer 20 |

Six of eight registered criteria held; of the expectations two held, two in part, two failed, one is not evaluable.

## Reading

Fine-tuned on four goals, a 0.5B language model is exactly optimal at 10 000 episodes per goal, where the models
from scratch need more than 30 000. Its policy is not additive in goal and history: the goal × history interaction is
a fifth of its logits, concentrated at the decisions where the state is known and the goal determines the route, and
the additive part, run as a policy, recovers 0.14 of the goal-directed return, against 0.81 in the maze10 study. The
post hoc result says why the registered comparison was mis-posed: the exact ceiling rule, optimal in 0.88 of the
optimum's decision-goal pairs, recovers 0.14 of the goal-directed return online, as the model's additive part does,
because its goal term is indexed by the step, which stands for position only along the optimum's own trajectories;
refitted on its own episodes until it meets no new set, the best additive rule of this form recovers about half. The
additive ceiling measured on a dataset is not the additive policy's reach in the task; the maze10 result lived in a
task whose ceiling (0.963) was the task's. Here the model computes the interaction the task forces, and it computes
it late: the decision token holds the belief from layer 13 and reads the goal's consequence for the decision in
layers 20–21 of 24, after which the goal's whole effect on the token's state can be installed by a fixed linear map
(0.90 of the natural swap) and none of it by a constant offset.

## Limits

One maze, four goals, one seed per model; the language model is trained by supervision from the exact optimum, so the
interaction it computes is the optimum's; the from-scratch models are not size-matched; the additive decomposition
uses the step as the only history variable in G, as the plan registered, so a history-dependent goal term with more
structure (the goal-free set itself, as in the ceiling) is tested only post hoc and only as an exact rule; part B
patches the decision token only and reads what it holds, not how the prefix tokens produced it; the plan's unique
attribution is undefined with three actions; the linear map at layer 23 is fitted on 6 144 decisions in 896
dimensions by ridge and its 0.67 R² for the state leaves room for a better map; MAP's miss (0.004) and ADD-OFF's
margin (0.006) are within sampling noise of their thresholds, and neither has an interval computed.
