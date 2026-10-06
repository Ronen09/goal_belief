# What carries the history: candidate mediators at the goal token (history mediator)

Run date: 2026-10-06. Brief: `BRIEF.md` (given in conversation). Candidates, arms and decision rule: `PLAN.md`, committed
(`d2cf1a8`) **before `run.py` was run on any trained model** (smoke-tested on the untrained checkpoint of seed 0 only).
Numbers from `tables.md`; data in `results.json` (registered) and `explore.json` (post hoc). Reproduce:
`reproduce.py history_mediator` (6 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen. The additive-ablation experiment found
that the history reaches the decision through the goal-averaged history part d(h) of the goal token's four attention
outputs. Here d(h) = x_Z(h) + r(h), where x_Z = E[d | Z, L] is estimated on fit-side histories and every test is on
held-out cells. All edits are at those four attention outputs.

| Z | what | x_Z estimator |
|---|---|---|
| top, rank2 | H's top action; its top two, ordered | table |
| H2, H | the goal-averaged logit profile (additive-code experiment), projected on 2 PCs, or whole | MLP |
| mix | the H-mixture experiment's code (P(optimal \| random goal), mean reachability) | table |
| b | the exact posterior | table |

* **mix and b give identical results.** The eight numbers of the H-mixture code identify the posterior, so their tables
  coincide. This was not anticipated in the plan.
* 14 670 of 71 996 cells were dropped, against an expected ~2 %. In those cells the (posterior, prefix length) value has
  fewer than five fit-side histories. The plan's coverage estimate ignored prefix length. The same cells are dropped
  for every candidate.

## 1. H and the posterior carry the transfer; compressions of H do not

Informative pairs (recipient's and donor's natural actions differ; 24 k per model):

| Z | R² of x_Z | swap_Z: takes donor's action | ratio to all history | swap_r (random donor): changes action | swap_r, same-Z donor: changes action |
|---|---|---|---|---|---|
| top action | 0.40 | 0.659 | 0.72 | 0.356 | 0.169 |
| top two | 0.49 | 0.750 | 0.81 | 0.263 | 0.117 |
| H, 2 PCs | 0.55 | 0.797 | 0.87 | 0.221 | — |
| **H** | 0.74 | **0.878** | **0.96** | **0.092** | — |
| **posterior (= mix)** | 0.82 | **0.866** | **0.94** | **0.115** | **0.037** |
| all history | 1 | 0.917 | 1 | — | — |

* **The swap transfers almost entirely through H or the posterior.** Swapping only the part of the state they predict
  takes the donor's action 0.94–0.96 as often as swapping the whole history part. The rotated control takes it in
  0.065–0.074.
* **Compressions of H lose transfer steadily.** Top action 0.72, top two 0.81, a 2-D projection 0.87. The decision
  needs H's margins, not just its ranking. This fits the additive-code experiment, where decisions turn on near-ties.
* H (3 numbers once centred) transfers as well as the posterior (13) while predicting less of the state (R² 0.74
  against 0.82). Being the network's own readout, H keeps exactly the part that reaches the logits.

## 2. The residual is unnecessary, but not inert

| | H | posterior |
|---|---|---|
| remove r: drop in optimal (all cells) | 0.010 | **−0.006** |
| swap r between histories with the same posterior: changes | — | **0.037** |
| swap r from a random donor: changes | 0.092 | 0.115 |
| *post hoc*: the same r-swap vectors rotated (same size): changes | 0.052 | 0.037 |
| remove x_Z: optimal (ceiling 0.625); history dependence (natural 0.288) | 0.568; 0.140 | 0.582; 0.138 |

* **Removing r costs nothing.** With only the posterior's part of the history left, the policy is as good as natural.
  Between two histories with the same posterior, swapping r changes 3.7 % of decisions.
* **But r moved to a history with a different posterior changes 9–12 %**, about twice the R0 threshold. Post hoc, the
  same vectors rotated to random directions change 3.7–5.2 %. So about a third of the effect is generic perturbation
  at r's size. The rest, 6–8 points, is history information that is decision-relevant once it sits on a different
  posterior. Toward the donor's action: 0.070 against 0.021 rotated.
* **Removing x_Z alone does not collapse the policy.** Accuracy is near the ceiling (0.57–0.58), but history dependence
  is about half of natural, not ≤ 0.25 ×. What r carries changes decisions when the posterior's part is absent. The
  whole history part removed at this site gives history dependence 0.077 (additive-ablation experiment: 0.068).

## 3. Decision rule

| Z | S90 (ratio ≥ 0.9) | R0 (swap_r changes ≤ 0.05 and ≤ 0.1 × swap_all's) | mediates |
|---|---|---|---|
| top action | 0.72, no | 0.356, no | no |
| top two | 0.81, no | 0.263, no | no |
| H, 2 PCs | 0.87, no | 0.221, no | no |
| H | **0.96, yes** | 0.092 (0.10 ×), no | no |
| posterior / mix | **0.94, yes** | 0.115 (0.12 ×), no | no |

**Registered result: no candidate mediates.** H and the posterior carry ≥ 0.94 of the transfer, but swapping the
residual still changes 9–12 % of informative decisions.

Secondary criteria: removing r is harmless for H and the posterior (yes); the posterior's matched-donor r swap is
≤ 0.05 (yes); removing x_Z does not reach the 0.25 × history-dependence criterion for any candidate (no).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | swap_all reproduces the additive-ablation experiment's attention swap within 0.01 (0.927) | in part | 0.917 (on 80 % of the cells) |
| E2 | b fails S90 (ratio 0.7–0.9) | **no** | 0.94: passes |
| E3 | b fails R0 | yes | 0.115 |
| E4 | H reaches S90; H2 does not | yes | 0.96; 0.87 |
| E5 | no candidate below H passes R0 | yes | 0.22–0.36 |
| E6 | top action's swap effect ≥ 0.5 | yes | 0.66 |

5 of 6, one in part. E2 failed: the H-mixture experiment's 0.68 for the posterior table came from edits of the whole
residual stream entering block 2. At the attention outputs, where the history enters, the posterior carries 0.94.

## Conclusions

1. **The history's effect on the decision passes, almost entirely, through a posterior-level code.** The posterior, or
   the network's own 3-number H, carries 0.94–0.96 of what swapping the whole history part does. Without r the policy
   is unchanged, and r makes little difference between histories with the same posterior (0.037). That is close to
   sufficiency.
2. **It is not a strict mediator by the two-criterion definition.** Moved to another posterior, r still changes 6–8
   points of informative decisions beyond a same-size random perturbation. The history leaves a small decision-relevant
   trace beyond the posterior, which matches the pair-types experiment (identical posteriors, actions differing by
   0.04–0.09).
3. **H cannot be compressed further.** Its ranking alone carries 0.72–0.81 of the transfer; the margins matter.
4. Within the candidates tested, the smallest Z that reaches the causal ceiling (S90) is H. Its residual is not causally
   zero, so by the registered rule there is no minimal mediator.

Next, if wanted: ask what the r trace is. Does it predict the history's errors (the pair-types experiment's
evidence-dependent errors)? Is it the per-history deviation the additive code mispredicts at near-ties? Or does a richer
Z, posterior plus prefix symbols, bring R0 within 0.05?

Limits: one site with natural vectors; 20 % of cells dropped for table coverage; the MLP for H is not deterministic
across runs (H's swap_r change 0.092 here, 0.100 in the explore run); the rotated r control is post hoc; first decision
at the goal token; reward models only.
