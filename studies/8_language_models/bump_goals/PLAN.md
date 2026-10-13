# Bump maze with four goals: is a fine-tuned language model's policy additive in goal and history, and how does it swap goals?

Written before any model was trained on the four-goal task. **Disclosure:** the design probe (`design.md`: goal
placement, the exact optimum per goal, the exact additive ceiling) was run first; the single-goal experiment
(`../bump_maze`) is complete and its fine-tuning recipe is reused unchanged. Brief: given in conversation ("can we
add more goal states?", following "have we seen a similar additive policy on fine-tuning?").

## The task

The bump maze (`goalgeo.bumpmaze`) with **four goals** placed by farthest-point sampling, the four corners of the
7 × 7 interior; one goal per episode, named in the prompt and marked on the map; the start uniform over the other
free cells and the four headings. Everything else as in the single-goal experiment: one bit of observation per step,
the exact belief a set, the exact optimum per goal by value iteration (13 871–22 399 reachable sets per goal).

Design probe: the optimum returns 0.512–0.616 with success 0.987–1.000 under the four goals; QMDP 0 under every
goal; random 0.06–0.12. **The additive ceiling**: on 83 864 decisions of the optimum's episodes scored under every
goal (312 892 decision-goal pairs), the optimal action depends on the goal in 0.567 of decisions; the best rule
argmax h[goal-free belief set] + g[goal, step] is optimal in **0.881** of pairs, the best goal-blind rule h[set] in
0.820, the best history-blind rule g[goal, step] in 0.598. The maze10 maze's exact ceiling was 0.963: this task
forces more goal × history interaction than the one in which the additive code was found.

## Part A. Training and the additive code

**Data and models** as in the single-goal experiment: episodes from the exact optimum under the episode's goal with
ε = 0.2 random actions, labels the exact optimal action under that goal; Qwen2.5-0.5B full fine-tuning, 2 epochs, at
40 000 and 120 000 episodes (10 000 and 30 000 per goal); the from-scratch transformer and GRU with the goal as the
start token, at the same sizes. The prompt names the goal ("The goal is at row y, column x") and marks it on the map.
Test: 2 048 fixed starts per goal; 2 048 teacher-forced decisions per goal from the optimum's trajectories.

**The additive decomposition** of the fine-tuned model, on 4 096 decisions from the optimum's trajectories (all
goals): the centred action logits under each of the four goals with the history fixed, L(h, g). H(h) is the mean
over goals, G(g, step) the mean over histories at that step of L(h, g) − H(h), I(h, g) the rest.
* The variance shares of H, G and I in L; the share of goal-dependent decisions (the model's argmax differs between
  goals) and of history-dependent ones.
* **The additive policy** argmax H(h) + G(g, step): offline, the share of decision-goal pairs whose action is in the
  exact optimal set, against the model's own and the ceiling's 0.881; the goal-blind policy argmax H(h) against
  0.820. Online, run in the environment on the test starts under every goal: return, and the **recovery** of the
  goal-directed return (additive − goal-blind) / (model − goal-blind), as in the maze10 study (0.81 there).
* The same decomposition on the exact optimum's own action logits is not available (it has no logits); the ceiling
  is the comparison.

## Part B. Goal swaps in the residual stream

Quartets as in the factorial-patch experiment: 2 048 pairs of same-step decisions (goal-free sets different), each
with two goals (all six goal pairs, both directions), the four natural runs recorded. Patches of the decision
token's residual (the last prompt token, the one whose logits give the action) entering every layer l of 24:
* **same-history goal swap**: the residual from the run with the other goal; **same-goal history swap**; **both**;
  measured as in the factorial experiment (to source / to q_h / to q_g / kept on quartets where the targets differ,
  logit transfer; unique attribution on four-distinct quartets);
* at the best layer for the goal swap, **the state-level mean goal offset** per step (x(h, A) + mean over histories
  of x(h′, B) − x(h′, A)) and **a linear map** from the state under A (plus the step) to the state under B, fitted on
  other episodes, installed in place of the natural state under B;
* random vectors of the natural change's norm as the floor.

## Decision rule

One model per size; comparisons on the same test starts and decisions, bootstrap 95 % intervals over episodes or
decisions; a criterion holds if its interval excludes the threshold.

| | criterion |
|---|---|
| **L-OPT** | the fine-tuned 0.5B at 40 000 episodes: success ≥ 0.98 (mean over goals) and teacher-forced decision optimality ≥ 0.99 |
| **SCRATCH** | the from-scratch transformer at 40 000 episodes: success < 0.5 |
| **ADD-OFF** | the fine-tuned model's additive policy is offline optimal in ≥ 0.80 of decision-goal pairs (the ceiling 0.881, the goal-blind ceiling 0.820) |
| **ADD-ON** | the additive policy's online recovery of the goal-directed return is between 0.5 and 0.85 |
| **INT** | the goal × history interaction's variance share in L exceeds 0.15 (the maze10 models' natural logits on random quartets: 0.11, the same definition) |
| **SWAP** | at the best layer, the same-history goal swap of the decision token's residual switches ≥ 0.85 of goal-dependent decisions to the other goal's |
| **G-OFF** | the mean goal offset at that layer switches ≤ 0.75 × what the whole-state swap switches |
| **MAP** | the linear map at that layer switches ≥ 0.90 × what the whole-state swap switches |

## Expectations

| | expectation |
|---|---|
| E1 | L-OPT and SCRATCH hold (the single-goal data efficiency carries over; four goals need no more than four times the data) |
| E2 | ADD-OFF holds and the additive policy is within 0.03 of the ceiling: the model's additive part is as good as any additive rule, and the interaction is what the ceiling says is missing |
| E3 | ADD-ON holds, with recovery about 0.7, below maze10's 0.81 because the ceiling is lower |
| E4 | INT holds; the interaction is concentrated in decisions after localisation (set size 1), where the goal determines the route, and absent in the localisation phase, where the optimal action is nearly goal-free (the goal-blind ceiling 0.82) |
| E5 | SWAP holds at a middle layer (12–20, where the single-goal probes found the belief set); the same-goal history swap works at the same layers; the goal token's own layers (the goal phrase is in the prefix, tens of tokens before the decision token) are not needed by the decision token after the middle layers |
| E6 | G-OFF and MAP hold: as in maze10, the goal's effect on the state is a linear change of coordinates of which the mean offset is about two thirds |
| E7 | in unique attribution the decision token's residual at the middle layers goes to the source (both history and goal), unlike maze10's block-1 input, which held the belief only: the language model's decision token holds both from the middle layers on |

## Limits known in advance

One maze, four goals, one seed per model; the language model is trained by supervision from the exact optimum, so its
goal × history interaction is the optimum's, not a learner's invention; the prompt's goal statement is one wording;
part B patches the decision token only (not the prefix or transcript tokens), so it reads what the token holds, not
how it got there; the additive ceiling is fitted on the optimum's own decisions (605 goal-free sets) and is an upper
bound for that sample.
