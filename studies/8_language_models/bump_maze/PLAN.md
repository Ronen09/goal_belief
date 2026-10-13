# Bump maze: a small language model on the most partially observed navigation task with an exact belief

Written before any language model was run on the task. **Disclosure:** the task module (`goalgeo/bumpmaze.py`), its
tests and the design probe (the numbers in "The task") were built and run first; no language model, zero-shot or
trained, had been run when this was written. Brief: given in conversation ("test a small LLM on this task; use
MiniGrid; as partially observable as possible while still allowing us to track the exact belief; test both training
on the task and zero-shot accuracy").

## The task

A MiniGrid maze (`goalgeo.bumpmaze`, a 9 × 9 grid, seed 0, 4 loops): 35 free cells, a known map, a known goal cell,
the agent's start cell and heading unknown (uniform over the 34 non-goal cells × 4 headings = 136 start states; 140
states in all). Actions: turn left, turn right, forward (MiniGrid's own). **The only observation is one bit after a
forward move: moved or bumped a wall**; a turn returns nothing. Entering the goal ends the episode with reward
γ^(t−1) at move t (γ = 0.97); after 40 moves it ends without reward. `minigrid_env` builds the same world through
minigrid's `MiniGridEnv`, and `test_bumpmaze` checks minigrid's dynamics against the module's transition table.

Why this is the most partially observed version that still allows an exact belief: transitions and observations are
deterministic, so the posterior is uniform over the set of states consistent with the history, updated by one set
operation per step, and because the reachable beliefs are finite (17 285 sets) the **exact optimal policy** is
computable by value iteration over them. Any richer observation (a symbol per cell, a view) would carry more
information; any noisier one would make the belief continuous and the exact optimum unavailable, as in the maze10
work. The map and goal are fixed, so the problem is pure localisation under a known plan.

Design probe (`design.md`): 2 048 episodes, uniform starts.

| policy | return | success | mean length |
|---|---|---|---|
| oracle (state known) | 0.695 | 1 | 13.5 |
| **exact optimum (belief sets)** | **0.512** | **0.995** | **24.4** |
| QMDP (belief-weighted shortest path) | 0.000 | 0.000 | 40 |
| random | 0.070 | 0.103 | 37.5 |

The optimum spends about eleven moves localising (the consistent set falls from 136 to 2.7 states by move 10; the
state is known exactly in 0.48 of its decisions). QMDP never succeeds: a belief-weighted value never pays for a move
whose only purpose is to find out where one is, and it turns in place. Along the optimum's trajectories a random
action is optimal in 0.34 of decisions (the optimal set has 1.02 actions on average), QMDP's action in 0.72.

## Part A. Zero-shot

**Models.** Qwen2.5-Instruct at 0.5B, 1.5B, 3B and 7B, bf16, local, greedy.

**Prompt.** The rules and the map as text (`bumpmaze.describe`: rows and columns numbered, the agent's ignorance of
its start stated, the three actions and the two observations explained, the aim stated as fewest actions "which may
require actions that tell you where you are"), then the transcript of the episode so far (one line per action, with
"moved" or "bumped" after a forward), then the question "Next action?". The chat template's system and user turns
carry this; the assistant turn is scored.

**Decision.** The action is the argmax over the three candidate replies "left", "right", "forward" of their
log-likelihood as the assistant's reply (so every decision is valid; no parsing). **Think variant** (3B and 7B only):
the model generates up to 256 tokens of reasoning ending in one of the three words, greedy; an unparseable reply counts
as "forward".

**Measures.**
* **Closed loop**: 128 episodes per model (fixed starts, the same for every model), return, success, mean length,
  against the optimum, QMDP and random; the action marginal (share of forward).
* **Decision optimality, teacher-forced**: 2 048 decisions sampled from the exact optimum's own trajectories (the
  model sees the optimum's transcript and chooses the next action): the share whose action is in the exact optimal
  set for the exact belief; chance 0.34, QMDP 0.72. Split by whether the state is already known (consistent set of
  size 1) or not.
* **Belief by questioning**: on 256 of those decisions, for each of the 35 free cells the question "Given the
  history, could you currently be in row r, column c? Answer yes or no." scored by P(yes) against P(no); the AUC of
  P(yes) against the exact consistent set (a cell is possible if any heading at it is consistent), per decision,
  averaged; the same for the four headings. Chance 0.5. Also the AUC of P(yes) against the *prior* possibility (every
  non-goal free cell), to see whether the model answers from the map alone.

## Part B. Training on the task

**Data.** Episodes from the exact optimum with ε = 0.2 random actions (so beliefs off the optimal path are seen),
every start equally likely; the label at every step is the **exact optimal action** for the exact belief (the
solver's argmax; ties broken by the first action), not the executed one. Training sets of 10 000, 30 000 and 100 000
episodes; the test set 2 048 episodes with disjoint random seeds.

**Models.**
* **LLM**: Qwen2.5-0.5B (base), full fine-tuning, bf16, AdamW 2e-5 with cosine decay, batch 64 sequences, 2 epochs;
  one sequence per episode = the rules-and-map prefix, then the transcript, loss on the action words only.
* **From scratch**: the project's small transformer (4 blocks, d = 128, 4 heads; one token per step carrying the
  action and the observation, a start token) trained on the same episodes and labels by cross-entropy, 2 epochs over
  the same data; and the same with a GRU (d = 128). Not size-matched to the LLM; the comparison is data efficiency at
  equal data, and the limit is stated.

**Measures.**
* Closed loop on the 2 048 test starts: return, success, length against the optimum; decision optimality on the
  optimum's trajectories; both as a function of training-set size.
* **Belief probes** on the fine-tuned LLM (every layer, the residual at the last transcript token, 8 192 decisions
  from the optimum's trajectories, held out by episode): a multi-label logistic probe to the 140 states' membership in
  the consistent set (AUC, and the share of decisions where the thresholded probe recovers the exact set), a ridge
  probe to the set's size (log), against the same probes on the **base model before fine-tuning** and on the
  from-scratch transformer's layers. A nonlinear (MLP) probe at the best layer as the ceiling.
* **Behaviour by uncertainty**: decision optimality split by consistent-set size (1, 2–4, 5–16, > 16).

## Decision rule

One task, one seed per model (the cost of fine-tuning); comparisons between models are on the same 2 048 test
starts and the same decisions, so paired; differences are reported with bootstrap 95 % intervals over episodes or
decisions, and a criterion holds if its interval excludes the threshold.

| | criterion |
|---|---|
| **Z-FLOOR** | every zero-shot model's closed-loop success ≤ 0.25 (random 0.10) |
| **Z-DEC** | every zero-shot model's teacher-forced decision optimality ≤ 0.55 (chance 0.34, QMDP 0.72) |
| **Z-SCALE** | the 7B's decision optimality exceeds the 0.5B's by ≥ 0.05 |
| **Z-BELIEF** | every zero-shot model's cell AUC ≤ 0.65 |
| **T-LEARN** | the fine-tuned 0.5B at 100 000 episodes: success ≥ 0.95 and return ≥ 0.9 × the optimum's |
| **T-DATA** | at 10 000 episodes the fine-tuned LLM's success exceeds the from-scratch transformer's by ≥ 0.10 |
| **T-PROBE** | the fine-tuned LLM's best-layer membership AUC ≥ 0.95, and ≥ 0.15 above the base model's at the same layer |
| **T-UNC** | the fine-tuned LLM's decision optimality on consistent sets > 16 is ≥ 0.10 below that on size 1 |

## Expectations

| | expectation |
|---|---|
| E1 | Z-FLOOR and Z-DEC hold: no small instruction model tracks a set of 136 hypotheses through one-bit observations in context; the think variant lifts the 7B's decision optimality by < 0.1 and leaves success < 0.4 |
| E2 | Z-SCALE holds, weakly (the 7B picks forward when the goal is ahead on the map more often) |
| E3 | Z-BELIEF holds; the AUC against the prior (map-only) possibility is higher than against the consistent set, so the models answer from the map, not the history |
| E4 | T-LEARN holds; the fine-tuned model's length is within 2 moves of the optimum's 24.4 |
| E5 | T-DATA holds: pretraining helps at 10 000 episodes; at 100 000 both reach T-LEARN's level |
| E6 | T-PROBE holds; the base model's probe at the same layer ≤ 0.80 (the history is in context, so some membership is linearly readable before any training, but not the set); the from-scratch transformer's best layer ≥ 0.95 as well |
| E7 | T-UNC holds: the residual errors of the trained models are in the information-seeking phase, as the maze10 agents were QMDP-optimal when certain and not when split |
| E8 | the GRU reaches T-LEARN's level too (the belief is a set the recurrence can carry; the bandit GRU's log-odds limitation does not apply to a deterministic filter) |

## Limits known in advance

One maze and one goal, so nothing about goal conditioning; one training seed per model; the from-scratch models are
not size-matched; the LLM is trained by supervision from the exact optimum, not by reward (the maze10 agents were
reward-trained), so what it learns is the optimum's policy, and the belief probes say what it computes to imitate it;
zero-shot prompts are one wording and one format each, and small models are sensitive to both; the think variant is
greedy and capped at 256 tokens.
