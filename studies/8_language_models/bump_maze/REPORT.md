# Bump maze: a small language model on the most partially observed navigation task with an exact belief

Run date: 2026-10-13. Task, prompts, models, measures and decision rule: `PLAN.md`, committed (`bfde9ec`) **before any
language model was run on the task**. Numbers from `tables.md`; data in `design.json`, `zero_shot.json`,
`runs/*/results.json`, `probe.json` (the 100k model) and `probe_llm10000.json` (the 10k model, the base model and
the from-scratch transformer). Reproduce: `reproduce.py bump_maze` (about 4 h on one GPU; the fine-tuned language
models are not committed, 2 GB each, and are rebuilt by `train.py` in 20–100 min).

**Disclosures.** (1) The zero-shot scorer as first written materialised full-vocabulary logits for every token and
took 44 GB per batch; the first sweep was killed after the 0.5B model, the scorer rewritten to score only the
candidate positions, and the sweep rerun from the start (the 0.5B's teacher-forced number was identical, 0.452).
(2) The plan's belief control, the AUC against the prior possibility, is degenerate (every non-goal cell is possible
a priori), so it was replaced by a **mismatched-history control**: the model's answers for another decision's
transcript scored against this decision's set. (3) The registered Z-DEC threshold (0.55) turned out to lie below a
constant policy: "always forward" is optimal in 0.707 of the teacher-forced decisions. That reference was added to the
design probe after the zero-shot results were seen and is used below; the registered criterion is scored as written.
(4) The 30k GRU run was interrupted by the memory incident and rerun alone. (5) One training seed per model.

**The task** (`goalgeo.bumpmaze`; design probe `design.md`). A 9 × 9 MiniGrid maze, 35 free cells, a known map and
goal, the start cell and heading unknown (136 start states, 140 states); actions left, right, forward; the only
observation is whether a forward move bumped. The belief is the set of states consistent with the history (17 285
reachable sets), and the exact optimum is value iteration over them. On 2 048 episodes: the optimum returns 0.512
with success 0.996 and 24.3 moves (the oracle that knows its state 0.695 and 13.5 moves), **QMDP 0.000** (it never
pays to localise and turns in place), random 0.057 (success 0.087), always forward 0.028, forward-and-turn-right-on-
bump 0.083. The optimum's consistent set falls from 136 to 2.7 states by move 10; its state is known in half of its
decisions. Teacher-forced references on its decisions: a random action is optimal in 0.34, QMDP's in 0.72, always
forward in 0.71.

## 1. Zero-shot: no model acts on the history, and none tracks where it could be

Qwen2.5-Instruct, greedy, the action chosen by scoring the three candidate replies; the think variant generates up to
256 tokens of reasoning first. Closed loop on 128 fixed starts (64 for the think variant); teacher-forced decision
optimality on 2 048 decisions of the optimum's trajectories (512 for think).

| model | closed loop: success (95 % CI) | return | length | teacher-forced decision optimality (95 % CI) | state known / unknown | forward share (teacher-forced) |
|---|---|---|---|---|---|---|
| 0.5B | 0.00 (0.00–0.00) | 0.000 | 40 | 0.452 (0.43–0.47) | 0.47 / 0.43 | 0.59 |
| 1.5B | 0.12 (0.06–0.17) | 0.091 | 36.5 | 0.664 (0.64–0.68) | 0.68 / 0.64 | 0.88 |
| 3B | 0.02 (0.00–0.05) | 0.021 | 39.2 | 0.391 (0.37–0.41) | 0.25 / 0.52 | 0.48 |
| 7B | 0.04 (0.01–0.07) | 0.028 | 38.9 | 0.552 (0.53–0.57) | 0.45 / 0.64 | 0.56 |
| 3B, think | 0.11 | 0.090 | 36.4 | 0.588 | 0.55 / 0.62 | 0.76 |
| 7B, think | 0.06 | 0.040 | 38.5 | 0.568 | 0.50 / 0.63 | 0.66 |
| references | random 0.087; always forward 0.028; wall-follower 0.083; **optimum 0.996** | | | chance 0.34; **always forward 0.71**; QMDP 0.72; optimum 1 | | |

* **Closed loop, every model is at or below a random walk.** The 1.5B's 0.12 is the best and is within the random
  walk's reach; the 0.5B never goes forward in its own episodes (it turns in place for 40 moves); the 3B and 7B
  succeed in 0.02–0.04. Thinking does not help: the 7B's reasoning mis-tracks its own heading ("you turned left
  twice, which brings you back to facing east") and both think variants stay at the random level.
* **Teacher-forced, no model beats "always forward".** The 1.5B's 0.66 comes from saying forward 0.88 of the time;
  the 7B's 0.55 and the 0.5B's 0.45 are below the constant policy's 0.71, the 3B's 0.39 near chance. Z-DEC as
  registered fails on the 1.5B (upper CI 0.68 > 0.55), but the threshold was set below the constant policy and the
  conclusion the criterion was meant to carry, that no model decides from the belief, stands. Z-SCALE holds (7B − 0.5B
  = 0.10), but 0.10 above a model that is itself below "always forward" is not evidence of scale helping.
* **The models do not use the history when asked where they could be.** Per-decision AUC of "could you be in row r,
  column c?" against the exact consistent set: 0.53, 0.57, 0.60, 0.53 for 0.5B to 7B; with another decision's
  transcript in place of the real one, 0.53, 0.57, 0.58, 0.54, the same. The answers come from the map, not the
  history. The heading question is the one thing the larger models partly track (AUC 0.70 for the 3B and 7B, 0.36 and
  0.44 for the smaller two): counting turns is within reach, intersecting a set of hypotheses with bump outcomes is
  not. The 1.5B answers "no" to every cell, the 0.5B "yes" to 0.82 of them.

## 2. Trained on the task: the fine-tuned 0.5B reaches the exact optimum at 10 000 episodes; the models from scratch need 100 000

Supervision from the exact optimum (labels: the optimal action for the exact belief, on episodes with ε = 0.2 random
actions), 2 epochs; closed loop on 2 048 test starts; teacher-forced decision optimality on 2 048 decisions.

| model, episodes | success (95 % CI) | return | length | decision optimality (95 % CI) | by set size 1 / 2–4 / 5–16 / > 16 |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B fine-tuned, 10 000** | **0.995** (0.99–1.00) | **0.514** | 24.3 | 0.999 | 1.00 / 1.00 / 1.00 / 1.00 |
| same, 30 000 | 0.995 | 0.515 | 24.2 | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 |
| same, 100 000 | 0.995 | 0.515 | 24.2 | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 |
| transformer from scratch (4 blocks, d 128), 10 000 | 0.153 | 0.132 | 34.8 | 0.790 | 0.71 / 0.71 / 0.88 / 1.00 |
| same, 30 000 | 0.308 | 0.206 | 32.6 | 0.880 | 0.78 / 0.92 / 1.00 / 1.00 |
| same, 100 000 | 0.887 | 0.468 | 25.4 | 0.988 | 0.98 / 1.00 / 1.00 / 1.00 |
| GRU (d 128), 10 000 | 0.134 | 0.119 | 35.3 | 0.783 | 0.71 / 0.73 / 0.85 / 0.97 |
| same, 30 000 | 0.383 | 0.231 | 32.3 | 0.876 | 0.83 / 0.79 / 0.98 / 1.00 |
| same, 100 000 | 0.979 | 0.508 | 24.4 | 0.990 | 0.99 / 0.98 / 1.00 / 1.00 |
| exact optimum | 0.996 | 0.512 | 24.3 | 1 | |

* **T-LEARN holds, and already at 10 000 episodes**: the fine-tuned model matches the optimum's return (0.514 against
  0.512, within noise), success and length, and makes the optimal decision in 0.999 of teacher-forced decisions and
  in 1.00 of its own. **T-DATA holds by 0.84**: at 10 000 episodes the transformer from scratch succeeds in 0.15, the
  GRU in 0.13. At 100 000 both approach the optimum (0.89 and 0.98 success), so the task is learnable from scratch
  with ten times the data. The language model's advantage is data efficiency, by at least an order of magnitude.
* **Where the from-scratch models fail is not the uncertain phase.** Their decision optimality is lowest when the
  state is already known (0.71 at 10 000 episodes) and highest on large consistent sets (0.97–1.00): the optimum's
  early moves are nearly always forward, easy to imitate, and what takes data is the map, the shortest-path move from
  each of 140 states once localised. **T-UNC fails** for the fine-tuned model because it has no errors anywhere
  (1.000 against 1.000), and the expected pattern is reversed for the models that do err.

## 3. Belief probes: the consistent set is linear in the fine-tuned model's middle layers, and half-present in the base model before any training

8 192 decisions of the optimum's trajectories, held out by episode (set size 14.9 on average, known in 0.48). A linear
multi-label probe from the residual stream at the last transcript token to the 140 states' membership in the
consistent set; "exact" is the share of decisions whose thresholded probe equals the exact set.

| model | best layer | membership AUC | exact set recovered (linear) | Jaccard | log set size R² | exact, MLP probe at the best layer |
|---|---|---|---|---|---|---|
| **Qwen2.5-0.5B fine-tuned, 100k** | 17–20 of 24 | **1.000** | **0.966** | 0.97 | 0.998 | 0.936 |
| fine-tuned, 10k | 19 | 1.000 | 0.906 | 0.93 | 0.994 | 0.903 |
| **Qwen2.5-0.5B base, no training on the task** | 16 | 0.994 | **0.585** | 0.64 | 0.987 | 0.764 |
| transformer from scratch, 100k | 3 of 4 | 0.980 | 0.317 | 0.45 | 0.960 | 0.627 |
| any model, the embedding layer | 0 | 0.749 | 0.000 | 0.00 | 0.000 | |

* **In the fine-tuned model the exact set is a linear readout of the middle layers.** At layers 16–20 the linear probe
  recovers the whole 140-state set exactly in 0.97 of decisions and the set's log size at R² 0.998; recovery rises
  from 0.33 at layer 4 to 0.78 at layer 12 to 0.97 at layer 17, and falls slightly toward the output (0.94 at layer
  24). The MLP probe adds nothing (0.936): the code is linear. With 10k episodes the same profile peaks at 0.91.
* **The base model, which cannot act on this task, already holds most of it.** Before any training a linear probe
  reads the membership at AUC 0.994 and the exact set in 0.585 of decisions at layer 16, the log size at R² 0.987;
  an MLP probe reaches 0.764. So **T-PROBE fails as registered** (the AUC gap is 0.007, not 0.15), because AUC
  saturates: 140 mostly-impossible states make membership easy to rank. On the demanding measure, exact recovery,
  fine-tuning moves the base model from 0.59 to 0.97. The from-scratch transformer, which does act near-optimally at
  100k, holds the set less linearly (0.32 exact, 0.63 by MLP) in a 128-dimensional state.
* The base model's result is the surprise of the experiment: a pretrained 0.5B model with the rules, the map and the
  transcript in context carries, in linear form in its middle layers, a representation from which the set of
  consistent positions can be read in most cases, while its own zero-shot behaviour and its answers to "could you be
  here" use none of it (the instruct model's cell AUC 0.53). Decodable is not used, as the first study of the project
  found for the belief in untrained networks, and here the gap is between a 0.59 exact-set readout and a policy at
  random. Fine-tuning closes it from both sides: the readout becomes nearly exact and the policy optimal.

## Scorecard

| | criterion | numbers | held |
|---|---|---|---|
| Z-FLOOR | every zero-shot success ≤ 0.25 | upper CIs 0.00, 0.17, 0.05, 0.07 | yes |
| Z-DEC | every zero-shot decision optimality ≤ 0.55 | 1.5B 0.66 (upper CI 0.68); always forward 0.71 | **no** (threshold below a constant policy) |
| Z-SCALE | 7B − 0.5B ≥ 0.05 | 0.10 | yes |
| Z-BELIEF | every zero-shot cell AUC ≤ 0.65 | 0.53–0.60 | yes |
| T-LEARN | fine-tuned at 100k: success ≥ 0.95, return ≥ 0.9 × optimum | 0.995; 1.01 × | yes |
| T-DATA | at 10k, LLM success − transformer's ≥ 0.10 | 0.995 − 0.153 = 0.84 | yes |
| T-PROBE | best-layer AUC ≥ 0.95 and ≥ 0.15 above the base model's | 1.000 against 0.993 | **no** (AUC saturates in the base model) |
| T-UNC | fine-tuned: sets > 16 ≥ 0.10 below size 1 | 1.000 against 1.000 | **no** (no errors) |

| | expectation | held |
|---|---|---|
| E1 | Z-FLOOR, Z-DEC; think lifts the 7B by < 0.1 and leaves success < 0.4 | in part: floor yes, Z-DEC no as registered; think +0.02 and 0.06 success |
| E2 | Z-SCALE, weakly | yes, and uninformative (both below always forward) |
| E3 | Z-BELIEF; answers from the map, not the history | yes (mismatched-history control equal to the real one) |
| E4 | T-LEARN; length within 2 moves of the optimum | yes (24.2 against 24.3) |
| E5 | T-DATA; both reach T-LEARN's level at 100k | in part: T-DATA yes; the transformer from scratch reaches 0.89 at 100k, the GRU 0.98 |
| E6 | T-PROBE; base ≤ 0.80; from-scratch ≥ 0.95 | no: the base model is at 0.994 AUC (0.59 exact); the from-scratch transformer 0.98 AUC (0.32 exact) |
| E7 | T-UNC: residual errors in the information-seeking phase | no, reversed: the models that err do so where the state is known |
| E8 | the GRU reaches T-LEARN's level | nearly (0.979 success, return 0.508 at 100k) |

Five of eight registered criteria held; of the expectations three held, two in part, three failed.

## Reading

Small instruction-tuned language models cannot do this task in context: through one-bit observations, none decides
better than a constant policy or walks better than a random one, and none uses the transcript when asked where it
could be, at 7B as at 0.5B, with or without a chain of thought. Trained on the task by supervision from the exact
optimum, a 0.5B model becomes exactly optimal with ten thousand episodes, where the same architecture family trained
from scratch needs a hundred thousand, and it holds the belief, the full set of 140 state hypotheses, as a linear
readout of its middle layers, recoverable exactly in 0.97 of decisions. The base model already holds that readout
at 0.59 before training, which is where the data efficiency comes from: pretraining supplies a representation of the
transcript from which the consistent set is close to linearly available, and fine-tuning turns it into the policy.

## Limits

One maze, one goal, one seed per model, one prompt wording and format per variant; the models from scratch are not
size-matched (0.8M against 494M parameters), so T-DATA measures data efficiency, not sample complexity per
parameter; the language model is trained by supervision, not reward; the probes read a 140-dimensional set from a
896-dimensional state at 7 150 fit decisions, so exact recovery in the base model may partly reflect the probe's
capacity to compute the set from a rich contextual embedding of the transcript rather than the model holding it; a
causal test (patching the probed subspace) was not run. The think variant is greedy and capped at 256 tokens.
