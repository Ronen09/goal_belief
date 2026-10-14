# Bump reward: does reward training give the small models a set-sufficient policy where imitation did not?

Written before any model was trained by reward on the task. **Disclosure:** the training script was smoke-tested for
six updates to check that it runs (returns at the random level; nothing else looked at). Brief: given in conversation
("what if we try just reward training the model", then "sure do tier 1"): the from-scratch transformer and GRU trained
by reward alone, measured with the policy study's battery (`../bump_policy`), against the same architectures trained
by supervision from the exact optimum (`../bump_maze`, `tfm_100000`, `gru_100000`).

## The question

The policy study found that the supervised models' policies are not functions of the belief set: the same action on
every history of a set in 0.59 of sets for the fine-tuned 0.5B and 0.49 for the from-scratch transformer (its baseline
for different sets 0.56), optimal on 0.65 and 0.53 of the 10 459 reachable sets the training data never visits, their
tables worth 0.95 and 0.24 of V* from the start although both are competent online, no general set code in the state,
nothing carried through the step boundary. Every one of those models imitated the optimum's trajectory distribution.
A reward-trained agent has no distribution to imitate: its training data is its own visitation, exploration included,
and its online value is by construction the value of the policy it runs. The question is whether that gives a policy
that is a function of the belief set, or whether set-insufficiency is a property of these sequence models on this task
however they are trained.

## Training

The single-goal bump maze (`goalgeo.bumpmaze`, seed 0; the exact optimum returns 0.512 with success 0.995; QMDP 0;
random 0.057 / 0.087; always forward 0.028; wall-follower 0.083). The single-goal experiment's `StepTransformer`
(4 blocks, width 128, 4 heads) and `StepGRU` (width 128) with a value head on the same trunk; PPO on reward only (1 at
the goal, γ = 0.97 as the task's; GAE λ 0.95, clip 0.2, 3 epochs × 4 minibatches, lr 1e-4 with 50 warm-up updates,
entropy 0.03 → 0.003, gradient norm 0.5; 4 096 environments × 2 000 updates ≈ 8 M episodes), **5 seeds per
architecture**, the study 7 settings unchanged. Evaluation greedy on the single-goal experiment's 2 048 fixed test
starts, every 50 updates. The registered object of each architecture is the seed with the median final greedy
return; all seeds are reported.

## The battery (the policy study's, `../bump_policy`)

Over the 17 285 reachable sets (`../bump_policy/sets.pt`, the shortest history to each; the histories of
`../bump_policy/histories.pt`, up to four per set):

* **A1, set-sufficiency**: the share of sets on which the greedy action is the same for every history; the baseline for
  two different sets of the same size.
* **A2, the table**: the greedy action on every set from its shortest history against the optimal set, grouped by the
  imitation training data's coverage (optimum-visited / training-visited / never visited, for comparability with the
  policy study) and by the agent's own coverage (sets visited in 100 000 of its own sampled episodes / not).
* **A3, the table's value** over the belief MDP from the start, against the agent's own greedy online return.
* **B1, probes** at the decision position on 8 192 decisions uniform over sets, held out by set: the set (exact,
  Jaccard), Q* (R²), P(bump | forward) (R²), at the residual entering each block and the final (GRU: the hidden state).
* **C1, swaps** of the decision position's residual entering each block between two histories of the same set and
  between two sets of the same size (1 024 pairs each; GRU: the hidden state at the decision).
* **D2, the prefix patch** (transformer): the policy study's pairs (same-length prefixes with different sets, the same
  3-step suffix, disjoint optimal actions at the end); the token ending the prefix, all prefix tokens, a middle token,
  at the residual entering each block. The GRU carries its state by construction, so D2 is not run for it.

The same B1, C1 and D2 are run on the supervised `tfm_100000` and `gru_100000` for the comparison (their A parts are
in the policy study).

## Decision rule

Bootstrap 95 % intervals over sets or pairs; a criterion holds if the median seed's interval excludes the threshold.

| | criterion |
|---|---|
| **LEARN** | the median seed's greedy return ≥ 0.30 and success ≥ 0.60 (QMDP 0; the supervised transformer 0.468 / 0.887, GRU 0.508 / 0.979) |
| **SUF-R** | set-sufficiency ≥ 0.70 and ≥ 0.15 above the same-size baseline (the supervised transformer 0.49 against 0.56; the 0.5B 0.59 against 0.43) |
| **OUT-R** | optimal on the sets the imitation data never visits ≥ 0.60 (the supervised transformer 0.53, the 0.5B 0.65) |
| **VALUE-R** | the shortest-history table's value from the start within 5 % of the agent's own greedy return |
| **CODE-R** | the set's exact recovery on held-out sets ≤ 0.10 at every block; P(bump) R² ≥ 0.60 at some block |
| **SAME-R** | same-set swaps change ≤ 0.10 of decisions at every block |
| **PRIOR-R** | transformer: the last prefix token patched switches ≤ 0.10 at every block; all prefix tokens ≥ 0.80 at block 0 |

## Expectations

| | expectation |
|---|---|
| E1 | LEARN holds, at 0.30–0.45: competent, below the supervised models (which had the optimum's labels) |
| E2 | SUF-R holds for both architectures: the agent's own visitation includes off-optimum histories of the same sets, and a self-consistent policy pays under reward; this is the central expectation |
| E3 | OUT-R holds weakly (0.60–0.70): better than the supervised transformer, about the 0.5B's level, still far from optimal on never-visited sets |
| E4 | VALUE-R holds (it follows from E2): for a reward-trained agent the table is the policy |
| E5 | CODE-R holds: no general linear set code in either architecture; whether forward will bump is present (R² ≥ 0.6), as in the 0.5B |
| E6 | SAME-R holds in part (≤ 0.15 at the last block): less history-specific content than the imitators' 0.21 |
| E7 | PRIOR-R holds: the transformer reads the prefix from its raw tokens, as the 0.5B did; the GRU is set-sufficient to the same degree as the transformer, so recurrence is not what decides it |

## Limits known in advance

One maze, one goal; the PPO recipe is study 7's with no tuning for this task, so a LEARN failure would be a failure of
this recipe under sparse reward over 40 steps, not a bound on reward training; "never visited" is defined by the
imitation data for comparability and the agent's own coverage is reported alongside; the shortest history is an
out-of-distribution history for the agent as for the imitators; one battery run per seed, the median seed registered.
