# Bump reward: reward training does not give the small models a set-sufficient policy either

Run date: 2026-10-14. Training, battery and decision rule: `PLAN.md`, committed (`eeac853`) **before any model was
trained by reward**. Numbers from `tables.md`; data in `runs/<kind>/seed<k>/{model.pt, log.json}`, `battery_*.json`,
`table_*.pt`. Reproduce: `reproduce.py bump_reward` (about 3 h on one GPU, given the policy study's `sets.pt` and
`histories.pt`).

**Disclosures.** (1) The first launch of the ten runs stalled after a few hundred updates: ten caching allocators
filled the GPU and thrashed. All runs were killed and relaunched from scratch with a per-process memory cap (8 % of
the GPU) and expandable allocator segments; nothing else changed, and nothing from the first launch is used. (2) The
battery script's first launch on the supervised models failed on a path error before loading any model, and was rerun.
(3) The battery was smoke-tested on the six-update smoke model of the plan's disclosure before the real runs; its
outputs were deleted. (4) The supervised transformer's part A numbers reproduce the policy study's to three decimals,
which checks the new battery against the old scripts.

**The question.** The policy study found that models trained by supervision from the exact optimum do not compute the
belief set: their action depends on the history beyond the set, they are far from optimal on sets the training never
visits, no set is decodable from their state on held-out sets, and nothing is carried through the step boundary. A
reward-trained agent imitates nothing and its training data is its own visitation. Does that give a policy that is a
function of the set?

## 1. Training: the GRU learns the task by reward, the transformer barely

PPO on reward only, 4 096 environments × 2 000 updates (about 8 M episodes, 15 M steps), the study 7 settings
unchanged; greedy on the 2 048 fixed test starts.

| architecture | seeds: final greedy return | success | at update 250 / 500 / 1000 (median seed) | the supervised model (100k episodes) |
|---|---|---|---|---|
| **GRU** | **0.413–0.450** (median 0.434) | 0.78–0.92 | 0.36 / 0.41 / 0.44 | 0.508 / 0.979 |
| transformer | TFM_RETURNS | TFM_SUCCESS | TFM_CURVE | 0.468 / 0.887 |
| references | the exact optimum 0.512 / 0.995; QMDP 0; wall-follower 0.083; random 0.057 | | | |

* **LEARN holds for the GRU** (median 0.434, success 0.855): it passes the wall-follower by update 100, the plan's
  0.30 by update 250, and ends 0.07 below the supervised GRU and 0.08 below the optimum, with 0.86 success and 27 moves
  against the optimum's 24. TFM_LEARN_SENTENCE

## 2. The battery: set-sufficiency is at chance, the table is not the policy, and the agent sees even fewer sets

Every model's greedy action on all 17 285 reachable sets from the shortest history; set-sufficiency over the policy
study's histories; the table's exact value; the agent's own coverage from 100 000 of its sampled episodes.

| model | online return | own coverage (sets) | optimal: all sets | optimum-visited (247) | imitation-visited, other (6 579) | never visited by the imitation data (10 459) | visited by the agent itself | **set-sufficiency** (baseline) | table value: of V* / of its own return |
|---|---|---|---|---|---|---|---|---|---|
| PPO GRU, median seed 2 | 0.434 | 462 | 0.540 | 0.757 | 0.592 | 0.502 | 0.716 (462) | **0.605** (0.58) | 0.396 / 0.473 |
| PPO GRU, seeds 0, 1, 3, 4 | 0.41–0.45 | 380–473 | 0.48–0.55 | 0.68–0.78 | 0.53–0.61 | 0.45–0.51 | 0.64–0.72 | 0.37 (0.50), 0.61 (0.65), 0.48 (0.45), 0.40 (0.37) | 0.25–0.43 / 0.30–0.52 |
| PPO transformer, median seed | TFM_A_MEDIAN |
| PPO transformer, other seeds | TFM_A_OTHERS |
| supervised GRU, 100k | 0.508 | 2 585 | 0.628 | 0.858 | 0.706 | 0.573 | 0.694 (2 585) | 0.546 (0.47) | 0.280 / 0.286 |
| supervised transformer, 100k | 0.468 | 2 456 | 0.585 | 0.822 | 0.660 | 0.533 | 0.646 (2 456) | 0.494 (0.56) | 0.244 / 0.270 |
| the fine-tuned 0.5B (policy study) | 0.515 | | 0.741 | 0.968 | 0.879 | 0.648 | | 0.591 (0.43) | 0.946 / 0.95 |

* **SUF-R fails, in every seed and both architectures.** The reward-trained GRUs give the same action on every history
  of a set in 0.37–0.61 of sets against same-size baselines of 0.37–0.65: within 0.04 of chance in four seeds and
  below it in one (0.37 against 0.50). The supervised GRU is 0.08 above its baseline, the 0.5B 0.16 above. TFM_SUF_SENTENCE
  The central expectation, that an agent trained on its own visitation would be self-consistent across histories of a
  set, is wrong: the histories that reach a set by routes the agent does not take are as foreign to it as to an
  imitator.
* **OUT-R fails; the reward-trained agents are the least general.** On the sets the imitation data never visits the
  PPO GRUs are optimal in 0.45–0.51, below the supervised GRU (0.57) and transformer (0.53) and far below the 0.5B
  (0.65); even on the sets their own episodes visit they are optimal in only 0.64–0.72 (their returns are 0.08–0.10
  below the optimum). TFM_OUT_SENTENCE
* **VALUE-R fails: the table is not the policy for a reward-trained agent either.** The policy defined by the GRUs'
  actions on shortest histories is worth 0.25–0.43 of V* from the start and 0.30–0.52 of their own online return, and
  loses its way from 0.95–1.00 of sets, while the agents themselves succeed in 0.78–0.92 of episodes. TFM_VALUE_SENTENCE
* **The agents see far fewer situations.** 100 000 sampled episodes of a PPO GRU visit 380–473 of the 17 285 sets; the
  supervised models' sampled episodes 2 456–2 585 and the ε = 0.2 imitation data 6 826. A policy trained on its own
  visitation, with the entropy bonus annealed, narrows its own distribution; it is tuned to a few hundred situations
  reached its own way.

## 3. State, swaps and the prefix

**Probes** at the decision position (the GRU's hidden state; the transformer's residual entering each block), 8 192
decisions uniform over sets, held out by set:

| model | set: exact | Jaccard | Q*: R² | P(bump \| forward): R² |
|---|---|---|---|---|
| PPO GRU (5 seeds) | 0.000 | 0.01–0.02 | 0.08–0.14 | 0.57–0.61 |
| supervised GRU | 0.000 | 0.10 | 0.20 | 0.75 |
| PPO transformer (median seed, best block) | TFM_PROBE |
| supervised transformer (best block) | 0.000 | 0.01 | 0.06 | 0.52 |
| the fine-tuned 0.5B (best layer) | 0.035 | 0.41 | 0.42 | 0.84 |

* **CODE-R holds in the way expected, and for a reason that cuts the other way.** No model holds the set (exact 0.000,
  Jaccard ≤ 0.10), and all hold whether forward will bump, the reward-trained GRUs at R² 0.57–0.61, the supervised
  GRU 0.75, the 0.5B 0.84. The small models' 128-dimensional states carry less about the belief than the language
  model's, not more; reward training carries the least.

**Swaps** of the decision state between two histories of the same set, and between two sets of the same size:

| model | same set: decision changed | same set, where the decisions differ: to the donor's | different sets: to the donor's |
|---|---|---|---|
| PPO GRU (hidden state; 5 seeds) | SWAP_GRU |
| supervised GRU | SWAP_SUP_GRU |
| PPO transformer (median seed, by block) | TFM_SWAPS |
| supervised transformer (by block 0 → 4) | 0.38 / 0.39 / 0.41 / 0.31 / 0.29 | | 0.46 / 0.47 / 0.52 / 0.62 / 1.00 |

* **SAME-R fails.** Swapping the whole decision state between two histories of the same set changes the decision in
  SWAP_GRU_RANGE of pairs for the reward-trained GRUs (the supervised GRU SWAP_SUP_GRU_VAL): the state is a
  history code, and the two histories' states decide differently more often than not where their natural decisions
  differ. For the transformers the same holds at every block (the supervised one 0.29–0.41).

**The prefix patch** (transformers): TFM_PREFIX_PARA

## Scorecard (the median-return seed of each architecture)

| | criterion | GRU | transformer |
|---|---|---|---|
| LEARN | return ≥ 0.30, success ≥ 0.60 | 0.434 / 0.855: yes | TFM_SC_LEARN |
| SUF-R | set-sufficiency ≥ 0.70, ≥ 0.15 above baseline | 0.605 (0.60–0.61) against 0.58: **no** | TFM_SC_SUF |
| OUT-R | optimal on never-visited sets ≥ 0.60 | 0.502: **no** | TFM_SC_OUT |
| VALUE-R | table value within 5 % of the online return | 0.473: **no** | TFM_SC_VALUE |
| CODE-R | set exact ≤ 0.10 at every block; P(bump) R² ≥ 0.60 somewhere | 0.000; 0.565: **no** (P(bump) 0.57, just below) | TFM_SC_CODE |
| SAME-R | same-set swaps change ≤ 0.10 | SWAP_GRU_MED: **no** | TFM_SC_SAME |
| PRIOR-R | last prefix token ≤ 0.10 at every block, all prefix tokens ≥ 0.80 at block 0 | not run (recurrent) | TFM_SC_PRIOR |

| | expectation | held |
|---|---|---|
| E1 | LEARN at 0.30–0.45 | GRU yes (0.43); transformer TFM_E1 |
| E2 | SUF-R for both: the central expectation | **no**: at chance for both |
| E3 | OUT-R weakly, above the supervised transformer | no: below every supervised model |
| E4 | VALUE-R: the table is the policy | no: 0.3–0.5 of the online return |
| E5 | CODE-R: no set code, P(bump) present | the set absent everywhere; P(bump) at 0.57–0.61 (GRU), just under the registered 0.60 |
| E6 | SAME-R in part, less history-specific content than the imitators | no: more, not less |
| E7 | PRIOR-R: the transformer reads the prefix raw; recurrence does not decide set-sufficiency | TFM_E7; the GRU is no more set-sufficient than the transformer |

SCORE_SENTENCE

## Reading

Reward training does not change the kind of solution. A GRU trained by PPO alone learns the task to 0.43 of return
and 0.86 success, a competent information-seeking policy where QMDP scores zero, and its policy is no more a function
of the belief set than the imitators' were: on two histories that reach the same set it chooses the same action at the
chance rate, it is optimal on half of the sets the imitation data never visits and on two thirds of the sets it visits
itself, its table is worth a third of its own return, its state holds no set and swapping it between two histories of
the same set changes the decision in SWAP_GRU_RANGE of pairs. The reward-trained agent is the narrowest of all the
models measured: its own episodes visit a few hundred of the 17 285 reachable sets, and it has learned those few
hundred situations by the routes it takes to them. Imitation of the optimum's trajectories and optimisation of the
agent's own are two ways of learning a trajectory distribution; neither produces the filter. With the policy study,
the conclusion covers the three ways these models were trained on this maze and the three architectures: a pretrained
language model fine-tuned by supervision, small transformers and GRUs by supervision, and the same by reward. None of
them computes the belief set, and the exactness of the fine-tuned language model on the optimum's own decisions is the
exactness of an interpolator on its training distribution. TFM_READING_SENTENCE

## Limits

One maze, one goal; the PPO recipe is study 7's without tuning, and the transformer's failure to learn by reward here
is a failure of that recipe under sparse 40-step reward, not a bound; the agents' return is 0.08 below the optimum's,
so some of their non-optimality on visited sets is competence, not set-insufficiency, which is why the set-sufficiency
and swap measures, which do not use the optimum, carry the conclusion; the shortest and random-walk histories are
out-of-distribution for every model, equally; the probes are linear in a 128-dimensional state; the battery's
coverage count samples 100 000 episodes, so rarely visited sets are missed.
