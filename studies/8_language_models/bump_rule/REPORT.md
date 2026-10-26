# Bump rule: what the reward-trained GRU does instead of computing the belief set

Run date: 2026-10-20. **No plan was registered**: the question ("what policy is reward training teaching the model?") was
asked after the reward experiment's battery (`../bump_reward`) and the analyses were run on request, so every number
here is exploratory. Scripts: `rule.py` (collection, memory, the window rule in the environment, the automaton, the
references) and `deviations.py` (where the deviations are); numbers in `tables.md`; data in `rule_<name>.json`,
`deviations_<name>.json`. Models: the five reward-trained GRUs (`../bump_reward/runs/gru/seed<k>`), the supervised GRU
(`../bump_maze/runs/gru_100000`) and the exact optimum put through the same analyses as controls. Each model: 20 000
greedy episodes from random starts with the exact filter alongside; lookups fitted on half the episodes and scored on
the other half; rules run on the 2 048 fixed test starts.

**Disclosures.** (1) The first launch of the analyses ran the GRU on all 20 000 episodes at once and three of the seven
processes died of GPU memory; the model calls were chunked and those three relaunched, nothing else changed. (2) The
lookup's prediction was rewritten from a Python loop to a vectorised form after the first (optimum) run for speed; the
optimum's first run (`rule_optimum_v1.json`) and the rerun agree. (3) Seeds 2 and 3 are different models (different
weights) whose greedy policies coincide almost everywhere on their own distribution; they are reported separately.

## 1. On its own episodes the agent is the optimum with one decision in seven replaced

| model | return / success / length | distinct sets visited | optimal decisions | at known states / unknown | = QMDP | forward | deviations: value given up each | its move's information − the optimum's, at deviations | mean information of its move / of an optimal action (bits) |
|---|---|---|---|---|---|---|---|---|---|
| the optimum | 0.518 / 0.993 / 24.0 | 247 | 1.000 | 1.000 / 1.000 | 0.703 | 0.694 | – | – | 0.290 / 0.290 |
| the supervised GRU | 0.511 / 0.979 / 24.2 | 260 | 0.989 | 0.987 / 0.992 | 0.696 | 0.693 | 0.041 | 0.000 | 0.288 / 0.288 |
| the reward-trained GRUs (5 seeds) | 0.42–0.45 / 0.78–0.92 / 27.0–27.5 | 239–304 | 0.84–0.86 | 0.82–0.88 / 0.84–0.87 | 0.57–0.61 | 0.68–0.71 | 0.036–0.043 | −0.08 to −0.19 | 0.253–0.258 / 0.268–0.286 |

* **Greedy, the agent visits as few sets as the optimum** (239–304 against 247; the battery's 380–473 counted sampled
  episodes), and its first moves are the optimum's: the top opening sequences coincide (`FLFFLFFF`, `FLFFLFLF`, …) and
  localisation is identical through step 4.
* **One decision in seven is suboptimal, and each costs little**: 0.14–0.16 of its decisions are not in the optimal
  set, each giving up 0.04 of the value at that set (one to four moves at γ = 0.97). The supervised GRU: 0.011, the same
  cost each. The deviation rate is the same at known states (set size 1, 0.41–0.47 of its decisions) as at unknown ones:
  the shortfall is not a failure to infer where it is.
* **Its deviations are less informative moves, not more.** Two thirds of the deviations go forward where the optimum
  turns, a fifth turn where the optimum goes forward. At deviations its move carries 0.08–0.19 bits less than the
  optimum's would; it is the more informative move in 0.02–0.07 of deviations and the less informative in 0.14–0.23.
  The opposite of maze10, where reward-trained agents deviated from QMDP towards information. Over all its decisions it
  gathers 0.253–0.258 bits per move against the 0.268–0.286 an optimal action would give at the same sets, and it
  localises slower: the state known in 0.35–0.42 of alive episodes at step 15 (the optimum 0.69), 0.66–0.86 at step 20
  (0.98).
* **Agreement with QMDP is 0.57–0.61** against the optimum's own 0.70: the agent is further from the myopic policy than
  the optimum is, in the direction of less information, not more.

## 2. Where the deviations are: a few dozen situations, three quarters of them fixed

| model | systematic (the set's majority action wrong) / occasional | sets wrong by majority | sets carrying half / 90 % of the deviations | sets with ≥ 20 decisions and more than one action, one of them wrong (share of decisions) | known states wrong by majority (of 140) |
|---|---|---|---|---|---|
| the supervised GRU | 0.53 / 0.47 | 11 of 260 | 7 / 17 | 10 (0.037) | 7 |
| the reward-trained GRUs | 0.61–0.73 / 0.27–0.39 | 45–53 of 239–304 | 13–19 / 42–53 | 24–41 (0.17–0.24) | 12–19 |

* **Three quarters of the deviations are systematic**: at a set the agent visits, its majority action is wrong, and 13
  to 19 sets carry half of all deviations. Two kinds recur across seeds. Early: at step 2 (seeds 2, 3; the set of 46
  states after `F L`) and step 4 (seeds 1–4; 24 states) the agent inserts a turn where the optimum goes forward, one
  move's cost, which is the difference between its openings and the optimum's. Late, at known states: it goes straight
  where the optimum turns at the same dozen places in every seed, (7,3), (5,2), (4,1), (3,1), (7,2) and the right-hand
  corridor, overshooting a turn by one to four moves. These are route errors with the position known, a map the agent
  has learned slightly wrong, not belief errors.
* **The remaining quarter is on-distribution history dependence at the same set**, mostly at known states late in the
  episode (sets #44, #47, #90, #148: size 1, steps 19–27, 0.3–0.4 of the decisions there suboptimal): the agent reaches
  the same cell and heading by different routes and acts differently, one of the actions wrong. In 24–41 of its sets,
  0.17–0.24 of its decisions, its action is not a function of the set on its own distribution; the supervised GRU's 10
  sets, 0.037. This is the battery's set-insufficiency where it costs return.

## 3. The function: as long a memory as the optimum, less use of each token, no small automaton

**Window lookups** (the last k tokens → the greedy action, fitted on half the episodes, scored on the other half):

| model | k = 2 | 4 | 6 | 8 | 12 | 20 | the window-k rule run in the environment, k = 6 / 8 / 12 / 20 |
|---|---|---|---|---|---|---|---|
| the optimum | 0.79 | 0.86 | 0.94 | 0.98 | 0.997 | 1.00 | 0.32 / 0.48 / 0.50 / 0.515 |
| the supervised GRU | 0.79 | 0.86 | 0.95 | 0.98 | 0.998 | 1.00 | 0.32 / 0.46 / 0.50 / 0.508 |
| the reward-trained GRUs | 0.79–0.82 | 0.87–0.90 | 0.94–0.97 | 0.98 | 0.99 | 1.00 | 0.27–0.36 / 0.37–0.41 / 0.41–0.42 / 0.41–0.45 |

* **The window test does not discriminate, and that is the result.** The exact optimum, a function of the belief set by
  construction, is reproduced on its own episodes by its last 12 tokens at 0.997 and its window-20 rule returns the full
  0.515 in the environment: a deterministic policy from 140 starts generates a tree of about 1 900 histories, and a
  window identifies the node. The reward-trained GRUs' curves are the same curves. Along its own trajectories the agent
  is no more reactive and no shorter in memory than the optimum; it is a different tree. Adding the step, bumps so
  far, steps since the last bump or forward moves so far moves every curve the same way.
* **Token patches** (replace the token at distance d from the decision with a random other token; share of held-out
  decisions that change): the supervised GRU 0.72 at d = 1, 0.33 at 4, 0.19 at 8, 0.07 at 12; the reward-trained GRUs
  0.49–0.53 at d = 1 (seed 4: 0.66), 0.17–0.20 at 4, 0.10–0.12 at 8, 0.04–0.06 at 12. The reward-trained policy
  depends less on any one past token at every distance: it uses the history less, consistent with gathering less
  information from it.
* **No small automaton reproduces it.** k-means on the decision-time hidden state with K = 2 … 256: action purity on
  held-out decisions reaches 0.99 by K = 128, but the transition (cluster, token) → next cluster is only 0.82–0.86
  deterministic, and the K = 256 machine run in the environment returns 0.29–0.39 (success 0.49–0.74) against the
  agent's 0.41–0.45. The supervised GRU's machine does no better (0.35 against 0.51), so the test is too crude to find
  the set automaton even where one is behaviourally present; it does show that the hidden state's geometry is not
  partitioned into functional states.

## Reading

Reward training teaches the optimum's trajectory tree with a few dozen edits. The agent opens as the optimum does,
turns at every bump and otherwise goes forward at the optimum's rate; its 0.08 of lost return comes from one decision
in seven, each costing one to four moves, three quarters of them fixed errors at a few dozen situations: an extra turn
at step 2 or 4, and going straight past a turn at the same dozen known positions in every seed. The remaining quarter
is history dependence at the same known cell late in the episode. Its errors are not more informative moves traded for
return, as maze10's agents' were; they are less informative, and it localises slower than the optimum because of them.
As a function of the history it has the same memory length as the optimum along its own trajectories (both are a
window of about 12 tokens over a tree of about 1 900 histories), uses each token less, and does not reduce to a small
finite-state machine over its hidden state. What reward training produced is not a different kind of policy from
imitation's but a worse copy of the same tree, tuned on the few hundred situations its own rollouts reach, with the
errors of a map learned from sparse reward at the places the gradient reached least.

## Limits

Exploratory, no registered criteria. One maze, one goal, the GRU only (the transformer did not learn by reward). The
window test is confounded on own-trajectory data for any deterministic policy, so it bounds nothing about mechanism;
the token patch creates histories that are not consistent with the maze, so its absolute values are not comparable to
the optimum's. k-means is one partition of the hidden state; a partition aligned with the sets was not tried. The
deviation analysis uses greedy actions; the sampled policy differs little (final entropy 0.009).
