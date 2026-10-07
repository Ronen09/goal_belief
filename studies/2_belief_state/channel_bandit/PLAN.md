# Channel bandit: matched-evidence counterfactuals — task, measures, decision rule and expectations

Written before any model of this experiment was trained. **Disclosure:** a design probe of the cue process (no network)
chose its parameters and is given below; `--quick` smoke runs of the training and measurement scripts checked the
code. Brief: `BRIEF.md`. Code: `goalgeo/bandit.py` (`channel`, `channel_posterior`, `episode_tables`; tests in
`tests/test_bandit.py` against the hidden-goal experiment's filter), the reward-bandit experiment's `train.py` and
`measure.py` with the task overridden, and `pairs.py` here.

## The question

In the reward-bandit task the cues are i.i.d., so their counts fix the posterior and a network that acts on the belief
cannot be told from one that applies a static count-to-probability lookup. Here the cues pass through the hidden-goal
experiment's sticky reliability channel: a run of consistent tokens is evidence that the channel is on, so **the same
counts in a different order give a different exact posterior**. Two histories with the same counts and different
beliefs are a matched-evidence counterfactual: does the policy, and does block 0's MLP output, separate them in the
Bayes direction?

## Task

As the reward bandit (three goals, four actions, Bernoulli rewards with the clipped matrix, six decisions, rewards as
evidence), with **8 cue tokens** drawn through the channel: c ∈ {off, on}, P(stay) = **0.8**, start (½, ½); when on,
token k has probability **0.7** under goal k and 0.15 under goal k − 1; when off, the cues are uniform. The exact
posterior after the cues is the joint filter over (goal, channel) (`channel_posterior`, checked against
`latentgoal.filter_joint`); the decision phase then depends only on that posterior and the outcome counts, so Q* is
computed per episode by value iteration over the 3 003 outcome-count states (`episode_tables`, checked against the
count graph).

Design probe (8 192 episodes; a random permutation of each episode's cues as its matched counterfactual):

| cues | P(stay) | hit when on | matched pairs: median \|Δb\|₁ | share ≥ 0.1 | share ≥ 0.2 | different optimal first action | counts' MAP ≠ channel MAP | entropy after the cues |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.95 (the hidden-goal experiment's) | 0.55 | 0.039 | 0.15 | 0.02 | 0.06 | 0.05 | 0.77 |
| **8** | **0.8** | **0.7** | **0.083** | **0.45** | **0.23** | **0.075** | 0.06 | 0.56 |
| 8 | 0.6 | 0.7 | 0.029 | 0.15 | 0.02 | 0.01 | 0.02 | 0.50 |

The chosen setting makes order matter in about half the pairs by 0.1 of belief mass and in a quarter by 0.2; the
optimal first action changes in 7.5 % of pairs. References on the evaluation episodes: optimal 4.97, myopic 4.95,
evidence-blind 3.4.

**Two arms**, six transformers each (the reward-bandit network, PPO settings and 600 updates):
* **channel**: trained on the channel cues;
* **iid**: trained on i.i.d. cues with the same 8-token length (the control: for it the counts are sufficient).

Both arms are measured on the **same channel-generated evaluation cues** and against the channel posterior. The i.i.d.
arm sees a cue distribution slightly unlike its training one; its role is the order-invariant reference. No GRU: it
did not learn the i.i.d. task.

## Measures

**A. The reward-bandit measures** (`../reward_bandit/measure.py`): behaviour, decodability, transplant. Its
equal-belief and node-table entries are not meaningful here (permuted histories no longer have equal beliefs, and
belief states are no longer shared across episodes) and are not reported.

**B. Matched-evidence counterfactuals** (`pairs.py`), at the first decision (cues only) of the evaluation episodes:
each episode's cues and a random permutation of them (pairs where the tokens differ), with the exact beliefs b, b′,
Δb = ‖b − b′‖₁, and the optimal first actions a*, a*′ from Q*.
* *Behaviour*: the JS divergence between the model's action distributions on the two orders; its Spearman
  correlation with Δb across pairs; on pairs where a* ≠ a*′, the share where the model's greedy action differs
  between the orders, and the share where it is right on both.
* *Representation*: at every site (res0 … res2), the belief decoded by the affine probe (fitted on the model's own
  decisions) from each order; the R² of Δb predicted from the decoded difference (a linear fit across pairs), and
  the Spearman correlation of their magnitudes. Where along the blocks the order-dependent belief first appears.
* Strata: all pairs; pairs with Δb ≥ 0.2.
* The same for the i.i.d. arm and for the untrained checkpoints (both expected near zero).

## Decision rule

Six models per arm; medians.

| | criterion |
|---|---|
| **LEARN** | channel arm: regret ≤ a quarter of the evidence-blind policy's |
| **ORDER-B** | channel arm: Spearman(JS, Δb) ≥ 0.5, and on pairs where the optimal first action differs the greedy action differs between the orders in ≥ 0.5; i.i.d. arm ≤ 0.1 on the latter |
| **ORDER-R** | channel arm, at mlp0: R² of Δb from the decoded difference ≥ 0.5; i.i.d. arm ≤ 0.1 |

| result | reading |
|---|---|
| ORDER-B and ORDER-R | **the MLP computes the belief, not a count lookup**: histories with the same counts and different posteriors are separated at its output and in the decision, in the Bayes direction |
| ORDER-B without ORDER-R at mlp0 (but ORDER-R at res2) | the order-dependent belief is assembled later than block 0's MLP |
| not ORDER-B | the channel arm acts on the counts: the order is learned away or not learned |
| not LEARN | the channel arm is not interpreted |

## Expectations

| | expectation |
|---|---|
| E1 | LEARN holds; the channel arm ends within 0.05 of the optimum, as the i.i.d. models did |
| E2 | ORDER-B holds: Spearman 0.5–0.8; the greedy action differs in 0.5–0.8 of the pairs where it should; the i.i.d. arm ≤ 0.1 |
| E3 | ORDER-R holds at res2 (R² ≥ 0.6) and in part at mlp0 (0.3–0.6): block 0's attention must first weight the cues by position and run, which takes both blocks |
| E4 | the order-dependent belief first appears at attn0 (R² of Δb ≥ 0.2 there, < 0.1 at res0), the i.i.d. arm and the untrained checkpoints < 0.1 at every site |
| E5 | the channel arm's permuted-history JS ratio (the reward-bandit measure) is above the i.i.d. arm's (0.09) |

## Limits known in advance

The i.i.d. arm is evaluated on channel-generated cues. One channel setting, chosen to make order matter; the effect
is still moderate (a quarter of the pairs differ by 0.2 of belief mass). The probes are affine; the counterfactual
pairs are random permutations, not the most diagnostic ones. Six models per arm.
