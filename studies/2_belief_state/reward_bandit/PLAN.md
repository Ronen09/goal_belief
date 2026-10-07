# Reward bandit: task, measures, decision rule and expectations

Written before any model of this experiment was trained. **Disclosure:** the exact solver's references and a design
probe (no network) came first and are given below; a learnability pilot (2 seeds per architecture, 300 then 600
updates, three PPO settings, returns and regrets only) chose the training settings; one pilot model per architecture
was run through `measure.py` to check the code, which also showed that with the brief's exact 0 and 1 reward
probabilities half of all decisions have a posterior with a zero entry (an outcome rules a goal out), which decided the
clipping below. All pilot models are discarded. Brief: `BRIEF.md`. Code: `goalgeo/bandit.py` (tests in
`tests/test_bandit.py`: the count filter against sequential Bayes, Q* against brute-force expectimax), `train.py`,
`measure.py`, `tables.py`.

## Task

A goal G ~ uniform over K = 3 is hidden. The agent sees **4 cue tokens** o ~ L[G] (the hidden-goal experiment's i.i.d.
emission: 5 symbols, token k has probability 0.30 under goal k, 0.20 under goal k − 1, and the rest share the remainder),
then makes **6 decisions** among A = 4 actions. Action a pays a Bernoulli reward with probability R[G, a], and the
reward is shown, so it is evidence too:

```
R = clip( (1, .6, .1, 0), 0.1, 0.9 )   = (.9, .6, .1, .1)      goal 1
          (.1, 1, .4, 0)               = (.1, .9, .4, .1)      goal 2
          (0, .2, .7, 1)               = (.1, .2, .7, .9)      goal 3
```

The brief's vectors, with every probability **clipped to [0.1, 0.9]**: with the exact 0s and 1s a single outcome can
rule a goal out, after which the posterior has zero entries, its log-odds are undefined, and the belief is trivial
in half of all decisions. Clipping keeps the ordering of every row and column; no outcome is conclusive. Return = the
sum of the 6 rewards (no discount). Tokens: `[BOS] [CUE o]×4 [EVT a r]×5`; the next action is predicted at the last cue
token and at every event token. The goal, the posterior and the solver's values are never inputs.

**Exact ground truth.** Every likelihood is a product of independent draws, so the posterior depends on the history
only through its counts (cue counts, and per action the numbers of successes and failures). The belief MDP therefore
has 210 210 states, and value iteration over them gives Q* exactly, the value of information included. References
on the 16 384 evaluation episodes (the same goals, cues and reward draws for every policy: the reward of (episode,
step, action) is decided by one uniform draw):

| | optimal (argmax Q*) | myopic (argmax_a Σ_g b(g) R[g, a]) | evidence-blind (the best fixed action, a2) | random |
|---|---|---|---|---|
| return | 4.552 | 4.470 | 3.392 | 2.590 |
| regret | 0 | 0.067 | 1.149 | 1.977 |

Design probe: the value of information is 0.07–0.08 per episode (1.5 % of the return); the optimal policy takes a
non-myopic action in 12 % of its decisions (40 % of first decisions, 13 % of second, 11 % of third), gaining 0.09 of
Q* on average there. The posterior's most likely goal is right at the last decision in 0.90 of episodes.

**Models.** A 2-layer pre-LN transformer (width 128, 4 heads; sites res0 = the embedding, res1, res2 = the final
residual the heads read) and a 1-layer GRU (width 128; site h = its state), the same embedding and heads. PPO on reward
only (the maze experiments' update, no discount), 4 096 episodes per update, **600 updates**, learning rate 3e-4,
entropy bonus 0.03 falling to 0.003 (the pilot's best of three settings for the transformer; the GRU ended at the same regret, 0.29, under every setting and with 1 000 updates or a learning rate of 1e-3). Six seeds per architecture.

## Measures (`measure.py`), on each model's own greedy episodes

**A. Behaviour.** Return, exact regret, the share of optimal actions; where information pays (the myopic action is
not optimal): the share of optimal and of myopic actions.

**B. Decodability of the posterior** (`goalgeo/beliefprobe.py`'s unregularised affine probes), at every site, at
decision positions: the log-odds y (2 numbers) and the probabilities b, 5-fold by episode (IID); the extrapolation
split EXT (fit on decisions after the first with every |log-odds| < 2 nats, test on those with one ≥ 3; the first
decision is left out because all its states lie in the fit region); V* and Q*; the optimal action; y within each
optimal-action class (the ladder of the hidden-goal causal study). The same for the untrained checkpoint of each
seed. With i.i.d. evidence the log-odds are exactly affine in the counts, so IID decodability is expected to be
near 1 in any network that can sum token embeddings, and so is EXT for y (an affine function of the counts extrapolates); EXT for b, which is not affine in the counts, against the untrained reference is the discriminating number.

**C. Does the policy act on the belief alone?**
* **Equal-belief pairs**: every history's cue tokens and event tokens are permuted at random (the counts, hence the
  belief and the Bayes-optimal action, are unchanged). The JS divergence between the model's action distributions on
  the original and the permuted history, against random pairs of histories at the same step; the ratio; the share of
  pairs where the greedy action changes.
* **Belief-node table**: the share of the centred logits' variance explained by a table over belief states (the
  rest is dependence on the history beyond the belief).

**D. Transplant along the probe at the state the heads read** (res2 / h; `goalgeo/beliefcausal.BeliefMap`). The
decoder and its reverse-regression encoder are fitted to y. For pairs (A, B) of decisions at the same step, A's state
is moved to B's log-odds along the encoder (probe_e), along the decoder's pseudo-inverse (probe_d), or by a random
direction of the same length; the heads' action distribution is compared with the model's own on B. Gap closed =
1 − JS(edited, B) / JS(A, B). For the transformer this is a complete cut for the immediate decision (nothing else reads
that position's final residual); the future is not re-run in either architecture.

## Decision rule

Six models per architecture; medians; exact one-sided Wilcoxon signed-rank tests against a reference's value, p < 0.05.

| | criterion |
|---|---|
| **LEARN** | regret ≤ a quarter of the evidence-blind policy's (0.287; the pilot's GRUs sit at 0.289–0.291, so this is a coin flip for them) |
| **SEEK** | return > the myopic policy's (4.470): the agent collects what only the value of information can give |
| **BELIEF** | final site: EXT R² of b (the probabilities, in which Q is linear) ≥ 0.9 |
| **STATE** | equal-belief JS ratio ≤ 0.1 |
| **TABLE** | belief-node table R² ≥ 0.95 |
| **STEER** | probe_e gap closed ≥ 0.8, with the random direction ≤ 0.2 |

| result | reading |
|---|---|
| BELIEF, STATE, TABLE and STEER | **reward alone produced a Bayesian belief and the policy acts through it**: the posterior is linearly available beyond the fit region, the policy depends on the history only through it, and moving the decoded belief moves the decision |
| STATE and TABLE without STEER | the policy is a function of the belief, but the decoded directions are not the ones the heads use (a nonlinear or distributed code) |
| BELIEF without STATE | the belief is decodable, but the policy acts on history beyond it (order effects) |
| SEEK | the agent seeks information; not SEEK: it is at most myopic |
| not LEARN (an architecture) | that architecture is not interpreted |

## Expectations

| | expectation |
|---|---|
| E1 | LEARN holds for the transformer (pilot regret 0.004–0.04) and fails, narrowly, for the GRU (0.29); the GRU is between the evidence-blind and the myopic policy |
| E2 | **SEEK holds for the transformer**: it ends within 0.03 of the optimum, above the myopic policy, and takes the optimal action where information pays in ≥ 0.5 of those decisions (pilot 0.53–0.93). SEEK fails for the GRU (pilot: 0.05–0.09 there) |
| E3 | IID R² ≥ 0.98 at the final site in both, and already ≥ 0.95 untrained; EXT R² of y is high untrained too (0.99 in the pilot's untrained transformer: y is affine in the counts); EXT R² of b separates them: trained ≥ 0.9, untrained < 0.5 (pilot: −2.9 transformer, 0.09 GRU); BELIEF holds. The interface is in probabilities, as the hidden-goal experiment's action-trained models' was |
| E4 | STATE holds in both; the GRU's ratio is below the transformer's (positions make permutation invariance a thing to learn) |
| E5 | TABLE holds in both |
| E6 | STEER fails in both by its threshold: probe_e closes 0.4–0.75 of the gap (the unclipped pilot's models: 0.62 transformer, 0.44 GRU), probe_d ≈ 0, random ≤ 0 |
| E7 | the ladder: within-action R² of y stays ≥ 0.95 at every site (no quotient by the action, as the value head needs the belief too) |

## Limits known in advance

One task, one reward matrix; i.i.d. cues, so counts are a sufficient statistic and IID decodability is not
diagnostic. The transplant moves one position's state and does not re-run the future. The equal-belief test permutes
tokens the network has seen in training-typical arrangements. Six models per architecture; one hyperparameter setting
chosen by the pilot.
