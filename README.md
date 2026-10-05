# goalgeo

Experiments on what small networks represent about a goal or a belief, and when that representation is the state the
network actually computes with. Each experiment states its predictions before training (`THEORY.md` / `PLAN.md`),
then reports which held (`REPORT.md`). [FINDINGS.md](FINDINGS.md) has every result on one page.

## The story

1. **[Representation geometry](studies/1_representation_geometry/).** A network's hidden geometry mirrors the
   distinctions its training target contains, and the size of a distinction is set by the output gap it must produce.
   Raw geometry and local metrics change between functionally equivalent models; decodability, readout contrast and
   interventions on a complete causal cut do not. Everything after this relies only on the invariant measures.
2. **[Belief state](studies/2_belief_state/).** With a goal that must be inferred, the exact posterior is decodable
   from every model, but only a recurrent network uses it as its state. A transformer recomputes the belief from the
   tokens, unless a carried state plus unreliable or priced access to the history make keeping it worthwhile.
3. **[Reward-trained agents](studies/3_reward_trained_agents/).** Trained by reward in a grid and an aliased maze, agents
   learn to use evidence, but the belief is decodable before training too. Editing the decoded belief does not steer
   the policy, and an observation-prediction objective improves regret without making the policy more belief-consistent.
4. **[Predictive pretraining](studies/4_predictive_pretraining/).** A backbone trained only to predict the maze is the
   best starting point for new goals, and its belief encoding controls a new head; the original policy recomputes the
   evidence from the raw tokens instead.
5. **[Goal × belief mechanism](studies/5_goal_belief_mechanism/).** Followed component by component, the maze policy reads a
   goal-free evidence estimate, receives the goal through one self-attention value, and decides, to a first
   approximation, additively: a goal-averaged usefulness profile of the history plus a fixed bias per goal.
6. **[Hard cases and harder tasks](studies/6_hard_cases_and_tasks/).** That additive code fails where two goals' biases
   point the wrong way at near-ties. Optimal-move supervision teaches the missing interaction and cuts regret 4×; a
   model-free screen and an exact multi-goal solver look for tasks that would force it under reward alone.
   Exploratory.

## Layout

```
studies/<study>/                one research thread; README.md lists its experiments in order
    <experiment>/               everything for one experiment:
        BRIEF.md                  the question as posed (none where it was given in conversation)
        THEORY.md / PLAN.md       design and predictions, committed before training
        REPORT.md                 results, with the scorecard of predictions
        tables.md, *.png, *.json  generated numbers, figures and per-model data; models/ checkpoints
        run.py, tables.py, …      training, measurement and the tables
goalgeo/                        shared library (environments, exact filters, models, probes, interventions)
reproduce.py                    the only entry point: a registry of every experiment's steps
tests/                          148 tests (exact identities, filter vs brute force, invariances)
FINDINGS.md                     every result, by study
```

## Reproduce

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv    # reuses system torch / numpy / scipy / matplotlib
uv pip install --python .venv/bin/python pytest

.venv/bin/python reproduce.py list                     # experiments by study: cost, dependencies, question
.venv/bin/python reproduce.py filter_state             # rerun one experiment end to end
.venv/bin/python reproduce.py filter_state --tables    # rebuild its tables and figures from the committed data (seconds)
.venv/bin/python reproduce.py all --quick              # smoke-run every experiment (minutes)
.venv/bin/python reproduce.py check                    # tests + rebuild every table from saved data (must match the
                                                       # committed ones) + smoke runs
```

Trained models are committed, so experiments that reuse another's models and every `--tables` rebuild run without
retraining.

## Earlier names

Until October 2026 the experiments were numbered rounds in `rounds/rNN_<name>/`; commit messages before then use those
numbers. Two lines of work both used 17: the read gate (now `read_cost`) and the navigate-and-commit agent
(`navigate_commit`).

| round | experiment | round | experiment | round | experiment |
|---|---|---|---|---|---|
| 1 | [occupancy](studies/1_representation_geometry/occupancy/) | 2 | [policy_quotient](studies/1_representation_geometry/policy_quotient/) | 3 | [supervision](studies/1_representation_geometry/supervision/) |
| 4 | [hmm_objectives](studies/1_representation_geometry/hmm_objectives/) | 5 | [prominence](studies/1_representation_geometry/prominence/) | 6 | [readout_scale](studies/1_representation_geometry/readout_scale/) |
| 7 | [allocation](studies/1_representation_geometry/allocation/) | 8 | [invariants](studies/1_representation_geometry/invariants/) | 9 | [interventions](studies/1_representation_geometry/interventions/) |
| 10 | [transformer](studies/1_representation_geometry/transformer/) | 11 | [implementation_freedom](studies/1_representation_geometry/implementation_freedom/) | 12 | [hidden_goal](studies/2_belief_state/hidden_goal/) |
| 13 | [filter_state](studies/2_belief_state/filter_state/) | 14 | [window_carry](studies/2_belief_state/window_carry/) | 15 | [prior_vs_recompute](studies/2_belief_state/prior_vs_recompute/) |
| 16 | [kv_dropout](studies/2_belief_state/kv_dropout/) | 17 (read gate) | [read_cost](studies/2_belief_state/read_cost/) | 17 (agent) | [navigate_commit](studies/3_reward_trained_agents/navigate_commit/) |
| 18 | [maze_belief](studies/3_reward_trained_agents/maze_belief/) | 19 | [maze_occupancy](studies/3_reward_trained_agents/maze_occupancy/) | 20 | [belief_edit](studies/3_reward_trained_agents/belief_edit/) |
| 21 | [pattern_specificity](studies/3_reward_trained_agents/pattern_specificity/) | 22 | [pair_types](studies/3_reward_trained_agents/pair_types/) | 23 | [obs_prediction](studies/3_reward_trained_agents/obs_prediction/) |
| 24 | [head_consistency](studies/3_reward_trained_agents/head_consistency/) | 25 | [balanced_prediction](studies/3_reward_trained_agents/balanced_prediction/) | 26 | [predictive_transfer](studies/4_predictive_pretraining/predictive_transfer/) |
| 27 | [belief_encoding_edit](studies/4_predictive_pretraining/belief_encoding_edit/) | 28 | [policy_belief_edit](studies/4_predictive_pretraining/policy_belief_edit/) | 29 | [goal_route_selection](studies/5_goal_belief_mechanism/goal_route_selection/) |
| 30 | [goal_swap_components](studies/5_goal_belief_mechanism/goal_swap_components/) | 31 | [cross_history_mlp](studies/5_goal_belief_mechanism/cross_history_mlp/) | 32 | [direct_belief_edit](studies/5_goal_belief_mechanism/direct_belief_edit/) |
| 33 | [attention_belief_edit](studies/5_goal_belief_mechanism/attention_belief_edit/) | 34 | [nonlinear_belief_edit](studies/5_goal_belief_mechanism/nonlinear_belief_edit/) | 35 | [block0_steps](studies/5_goal_belief_mechanism/block0_steps/) |
| 36 | [query_swap](studies/5_goal_belief_mechanism/query_swap/) | 37 | [self_value](studies/5_goal_belief_mechanism/self_value/) | 38 | [self_value_interaction](studies/5_goal_belief_mechanism/self_value_interaction/) |
| 39 | [mlp_bilinear](studies/5_goal_belief_mechanism/mlp_bilinear/) | 40 | [mlp_depth](studies/5_goal_belief_mechanism/mlp_depth/) | 41 | [interaction_removal](studies/5_goal_belief_mechanism/interaction_removal/) |
| 42 | [additive_code](studies/5_goal_belief_mechanism/additive_code/) | 43 | [what_is_H](studies/5_goal_belief_mechanism/what_is_H/) | 44 | [H_mixture](studies/5_goal_belief_mechanism/H_mixture/) |
| 45 | [hard_cases](studies/6_hard_cases_and_tasks/hard_cases/) |  |  |  |  |

## Library

| modules | used by | contents |
|---|---|---|
| `gridworld`, `envs`, `planning`, `occupancy` | occupancy-geometry to supervision-bottleneck | gridworlds, value iteration, exact occupancy / successor measures |
| `model`, `train`, `models2`, `targets` | occupancy-geometry to supervision-bottleneck | goal-conditioned MLPs and variants, behavioural cloning, supervision targets |
| `geometry`, `analysis`, `quotient`, `supervision` | occupancy-geometry to supervision-bottleneck | RDMs / RSA / CKA / decoding, and each experiment's analyses |
| `hmm`, `hmm4`, `seqmodels`, `prominence` | HMM-objectives to intervention-equivalence | HMMs with exact inference, GRUs and objectives, prominence measures |
| `invariants`, `steering` | invariants to intervention-equivalence | invariant measures under exact coordinate changes, propagation-based steering |
| `tfm`, `tfm_batched`, `tfm_measure`, `cuts`, `factorize` | transformer-reproduction to read-cost | causal transformer, stacked GPU trainer, complete-cut patching, readout factorisation |
| `latentgoal`, `belief_train`, `beliefprobe`, `beliefcausal`, `filterstate` | hidden-goal to window-carry | hidden-goal environments with the exact joint filter; training; probes; transplant / equivalence tests; full-state coordinates |
| `wtfm`, `kvprior`, `readgate` | window-carry to read-cost | windowed transformer with a recurrent carry, K/V dropout and a priced read gate; K/V-source splicing for position t+1 |
| `navcommit`, `navmodel`, `navppo`, `navbank`, `navprobe`, `navcausal` | navigate-commit | navigate / investigate / commit: exact solver, token format and transformer, vectorised environment and PPO, fixed evaluation histories, decoders, matched-pair patches |
| `mazeedit` | belief-edit to pair-types | edits of prefix-token states in decoder-defined and covariance-defined subspaces |
| `mazepred` | predictive-transfer to belief-encoding-edit | prediction-only backbone (k-step heads, random-walk targets), exact k-step predictions, small goal-conditioned heads trained side by side on frozen features |
| `mazeaux` | observation-prediction to balanced-prediction | next-symbol prediction head, its loss against the exact predictive distribution, PPO update with the auxiliary term |
| `mazeocc` | maze-occupancy | exact occupancy under the solver's policy; a model's own occupancy by rollouts |
| `multigoal` | hard-cases | multi-goal collection with random values: exact belief graph over (moves left, collected set, belief), Q* for every value setting |
| `mazebelief`, `mazegraph`, `mazemodel`, `mazeppo`, `mazemeasure` | maze-belief to maze-occupancy | aliased maze with a hidden location: exact filter and solver, belief graph, tokens and transformer, vectorised environment and PPO, decoders and cross-goal patches |
| `plotting`, `style` | all | shared figure style (the representation geometry study, the hidden-goal to K/V-dropout experiments) |
