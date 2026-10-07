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
   tokens, unless a carried state plus unreliable or priced access to the history make keeping it worthwhile. Trained
   by reward alone on a hidden-goal bandit, a transformer plays near the exact optimum, acts on the belief and nothing
   else, and comes to hold the posterior's probabilities, the form its decision needs: block 0's MLP makes that code
   from counts that attention gathers for free, and a decision affine in the log-odds costs three times the regret;
   a GRU keeps the log-odds and stops short of the myopic policy.
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
   Exploratory. With spawns anywhere in the
   small maze the policy stays additive and loses accuracy where the task stops being additive.
7. **[Information seeking in a larger maze](studies/7_information_seeking/).** In a 55-cell maze with no solver, no
   passive prefix and random spawns, reward-trained agents reach the level of QMDP and localise faster than it does.
   The additive policy, run in the environment, keeps four fifths of the goal-directed return; the goal × history
   interaction carries the rest. Moving the goals to interior junctions changes little (three quarters); collecting
   two goals per episode, or drawing the goal from every cell, lowers it to two thirds, although the fully observed
   task is as additive as ever; with a goal from anywhere the agents pass QMDP.

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


## Library

| modules | used by | contents |
|---|---|---|
| `gridworld`, `envs`, `planning`, `occupancy` | occupancy-geometry to supervision-bottleneck | gridworlds, value iteration, exact occupancy / successor measures |
| `model`, `train`, `models2`, `targets` | occupancy-geometry to supervision-bottleneck | goal-conditioned MLPs and variants, behavioural cloning, supervision targets |
| `geometry`, `analysis`, `quotient`, `supervision` | occupancy-geometry to supervision-bottleneck | RDMs / RSA / CKA / decoding, and each experiment's analyses |
| `hmm`, `hmm4`, `seqmodels`, `prominence` | HMM-objectives to intervention-equivalence | HMMs with exact inference, GRUs and objectives, prominence measures |
| `invariants`, `steering` | invariants to intervention-equivalence | invariant measures under exact coordinate changes, propagation-based steering |
| `tfm`, `tfm_batched`, `tfm_measure`, `cuts`, `factorize` | transformer-reproduction to read-cost | causal transformer, stacked GPU trainer, complete-cut patching, readout factorisation |
| `bandit` | reward-bandit | a hidden goal with cues and Bernoulli rewards as evidence: count-based exact filter, exact belief-MDP solver, GPU environment with common random numbers, transformer and GRU policies, PPO |
| `latentgoal`, `belief_train`, `beliefprobe`, `beliefcausal`, `filterstate` | hidden-goal to window-carry | hidden-goal environments with the exact joint filter; training; probes; transplant / equivalence tests; full-state coordinates |
| `wtfm`, `kvprior`, `readgate` | window-carry to read-cost | windowed transformer with a recurrent carry, K/V dropout and a priced read gate; K/V-source splicing for position t+1 |
| `navcommit`, `navmodel`, `navppo`, `navbank`, `navprobe`, `navcausal` | navigate-commit | navigate / investigate / commit: exact solver, token format and transformer, vectorised environment and PPO, fixed evaluation histories, decoders, matched-pair patches |
| `mazeedit` | belief-edit to pair-types | edits of prefix-token states in decoder-defined and covariance-defined subspaces |
| `mazepred` | predictive-transfer to belief-encoding-edit | prediction-only backbone (k-step heads, random-walk targets), exact k-step predictions, small goal-conditioned heads trained side by side on frozen features |
| `mazeaux` | observation-prediction to balanced-prediction | next-symbol prediction head, its loss against the exact predictive distribution, PPO update with the auxiliary term |
| `bigcollect` | interior-goals | two goals to collect per episode in the larger maze: exact filter with announced pickups, fully observed values by value iteration, reference policies |
| `bigmaze` | maze10, interior-goals | a 55-cell aliased maze without a solver: layout, vectorised environment with the exact filter, reference policies (oracle, QMDP, QMDP with lookahead, most likely cell), expected information gain |
| `mazeocc` | maze-occupancy | exact occupancy under the solver's policy; a model's own occupancy by rollouts |
| `multigoal` | hard-cases | multi-goal collection with random values: exact belief graph over (moves left, collected set, belief), Q* for every value setting |
| `mazebelief`, `mazegraph`, `mazemodel`, `mazeppo`, `mazemeasure` | maze-belief to maze-occupancy | aliased maze with a hidden location: exact filter and solver, belief graph, tokens and transformer, vectorised environment and PPO, decoders and cross-goal patches |
| `plotting`, `style` | all | shared figure style (the representation geometry study, the hidden-goal to K/V-dropout experiments) |
