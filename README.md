# goalgeo

A sequence of pre-registered experiments on what small networks represent about a goal or a belief,
and when that representation is the state the network actually computes with. It starts from goal-conditioned
occupancy geometry (round 1) and continues through when a sufficient statistic becomes the causal computational
state of a transformer (round 16) to how a reward-trained transformer comes to use a belief (round 17) and whether a prediction-trained one can be reused for control (round 26). `SUMMARY.md` gives every round's results on one page.

## Reproduce

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv    # reuses system torch / numpy / scipy / matplotlib
uv pip install --python .venv/bin/python pytest

.venv/bin/python reproduce.py list            # rounds, cost, dependencies
.venv/bin/python reproduce.py r13             # rerun one round end to end
.venv/bin/python reproduce.py r13 --tables    # rebuild its tables and figures from the committed data (seconds)
.venv/bin/python reproduce.py all --quick     # smoke-run every round (minutes)
.venv/bin/python reproduce.py check           # tests + rebuild every table from saved data (must match the
                                              # committed ones) + smoke runs
```

Trained models are committed, so rounds that reuse another round's models (r09, r13, r14) and every
`--tables` rebuild run without retraining.

## Layout

```
reproduce.py          the only entry point: a registry of every round's steps
goalgeo/              shared library (environments, exact filters, models, probes, interventions)
rounds/rNN_<name>/    everything for one round:
    BRIEF.md            the question as posed (rounds whose brief was given in conversation have none)
    DESIGN.md / THEORY.md   design and pre-registered predictions, committed before training
    REPORT.md           results, with the scorecard of predictions
    tables.md           generated numbers;  *.png figures;  *.json per-model data;  models/ checkpoints
    run.py              training + measurement;  followup.py / tables.py / … further steps
    plots.py            the round's figures
tests/                114 tests (exact identities, filter vs brute force, invariances)
```

## Rounds

| round | question | report | cost |
|---|---|---|---|
| r01 | Does a goal-conditioned policy learn goal-occupancy geometry? | [report](rounds/r01_occupancy/REPORT.md) | 6 min CPU |
| r02 | Policy quotient vs occupancy geometry; soft targets | [report](rounds/r02_policy_quotient/REPORT.md) | 15 min CPU |
| r03 | The supervision bottleneck: what the target keeps | [report](rounds/r03_supervision/REPORT.md) | 18 min CPU |
| r04 | One-step vs k-step vs sequential objectives on an HMM | [report](rounds/r04_hmm_objectives/REPORT.md) | 5 min |
| r05 | What makes information geometrically prominent? | [report](rounds/r05_prominence/REPORT.md) | 20 min, 8 workers |
| r06 | Readout gain vs hidden separation | [report](rounds/r06_readout_scale/REPORT.md) | 8 min |
| r07 | What sets the gain / separation split | [report](rounds/r07_allocation/REPORT.md) | 6 min |
| r08 | Which representation measures are invariant | [report](rounds/r08_invariants/REPORT.md) | 13 min |
| r09 | Intervention equivalence and local metrics | [report](rounds/r09_interventions/REPORT.md) | 5 min |
| r10 | Transformer reproduction of rounds 5–9 | [report](rounds/r10_transformer/REPORT.md) | 10 min GPU |
| r11 | Cut identifiability and factorisation freedom | [report](rounds/r11_implementation_freedom/REPORT.md) | 1.5 h GPU |
| r12 | A hidden goal: is the posterior recoverable, and is it the causal state? | [report](rounds/r12_hidden_goal/REPORT.md), [causal](rounds/r12_hidden_goal/causal/REPORT.md) | 45 min |
| r13 | Is the recurrent state the environment's minimal predictive state? | [report](rounds/r13_filter_state/REPORT.md) | 7 min |
| r14 | Does a narrow attention window force a steerable belief state? | [report](rounds/r14_window_carry/REPORT.md) | 31 min |
| r15 | Does a next-token transformer use its previous belief as a prior? | [report](rounds/r15_prior_vs_recompute/REPORT.md) | 11 min GPU |
| r16 | Can K/V dropout induce recurrence continuously? | [report](rounds/r16_kv_dropout/REPORT.md) | 1 h |
| r17 | How does a reward-trained transformer come to build and use a belief? | [report](rounds/r17_navigate_commit/REPORT.md) | 4 h GPU |
| r18 | Does an inferred location belief support goal-dependent decisions? | [report](rounds/r18_maze_belief/REPORT.md) | 1.5 h GPU |
| r19 | Is goal-conditioned occupancy represented beyond the posterior and the action values? | [report](rounds/r19_occupancy/REPORT.md) | 25 min GPU |
| r20 | Does the policy use the decoded belief? | [report](rounds/r20_belief_edit/REPORT.md) | 20 min GPU |
| r21 | Is the posterior-associated edit more specific than replacing the dominant history representation? | [report](rounds/r21_pattern_specificity/REPORT.md) | 2 min GPU |
| r22 | Does history matter beyond the belief, and is evidence kept beyond one goal's action? | [report](rounds/r22_pair_types/REPORT.md) | 2 min GPU |
| r23 | Does an observation-prediction objective reduce history dependence, incorrect action changes and regret? | [report](rounds/r23_obs_prediction/REPORT.md) | 2.5 h GPU |
| r24 | Does the prediction head agree on identical-posterior histories? | [report](rounds/r24_head_consistency/REPORT.md) | 1 min GPU |
| r25 | Does balanced candidate-action supervision make predictions, and then the policy, more belief-consistent? | [report](rounds/r25_balanced_prediction/REPORT.md) | 1.5 h GPU |
| r26 | Does learning to predict the maze give a representation from which different goals are solved efficiently? | [report](rounds/r26_predictive_transfer/REPORT.md) | 30 min GPU |
| r27 | Can the transferable representation support a selective causal belief edit? | [report](rounds/r27_belief_encoding_edit/REPORT.md) | 5 min GPU |
| r28 | Does the same belief-associated change control the new head and the original policy? | [report](rounds/r28_policy_belief_edit/REPORT.md) | 3 min GPU |
| r29 | Does the goal change which evidence route the policy relies on? | [report](rounds/r29_goal_route_selection/REPORT.md) | 2 min GPU |
| r30 | With the history fixed, which components after the interface carry a goal swap? | [report](rounds/r30_goal_swap_components/REPORT.md) | 4 min GPU |
| r31 | Do the goal token's MLPs carry a goal instruction, an evidence–goal combination, or an action preference? | [report](rounds/r31_cross_history_mlp/REPORT.md) | 1 min GPU |
| r32 | Does a shared belief edit at the goal token's state after block 0 transfer donor behaviour across goals? | [report](rounds/r32_direct_belief_edit/REPORT.md) | 2 min GPU |
| r33 | Before block 0's MLP, is the belief map shared across goals, and does a shared edit control the policy? | [report](rounds/r33_attention_belief_edit/REPORT.md) | 2 min GPU |
| r34 | Does a nonlinear encoding of the posterior close the gap on goal-dependent decisions? | [report](rounds/r34_nonlinear_belief_edit/REPORT.md) | 6 min GPU |
| r35 | Is the goal-conditioned belief already in block 0's attention output, and does anything change it before the MLP? | [report](rounds/r35_block0_steps/REPORT.md) | 2 min GPU |
| r36 | Does the goal-dependent belief part move with block 0's query, and does the decision depend on it? | [report](rounds/r36_query_swap/REPORT.md) | 1 min GPU |
| r37 | Which part of the goal token's block-0 self-attention carries the goal's identity: value or key, which head? | [report](rounds/r37_self_value/REPORT.md) | 1 min GPU |
| r38 | Does swapping the goal's self value move the goal × belief interaction at the MLP input, with the shared belief fixed? | [report](rounds/r38_self_value_interaction/REPORT.md) | 20 min GPU |

## Library

| modules | used by | contents |
|---|---|---|
| `gridworld`, `envs`, `planning`, `occupancy` | r01–r03 | gridworlds, value iteration, exact occupancy / successor measures |
| `model`, `train`, `models2`, `targets` | r01–r03 | goal-conditioned MLPs and variants, behavioural cloning, supervision targets |
| `geometry`, `analysis`, `quotient`, `supervision` | r01–r03 | RDMs / RSA / CKA / decoding, and each round's analyses |
| `hmm`, `hmm4`, `seqmodels`, `prominence` | r04–r09 | HMMs with exact inference, GRUs and objectives, prominence measures |
| `invariants`, `steering` | r08–r09 | invariant measures under exact coordinate changes, propagation-based steering |
| `tfm`, `tfm_batched`, `tfm_measure`, `cuts`, `factorize` | r10–r16 | causal transformer, stacked GPU trainer, complete-cut patching, readout factorisation |
| `latentgoal`, `belief_train`, `beliefprobe`, `beliefcausal`, `filterstate` | r12–r14 | hidden-goal environments with the exact joint filter; training; probes; transplant / equivalence tests; full-state coordinates |
| `wtfm`, `kvprior` | r14–r16 | windowed transformer with a recurrent carry and K/V dropout; K/V-source splicing for position t+1 |
| `navcommit`, `navmodel`, `navppo`, `navbank`, `navprobe`, `navcausal` | r17 | navigate / investigate / commit: exact solver, token format and transformer, vectorised environment and PPO, fixed evaluation histories, decoders, matched-pair patches |
| `mazeedit` | r20–r22 | edits of prefix-token states in decoder-defined and covariance-defined subspaces |
| `mazepred` | r26–r27 | prediction-only backbone (k-step heads, random-walk targets), exact k-step predictions, small goal-conditioned heads trained side by side on frozen features |
| `mazeaux` | r23–r25 | next-symbol prediction head, its loss against the exact predictive distribution, PPO update with the auxiliary term |
| `mazeocc` | r19 | exact occupancy under the solver's policy; a model's own occupancy by rollouts |
| `mazebelief`, `mazegraph`, `mazemodel`, `mazeppo`, `mazemeasure` | r18–r19 | aliased maze with a hidden location: exact filter and solver, belief graph, tokens and transformer, vectorised environment and PPO, decoders and cross-goal patches |
| `plotting`, `style` | all | shared figure style (rounds 1–11, rounds 12–16) |
