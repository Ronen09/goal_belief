# goalgeo

A sequence of pre-registered experiments on what small networks represent about a goal or a belief,
and when that representation is the state the network actually computes with. It starts from goal-conditioned
occupancy geometry (round 1) and ends with when a sufficient statistic becomes the causal computational
state of a transformer (round 17). `SUMMARY.md` gives every round's results on one page.

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
tests/                130 tests (exact identities, filter vs brute force, invariances)
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
| r17 | Does a price on reading the history induce a selective, recurrent belief state? | [report](rounds/r17_read_cost/REPORT.md) | 3 h |

## Library

| modules | used by | contents |
|---|---|---|
| `gridworld`, `envs`, `planning`, `occupancy` | r01–r03 | gridworlds, value iteration, exact occupancy / successor measures |
| `model`, `train`, `models2`, `targets` | r01–r03 | goal-conditioned MLPs and variants, behavioural cloning, supervision targets |
| `geometry`, `analysis`, `quotient`, `supervision` | r01–r03 | RDMs / RSA / CKA / decoding, and each round's analyses |
| `hmm`, `hmm4`, `seqmodels`, `prominence` | r04–r09 | HMMs with exact inference, GRUs and objectives, prominence measures |
| `invariants`, `steering` | r08–r09 | invariant measures under exact coordinate changes, propagation-based steering |
| `tfm`, `tfm_batched`, `tfm_measure`, `cuts`, `factorize` | r10–r17 | causal transformer, stacked GPU trainer, complete-cut patching, readout factorisation |
| `latentgoal`, `belief_train`, `beliefprobe`, `beliefcausal`, `filterstate` | r12–r14 | hidden-goal environments with the exact joint filter; training; probes; transplant / equivalence tests; full-state coordinates |
| `wtfm`, `kvprior`, `readgate` | r14–r17 | windowed transformer with a recurrent carry, K/V dropout and a priced read gate; K/V-source splicing for position t+1 |
| `plotting`, `style` | all | shared figure style (rounds 1–11, rounds 12–17) |
