# goalgeo — does a goal-conditioned agent learn occupancy geometry?

Code and results for the research programme in `TASK.md`: derive the
goal-conditioned discounted occupancy geometry of a small gridworld exactly,
train a network by behavioural cloning to act optimally toward a supplied
goal, and test whether the network's hidden geometry follows occupancy
geometry, especially where it conflicts with spatial geometry.

Two rounds of experiments:

| round | task | design spec | run script | results |
|---|---|---|---|---|
| 1 | `TASK.md` (Phases 1–9) | `docs/superpowers/specs/2026-09-17-goal-occupancy-geometry-design.md` | `scripts/run_all.py` | `results/REPORT.md`, `results/tables.md` |
| 2 | `TASK2.md` (Tasks 1–10: policy quotient vs occupancy) | `docs/superpowers/specs/2026-09-17-policy-quotient-design.md` | `scripts/run_task2.py` | `results2/REPORT2.md`, `results2/tables2.md`, `results2/boltzmann/` |
| 3 | `TASK3.md` (Tasks 1–14: the supervision bottleneck) | `docs/superpowers/specs/2026-09-17-supervision-bottleneck-design.md`, theory in `docs/task12_theory.md` | `scripts/run_task3.py` | `results3/REPORT3.md`, `results3/tables3.md` |
| 4 | HMM: one-step vs k-step vs sequential objectives on one-step-equivalent beliefs | `docs/superpowers/specs/2026-09-17-hmm-objectives-design.md` | `scripts/run_hmm.py` (`--device cuda`) | `results_hmm/REPORT_HMM.md`, `results_hmm/tables_hmm.md` |
| 5 | `TASK4.md` (what makes information geometrically prominent: frequency, strength, loss-weight and delay sweeps) | `docs/superpowers/specs/2026-09-18-geometric-prominence-design.md` | `scripts/run_task4.py --jobs 8 --grid --dynamics` | `results4/REPORT4.md`, `results4/tables4.md` |
| 6 | TASK5 (readout scale: fixed-gain readout, does hidden separation compensate as 1/‖w‖?) | design in `results5/REPORT5.md` | `scripts/run_task5.py --jobs 8` | `results5/REPORT5.md`, `results5/tables5.md` |
| 7 | TASK6 (allocation: readout lr ratio, readout init scale, control contrast) | design in `results6/REPORT6.md` | `scripts/run_task6.py --jobs 8` | `results6/REPORT6.md`, `results6/tables6.md` |
| 8 | TASK7 (which representation measures are invariant to function-preserving coordinate changes and optimisation history) | `docs/superpowers/specs/2026-09-18-representation-invariants-design.md`, theory in `docs/task7_theory.md` | `scripts/run_task7.py --jobs 8`, then `scripts/task7_followup.py` | `results7/REPORT7.md`, `results7/tables7.md`, `results7/models/` |
| 9 | TASK8 (intervention equivalence, on/off-manifold local metrics, propagation depth on the TASK7 models) | `docs/task7_theory.md` §4 | `scripts/run_task8.py --jobs 6` | `results8/REPORT8.md`, `results8/tables8.md` |
| 10 | TASK9 (transformer reproduction: 2-layer pre-LN causal decoder, with / without final LayerNorm; core dissociation, readout-scale allocation, invariance CVs, patching depth) | `results9/REPORT9.md` | `scripts/run_task9.py --jobs 8` | `results9/REPORT9.md`, `results9/tables9.md`, `results9/models/` |
| 11 | TASK10 (implementation freedom: a cut-identifiability claim and a factorisation-freedom claim, each with an experiment designed to falsify it) | theory and pre-registered predictions in `docs/task10_theory.md` | `scripts/run_task10.py --exp both --jobs 12`, then `scripts/task10_followup.py` | `results10/REPORT10.md`, `results10/tables10.md`, `results10/models/` |
| 12 | `TASK11.md` (hidden goal: is the exact Bayesian posterior over a latent goal affinely recoverable, out of distribution? claim 1 of the brief) | theory and pre-registered predictions in `docs/task11_theory.md` | `scripts/run_task11.py`, then `scripts/task11_followup.py`, `scripts/task11_tables.py` | `results11/REPORT11.md`, `results11/tables11.md`, `results11/models/` |
| 12b | `TASK11.md` claim 2 (is the decoded posterior the causal state? posterior transplant, equal-belief equivalence, same action / different belief) | cut analysis and pre-registered predictions in `docs/task11_causal_theory.md` | `scripts/run_task11_causal.py`, then `scripts/task11_causal_tables.py` (uses `results11/models/`) | `results11/causal/REPORT_causal.md`, `results11/causal/tables_causal.md` |
| 13 | `TASK12.md` (the full filter state: is a recurrent network's future a function of the environment's minimal predictive state? full-state decoding, matched two-block interventions, an equivalence hierarchy, a hidden-size bottleneck) | minimal-statistic analysis and pre-registered predictions in `docs/task12_filter_theory.md` | `scripts/run_task12.py`, then `scripts/task12_tables.py` (uses `results11/models/`) | `results12/REPORT12.md`, `results12/tables12.md`, `results12/models/` |
| 14 | TASK13 (windowed transformers: does restricting attention to the last l tokens force a steerable belief state? with and without a recurrent carry of the readout interface) | predictions in `docs/task13_window_theory.md` | `scripts/run_task13.py`, then `scripts/task13_tables.py` | `results13/REPORT13.md`, `results13/tables13.md`, `results13/models/` |
| 15 | `TASK14.md` (prior or recomputation: does a next-token transformer's position t+1 use the belief exported through position t's K/V as a prior? 2×2 of edited prior / edited evidence) | predictions in `docs/task14_prior_theory.md` | `scripts/run_task14.py`, `scripts/task14_recompute.py`, then `scripts/task14_tables.py` | `results14/REPORT14.md`, `results14/tables14.md`, `results14/models/` |
| 16 | `TASK15.md` (inducing recurrence by incentive: random historical K/V dropout during training, plain and carry transformers; does the prior's weight vary continuously with the cost of recomputation?) | predictions in `docs/task15_incentive_theory.md` | `scripts/run_task15.py` (+ `--out results15/fine --ps 0.05 0.1 0.2 0.3 --plain-ps 0.1 0.25`), then `scripts/task15_tables.py` | `results15/REPORT15.md`, `results15/tables15.md`, `results15/models/` |

## Setup

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv   # reuses system torch/numpy/scipy/matplotlib
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest            # 114 tests, ~70 s
.venv/bin/python scripts/run_all.py     # round 1, ~6 min on CPU -> results/
.venv/bin/python scripts/run_task2.py   # round 2, ~8 min -> results2/
.venv/bin/python scripts/run_task2.py --targets boltzmann --sweep-only --out results2/boltzmann   # soft-target sweep
.venv/bin/python scripts/run_task3.py   # round 3, ~18 min -> results3/
.venv/bin/python scripts/run_hmm.py --device cuda   # round 4, ~5 min -> results_hmm/ (GPU and CPU take about the same time)
.venv/bin/python scripts/run_task4.py --jobs 8 --grid --dynamics   # round 5, 105 runs + 12 dense-checkpoint runs, ~20 min on 8 CPU workers -> results4/
.venv/bin/python scripts/run_task5.py --jobs 8     # round 6, 42 runs, ~8 min -> results5/
.venv/bin/python scripts/run_task6.py --jobs 8     # round 7, 30 runs, ~6 min -> results6/
.venv/bin/python scripts/run_task7.py --jobs 8 && .venv/bin/python scripts/task7_followup.py   # round 8, 64 runs, ~11 min + 2 min -> results7/
.venv/bin/python scripts/run_task8.py --jobs 6     # round 9, needs results7/models; 6 new delay-4 models, ~5 min -> results8/
.venv/bin/python scripts/run_task9.py --jobs 4 && .venv/bin/python scripts/task9_followup.py   # round 10, 60 transformer runs on the GPU (~10 min; --only/--steps redo seeds) -> results9/
.venv/bin/python scripts/run_task10.py --exp both --jobs 12 && .venv/bin/python scripts/task10_followup.py   # round 11, 122 transformer runs on the GPU (~1.5 h) -> results10/
.venv/bin/python scripts/run_task11.py && .venv/bin/python scripts/task11_followup.py && .venv/bin/python scripts/task11_tables.py   # round 12, 160 + 14 models (~40 min, GPU + 96 CPU workers) -> results11/
.venv/bin/python scripts/run_task11_causal.py && .venv/bin/python scripts/task11_causal_tables.py   # round 12 part 2, causal tests on the round-12 models (~6 min) -> results11/causal/
.venv/bin/python scripts/run_task12.py && .venv/bin/python scripts/task12_tables.py   # round 13, 60 bottleneck GRUs + measurements (~7 min, 96 CPU workers) -> results12/
.venv/bin/python scripts/run_task13.py && .venv/bin/python scripts/task13_tables.py   # round 14, 60 windowed transformers (~31 min, CPU workers) -> results13/
.venv/bin/python scripts/run_task14.py && .venv/bin/python scripts/task14_recompute.py && .venv/bin/python scripts/task14_tables.py   # round 15, 12 next-token transformers (~11 min, GPU) -> results14/
.venv/bin/python scripts/run_task15.py && .venv/bin/python scripts/run_task15.py --out results15/fine --ps 0.05 0.1 0.2 0.3 --plain-ps 0.1 0.25 && .venv/bin/python scripts/task15_tables.py   # round 16, 54 models (~1 h, GPU + CPU workers) -> results15/
.venv/bin/python scripts/run_all.py --quick --out /tmp/quick   # smoke runs (also for run_task2.py)
```

## Layout

| path | contents |
|---|---|
| `goalgeo/gridworld.py` | `GridWorld`: ASCII maps, 4 actions, walls, thin edge walls, one-way portals, uniform or per-cell slip, hazard cells with knock-back; exact transition tensor `P[s,a,s']` and geodesic distances |
| `goalgeo/envs.py` | layouts `base`, `barrier`, `portal`, `twins`, `ice`, plus `make_switch_env(λ)` (policy-switch ring with hazards) and `make_corridor_env()` |
| `goalgeo/planning.py` | value iteration per goal (absorbing goal), `π*_g` uniform over ties, rollouts, BC dataset |
| `goalgeo/occupancy.py` | exact `ρ(g|s)`, `ρ(g|s,a)`, `Z_sa`, `Z_state`, `Q_g`; successor representation; shortest-path surrogate |
| `goalgeo/geometry.py` | RDMs, RSA, partial RSA, RDM regression, linear CKA, PCA / participation ratio, Ward clustering, group-CV ridge decoding, two-way decomposition |
| `goalgeo/model.py` | `PolicyNet` (one-hot embeddings or coordinates → 3×128 MLP → 4 logits), per-layer activations, `forward_from` for patching |
| `goalgeo/train.py` | full-batch Adam behavioural cloning with soft targets and checkpoints |
| `goalgeo/analysis.py` | round-1 analyses (hypothesis RDMs, layer reports, pair-level RSA, discriminating pairs, goal decomposition, steering) |
| `goalgeo/quotient.py` | round-2 analyses (λ-sweep ground truth, quotient pairs, six-way regression, interaction ablation, steering predictors, dimensionality) |
| `goalgeo/targets.py` | round-3 supervision targets built from Q (hard, Boltzmann(τ), advantage, Q, occupancy), losses (CE, KL, MSE on probabilities or logits), generic trainer |
| `goalgeo/models2.py` | deeper MLP, residual MLP, tiny transformer, synthetic fixed-argmax task |
| `goalgeo/supervision.py` | round-3 analyses (target geometry, information bottleneck of T, class compression, unique interaction variance, ring A/B/C pairs) |
| `goalgeo/hmm.py`, `goalgeo/seqmodels.py` | round-4 HMM (exact forward inference, k-step joint predictives, sampling) and GRU / window-MLP models trained under one-step, k-step or sequential objectives; `SeqNet(gain=c, out_scale=s)` gives the fixed-gain / rescaled-init readout and `train_weighted(lr_out=...)` the per-position-weighted sequential objective with checkpoints and a separate readout learning rate (rounds 5–7) |
| `goalgeo/tfm.py`, `goalgeo/tfm_measure.py` | round-10 causal pre-LN transformer (residual-stream patching as the downstream computation) and its measurements at the layer-1 residual and the post-norm readout interface |
| `goalgeo/cuts.py`, `goalgeo/factorize.py` | round-11 experiments: minimal counterfactual pairs and interchange effects on complete vs incomplete causal cuts, attention route restrictions (hard masks in `tfm.py`'s `attn_diag`, soft attention penalties in `train_routed`); the readout-interface factorisation C = g·D·cos θ with the frozen-LayerNorm ceiling |
| `goalgeo/latentgoal.py`, `goalgeo/belief_train.py`, `goalgeo/beliefprobe.py`, `goalgeo/beliefcausal.py`, `goalgeo/filterstate.py` | round-12 hidden-goal environment (iid or latent-channel evidence) with the exact joint filter and targets for four objectives; stacked transformer training with per-objective output masks and GRU training; affine log-odds probes with IID / EXT / CONF / TIME splits and the gain over the count-affine predictor; filtering from an arbitrary joint state, belief interventions along encoder or minimum-norm directions, equal-belief and same-action pair construction; round 13's full-filter-state coordinates, matched two-block interventions and equal-state pairs (same and different lengths) |
| `goalgeo/wtfm.py` | round-14 windowed transformer: attention to the last l positions with a learned relative bias, optional recurrent carry of the readout interface, position-by-position execution with a key/value cache, edits of u_t or res1(t) |
| `goalgeo/kvprior.py` | round-15 next-token training on sampled tokens; `forward_query` computes position t+1 from arbitrary exported per-layer residuals (the K/V sources) with optional attention masking; λ along A⁺ → B⁺ |
| `goalgeo/steering.py` | round-9 primitives: propagate a (perturbed) state through the network's own future steps, effect curves, matched-effect steering, directional logit derivatives, depth curves |
| `goalgeo/invariants.py` | round-8 measures from extracted arrays (OLS decoding, exact rank, sample-space projection, whitened RSA, readout/Jacobian/Fisher/finite-propagation functional measures), exact interface transforms h→Ah, W→WA⁻¹, J→JA⁻¹ |
| `goalgeo/hmm4.py`, `goalgeo/prominence.py` | round-5 parametrised HMM family (relevance frequency r, strength δ, delay k, fixed immediate-relevance control branch, forgetful-filter cost) and measurements (pairwise metric prominence, decodability, RSA, cue-state gradients of the delayed loss) |
| `goalgeo/plotting.py` … `goalgeo/plotting9.py` | figures per round |
| `scripts/run_all.py` … `scripts/run_task4.py` | one runner per round |
| `tests/` | 114 tests: analytic checks (e.g. `ρ(g|s,a) = γ^T` exactly), geometry invariances, model/patching identities, switch-env boundary, linearised steering threshold exact at the last layer |
