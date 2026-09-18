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

## Setup

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv   # reuses system torch/numpy/scipy/matplotlib
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest            # 35 tests, ~3 s
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
| `goalgeo/steering.py` | round-9 primitives: propagate a (perturbed) state through the network's own future steps, effect curves, matched-effect steering, directional logit derivatives, depth curves |
| `goalgeo/invariants.py` | round-8 measures from extracted arrays (OLS decoding, exact rank, sample-space projection, whitened RSA, readout/Jacobian/Fisher/finite-propagation functional measures), exact interface transforms h→Ah, W→WA⁻¹, J→JA⁻¹ |
| `goalgeo/hmm4.py`, `goalgeo/prominence.py` | round-5 parametrised HMM family (relevance frequency r, strength δ, delay k, fixed immediate-relevance control branch, forgetful-filter cost) and measurements (pairwise metric prominence, decodability, RSA, cue-state gradients of the delayed loss) |
| `goalgeo/plotting.py` … `goalgeo/plotting9.py` | figures per round |
| `scripts/run_all.py` … `scripts/run_task4.py` | one runner per round |
| `tests/` | 95 tests: analytic checks (e.g. `ρ(g|s,a) = γ^T` exactly), geometry invariances, model/patching identities, switch-env boundary, linearised steering threshold exact at the last layer |
