# Representation Invariants (TASK7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the equivalence set of same-function GRUs, compute raw / information / subspace / functional measures, and test each for coordinate invariance, optimisation invariance and functional sensitivity.

**Architecture:** `goalgeo/invariants.py` extracts per-model arrays and computes every measure from arrays (so coordinate transforms are applied to arrays); `rounds/r08_invariants/run.py` trains and saves the models, runs the analyses and writes tables/JSON; `rounds/r08_invariants/plots.py` draws the four figures.

**Tech Stack:** numpy, scipy, torch, matplotlib, pytest. Executed inline in this session; the code is written directly into the files named below.

**Spec:** `rounds/r08_invariants/DESIGN.md`

## Global Constraints
- `.venv/bin/python`, tests via `.venv/bin/python -m pytest`; not a git repo.
- Reuse `SeqNet(gain, out_scale)`, `train_weighted(lr_out)`, `prominence.make_eval/positions/cross_distance`, `hmm4`.

---

### Task 1: `goalgeo/invariants.py` + `tests/test_invariants.py`
- [ ] Tests: exact invariance of `ols_cv_r2`, `exact_rank`, `subspace_basis` projection, whitened RDM, `W@dh`, `J@dh` under random invertible A; PR / Euclidean change under diagonal A; explicit Jacobian equals a JVP; `js_divergence(p,p)=0`.
- [ ] Implement `ols_cv_r2`, `ols_cv_acc`, `exact_rank`, `subspace_basis`, `whitened_rdm`, `subspace_overlap`, `future_jacobian`, `finite_js`, `extract(net, m, ev, sub, dec, seed)`, `transform(ex, A)`, `metrics_from_arrays(ex)`, `random_transform(kind, d, rng, cond)`.
- [ ] Suite green.

### Task 2: `rounds/r08_invariants/run.py`
- [ ] Conditions table (spec), `run_one` saving `rounds/r08_invariants/models/<label>_s<seed>.pt` and returning metrics + ex-lite (Q, RDM, logp subset) for cross-model analyses.
- [ ] Analyses 1–6, tables.md, results7.json; `--quick` smoke.

### Task 3: `rounds/r08_invariants/plots.py`
- [ ] Figures 1–4; check PNGs.

### Task 4: full run, REPORT.md, README, memory, tests.
