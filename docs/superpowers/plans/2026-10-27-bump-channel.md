# Bump channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the `bump_channel` experiment: does a change of belief move the decision through the same few
directions the goal is read from, in the fine-tuned four-goal 0.5B and the from-scratch four-goal transformer and GRU.

**Architecture:** One study directory `studies/8_language_models/bump_channel/` in the house style (dense scripts, a
`common.py` of pure helpers plus model runners, `--quick` smoke mode writing to `_smoke/`, `tables.py` → `tables.md`,
a `reproduce.py` entry). It reuses `bump_coords/common.py` (ridge, projectors, switch, cluster bootstraps, the decision
pool, the hooked fine-tuned model) and `bump_coords/cache/states.pt` (the pool's recorded states). Three measurement
scripts: `subspace_small.py` (part 0, the from-scratch models' goal subspaces), `channel.py --model {llm,tfm,gru}`
(parts 1–3), `tables.py`. Pure helpers are unit-tested on CPU in `tests/test_bump_channel.py`; GPU scripts are verified
by their `--quick` runs.

**Tech Stack:** Python 3, torch 2.8 + CUDA, transformers (Qwen2.5-0.5B), numpy; the repo's `.venv` on the pod at
`/workspace/goal_belief`. Everything runs over SSH on the pod.

**Spec:** `studies/8_language_models/bump_channel/PLAN.md` (commit `88e6f4d`).

## Global Constraints

- Run everything on the pod: `cd /workspace/goal_belief && PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_channel .venv/bin/python <script>`; tests with `.venv/bin/python -m pytest tests/test_bump_channel.py -q`.
- House style: one dense module-level docstring with the run command, `argparse` with `--device`, `--out`, `--quick`; a quick run never writes into committed results (it writes to `_smoke/`).
- `cache/` is git-ignored (add `.gitignore` with `cache/` and `_smoke/` as `bump_coords` does); JSON results, `tables.md`, scripts and the PLAN are committed. No checkpoints in git.
- Causal convention (spec): the receiving history h1 is run under goal g; **switch** = share of history-dependent cells whose patched argmax equals h2's natural decision under g. Bootstraps cluster by h1 (`CC.boot_mean(..., groups=)`, `CC.boot_ratio(..., groups=)`).
- Registered k: fine-tuned model 4 at 23, 8 at 21, 64 at 19, 128 at 15; from-scratch models from part 0 (`registered_k`). Sweep grids: `KS_LLM = (1, 2, 4, 8, 16, 32, 64, 128, 896)`, `KS_SMALL = (1, 2, 4, 8, 16, 32, 64, 128)`.
- A through ratio whose reference switch is below 0.10 is reported as undefined (`null` in JSON), never as a number.
- Commit style: `goalgeo: bump channel — <descriptive sentence(s)>`, with the Co-Authored-By trailer. Never force-push.

## Review Focus

1. A (pair, goal) cell set with no history-dependent decisions (small quick runs, known-state bins) must give `nan` means and `null` ratios, not a crash. Test in Task 1 (`through` on empty arrays).
2. The GRU's `h_{t−2}` site for a decision at step 1 is the zero initial hidden state; patching it must still run the last two tokens. Test in Task 2.
3. Patching with the recorded natural state must reproduce the natural logits exactly (bf16 casting in the fine-tuned hook, position indexing in the small runners). Tests in Task 2 (small) and the Task 3 smoke check (fine-tuned).
4. The registered-k median with undefined pairs: `None` entries are dropped; more than six → no registered k. Test in Task 1.
5. The readout Jacobian at a linear site must equal the head's centred rows (the plan calls those sites ceilings for this reason). Test in Task 2.

---

### Task 1: Pure helpers (`common.py` part 1) and their tests

**Files:**
- Create: `studies/8_language_models/bump_channel/common.py`, `studies/8_language_models/bump_channel/.gitignore` (`cache/`, `_smoke/`, `__pycache__/`)
- Test: `tests/test_bump_channel.py` (same import pattern as `tests/test_bump_coords.py`: `sys.path.insert(0, <bump_channel dir>)`; also insert the `bump_coords` dir, which `common.py` needs)

**Interfaces:**
- Consumes: `bump_coords/common.py` as `CC` (`CC.basis`, `CC.projector`, `CC._centre`, `CC.random_projector`, `CC.spectrum_ranks`, `CC.subspace_overlap`, `CC.boot_mean`, `CC.boot_ratio`, `CC.causal_rank`, `CC.K`, `CC.H`).
- Produces (all in `common.py`):
  - constants `SITES_LLM = (15, 19, 21, 23)`, `K_LLM = {15: 128, 19: 64, 21: 8, 23: 4}`, `KS_LLM`, `KS_SMALL`, `SITES_TFM = (1, 2, 3, 4)` (4 = the final residual), `SITES_GRU = (-2, -1, 0)` (offset from the decision position), `N_PAIRS = 512`, `D_SMALL = 128`.
  - `set_key(w, acts, obss) -> bytes` — `np.packbits(BMZ.common_mask(w, acts, obss)).tobytes()`.
  - `draw_pairs(steps, keys, n_pairs=N_PAIRS, seed=1) -> list[tuple[int, int]]` — same-step pairs with different keys, drawn as `CC.swap_pairs` draws (uniform i, uniform j among i's step; histories may repeat across pairs).
  - `registered_k(kc_list, ks) -> int | None` — median over the defined entries, rounded up to the next value in `ks`; `None` if more than 6 entries are `None` or the list is empty.
  - `goal_basis(X_fit: dict[int, Tensor[n, D]], steps_fit) -> tuple[Tensor[D, D], dict]` — the pooled per-step-centred differences of all 12 ordered pairs, `CC.basis(pooled)` and `CC.spectrum_ranks(pooled)`.
  - `history_basis(X_fit_by_goal, steps_fit, keys_fit, n_pairs=N_PAIRS, seed=2) -> Tensor[D, D]` — pairs drawn by `draw_pairs` on the fit histories, differences `X[g][j] − X[g][i]` per goal, per-step centred, pooled over goals, `CC.basis`.
  - `readout_basis(J: Tensor[n, 3, D]) -> Tensor[D, D]` — `CC.basis` of the stacked centred Jacobian rows (`J − J.mean(1, keepdim)`), `centre=False`.
  - `projectors(Vh, k, d, seed=0) -> dict[str, Tensor[d, d]]` with keys `goal`, `comp` (I − goal), `bottom`, `random0`, `random1`, `random2`.
  - `candidate(x1, delta, P) -> Tensor` — `x1 + delta @ P`.
  - `norm_share(delta, P) -> float` — mean over rows of ‖δP‖² / ‖δ‖².
  - `through(per_var: ndarray, per_ref: ndarray, groups) -> dict` — `{"switch": mean, "switch_ci", "ref": mean, "ratio": mean(var)/mean(ref) or None if mean(ref) < 0.10 or empty, "ratio_ci"}`.
  - `split_known(per_var, per_ref, groups, known: ndarray[bool]) -> dict` — `through` on each half plus `diff_ci` (bootstrap of ratio_known − ratio_uncertain, clustered), `None` where either is undefined.

- [ ] **Step 1: Write the failing tests** (`tests/test_bump_channel.py`): `test_draw_pairs_same_step_different_key` (steps/keys arrays with 3 steps × 4 keys; every pair has equal steps, different keys, 20 pairs), `test_registered_k_rounds_up_and_drops_none` (`[4]*10 + [8]*2 → 4`; `[4]*6 + [8, 16, 16, 16, 16, 32] → 8`; `[4]*5 + [None]*7 → None`), `test_projectors_identities` (random `X_fit` for 4 goals, n = 60, D = 16: `goal` is symmetric idempotent of trace k; `goal + comp` is the identity; `bottom` orthogonal to `goal`), `test_through_undefined_below_floor_and_on_empty` (ref mean 0.05 → `ratio is None`; empty arrays → `ratio is None` and `switch` is nan), `test_split_known_diff` (constructed arrays where known ratio 1.0, uncertain 0.5 → `diff_ci` excludes 0).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_bump_channel.py -q` → FAIL (`No module named common` / attribute errors).
- [ ] **Step 3: Implement `common.py` part 1** with the signatures above; the module docstring names the scripts and the run command; `sys.path` set up so `import common as CC` resolves to `bump_coords/common.py` **before** this module shadows it (import it under the name `CC` by path: `importlib.util.spec_from_file_location("coords_common", COORDS / "common.py")`, since both files are called `common.py`).
- [ ] **Step 4: Run the tests** → PASS.
- [ ] **Step 5: Commit** `goalgeo: bump channel — shared helpers: pair drawing, the registered rank, goal / history / readout bases, projectors, through ratios with cluster bootstraps and the known / uncertain split; tests`.

### Task 2: Small-model runners (`common.py` part 2) and their tests

**Files:**
- Modify: `studies/8_language_models/bump_channel/common.py`
- Test: `tests/test_bump_channel.py`

**Interfaces:**
- Consumes: `bump_maze/train.py` `StepTransformer`, `StepGRU`; `bump_goals/train_goals.py` `step_tokens(acts, obss, length, goal, H)` (goal token `7 + g`; `model.tok = nn.Embedding(7 + K, d)` as `train_small` builds it).
- Produces: `class Small` with `__init__(self, kind: str, path: Path | None, H: int, device, sites=None)` (path `None` → random weights, for tests; `sites` default `SITES_TFM` / `SITES_GRU`), and
  - `record(self, hists, g) -> (logits[n, 3] float64 centred, X[n, n_sites, 128] float32)`;
  - `patched(self, hists, g, site, new[n, 128]) -> logits`;
  - `patched_record(self, hists, g, site, new) -> (logits, X)`;
  - `jacobian(self, hists, g, site) -> J[n, 3, 128]` (gradient of each centred logit with respect to the site's state, by three `autograd.grad` calls on batch sums).
  - Transformer semantics: site `l ∈ {1, 2, 3}` is the residual entering block `l` at the decision position (`rec[l]`), site 4 the final residual before `model.ln`; a patch writes `x[row, t_row] = new[row]` at that point. GRU semantics: site `s ∈ {−2, −1, 0}` is the output at position `t + s` (position 0 is the goal token); a position below 0 is the zero initial state; `patched` runs `nn.GRU` on the prefix up to that position, installs `new` as the hidden state, and runs the remaining `−s` tokens from it; the decision reads `head(h_t)`.
  - `load_small(kind, run_name="…_120000", H, device) -> Small` loading `bump_goals/runs/<kind>_120000/model.pt`.

- [ ] **Step 1: Write the failing tests**: `test_small_tfm_patch_identity` (random `Small("tfm", None, H=40)`, 6 histories of steps 1..6 under goals 0..3: `patched(hists, g, l, record(...)[1][:, i])` equals `record` logits for every site, `atol 1e-5`), `test_small_gru_patch_identity` (sites −2, −1, 0, including a step-1 history, whose site −2 state is zeros), `test_small_gru_jacobian_at_h_t_is_head_rows` (`jacobian(..., site=0)` equals `head.weight − head.weight.mean(0)` for every row, `atol 1e-5`), `test_small_patched_record_matches_natural_when_patch_is_natural` (both models: `patched_record` with the natural state returns the natural `X` at every later site).
- [ ] **Step 2: Run** → FAIL (`Small` undefined).
- [ ] **Step 3: Implement `Small`** (a re-implementation of `StepTransformer.forward` with a patch point, not a monkey-patch; the GRU runner uses `self.model.gru(emb, h0)` on slices).
- [ ] **Step 4: Run the tests** → PASS.
- [ ] **Step 5: Commit** `goalgeo: bump channel — the from-scratch four-goal models with recording, patching and Jacobians at the decision position's block entrances and the GRU's last three hidden states; tests`.

### Task 3: The fine-tuned model runner and the pool loader (`common.py` part 3)

**Files:**
- Modify: `studies/8_language_models/bump_channel/common.py`

**Interfaces:**
- Consumes: `CC.Model` (`record`, `patched`, `.P` the `Patched` hook with `_patch` / `_rec`), `CC.load_cache()` (bump_coords' `states.pt`: `swap["X_fit"][g][n, 4 sites, 896]`, `X_h1`, `L_fit`, `L_h1`, `steps_fit`, `steps_test`, `hists_fit`, `hists_h1`, `dec_fit_idx`, `dec_test_i1`), `CC.worlds()`, `CC.swap_decisions`, `CC.split_swap_pool`.
- Produces:
  - `class LLM(CC.Model)` adding `patched_record(hists, g, site, new) -> (logits, X[n, len(sites), 896])` (one `P.run(ids, att, patch=…, record=True)` per batch) and `jacobian(hists, g, site) -> J[n, 3, 896]` (the patch leaf is the natural state in float32 with `requires_grad`, run under `torch.enable_grad()` through the same hook; three `autograd.grad` calls on batch sums; batches of 8).
  - `pool(worlds, sols, quick=False) -> dict` with `dec`, `fit_idx`, `test_idx`, `hists_fit`, `hists_test`, `steps_fit`, `steps_test`, `keys_fit`, `keys_test`, `sizes_test` (goal-free set size of each test decision, `dec[i]["size"]`) — the coordinates study's pool (`CC.swap_decisions(worlds, sols, 64 if quick else 1024)`, `CC.split_swap_pool(dec, 0.25, 16 if quick else 1024)`).
  - `load_states(kind, cache_dir) -> dict` returning the `swap` dict shape above for `kind="llm"` from `bump_coords/cache/states.pt` (raising a clear error naming `bump_coords/collect.py` if absent) and for `tfm` / `gru` from `bump_channel/cache/states_<kind>.pt` (written by Task 4).

- [ ] **Step 1: Implement** `LLM`, `pool`, `load_states`.
- [ ] **Step 2: Smoke-verify on the GPU** with a throwaway script in `_smoke/` (not committed): load `LLM`, 4 test histories under goal 0, check `patched(…, 23, natural_23)` equals `record` logits (`atol 1e-2`, bf16), `patched_record(…, 15, natural_15)[1][:, site 23]` equals the natural site-23 state within `1e-2` relative, and `jacobian(…, 23)` has shape `[4, 3, 896]` with rows summing to ~0 (centred). Print the three checks.
- [ ] **Step 3: Commit** `goalgeo: bump channel — the fine-tuned model with patched recording and Jacobians through the goal-swap hook; the coordinates study's decision pool and recorded states as the loader`.

### Task 4: Part 0, `subspace_small.py`

**Files:**
- Create: `studies/8_language_models/bump_channel/subspace_small.py`

**Interfaces:**
- Consumes: `Small`, `load_small`, `pool`, `goal_basis`, `registered_k`, `CC.tune_lambda`, `CC.ridge`, `CC.apply_map`, `CC._centre`, `CC.spectrum_ranks`, `CC.projector`, `CC.bottomk_projector`, `CC.random_projector`, `CC.causal_rank`, `CC.switch`, `CC.boot_mean`, `CC.boot_ratio`.
- Produces: `cache/states_<kind>.pt` (the `swap` dict shape of Task 3 at that model's sites), `cache/subspaces_<kind>.pt` (`{site: {"Vh": [128,128], "k": registered k or None, "k_c_pooled": [12], "natural": float}}`), `subspace_<kind>.json` (per site: natural goal swap with CI, per pair: full-map switch, `switch_by_k` for `top`, `pooled`, `bottom`, `random0..2`, `k_c`, `k_c_pooled`, `k50/k90/k99`, offset share; summary: `k_registered`, `k_c_pooled_median_raw`, `natural`, `rank_at_k_ratio` with CI, `registered_site` (the non-linear site with the largest natural goal swap ≥ 0.10, else `None`)).

- [ ] **Step 1: Write `subspace_small.py --kind {tfm,gru} [--quick]`** mirroring `bump_coords/rank.py`'s loop (fit maps per ordered pair on the fit histories; the causal sweep on the test histories' goal-dependent decisions under the source goal; `KS_SMALL`; all variants at every site, since the small models are cheap). The registered site excludes the linear site (tfm 4, gru 0) and applies the 0.10 floor.
- [ ] **Step 2: Quick run** both kinds: `… subspace_small.py --kind tfm --quick --out studies/8_language_models/bump_channel/_smoke/subspace_tfm` (and `gru`); expected: finishes in under a minute, writes the three files, prints one line per site with the natural goal swap and `k_c` per pair, no nan outside empty cells.
- [ ] **Step 3: Commit** `goalgeo: bump channel — part 0: the from-scratch four-goal models' goal subspaces by the coordinates study's rank analysis at the block entrances and the GRU's last three hidden states; the registered site and rank`.

### Task 5: Parts 1–3, `channel.py`

**Files:**
- Create: `studies/8_language_models/bump_channel/channel.py`

**Interfaces:**
- Consumes: Tasks 1–3; for `tfm` / `gru` the subspaces of Task 4; for `llm` the goal bases computed here from `load_states("llm")` with `K_LLM`.
- Produces: `channel_<kind>.json`:
  ```
  {"kind", "sites": [...], "k": {site: k}, "n_pairs", "n_cells", "dep_share_by_goal": [4],
   "overlap": {site: {"goal_readout", "goal_history", "history_readout", "random"}},
   "part2": {site: {variant: through-dict + {"norm_share": float, "known": split-dict}}},   # variants: natural, goal, comp, bottom, random0-2, readout, history, floor, k1, k2, …
   "part1": {"<s_in>-><s_out>": {"switch_in": …, variant: through-dict…}},
   "registered": {"site", "s_in", "THRU-D", "THRU-U", "COMP", "CTRL", "DISTINCT", "KNOWN"}}   # each a {"value", "ci", "holds"} or null
  ```

- [ ] **Step 1: Write `channel.py --model {llm,tfm,gru} [--quick] [--out]`**: load states and subspaces (`llm`: `goal_basis` per site from `X_fit`, `k = K_LLM[site]`); `history_basis` per site; `readout_basis` from `jacobian` on 512 fit decisions per goal (seed 3; quick: 8); `draw_pairs` on the test decisions; for each goal the natural decisions and the history-dependent mask; **part 2** at every site: candidates `natural` (x2), `goal`, `comp`, `bottom`, `random0-2`, `readout`, `history`, `floor` (random vector of |Δ|), and `k<k>` for the sweep; **part 1** for the model's (s_in, s_out) pairs (llm: {15, 19} × {21, 23}; tfm: {1, 2} × {3, 4}; gru: (−2, −1) and (−2, 0)): `patched_record` at s_in, then the same candidate set on δ at s_out plus `full` (x1 + δ, the check); **part 3** via `split_known` with `known = sizes_test[i] == 1`; the registered block per the spec's decision rule (llm: site 23, s_in 19; small: `registered_site` from part 0, s_in the earlier site); print one line per (part, site, variant).
- [ ] **Step 2: Quick runs** for all three models into `_smoke/channel_<kind>` (llm sites (19, 23), 16 pairs, ks (1, 4, full)); expected: `full` through ratio 1.0 (±1e-6) in part 1 on every (s_in, s_out), `natural` ratio 1.0 in part 2, finishes in a few minutes for the llm.
- [ ] **Step 3: Commit** `goalgeo: bump channel — parts 1–3: the upstream belief swap's induced change and the natural history difference restricted to the goal subspace, its complement, bottom, random, readout and history directions; the known / uncertain split; the registered scorecard`.

### Task 6: `tables.py`, `run_all.sh`, the `reproduce.py` entry

**Files:**
- Create: `studies/8_language_models/bump_channel/tables.py`, `run_all.sh`
- Modify: `reproduce.py` (after the `bump_coords` Experiment, line 218)

**Interfaces:**
- Consumes: `subspace_<kind>.json`, `channel_<kind>.json`.
- Produces: `tables.md` with sections: 0 (part 0 per small model and site: natural swap, full map, rank-at-k ratio, k_c per pair, registered site and k), 1 (part 1 per model and (s_in, s_out): switch_in, then through ratios of full / goal / comp / bottom / random / readout / history with CIs, norm shares), 2 (part 2 per model and site: the same for the natural difference, plus the sweep as one row of `k: ratio`), 3 (known / uncertain for goal and comp, with the difference's CI), overlaps, and the scorecard (one row per criterion × model).
- `reproduce.py`: `Experiment("bump_channel", "8_language_models/bump_channel", "<one-sentence question from the PLAN>", "2–3 h GPU", [Step("subspace_small.py", ("--kind", "tfm"), quick=True), Step("subspace_small.py", ("--kind", "gru"), quick=True), Step("channel.py", ("--model", "llm"), quick=True), Step("channel.py", ("--model", "tfm"), quick=True), Step("channel.py", ("--model", "gru"), quick=True), Step("tables.py", tables=True)], needs=("bump_goals", "bump_coords"))`.

- [ ] **Step 1: Write `tables.py`** (reads from `HERE` or `--out`; `--quick` reads the `_smoke/` outputs) and `run_all.sh` (the six steps in order with logs under `logs/`).
- [ ] **Step 2: Run** `tables.py --quick` on the smoke outputs → a `tables.md` in `_smoke/` with every section present; then `.venv/bin/python reproduce.py list | grep bump_channel` shows the entry.
- [ ] **Step 3: Commit** `goalgeo: bump channel — tables, the run script and the reproduce entry`.

### Task 7: The full run and the results commit

- [ ] **Step 1: Launch** in tmux: `tmux new -d -s bump_channel 'bash studies/8_language_models/bump_channel/run_all.sh > studies/8_language_models/bump_channel/logs/run_all.log 2>&1'`; check `nvidia-smi` shows the process and the log advances.
- [ ] **Step 2: Monitor** until `tables done` appears in the log (expected 2–3 h); on a crash, fix and relaunch from the failed step only (the cache keeps earlier steps), and record the fact as a disclosure for the report.
- [ ] **Step 3: Verify** `tables.md`: part 1's `full` ratio 1.00 everywhere; part 2's `natural` 1.00; the fine-tuned model's natural goal swap at 23 ≈ 0.98 and natural history swap ≈ 0.99 (the swap study's numbers); the registered block filled for every model (or `null` with the reason).
- [ ] **Step 4: Commit** the JSONs, `tables.md` and logs: `goalgeo: bump channel results — <the headline numbers: THRU-D per model, DISTINCT, KNOWN>`.

### Task 8: REPORT.md, the study README row, the FINDINGS entry

- [ ] **Step 1: Write `REPORT.md`** in the house form (run date, disclosures, the question, one section per part with the key table, the scorecard against the PLAN's criteria and expectations, Reading, Limits), every number from `tables.md`.
- [ ] **Step 2: Add** the `Bump channel` row to `studies/8_language_models/README.md` (question, result, cost) and the reproduce command; add the `**Bump channel …**` paragraph to `FINDINGS.md` section 8 after bump coordinates.
- [ ] **Step 3: Commit** `goalgeo: bump channel report — <one-sentence reading>; README row, FINDINGS entry`.
