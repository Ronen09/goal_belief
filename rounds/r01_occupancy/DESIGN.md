# Goal-occupancy geometry in goal-conditioned agents — design spec

Date: 2026-09-17. Source of requirements: `rounds/r01_occupancy/BRIEF.md`.

## 1. Question

Does a network trained only to act optimally toward a supplied goal learn a
hidden geometry predicted by the goal-conditioned discounted occupancy of the
decision problem? The deliverable is a runnable, tested Python package plus a
results report (figures + numbers) covering Phases 1–8 of `rounds/r01_occupancy/BRIEF.md`, and the
first two items of Phase 9 (stochastic transitions).

## 2. Decisions and assumptions (things `rounds/r01_occupancy/BRIEF.md` leaves open)

| Topic | Decision | Why |
|---|---|---|
| Goal semantics | Goal cell is **absorbing**: once at `g` the agent stays there. Reward 1 on the arrival step, 0 otherwise. | Makes occupancy well-defined and exactly `ρ(g|s,a)=γ^{T}` in the deterministic case (T = arrival time after taking `a`). Standard in goal-conditioned occupancy work. |
| Discount | `γ = 0.9` | 8×8 with walls has paths up to ~25 steps; 0.9^25 ≈ 0.07 keeps far goals distinguishable from unreachable ones (0). |
| Invalid moves | Bumping into a wall/border keeps the agent in place. | Keeps the action set fixed at 4 everywhere. |
| Optimal policy | Uniform distribution over the argmax actions of `Q_g`. | `rounds/r01_occupancy/BRIEF.md` asks for `π*_g(a|s)` as a distribution; ties are common on grids. |
| BC dataset | Every `(s, g)` with `s` free and `s ≠ g`. Target = `π*_g(·|s)` (soft cross-entropy). | Exhaustive dataset removes sampling noise; soft targets avoid arbitrary tie-breaks. |
| Network | `Embedding(n_cells, 32)` for state and for goal, concatenated → MLP `64→128→128→128→4`, ReLU. Layers analysed: `emb` (concat embeddings), `h1`, `h2`, `h3` (post-ReLU), `logits`. | Matches the "state embedding + goal embedding + 3–4 layer MLP, width 128" prescription. One-hot input gives the network **no** spatial prior, so any spatial geometry that appears must be learned. |
| Input variant | A second encoding `coord` feeds normalised `(x, y)` of state and goal through a linear layer to 32 dims each. | Gives `H_spatial` its best shot: spatial geometry is *supplied* and the question is whether occupancy geometry overrides it. Used in Phase 6 alongside `onehot`. |
| Training | Adam, lr 1e-3, batch = full dataset, 3000 steps, weight decay 0, 5 seeds. Checkpoints at log-spaced steps for the "emerges during training" analysis. | Tiny dataset (~800 rows); full batch is deterministic and fast on CPU. |
| Theoretical representations | Computed from exact dynamics, see §4. | Task: "Keep the transition dynamics fully known". |

## 3. Environments (Phase 1 and 6)

All are `H×W` grids (default 8×8) described by ASCII maps. Symbols:
`#` wall, `.` free, `G` free cell that is also a candidate goal, `>`/`<`/`^`/`v`
one-way portal source with a named destination (see `envs.py`).

| name | purpose | design |
|---|---|---|
| `base` | Phases 1–5, 7, 8 | Open 8×8 with an L-shaped wall and a short wall stub; 16 goals spread over the free cells. |
| `barrier` | Phase 6: adjacent-but-far | A near-full vertical wall with a single gap at the bottom. Cells on opposite sides of the wall are at Euclidean distance 1 but geodesic distance up to ~12. |
| `portal` | Phase 6: far-but-same-future | Two dead-end pocket cells at opposite corners; every action from a pocket teleports to the same hub cell. The pockets have **identical** `Z_sa` and coordinates 7+ apart. |
| `twins` | Phase 6: duplicated regions | Two mirror-image rooms (top/bottom) joined to a central goal strip by same-length corridors. Goals only in the strip. Mirror pairs have identical `Z_state` and `Z_sa` up to the U↔D block swap, at Euclidean distance up to 7. |

For each environment we also record, for every free cell, its coordinates and
the geodesic (shortest-path) distance matrix.

## 4. Theoretical representations (Phase 3)

With `P_g` the transition matrix under `π*_g` (goal absorbing) and
`D_g = (1−γ)(I − γP_g)^{-1}`:

- `ρ(g|s)` = `D_g[s, g]` — state occupancy.
- `ρ(g|s,a)` = `(1−γ)·1[s=g] + γ Σ_{s'} P(s'|s,a) D_g[s', g]` — state-action occupancy as in `rounds/r01_occupancy/BRIEF.md`.
- `z(s,a) ∈ R^k` = `[ρ(g_1|s,a), …, ρ(g_k|s,a)]`.
- `Z_sa(s) ∈ R^{4k}` = concat over actions of `z(s,a)`. **Primary theoretical representation of a state**: it is exactly the information needed to act optimally for every goal.
- `Z_state(s) ∈ R^k` = `[ρ(g_i|s)]_i`.
- `Q_g(s,·) ∈ R^4` = `[ρ(g|s,a)]_a` — goal-specific action relevance (the interaction term for Phase 7).

Geometry of `Z_sa` is characterised by: pairwise Euclidean RDM, cosine
similarity, PCA spectrum, participation ratio and entropy effective rank, and
hierarchical clustering (Ward) with a dendrogram + cluster map on the grid.

Competing geometries (used as regressors in Phases 5–6):

| hypothesis | representation | distance |
|---|---|---|
| `H_occupancy` | `Z_sa(s)` (also `Z_state`) | Euclidean |
| `H_spatial` | `(x, y)` | Euclidean |
| `H_geodesic` | graph shortest path | shortest-path length |
| `H_SR` | successor representation under uniform random policy, same γ | Euclidean |
| `H_identity` | one-hot | 0 if same cell else 1 (null) |

## 5. Learned representation and comparison (Phase 5)

For every layer `l`, activation tensor `H_l[s, g, :]` over all free `s`
and all goals `g`. Two views:

- **State view** `h̄_l(s)` = mean over `g` of `H_l[s,g,:]`, and the concatenated view `H_l[s, :, :]` flattened. RDMs over states compared to §4 models.
- **Pair view** `H_l[s,g,:]` over `(s,g)` pairs compared to a pair-level theoretical representation `[Z_sa(s), Q_g(s,·), onehot(g)]`-derived RDMs.

Analyses per layer, per seed, with an untrained (random-init) network as control:

1. RSA: Spearman ρ between upper-triangular RDMs (activation vs each hypothesis); also partial Spearman of occupancy controlling for spatial and vice versa.
2. Linear CKA between activations and each theoretical matrix.
3. Ridge decoding `h → Z_sa`, R² with **leave-states-out** 5-fold CV (a one-hot-based network can only pass this if its geometry generalises across cells).
4. Dimensionality: participation ratio of `h̄_l` vs `Z_sa`; overlap of top-k principal subspaces.
5. Trajectory across training checkpoints of items 1–3.

## 6. Discriminating test (Phase 6)

On `barrier`, `portal`, `twins` (and `base` for reference), with both input
encodings: fit `d_act = β_occ d_occ + β_sp d_sp + β_geo d_geo + β_SR d_SR + c`
on z-scored upper-triangular RDMs; report β's, and Spearman on the subset of
pairs where the hypotheses **disagree** (top/bottom quartile of
`d_occ − d_sp` after z-scoring). Also report the designed pairs directly
(pockets in `portal`, wall-adjacent pairs in `barrier`, mirror pairs in
`twins`) as activation distance normalised by the median pairwise distance.

## 7. Goal dependence (Phase 7)

Two-way additive decomposition of `H_l[s,g,:]`: `μ + a_s + b_g + r_{sg}`.
Report variance fractions (state, goal, interaction) per layer. Test that
`a_s` correlates (RSA) with `Z_sa`, `b_g` with the goal's own coordinates /
`Z_state(g)`, and `r_{sg}` with `Q_g(s,·)` and with the optimal action.

## 8. Causal intervention (Phase 8)

For layer `l` and goal pair `(g1, g2)`: `v = mean_s H_l[s,g2] − H_l[s,g1]`.
Patch `H_l[s,g1] + α v`, run the remaining layers, decode the action. For
`α ∈ [0, 2]` report agreement with `π*_{g2}` and with `π*_{g1}`, restricted to
states where the two policies differ. Controls: random direction with the same
norm; steering with a *different* goal's vector. Compare the per-state flip
against the occupancy model's prediction (`argmax_a ρ(g2|s,a)`), and correlate
the α at which a state flips with the occupancy margin
`ρ(g2|s,a*_{g2}) − ρ(g2|s,a*_{g1})`.

## 9. Stochastic extension (Phase 9, items 1–2)

`slip ∈ {0.0, 0.2}`: with prob. `slip` the action is replaced by a uniformly
random one. Occupancy is computed by the exact linear solve of §4, so it now
differs from `γ^{shortest path}`. Test which of `Z_sa(stochastic)` or
`Z_sa(deterministic shortest-path)` better explains the network trained on the
stochastic MDP. Partial observability and recurrent models are documented as
future work, not implemented.

## 10. Code layout

```
goalgeo/
  gridworld.py   GridWorld: parse map, free cells, transition tensor P[s,a,s'], portals, slip
  envs.py        named layouts + goal sets
  planning.py    value iteration per goal, π*_g, rollouts
  occupancy.py   D_g, ρ(g|s), ρ(g|s,a), Z_sa, Z_state, Q_g, SR, geodesic
  geometry.py    RDM, cosine, PCA, effective rank, clustering, RSA, partial RSA, CKA, ridge CV
  model.py       PolicyNet (onehot/coord), activations per layer, forward_from_layer for patching
  train.py       BC training, checkpoints, evaluation (accuracy, rollout success)
  analysis.py    phase-level analysis functions producing dicts + figures
rounds/r01_occupancy/run.py   runs every phase, writes rounds/r01_occupancy/<env>/... and rounds/r01_occupancy/REPORT.md
tests/               pytest: exact analytic checks for occupancy, geometry, model
```

Results are written under `rounds/r01_occupancy/` as JSON plus PNG figures; the report
`rounds/r01_occupancy/REPORT.md` is generated by the run script and then edited by hand
with interpretation.

## 11. Success criteria

- All tests pass; BC accuracy ≥ 99% on the training set for every seed.
- Every phase produces at least one figure and a numeric table.
- The report states explicitly, per phase, whether the occupancy hypothesis is supported, with the controls (random-init net, competing geometries) shown alongside.
