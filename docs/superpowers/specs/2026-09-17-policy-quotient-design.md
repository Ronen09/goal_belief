# Policy-quotient versus occupancy geometry — design spec (TASK2)

Date: 2026-09-17. Requirements: `TASK2.md`. Builds on the `goalgeo` package and
the findings in `results/REPORT.md`.

## 1. Question

Does the network preserve the metric geometry of future occupancy (H1), collapse
states into decision-equivalence classes (H2), or encode a smooth value-like
geometry (H3)? The deliverable is `scripts/run_task2.py` writing `results2/`
(figures, JSON, `tables2.md`) plus `results2/REPORT2.md`.

## 2. Decisions

| Topic | Decision | Why |
|---|---|---|
| Failure mechanism (Task 1) | Hazard cells: at such a cell every action is replaced, with probability λ, by a knock-back to a fixed cell (the entrance of the short route). | Keeps every state ordinary (no trap state), occupancy varies continuously in λ, and the switch is sharp. |
| Switch environment | 5×13 ring: safe corridor on row 0, walls rows 1–3 except columns 0 and 12, short corridor on row 4 with hazards at columns 5–7 knocking back to (4,0). Goals: 5 on the right column, 2 on each corridor, 1 on the left column (10 goals). | Route A (row 4) is 12 steps with 3 hazards, route B is 20 steps. Different goals and start rows switch at different λ_c, giving many policy boundaries. |
| λ sweep | 21 values, 0 to 1 in steps of 0.05; 5 seeds each, 3000 full-batch steps as before. | Enough resolution to locate boundaries; seeds separate geometry from optimisation noise. |
| Aligning networks across λ (Task 4) | Three complementary measures: (a) RDM change (1 − Spearman between successive RDMs), alignment-free; (b) per-(s,g) activation change after orthogonal Procrustes alignment of independently trained networks; (c) a warm-start chain (same seed, network at λ initialised from λ−Δ and trained 1000 steps) giving raw ‖h_{λ+Δ} − h_λ‖. | Independently trained networks live in different bases; (a)/(b) handle that, (c) is the literal quantity asked for. |
| Q geometry | With reward 1 on arrival and absorbing goals, `Q_g(s,a) = ρ(g|s,a)/γ` exactly, so D_Q ≡ D_occupancy. Reported once; the regression uses occupancy, SR, policy, advantage, margin and spatial instead. | Avoids a perfectly collinear regressor. |
| Advantage / margin geometries (Task 6) | Advantage: `A(s,a,g) = Q(s,a,g) − max_a Q` flattened over (g,a). Margin: `m(s,g) = Q(a1) − Q(a2)` over g. Value: `V(s,g)` = `Z_state`. | These are the "smooth decision-relevant" competitors of H3. |
| Quotient pairs (Task 5) | Constructed in a dedicated 1×15 corridor (goals at 0, 4, 7, 10, 14, giving four runs of identical policy tables) and in the switch env at λ ∈ {0.1, 0.5}. Case A = identical policy table and occupancy distance in the top quartile; Case B = policy tables differ in ≥ 1 goal and occupancy distance in the bottom quartile. Hidden distances reported relative to the median and relative to a random-init network. | Deliberate construction as requested; the random-init ratio separates "not amplified" from "collapsed". |
| Interaction ablation (Task 7) | At layer l: (i) subtract the exact interaction component `r_sg`; control: subtract a random vector with the same norm per (s,g). (ii) project out the top-k PCA subspace of `r` (k = 90% of interaction variance); control: random k-dim subspace. Measures: action accuracy, goal sensitivity (fraction of goal pairs per state with different argmax), steering success at α = 1. | Both the exact-component and subspace versions of "remove h_sg". |
| Steering predictors (Task 8) | Per (s, g1→g2): α* (first g2-optimal action), predictors: m1, m2, occupancy-vector difference ‖ρ(g2|s,·) − ρ(g1|s,·)‖, network logit margin at α = 0, and the linearised boundary crossing α_lin = logit margin / directional derivative of the margin along v (exact for the last layer). Spearman of each with α*; multiple regression. | Distinguishes occupancy/value geometry from direct boundary crossing. |
| Stochastic test (Task 9) | In the sweep: RSA and partial RSA of each layer with the policy table of π*_λ versus the deterministic (λ = 0) policy table, as a function of λ. | The switch env is exactly the asymmetric-risk design requested. |
| Dimensionality (Task 10) | PR, entropy rank, PCA spectrum, two-NN intrinsic dimension (Facco et al. 2017) for hidden layers, Z_sa, policy table, SR, random init, across base, twins, portal, barrier, corridor and switch (λ ∈ {0, 0.5, 1}). Correlation across environments between PR_hidden and PR_occupancy vs PR_policy. | Tests whether occupancy predicts degrees of freedom even where it misses the metric. |

## 3. Code

- `goalgeo/gridworld.py`: add `hazards: dict[cell -> (prob, dest)]`.
- `goalgeo/envs.py`: `make_switch_env(lam)`, `make_corridor_env()`.
- `goalgeo/geometry.py`: `procrustes_distance`, `two_nn_dimension`.
- `goalgeo/quotient.py`: switch-env ground truth per λ (occupancy, Q, π*, margins, boundaries), quotient pairs, extended regression, interaction ablation, steering predictors.
- `scripts/run_task2.py`: Tasks 1–10, writes `results2/`.
- Tests for hazards, boundary detection, Procrustes, two-NN, pair construction, ablation identity (removing nothing changes nothing).

## 4. Success criteria

Same as `TASK2.md` "Success Criteria"; the report states per criterion whether it
was met, with controls.
