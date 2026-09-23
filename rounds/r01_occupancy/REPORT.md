# Does a goal-conditioned agent learn goal-occupancy geometry?

Run date: 2026-09-17. All numbers below are means over seeds from `rounds/r01_occupancy/tables.md`
(5 seeds for the base grid, 3 for the discriminating and stochastic environments).
Design and assumptions: `rounds/r01_occupancy/DESIGN.md`.
Reproduce with `.venv/bin/python rounds/r01_occupancy/run.py` (about 6 minutes on CPU).

## Summary

**The strong form of the hypothesis is not supported; a quotient of it is strongly supported.**

- A 3-layer MLP trained by behavioural cloning to act optimally for a supplied goal
  learns a hidden state geometry that is only weakly predicted by the Euclidean
  geometry of the occupancy vectors `z(s,a) = [ρ(g|s,a)]_g` (RSA 0.29–0.35 at the
  last two layers with one-hot inputs, versus 0 at initialisation). Smoother
  distance-like geometries (spatial coordinates, geodesic distance, the successor
  representation) fit better than the occupancy vectors themselves, and the fit to
  occupancy improves monotonically as the discount used to build it approaches 1.
- The geometry *is* strongly predicted by the **policy table**: representing each
  state by its optimal action distribution for every goal, which is exactly the
  argmax structure of the occupancy vectors. RSA reaches 0.73–0.78 in the base grid
  and 0.77–0.85 in the "twins" maze where every other hypothesis is near zero. In a
  three-way RDM regression against spatial and occupancy geometry, the policy table
  carries most of the weight in every environment.
- Where occupancy geometry conflicts with spatial geometry, the one-hot network
  follows occupancy when the two states have identical futures (the portal pockets
  end up at 0.35 of the median pairwise distance while being at the opposite corners
  of the grid) and separates wall-adjacent cells more than spatially matched pairs.
  It does not, however, place mirror-image states together, because they require
  opposite actions. All of these outcomes are what the policy table predicts.
- The representation decomposes additively: at the middle layer about 44% of the
  variance is a state term, 46% a goal term and only 11% interaction. Training makes
  the representation *more* additive than a random network. The interaction term is
  where the goal-specific action relevance `ρ(g|s,·)` lives (decodable, R² 0.27).
- Causal steering works. Adding the mean goal-difference vector
  `E_s[h(s,g2) − h(s,g1)]` at any hidden layer with α = 1 makes 88–100% of the
  states whose optimal actions differ act as if the goal were `g2`, against 41–49%
  for a random direction of the same norm. The occupancy model's linear-interpolation
  prediction of the per-state flip threshold matches on average but ranks states only
  weakly (Spearman 0.26–0.30).
- With stochastic transitions, the network's geometry follows the policy table of
  the stochastic-optimal policy over the deterministic shortest-path policy
  (partial RSA 0.61–0.69 versus 0.12). At the level of the occupancy vectors the two
  hypotheses are too correlated (0.91–0.999) to be told apart in a grid.

The short answer to the main research question: the internal geometry of this agent
can be predicted from the goal-conditioned occupancy structure only after that
structure is collapsed to what it implies for action. Representational similarity
tracks "the same actions are optimal for the same goals", not "the discounted
occupancy vectors are close".

## Phase 1–2: environment and optimal behaviour

The base grid is 8×8 with an L-shaped wall and a stub (57 free cells, 15 goals),
four actions, deterministic moves, bumping leaves the agent in place, and the goal
cell is absorbing with reward 1 on arrival. Value iteration gives `V_g(s) = γ^(d−1)`
with `γ = 0.9`; the optimal policy is uniform over tied argmax actions. Rollouts
from every start reach every goal in the shortest number of steps.
Figure: `base/env.png`.

## Phase 3: geometry of the theoretical representation

`Z_sa(s)` (60 dimensions per state) is almost three-dimensional: participation
ratio 2.79, 4 principal components explain 90% of the variance, mean pairwise cosine
0.91. Ward clusters are contiguous regions of the grid. In the base grid the
occupancy RDM correlates at 0.94 with Euclidean coordinates and 0.95 with geodesic
distance, so the base grid cannot by itself distinguish the hypotheses; that is what
Phase 6 is for. The policy-table RDM correlates only 0.31 with the occupancy RDM.
Figure: `base/occupancy_geometry.png`.

## Phase 4: training

Both input encodings reach 100% argmax-in-optimal-set accuracy on all 840 (s, g)
pairs, 100% rollout success and 100% optimal-length paths. Figure: `base/training.png`.

## Phase 5: learned versus predicted geometry (base grid, one-hot inputs)

State-level view (mean over goals of `h_l(s, g)`), Spearman RSA at layer h3, with
the random-initialisation control in brackets:

| hypothesis | RSA | held-out-state decoding R² |
|---|---|---|
| policy table | 0.78 (0.01) | — |
| successor representation | 0.49 (0.00) | — |
| spatial (x, y) | 0.47 (0.01) | 0.52 (−0.03) |
| geodesic distance | 0.41 (0.00) | — |
| log occupancy | 0.38 (0.01) | — |
| occupancy `Z_sa` | 0.35 (0.01) | 0.20 (−0.03) |

Partial RSA of occupancy controlling for spatial geometry is negative (−0.30) while
spatial controlling for occupancy stays positive (+0.44). The γ sweep
(`base/phase5_gamma_sweep_onehot.png`) shows the fit to an occupancy RDM rising
monotonically from γ = 0.5 to 0.99: the network's geometry is closer to linear in
distance than to the exponential profile of the trained discount.

Dimensionality is the one place where the occupancy prediction is met directly: the
participation ratio of the state representation falls from 20 at the embedding to
3.1 at h3 (Z_sa: 2.79; random network: 13.5), and the top-3 principal subspace
overlap with Z_sa rises from 0.04 to 0.32. All of this structure emerges during
training in step with accuracy and keeps growing slowly after accuracy saturates
(`base/phase5_emergence_onehot.png`).

At the (s, g) pair level (`base/phase5_pairlevel_onehot.png`) goal identity
dominates the embedding layer (0.41) and the optimal-action geometry dominates the
last layer (0.56); the goal-specific action relevance `ρ(g|s,·)` has near-zero RSA
(0.04) but is linearly decodable from h2/h3 (R² 0.25–0.27).

With coordinate inputs the picture is uninformative in the base grid: a random
network already has RSA 0.93 with occupancy and 0.99 with spatial geometry because
the input carries it. Training does raise held-out decoding of `Z_sa` from 0.59 to
0.93 at h2, but adds no occupancy structure beyond what spatial geometry explains
(partial RSA about 0).

## Phase 6: discriminating environments

Three layouts decorrelate the hypotheses (occupancy~spatial RSA: barrier 0.70,
portal 0.79, twins 0.32; maps in `rounds/r01_occupancy/phase6/env_*.png`). Three-way RDM
regression at h2 (standardised β; one-hot input):

| environment | β occupancy | β spatial | β policy |
|---|---|---|---|
| base | −0.29 | 0.40 | 0.65 |
| barrier | 0.14 | 0.30 | 0.44 |
| portal | −0.12 | 0.33 | 0.49 |
| twins | 0.00 | 0.05 | 0.76 |

Designed pairs, activation distance divided by the median pairwise distance at h3
(one-hot input; lower means the pair is represented as more similar):

| pair type | designed pairs | same spatial distance | same occupancy distance | random init |
|---|---|---|---|---|
| portal pockets (identical futures, opposite corners) | 0.35 | 1.24 | 0.35 | 1.13 |
| barrier (adjacent cells, wall between) | 0.72 | 0.52 | 1.14 | 0.98 |
| twins (mirror images, opposite actions) | 0.94 | 0.98 | 0.75 | 1.02 |

The portal result is the cleanest occupancy win: two cells at opposite corners with
identical futures collapse onto nearly the same representation. It is also exactly
what the policy table predicts, since the pockets have identical optimal actions for
every goal. Wall-adjacent cells are pushed apart relative to spatially matched pairs,
but not as far as occupancy alone would predict. Mirror pairs in the twins maze are
not merged: they share `Z_state` but need opposite up/down actions, and both `Z_sa`
and the policy table predict separation.

With coordinate inputs, spatial geometry persists (β spatial ≈ 1 in base and
portal; the pockets stay 2.8 median distances apart). The exception is the barrier
maze, where training pushes wall-adjacent pairs apart (0.27 at initialisation to
0.44 at h2) and occupancy beats spatial on the disagreement pairs (Spearman 0.65
versus 0.56). Figures: `phase6/phase6_h{1,2,3}.png`.

## Phase 7: goal dependence

Two-way decomposition of `h_l(s, g)` into state, goal and interaction variance
(one-hot input):

| layer | state | goal | interaction | interaction, random init |
|---|---|---|---|---|
| emb | 0.52 | 0.48 | 0.00 | 0.00 |
| h1 | 0.51 | 0.37 | 0.13 | 0.13 |
| h2 | 0.44 | 0.46 | 0.11 | 0.22 |
| h3 | 0.37 | 0.52 | 0.11 | 0.27 |

The representation is close to additive at every hidden layer and training makes it
more additive than a random ReLU network. The goal component's geometry tracks the
goal's coordinates (RSA 0.82–0.86) slightly more than the goal's own occupancy
vector (0.71–0.76). The interaction term carries the goal-specific action relevance:
`ρ(g|s,·)` is decodable from it with R² 0.25–0.27 (0 at initialisation) and its RSA
with the optimal action rises to 0.20 at h3. A state's representation is largely
goal-invariant (mean cosine across goals 0.81 at h2). Figure: `base/phase7.png`.

## Phase 8: causal intervention

Fraction of states with differing optimal actions that act as if the goal were g2
after adding `α · E_s[h(s,g2) − h(s,g1)]` while the input goal is g1:

| layer | α = 0 | α = 0.5 | α = 1 | random direction, α = 1 | another goal's direction, α = 1 |
|---|---|---|---|---|---|
| emb | 0.45 | 0.82 | 1.00 | 0.49 | 0.57 |
| h1 | 0.45 | 0.74 | 0.92 | 0.41 | 0.53 |
| h2 | 0.45 | 0.72 | 0.90 | 0.46 | 0.53 |
| h3 | 0.45 | 0.71 | 0.88 | 0.46 | 0.60 |

At the embedding layer the mean difference is exactly the goal-embedding difference,
so α = 1 is an identity check. At the hidden layers a single averaged vector moves
about 90% of states to the g2-appropriate action, with only a small generic effect
from another goal's direction. The occupancy model with linear interpolation of
`ρ(g|s,·)` predicts a per-state flip threshold `α* = m1 / (m1 + m2)` from the two
goals' occupancy margins; observed thresholds correlate with it at Spearman
0.26–0.30 with mean absolute error 0.19–0.33 (`base/phase8_flip_vs_margin.png`).
Behaviour is steerable along goal directions, but the fine structure of when a state
flips is only weakly captured by occupancy magnitudes. Figure: `base/phase8_steering.png`.

## Phase 9: stochastic transitions

With uniform slip 0.2 the exact occupancy and the shortest-path surrogate `γ^T`
correlate at 0.999, so nothing can be learned about which one the network encodes.
The "ice" variant (slip 0.9 on alternating rows) separates them to 0.91 and changes
17% of optimal-action entries. At the occupancy-vector level the result is still
inconclusive (partial RSA 0.04 and 0.05 either way). At the policy-table level it is
decisive: the network's geometry follows the stochastic-optimal policy table over the
deterministic one (partial RSA 0.61–0.69 versus 0.12; the two tables correlate at
0.43). The network encodes the action consequences of the transition probabilities,
not the probabilities themselves. Figures: `stochastic/phase9_{slip0.2,ice}.png`.

Partial observability, belief states and recurrent models (Phase 9, items 3–5) were
not run; see "Next steps".

## Interpretation and caveats

1. **The learned geometry is a policy-table geometry.** Behavioural cloning of
   argmax actions gives the network no reason to preserve the magnitudes of
   `ρ(g|s,a)`, only their ordering per goal. Everything that occupancy geometry
   predicts correctly here (pocket merging, wall separation, dimensionality, steering
   direction) is also predicted by the policy table, and where the two differ
   (metric distances, mirror pairs, flip thresholds) the policy table wins.
2. **Occupancy magnitudes might appear under value-based training.** An agent that
   must estimate values or successor features would be pushed to encode magnitudes;
   the imitation-only setup was chosen to avoid optimisation confounds and
   deliberately removes that pressure.
3. **Linear decodability of `Z_sa` is low from held-out states** (R² 0.20) and
   negative at the embedding, so decodability is not driving any of the conclusions.
4. **The base grid confounds hypotheses** (occupancy~spatial 0.94). Conclusions
   about "which geometry" rest on Phase 6, where the hypotheses were decorrelated by
   construction.
5. **Coordinate inputs preserve input geometry** through all layers; the
   one-hot condition is the clean test because the network has no spatial prior.

## Next steps

- Train the same network with value learning (or an auxiliary successor-feature
  loss) and repeat Phases 5–6 to test whether occupancy magnitudes appear when the
  objective depends on them.
- Add a pure "policy table" competitor to the steering analysis: predict flip
  thresholds from logit margins rather than occupancy margins.
- Partially observable variant with exact belief states and a recurrent policy,
  testing whether belief-state and policy-table geometry factorise.

## Files

- `rounds/r01_occupancy/tables.md`: every number, all layers, both encodings.
- `rounds/r01_occupancy/base/*.json`, `rounds/r01_occupancy/phase6/phase6.json`, `rounds/r01_occupancy/stochastic/phase9.json`: raw per-seed values.
- `rounds/r01_occupancy/run_log.txt`: console log of the run.
