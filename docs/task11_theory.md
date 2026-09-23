# A hidden goal: is the Bayesian posterior affinely recoverable? (TASK11, round 12)

Brief: `TASK11.md`. Environment and exact filter: `goalgeo/latentgoal.py` (tests in
`tests/test_latentgoal.py`, which check the filter against brute-force enumeration over
channel paths). The predictions in §5 were written and committed **before any model of this
round was trained**. The brief lists three claims but only states the first. This round tests
claim 1 (recoverability) only.

---

## 1. The environment

A latent goal G ∈ {1..K} is drawn from a uniform prior at the start of an episode. The agent
then sees T = 24 observation tokens o_1..o_T from an alphabet of M = K+2 symbols: token k
favours goal k (P = HIT under goal k, NEIGH under goal k−1, the rest shared), and two neutral
tokens have the same probability under every goal. Two evidence processes:

* **iid**: o_t ~ L[G] independently (HIT 0.30, NEIGH 0.20).
* **channel**: a hidden sticky reliability channel c_t ∈ {off, on}, P(stay) = 0.95,
  starting at (½, ½). When on, o_t ~ L_on[G] (HIT 0.55, NEIGH 0.15). When off, o_t is uniform
  over the alphabet. The exact filter runs over (G, c_t) jointly (2K states).

The object of interest is

    b_t(g) = P(G = g | o_≤t)        (actions do not affect observations, so conditioning
                                     on a_<t changes nothing in this design)

and its K−1 log-odds coordinates y_t = (log b_t(g_k)/b_t(g_K))_{k<K}.

**Objectives.** Every model reads [BOS, o_1..o_T] causally and is trained at every position t
against an exact target computed from the filter:

| objective | target at position t | output dim |
|---|---|---|
| `goal` | b_t (the posterior itself; same minimiser as cross-entropy against the sampled G) | K |
| `act_soft` | softmax(Q_b/τ), τ = 0.1, with Q_b = (b_t(1), …, b_t(K), SAFE) | K+1 |
| `act_hard` | uniform over argmax_a Q_b(a) | K+1 |
| `next_obs` | P(o_{t+1} \| o_≤t), the posterior predictive | M |

Actions: "commit to goal k" pays 1 if G = k, and "safe" pays SAFE = 0.5. The optimal action
depends on b_t: commit to the MAP goal once its posterior exceeds ½, and stay safe until then.

**Environment facts (K = 4, 4000 sequences, positions t ≥ 1, computed from the filter only).**

| | iid | channel |
|---|---|---|
| \|y\| median / 95th percentile (nats) | 1.35 / 4.9 | 0.58 / 7.4 |
| MAP accuracy at t = T | 0.78 | 0.77 |
| fraction of states where "safe" is optimal | 0.32 | 0.40 |
| best affine-in-counts predictor of y, out-of-sample R² | **1.000** | **0.872** |
| best affine-in-counts predictor of b, R² | 0.803 | 0.788 |
| best affine predictor of y from (counts/t, t), R² | 0.560 | 0.554 |
| sequences / states in the conflict region | 0.13 / 0.009 | 0.20 / 0.016 |

K = 3 and K = 5 give the same pattern: counts→y is 1.000 in iid and 0.897 / 0.857 in the
channel env.

---

## 2. What affine recoverability does and does not measure

**2.1 It is an availability invariant.** The out-of-sample R² of an unregularised affine probe
h ↦ Ah + c is exactly invariant to every invertible affine change of coordinates of h. It is
also invariant to the choice of reference goal, because log-odds against another reference
are an invertible linear map of y. It is on the "availability" line of the round-8 table, not
the geometry line, and does not depend on the metric geometry the brief warns about. Round 8
found such measures stable across functionally equivalent models (CV ≤ 0.02).

**2.2 In the iid environment it cannot tell a belief from a counter.** With i.i.d.
evidence,

    y_t = y_0 + Λ n_t,     Λ[k, m] = log L[g_k, m] / L[g_K, m],

where n_t ∈ ℕ^M are the running token counts (`test_iid_log_odds_are_affine_in_counts`). Any
representation from which the counts are affinely readable yields y affinely. In iid,
"y is affinely recoverable" is the same statement as "the Λ-projection of the counts is
affinely recoverable". Counting is exactly the sufficient statistic here, so this is not a
flaw of the probe. It does mean the iid result cannot distinguish "represents the posterior"
from "represents a running tally". In contrast, b is *not* affine in the counts (R² 0.80).

**2.3 The channel environment makes the posterior a nonlinear filter.** With a latent sticky
channel, a run of consistent tokens is evidence that the channel is on, which raises the
weight of every later token. The best affine-in-counts predictor explains R² 0.872 of y, and
the remaining 13% is the part of the posterior that no running tally carries. The diagnostic
statistic is the **gain over counts**

    G_count(h) = 1 − SSE(y − ŷ_h) / SSE(y − ŷ_counts),

both probes fit on the same training states and scored on the same held-out states. G = 1
means the representation carries the exact posterior. G = 0 means it carries no more than a
tally. G < 0 means it carries less.

**2.4 What the objective forces at the readout interface.** Let u_t be the vector the (affine)
unembedding reads: the post-final-LayerNorm vector of the transformer, or the GRU hidden state.
If a model attains its objective's minimiser exactly, then:

* `goal`: z = Wu + c with softmax(z) = b_t, so log b_t(g) = z_g − logsumexp(z) and
  **y_t = (W_{1:K−1} − W_K) u_t + const, an exact affine identity** on every reachable input,
  with no restriction to the training region.
* `act_soft`: softmax(z) = softmax(Q_b/τ), so z = Q_b/τ + const and **b_t is exactly affine in
  u_t**. y is then a nonlinear (log-ratio) function of an affine image of u. It is decodable
  near a fixed point but not globally, and worst where some b_t(g) → 0 (large |y|).
* `act_hard`: the minimiser is not attained at finite logits. Only the decision region (which
  action is optimal) is fixed, and nothing forces any coordinate of b.
* `next_obs`: log P(o_{t+1}) is affine in u. The predictive is a linear image of the joint
  belief, so in iid b is recoverable from it by a pseudo-inverse, but neither b nor y is affine
  in its logarithm. In the channel env the predictive also depends on the channel belief, so
  the model must run the joint (G, c) filter even though G is never a target.

Upstream of the interface, nothing is forced: the information must be present (every objective
except `act_hard` pins down the belief uniquely) but may sit in any coordinates.

**2.5 Transformer mechanism.** Causal attention computes weighted averages over the prefix, so
the cheapest linear statistic a layer can produce is the running mean n_t/t (and t, from
position). y_t = t·Λ(n_t/t) is bilinear in these, and (mean, t) alone gives R² 0.56. A
transformer that holds y affinely has computed that product in an MLP. The GRU instead
accumulates in a bounded state (tanh), so large |y| must be carried by a large readout, not a
large state.

---

## 3. Measurements

Evaluation set: 8000 fresh sequences per environment, shared by every model, with states at
positions t = 1..24. Sites: transformer `res0` (embedding + position; no mixing across
positions, a floor), `res1`, `res2` (after blocks 1 and 2, pre-LayerNorm) and `u` (the readout
interface). GRU: `h` (its only state, which is also its readout interface).

Probe: unregularised least squares on [h, 1] to y (K−1 targets) or to b (K targets). Reported:
R² pooled over coordinates with the test set's own mean, RMSE in nats, G_count, and the
functional error KL(b ‖ softmax([ŷ, 0])) in nats.

Splits, each fitting on one set of states and scoring on a disjoint one:

| split | fit on | score on | what it tests |
|---|---|---|---|
| IID | 4/5 of sequences | the other 1/5 (5-fold) | ordinary recoverability |
| EXT | states with max\|y\| < 2 | states with max\|y\| ≥ 3 | extrapolation to confident posteriors the probe never saw |
| CONF | states outside the conflict region | states inside it | extrapolation to posterior values the probe never saw |
| TIME | t ≤ 12 | t ≥ 13 | one code across time vs a time-dependent one |
| NETHO | CONF split on models *trained* without any sequence that enters the conflict region | | posterior values neither the network nor the probe ever saw |

Conflict region: the second-largest posterior is ≥ 0.45 (iid) or ≥ 0.35 (channel), so two
goals are both strongly supported. Thresholds were chosen so that 13–20% of K = 4 sequences
enter the region. It holds 0.9–1.6% of states (~1.5k evaluation states).

Controls: the same architectures at initialisation (untrained); the counts baseline and the
(mean, t) baseline above.

---

## 4. Grid

Transformer: the round-10/11 2-layer pre-LN causal decoder (d = 64, 4 heads, MLP 128, learned
final LayerNorm), T = 25, trained as stacked models on the GPU (`tfm_batched`). GRU: the
round-4 `SeqNet` (64 hidden, 16-dim embedding). Exact soft targets and Adam.

* main: {iid, channel} × 4 objectives × {transformer, GRU} × 4 seeds
* NETHO: {iid, channel} × 4 objectives × {transformer, GRU} × 3 seeds, trained on pools that
  avoid the conflict region
* K: K ∈ {3, 5} × {iid, channel} × {goal, act_hard} × {transformer, GRU} × 3 seeds
* init controls: each architecture, each environment and K, 4 seeds

A model counts as converged when its mean per-position KL to the exact target is < 0.01 nats
(soft objectives) or its argmax agrees with the optimal action on ≥ 99% of non-tied states
(`act_hard`). Unconverged models are reported and left out of the prediction tests.

---

## 5. Pre-registered predictions

Seed means, K = 4 unless stated. "Every cell" means both architectures × both environments.

* **P1 (forced identity).** `goal` models, interface: R²_y ≥ 0.99 on IID, and ≥ 0.97 on EXT,
  CONF and TIME, in every cell.
* **P2 (coordinates follow the target).** At the interface, `goal` models decode y better than
  b (R²_y > R²_b) and `act_soft` models decode b better than y (R²_b > R²_y), in every cell.
* **P3 (extrapolation exposes the coordinate).** `act_soft` interface: EXT R²_y is at least
  0.2 below the `goal` interface's EXT R²_y, in every cell.
* **P4 (hard targets keep the region, not the posterior).** `act_hard` interface: IID R²_y at
  least 0.1 below `goal`'s in every cell, while the optimal action is linearly decodable
  (least-squares one-vs-rest, argmax) on ≥ 95% of held-out states.
* **P5 (iid is not diagnostic).** Transformer, iid: the best site among res1, res2 and u has
  IID R²_y ≥ 0.95 for all four objectives.
* **P6 (channel is diagnostic).** Channel env: `goal` and `next_obs` models reach
  G_count ≥ 0.7 at their best site, in both architectures. `act_hard` models have lower
  G_count at the interface than `goal` models.
* **P7 (init control).** Untrained networks: IID R²_y ≤ 0.8 at every site, both architectures,
  both environments. In the channel env, G_count ≤ 0.3.
* **P8 (network-level held-out).** NETHO `goal` models: interface R²_y on the conflict states
  ≥ 0.9 (probe fit outside), and the model's own KL(b ‖ p_model) inside the region is at most
  3× its value outside, in every cell.
* **P9 (the transformer's first block codes a running mean).** Transformer, `goal`, iid, res1:
  TIME-split R²_y is at least 0.1 below IID R²_y.
* **P10 (seed stability).** CV across seeds of interface IID R²_y ≤ 0.02 in every trained
  (objective, environment, architecture) cell.
* **P11 (K).** P1 and the `goal` half of P6 hold at K = 3 and K = 5.
