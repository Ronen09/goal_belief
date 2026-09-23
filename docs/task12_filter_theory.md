# The full filter state: does a recurrent network learn the minimal predictive state? (TASK12, round 13)

Brief: `TASK12.md`. Follows round 12 (`results11/REPORT11.md`, `results11/causal/REPORT_causal.md`),
whose channel-env results showed that equal-goal-posterior histories diverge by exactly the Bayes
amount. This round decodes and intervenes on the **whole** filter state. The predictions in §5
were committed **before any measurement or training of this round**. Environment facts (§1) come
from the filter alone.

---

## 1. The minimal sufficient statistic of the channel environment

The generative model has hidden state (G, c_t) with 2K = 8 values. The filter α_t(g, c) =
P(G = g, c_t = c | o_≤t) is sufficient for every future quantity. It is also **minimal**. The
observability matrix O[(g, c), s] = P(o_{t+1..t+L} = s | g, c_t = c) has rank 5 at L = 1 and
full rank 8 at L ≥ 2. Distinct filter states therefore predict distinct futures, and the
minimal predictive state is the whole joint, with 7 degrees of freedom (K = 4).

Coordinates. The factorised log-odds

    y_g = log b(g)/b(K)            (3, the goal block; the round-12 target)
    ℓ_g = logit P(c_t = on | G = g) (4, the channel block, one per goal)

are an invertible reparametrisation of α. The pair of marginals (b, P(on)) suggested in the brief
is **not** sufficient. P(on | g) differs across goals by a median of 0.56 (90th percentile 0.90),
and the marginals predict the channel block with affine R² of only 0.68. Pairs with equal
marginals but channel blocks differing by ≥ 0.5 logit are common (57k in a 200k-history pool).

**Supervision exposes only projections.** The round-12 objectives' targets are functions of α of
lower rank: `goal` and `act_soft` see b (3 degrees of freedom), and `next_obs` sees the one-step
predictive (rank 5). No objective's target determines α. A model that carries α anyway has learned
what the future of its own targets requires, which is the full state, since future posteriors and
predictives depend on all 7 coordinates.

## 2. What would count as "learned the minimal predictive state"

The equivalence to test is the one the brief names. For a model M with state h at a complete cut
(the GRU):

    α(H₁) = α(H₂)  ⇒  M's outputs after H₁⊕c and H₂⊕c agree for every continuation c.   (⇐ also, by §1)

"Functionally equivalent to the minimal predictive state" means this holds with the model's own
divergence no larger than the Bayes divergence the matching tolerance allows, and that h moves
through α's coordinates causally: editing the decoded α moves the future to the Bayes future of
the edited α. The transformer has no complete-cut state (round 12, part 2) and is used here for
recoverability only.

## 3. Tests

GRU `goal`, `act_soft`, `next_obs` channel models from round 12 (K = 4, 4 seeds; `act_hard`
excluded, not converged), and new bottleneck GRUs (§4).

**R. Recoverability of the full state.** An affine probe from h to the 7 factorised coordinates,
scored per block (goal y; channel ℓ) on IID, TIME (t ≤ 12 → t ≥ 13) and EXT-channel (fit on
max|ℓ| < 2, score on max|ℓ| ≥ 3) splits. Baselines: the channel block predicted from the true
goal block alone and from the true marginals; untrained GRUs. Transformer sites res1, res2, u for
comparison.

**I. Matched interventions in both coordinates.** Pairs (A, B), 1500 per t ∈ {6, 12}. With the
7-dim map (PROBE-E: move along the encoder directions so that the decoded coordinates hit their
target exactly), A's state is moved to:
* **G-only** (y_B, ℓ_A), with Bayes target α = b_B(g)·P_A(c | g);
* **R-only** (y_A, ℓ_B), with Bayes target α = b_A(g)·P_B(c | g);
* **both** (y_B, ℓ_B), with Bayes target α_B, whose model reference is the genuine-B run;
* controls: SWAP (B's state) and RAND (a random direction with the norm of the "both" shift).

Each is scored by the fraction of the gap closed toward its own Bayes future at k ≥ 1, and "both"
also toward the genuine-B model run. Round 12's goal-only probe (3 coordinates, channel left
uncontrolled) closed 0.950 toward genuine-B at t = 6. That is the number to beat.

**E. Equivalence hierarchy.** Pairs at t = 12 (8 continuations each, from H₁'s predictive):
* **Eb**: equal b (|Δy|∞ < 0.05), marginal P(on) differing by ≥ 0.15 (round 12's set);
* **Em**: equal marginals (|Δy|∞, |Δ logit P(on)| < 0.05), channel block differing by
  ≥ 0.5 logit;
* **Ej**: equal full state (|Δy|∞, |Δℓ|∞ < 0.02), different token sequences;
* **Ex**: equal full state across different lengths: H₁ at t = 8, H₂ at t = 16 (same
  tolerance). Both states run the same 8-token continuation from the GRU cut. Nothing in α
  records time, so a network whose state is α must forget how long it has been running;
* **random** pairs (same t; and for Ex, random cross-length pairs).

Reported: model divergence (JS, k ≥ 1) ÷ random, model ÷ Bayes divergence of the same pairs, and
the pair-level Spearman.

## 4. Bottleneck

GRUs with hidden size n ∈ {2, 3, 4, 5, 6, 7, 8, 12, 16, 32} (plus the round-12 n = 64), objective
`goal`, both environments, 3 seeds, the round-12 training recipe (15k steps). The iid minimal
state has 3 degrees of freedom, the channel one 7. Measured: KL to the exact target, and the
full-state probe per block.

## 5. Pre-registered predictions

GRU, channel env, K = 4, seed means, unless stated.

* **F1 (the full state is recoverable although only a projection is supervised).** `goal` and
  `next_obs` models: channel-block IID R² ≥ 0.85 and goal-block ≥ 0.98. The channel block from
  h is better than from the true marginals (0.68). Untrained GRUs: channel block ≤ 0.5.
* **F2 (it generalises).** `goal` models: channel-block TIME R² ≥ 0.8 and EXT-channel R² ≥ 0.6.
* **F3 (both coordinates are causal, separately).** `goal` models, PROBE-E 7-dim: G-only,
  R-only and both close ≥ 0.9 of the gap to their own Bayes futures (k ≥ 1, t = 6 and 12). RAND
  closes ≤ 0.1.
* **F4 (controlling both coordinates beats controlling one).** "Both" closes more of the gap to
  the genuine-B model run than round 12's goal-only probe did (≥ 0.97, against 0.950 / 0.95 at
  t = 6 / 12).
* **F5 (the hierarchy).** `goal` models: model ÷ Bayes in [0.5, 2] for Eb and Em (Spearman
  ≥ 0.5). For Ej, model divergence ≤ 0.002 of random and ≤ 3× its own Bayes divergence.
* **F6 (time is forgotten).** Ex: `goal` model divergence ≤ 0.01 of cross-length random, and
  ≤ 3× its Bayes divergence.
* **F7 (not only the goal objective).** F5's Ej and F6 hold for `next_obs` and `act_soft`
  models.
* **F8 (bottleneck).** `goal` KL < 0.01 nats is reached at n ≤ 4 in iid and needs n ≥ 6 in the
  channel env.
* **F9 (the supervised projection is kept first).** Channel env, n ∈ {3, 4, 5}: goal-block R²
  ≥ 0.9 while channel-block R² ≤ 0.6.
* **F10 (transformer, descriptive contrast).** Transformer `goal` (channel): channel-block R² at
  res2 ≥ the value at u. The interface keeps less of the channel state than the layer that
  computes the posterior.
