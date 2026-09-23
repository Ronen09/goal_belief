# Prior or recomputation? Belief transplants through the K/V cache of a next-token transformer (TASK14, round 15)

Brief: `rounds/r15_prior_vs_recompute/BRIEF.md`. The predictions in §5 were committed **before any model of this round was
trained**.

## 1. What position t exports

In a pre-LN causal transformer with L blocks, position t+1 reads position t only through the keys
and values computed from position t's residual stream: block i attends to LN₁ⁱ(res_i(s)) for s ≤ t+1,
where res_0 = token + position embedding. So position t exports L residual vectors
res_0(t), …, res_{L−1}(t):
* **res_0(t)** is the embedding of the raw token x_t and its position. It carries evidence, not
  belief.
* **res_1(t), …, res_{L−1}(t)** are computed from the prefix and can carry a belief. In a 2-layer
  model there is exactly one of them, res_1(t), read by block 2.

"Editing the exported K/V" is therefore editing res_i(t) for i ≥ 1 as the source of the keys and
values, with everything else fixed. Position t+1's own stream then follows.

`goalgeo/kvprior.py` computes position t+1 from an arbitrary set of exported residuals for
positions 0..t (per layer), optionally hiding some positions from its attention. Every cell of
the 2×2 is an instance of that one function.

## 2. The 2×2 (and its variants)

Histories A and B with very different full filter states at t (round 13's 7 coordinates), and the
next observation x = x_{t+1} taken from A. A⁺ = BayesUpdate(α_A(t), x), B⁺ = BayesUpdate(α_B(t), x).

| cell | exported by positions < t | exported by position t (layers ≥ 1) | token at t | expected if pure recomputation / pure prior |
|---|---|---|---|---|
| R1 baseline | A | A | A | A⁺ / A⁺ |
| R2 prior edited (**swap**) | A | B's res_i(t) | A | A⁺ / B⁺ |
| R2 prior edited (**probe**) | A | A's res_i(t) moved along the probe's encoder directions to B's decoded full state, per layer | A | A⁺ / B⁺ |
| R3 evidence edited (**corrupt**) | B | A | A | B⁺ / A⁺ |
| R3 evidence removed (**mask**) | hidden from t+1 | A | A | (prior only) / A⁺ |
| R4 consistent | B | B | B | B⁺ / B⁺ |

**Read-out.** The full state is decoded at t+1 from each site (res_1(t+1), …, u(t+1)) with the
affine probe fit on genuine states. λ is the position of the decoded state along A⁺ → B⁺ in the
7-coordinate space, pooled over pairs: λ = Σ(z − z_{A⁺})·Δ / Σ‖Δ‖². The perpendicular residual is
reported as a fraction of ‖Δ‖. The same λ is computed for the model's output, the next-token
predictive in centred-log coordinates, against the exact predictives of A⁺ and B⁺.

**Mechanistic correlate.** The attention weight of query t+1 on key t, per layer (genuine runs).

## 3. Models

Standard next-token transformers on the channel environment, trained on **sampled** next tokens
(plain cross-entropy; no exact targets, no carry). d = 64, 4 heads, MLP 128, learned absolute
positions, final LayerNorm. L ∈ {2, 4} × context T ∈ {24, 64} observations × 3 seeds, 30k Adam
steps at 1e-3, batch 128, a pool of 100k sequences. Convergence: KL of the model's predictive to
the exact predictive, averaged over positions.

Intervention positions: t ∈ {2, 4, 8, 16, 23} for T = 24 and t ∈ {2, 4, 8, 16, 32, 48, 63} for
T = 64. 1000 pairs per t with full-state distance ≥ 2 between A and B.

## 4. What the architecture already implies

* **An attention average dilutes position t by the number of positions.** If block i ≥ 2 at t+1
  attends roughly uniformly, position t's export has weight ~1/(t+1) against the other t
  positions, which all carry the same history's evidence. Pure recomputation then gives λ → 0 as
  t grows, and any sizeable λ at large t requires attention concentrated on the previous
  position.
* **The 2-layer model has one belief-carrying export**, res_1(t), and res_1 is itself computed by
  a single attention layer over tokens, so its belief is partial (round 12: goal-block R² 0.90 at
  res_1). A 4-layer model has three.
* **The brief's hypothesis** is that the recursive share grows when recomputation is harder,
  i.e. with longer histories. The dilution argument predicts the opposite in this architecture,
  and so do the predictions below.

## 5. Pre-registered predictions

Seed means over converged models (predictive KL < 0.01), decoded at u(t+1) unless stated.

* **Q1 (positive control).** R4: λ ≥ 0.9 for every model and t.
* **Q2 (mostly recomputation).** 2-layer models, R2-swap: λ ≤ 0.3 at every t ≥ 8, both T.
* **Q3 (the prior's weight falls with context).** R2-swap λ is larger at t = 4 than at t = 16
  (T = 24 models), and larger at t = 8 than at t = 48 (T = 64 models), for both depths.
* **Q4 (depth).** 4-layer R2-swap λ > 2-layer R2-swap λ at t = 8 and t = 16, both T.
* **Q5 (the belief component carries the effect).** R2-probe λ ≥ 0.7 × R2-swap λ, averaged over
  t ≥ 4, every model class.
* **Q6 (sum rule: two sources, added).** λ(R2-swap) + (1 − λ(R3-corrupt)) is within [0.8, 1.2]
  at t = 8 and 16. That is, the weights on the two conflicting sources sum to one.
* **Q7 (hybrid on the line).** For R2-swap and R3-corrupt, the perpendicular residual is ≤ 0.3 of
  ‖B⁺ − A⁺‖ at t ≥ 8.
* **Q8 (little recursion without rereading).** R3-mask at t ≥ 8: the decoded state at t+1 is
  closer to the prior-only update (BayesUpdate(initial joint, x)) than to A⁺, i.e. retention
  1 − ‖z − z_{A⁺}‖ / ‖z_{prior⁺} − z_{A⁺}‖ ≤ 0.5, for the 2-layer models.
* **Q9 (attention share predicts the prior's weight).** Across model classes and t, Spearman
  between R2-swap λ and the mean attention of t+1 on t (layers ≥ 2) is ≥ 0.7.
