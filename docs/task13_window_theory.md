# A windowed transformer: does restricting attention force a steerable belief state? (TASK13, round 14)

Request: train a transformer whose attention reaches only the last l tokens, so that it has to form a
stronger, more steerable belief representation. Follows rounds 12–13: round 12 found that a
full-attention transformer has no state that carries the belief forward (single-position edits
move ≤ 16% of the future), while the GRU state is a complete cut and is fully steerable. The
predictions in §4 were committed **before any model of this round was trained**.

## 1. Why a window alone is not enough

With window l, position t in layer 1 reads tokens t−l+1..t, and layer 2 reads layer-1 residuals of
positions t−l+1..t. A 2-layer model therefore sees at most 2(l−1)+1 tokens. For l ≤ 8 that is
≤ 15 < 24, so evidence older than the receptive field is simply lost, and the posterior cannot be
computed at late positions. A window restricts access; it does not create a place where a belief
must be kept.

## 2. The carry

The recurrent variant adds the previous position's readout interface to the next position's input:
x₀(t) = emb(o_t) + W_c u_{t−1}, with u = LN_f(x_top). Now:
* **l = 1 with carry**: attention is to the current position only, and u_{t−1} is the only path
  from the past to the future. u_t is a complete cut, as the GRU state is.
* **l > 1 with carry**: the past reaches t through u_{t−1} and through the window. The complete
  cut is u_t together with the window's residuals, and a single-site edit of u_t controls only
  the carry's share.
* **full window (l = 25) without carry**: the round-12 transformer, except that positions are a
  learned relative bias, not absolute embeddings.

No model has absolute positions, so no state records elapsed time.

## 3. Grid and measurements

Window l ∈ {1, 2, 4, 8, 25} × carry {no, yes} × {iid, channel} × 3 seeds, objective `goal` (exact
posterior targets), K = 4, the round-12 environments, 15k Adam steps at 1e-3, batch 128.

* **Accuracy**: KL to the exact posterior, overall and per position.
* **Recoverability**: affine R² of the goal block (and, in the channel env, of the channel block,
  as in round 13) from u, IID split.
* **Steerability**: round 12's transplant pairs (t = 6, 12; 1000 pairs). Edits at one site at
  position t: SWAP u_t with B's; PROBE-E on u_t to B's decoded state (goal block in iid, full
  7-coordinate state in channel, fit on u); SWAP res1(t) with B's. F = fraction of the gap to the
  genuine-B model future closed (k ≥ 1).
* **Equivalence** (channel): equal full state at lengths 8 vs 16 (round 13's Ex, tolerance 0.05),
  as a behavioural test: model divergence of H₁⊕c and H₂⊕c relative to the Bayes divergence.

## 4. Pre-registered predictions

Seed means.

* **W1.** Without carry, l ∈ {1, 2, 4, 8} fail (overall KL ≥ 0.01) in both environments; the full
  window converges (KL < 0.01).
* **W2.** With carry, every window converges (KL < 0.01) in both environments.
* **W3.** Carry, l = 1: SWAP u_t closes ≥ 0.95 of the future gap to genuine-B and PROBE-E ≥ 0.9,
  in both environments.
* **W4.** Carry: SWAP-u_t future F decreases as l grows (l = 1 > 2 > 4 > 8 > 25), and is ≤ 0.5 at
  l = 25.
* **W5.** Without carry: editing u_t moves the future by exactly 0 (nothing reads it), and SWAP
  res1(t) closes ≤ 0.3 of the future gap for every converged model.
* **W6.** Carry, l = 1, channel: the cross-length equal-state divergence is ≤ 3× Bayes.
* **W7.** Carry, l = 1, channel: from u, goal-block R² ≥ 0.98 and channel-block R² ≥ 0.9.
