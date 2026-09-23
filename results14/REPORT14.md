# Prior or recomputation? Belief transplants through the K/V cache of a next-token transformer (TASK14, round 15)

Run date: 2026-09-23. Brief: `TASK14.md`. Theory and 9 predictions: `docs/task14_prior_theory.md`,
committed (`c2fbd70`) **before any model was trained**. Code: `goalgeo/kvprior.py`. Numbers from
`results14/tables14.md` (§1–4 as pre-registered, §5 post hoc), per-model values in
`results14.json`, weights in `results14/models/`, figure `fig1_prior_vs_recompute.png`.
Reproduce: `scripts/run_task14.py` (11 min on one GPU), `scripts/task14_recompute.py`, then
`scripts/task14_tables.py`.

## Setup

**Models.** Standard full-attention next-token transformers (pre-LN, d = 64, learned absolute
positions, no carry) trained on **sampled** tokens from the channel environment with plain
cross-entropy. 2 or 4 layers × context 24 or 64 × 3 seeds, 30k steps. All 12 reach the exact
predictive to KL 0.0014–0.0080.

**What position t exports.** Position t+1 reads position t only through keys and values computed
from res_i(t), one per block. res_0(t) is the embedding of the raw token x_t; res_1(t) … res_{L−1}(t)
carry whatever position t inferred. `forward_query` recomputes position t+1 from any set of exported
residuals, and matches the full forward pass to 10⁻⁶. Every cell of the brief's 2×2 is one call:

| cell | positions < t export | position t exports (layers ≥ 1) | raw tokens |
|---|---|---|---|
| R1 baseline | A | A | A |
| R2 prior edited | A | B's (swap), or A's moved to B's decoded full state per layer (probe) | A |
| R3 evidence edited | B | A | B before t, A at t |
| R3 evidence removed | hidden from t+1 | A | A at t |
| R4 consistent | B | B | B |

x_{t+1} is A's in every cell. Pairs have full filter states ≥ 2 apart at t (1000 pairs per t). The
read-out is λ, the position of position t+1's state along A⁺ → B⁺ (the exact Bayes updates of A's
and B's joint filter states by x_{t+1}). It is measured two ways: on the next-token predictive the
model outputs (needs no probe), and on the full state decoded from u(t+1).

**One correction to the read-out.** The pre-registered λ is raw. In these models the full-state
probe is imperfect (R² 0.79–0.89 at u), so the raw decoded λ drifts with t even for the baseline and
the positive control (R1 from −0.27 to +0.19; R4 from 1.23 to 0.82). §5 of the tables rescales every
λ so that R1 = 0 and R4 = 1, per seed. On the output predictive, R1 and R4 are already 0.00 ± 0.01
and 1.00 ± 0.02, so the output λ needs no correction and is the number to read. Everything below
uses it unless stated.

## 1. The transplanted belief is almost ignored

Weight on B's belief when position t exports B's state but every token is A's (R2):

| t | 2 | 4 | 8 | 16 | 23 / 32 | 48 | 63 |
|---|---|---|---|---|---|---|---|
| 2 layers, T = 24 | 0.012 | 0.004 | 0.010 | 0.029 | 0.039 | | |
| 2 layers, T = 64 | 0.039 | 0.022 | 0.018 | 0.032 | 0.048 | 0.036 | 0.022 |
| 4 layers, T = 24 | 0.020 | 0.007 | 0.009 | 0.023 | 0.029 | | |
| 4 layers, T = 64 | 0.078 | 0.041 | 0.031 | 0.041 | 0.064 | 0.058 | 0.055 |

Swapping every belief-carrying export of position t (one layer in the 2-layer model, three in the
4-layer) moves position t+1 by 0.4–8% of the way to B⁺. Editing only the decoded belief component
with the probe gives the same or less (−0.01 to 0.08). The decoded state at u agrees (0.00–0.09).
Position t+1 does not treat position t's inferred state as a prior.

## 2. Position t+1 recomputes from the tokens it can see, almost exactly

With B's tokens before t and A's everything at t (R3), pure recomputation from tokens predicts a
specific, parameter-free λ: the exact Bayes update of the hybrid token sequence
[B_{<t}, x_t^A, x_{t+1}^A]. The model's output matches it (fig. 1, middle):

| t | 2 | 4 | 8 | 16 | 32 | 48 | 63 |
|---|---|---|---|---|---|---|---|
| pure recomputation (exact) | 0.450 | 0.693 | 0.822 | 0.861 | 0.898 | 0.919 | 0.916 |
| 2 layers, T = 64 | 0.450 | 0.701 | 0.830 | 0.861 | 0.885 | 0.915 | 0.916 |
| 4 layers, T = 64 | 0.475 | 0.720 | 0.844 | 0.877 | 0.912 | 0.930 | 0.926 |

Across all 24 (model, t) cells the model is within −0.016 to +0.029 of the recomputation prediction.
Position t's influence on t+1 in the conflict is almost entirely its raw token x_t, one piece of
evidence among t. At t = 2 that token is a large share of the evidence (λ = 0.45), and by t = 63 it
is small (0.92). Its inferred state adds the few percent of §1.

## 3. Without the older evidence, the exported belief cannot carry the inference

Hiding every position before t from t+1's attention (R3 mask), with position t's genuine export
intact, leaves t+1 further from A⁺ than an uninformed update of x_{t+1} would be (retention
−0.10 to −2.9). The exported state does not stand in for the history it summarises. When the
tokens are removed, the network has nothing it knows how to use.

## 4. Context length and depth

* **The brief's hypothesis, recursion growing with context, points the right way but is
  small.** The prior's weight rises from ≤ 0.02 at t ≈ 4–8 to 0.03–0.06 at t ≥ 16, most in
  the 4-layer, T = 64 models. My pre-registered prediction (Q3: it falls) was wrong in direction,
  but the effect never exceeds 8%.
* **More depth adds a little.** The 4-layer, T = 64 models give the prior the most weight
  (0.03–0.08). The 4-layer, T = 24 ones do not exceed the 2-layer ones.
* **The prior's weight does not track the attention position t receives.** Attention of t+1 on t
  falls from 0.27 (t = 2) to 0.08 (t = 63) while the prior's weight rises slightly, so the
  correlation is negative (ρ = −0.61). The small recursive contribution does not come from
  attending more to the previous position.

## Pre-registered predictions

| | prediction | held | numbers (raw, as registered) |
|---|---|---|---|
| Q1 | R4 λ ≥ 0.9 | no | decoded min 0.819 (probe drift); output 0.99–1.02 |
| Q2 | 2-layer R2 swap λ ≤ 0.3 at t ≥ 8 | **yes** | max 0.161 |
| Q3 | the prior's weight falls with context | no | rises: calibrated 0.004 → 0.029 (T = 24), 0.018 → 0.036 (T = 64) |
| Q4 | 4 layers > 2 layers | no | only for T = 64 in calibrated terms; raw values mixed |
| Q5 | R2 probe λ ≥ 0.7 × R2 swap λ | no | both ≈ 0 (0.00–0.06), so the ratio is noise |
| Q6 | λ(R2) + 1 − λ(R3) ∈ [0.8, 1.2] | no | −0.21 to 0.26: see below |
| Q7 | perpendicular residual ≤ 0.3 | no | 0.16–0.57, equal to the baseline's own probe floor |
| Q8 | 2-layer R3 mask retention ≤ 0.5 | **yes** | −0.23 to −2.5 |
| Q9 | Spearman(λ, attention on t) ≥ 0.7 | no | −0.61 |

2 of 9 held. What the failures show:
* **Q1, Q4 and Q7** are artefacts of the decoded read-out. The probe of the full state is
  imperfect in these models, and it biases raw λ and inflates the perpendicular residual by the
  same amount for the baseline as for the interventions. I should have calibrated against R1 and R4
  in the pre-registration.
* **Q6** was mis-specified. R3 keeps A's raw token at position t, so pure recomputation predicts
  λ(R3) < 1 (§2), and the sum rule does not apply. Once the token is accounted for, the result is
  cleaner than the sum rule would have been: prior weight ≈ 0, token-level recomputation ≈ exact.
* **Q5** is undefined when both quantities are near zero.
* **Q3** failed on substance: the direction was the brief's, not mine.
* **Q9** failed on substance: attention to t is not what carries the small prior weight.

## Conclusions

1. **Answer to the brief's question: position t+1 recomputes.** In a standard next-token
   transformer on this process, the belief inferred at position t is exported through the K/V
   cache, and position t+1 gives it 0.4–8% weight. Given conflicting sources, t+1 follows the
   tokens, and it does so quantitatively: its output equals the exact Bayes update of the token
   sequence it can attend to, to within ±0.03 at every context length, depth and position tested.
2. **The exported belief is not a sufficient summary for the next position.** Removing the
   older evidence breaks the inference even though position t's state is intact.
3. **Recursion grows slightly with context, as the brief guessed,** but reaches only ~6% at 63
   tokens in this architecture. It is not carried by increased attention to the previous
   position.
4. **This completes the architectural contrast of rounds 12–15.** GRUs, and transformers given a
   carry and a narrow window, use their state as a prior (future controlled 1.000 by one state
   edit). A full-attention transformer given the same task and the same belief geometry
   recomputes at every position. The belief is represented in both; whether it is used
   recursively is set by whether the architecture leaves a route around it.
