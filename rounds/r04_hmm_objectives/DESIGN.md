# One-step vs k-step vs sequential objectives in an HMM — design (TASK4)

Date: 2026-09-17. Tests the supervision-bottleneck prediction on belief states.

## HMM

Six hidden states over the alphabet {p, q, x, y}:

| state | emits | goes to |
|---|---|---|
| P | p | A |
| Q | q | B |
| A | x 0.5, y 0.5 | C |
| B | x 0.5, y 0.5 | D |
| C | x 0.9, y 0.1 | P 0.5, Q 0.5 |
| D | x 0.1, y 0.9 | P 0.5, Q 0.5 |

After observing p the belief is b₁ = δ_A; after q it is b₂ = δ_B. Both predict the
next symbol as (x 0.5, y 0.5), but two steps ahead they predict x 0.9 versus y 0.9.
So b₁ ≠ b₂, P(X_{t+1}|b₁) = P(X_{t+1}|b₂), P(X_{t+1:t+2}|b₁) ≠ P(X_{t+1:t+2}|b₂).
Control pair: δ_C versus δ_D, whose one-step predictions already differ.

A `noisy` variant makes the cues unreliable (P emits p 0.8 / q 0.2, Q the reverse)
and the return transitions state-dependent (C → P 0.7, D → Q 0.7), so beliefs
are mixtures and the belief space is richer; the one-step equivalence of the two
cue-beliefs still holds exactly because A and B share an emission distribution.

## Objectives (same GRU, 64 hidden units, 16-dim token embedding, Adam, 3000 steps)

1. **one-step**: input = window of the last 6 tokens, target = exact
   P(X_{t+1} | window) (forward algorithm from the stationary prior), soft cross-entropy.
2. **k-step**: same input, target = exact joint P(X_{t+1:t+k} | window) over the
   4^k continuations, k ∈ {2, 3}; also a `marginals` variant (k separate one-step
   heads for steps 1..k).
3. **sequential**: next-token loss at every position of 32-token sequences,
   exact per-position conditionals as targets (and a sampled-token variant, the
   standard LM loss).

An MLP on the concatenated one-hot window is run for objectives 1 and 2 as an
architecture check. 5 seeds.

## Measurements

Evaluation histories: 3000 sampled 12-token prefixes. Reference quantities per
history (computed from the tokens the model actually sees): belief over hidden
states, one-step predictive, 2- and 3-step joint predictives.

- Key pair: hidden distance between histories ending in p and in q, divided by the
  distance between histories ending in (p, x/y) and (q, x/y) (beliefs C vs D), and
  by the median pairwise distance. Prediction: one-step ≈ 0; k-step and sequential
  comparable to the control.
- RSA of the hidden RDM with belief, one-step, 2-step and 3-step RDMs; ridge-CV
  decoding of the belief; within/between one-step-equivalence-class distance.
