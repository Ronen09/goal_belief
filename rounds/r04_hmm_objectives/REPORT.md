# One-step, k-step and sequential objectives on an HMM (TASK4)

Run date: 2026-09-17, on the ROCm GPU (RX 7800 XT). 3 seeds, 1500 Adam steps,
batch 128. Numbers from `rounds/r04_hmm_objectives/tables.md`; figures `hmm_key_pair.png`,
`hmm_rsa.png`. Design: `rounds/r04_hmm_objectives/DESIGN.md`.
Reproduce with `rounds/r04_hmm_objectives/run.py --device cuda` (about 5 minutes; CPU is as fast).

## The HMM

Six hidden states over {p, q, x, y}. P emits p and moves to A; Q emits q and moves
to B; A and B both emit x or y with probability 0.5 and move to C and D
respectively; C emits x with 0.9, D emits y with 0.9, and both return to P or Q.
After a p the belief is b₁ = δ_P (next state A), after a q it is b₂ = δ_Q (next
state B). Both predict the next symbol as (x 0.5, y 0.5); two steps ahead they
predict x 0.9 versus y 0.9. The control pair is the beliefs after (p, x/y) and
(q, x/y), which already differ at one step. A noisy variant makes the cues 80%
reliable and the return transitions state-dependent, so beliefs are mixtures; the
one-step equivalence of b₁ and b₂ is preserved exactly because A and B share an
emission distribution.

Reference geometry facts: the one-step and two-step predictive RDMs correlate at
only 0.37 (clean) and 0.52 (noisy), so the objectives are genuinely different
targets.

## Result

Hidden distance between b₁ and b₂ histories divided by the distance between the
control pair, and RSA of the hidden RDM with each predictive (clean HMM; the noisy
HMM gives the same ordering):

| model / objective | d(b₁,b₂) / d(C,D) | d(b₁,b₂) / median | RSA 1-step | RSA 2-step | RSA 3-step | belief decodable, R² |
|---|---|---|---|---|---|---|
| MLP, one-step | 0.24 | 0.33 | 0.88 | 0.41 | 0.75 | 0.98 |
| MLP, 2-step joint | 1.05 | 0.30 | 0.50 | 0.80 | 0.83 | 0.99 |
| GRU, one-step | 0.85 | 0.55 | 0.75 | 0.52 | 0.83 | 1.00 |
| GRU, 2-step joint | 1.06 | 0.44 | 0.46 | 0.83 | 0.83 | 1.00 |
| GRU, 3-step joint | 1.17 | 0.51 | 0.46 | 0.82 | 0.83 | 0.98 |
| GRU, 2-step marginals | 1.06 | 0.43 | 0.76 | 0.67 | 0.83 | 1.00 |
| GRU, sequential (exact targets) | 0.84 | 0.55 | 0.74 | 0.52 | 0.83 | 1.00 |
| GRU, sequential (sampled tokens) | 0.83 | 0.70 | 0.76 | 0.52 | 0.83 | 1.00 |
| GRU, random init | 1.34 | 1.27 | 0.09 | 0.27 | 0.30 | — |
| MLP, random init | 1.11 | 0.92 | 0.30 | 0.43 | 0.49 | — |

**One-step versus k-step: the predicted hierarchy holds.** With a window MLP,
b₁ and b₂ end up at 0.24 of the control distance under one-step training and at
1.05 under 2-step training (0.38 and 1.09 in the noisy HMM). Each model's geometry
is best explained by its own target: the one-step model by the 1-step predictive
(0.88), the 2-step model by the 2-step predictive (0.80), and the marginals model
by the 1-step predictive with the 2-step in between. The 3-step joint RDM in the
clean HMM is nearly the belief-class RDM, which is why it correlates at 0.83 with
every trained GRU.

**The GRU does not collapse b₁ and b₂ under one-step training, it just stops
amplifying them.** At initialisation the two groups sit at 1.34 of the control
distance because the last input token differs. One-step and sequential training
bring that to 0.84–0.85 while k-step training keeps it at 1.06–1.17. The belief is
linearly decodable from every model at R² ≈ 1, including the one-step model, so
the distinction survives as an unamplified direction of the recurrent state.

**Sequential next-token training behaves like one-step in geometry, not like
k-step.** Its RSA profile (0.74 with the 1-step predictive, 0.52 with the 2-step)
and its b₁/b₂ ratio (0.84) match the one-step model, with exact or sampled targets.
The recurrent state after p must carry the cue for the next update, and it does
(the belief decodes perfectly and the model's loss is at the entropy floor), but
the *geometry* of that state is organised by the immediate target. The
information required by the future is preserved; it is not what the
representation's metric is built around.

## Reading against the hypothesis

The supervision-bottleneck prediction was: one-step gives h(b₁) ≈ h(b₂), k-step
separates them whenever their k-step predictions differ. Both parts hold in the
cleanest setting (independent examples, feed-forward model). Two refinements:

1. "No reason to separate" produces collapse only relative to the distinctions the
   objective does amplify, and only where the architecture does not carry input
   differences forward for free. The MLP collapses; the GRU shrinks the ratio from
   1.34 to 0.85 and keeps the information.
2. Sequential training does not sit with k-step in the hierarchy as far as
   geometry goes. It sits with one-step: the per-position loss is one-step, and the
   recurrence only requires that the cue be recoverable one update later, not that
   it be expressed in the readout-visible geometry now. Explicit k-step targets are
   what put the future into the metric.

## Files

- `rounds/r04_hmm_objectives/tables.md`: all numbers, both HMM variants, within-group
  distances, one-step-class compression and belief/2-step decoding.
- `rounds/r04_hmm_objectives/results_hmm.json`: per-seed values.
- `goalgeo/hmm.py` (exact inference, k-step joint predictives, sampling),
  `goalgeo/seqmodels.py` (GRU and window MLP, the three objectives), `tests/test_hmm.py`.
