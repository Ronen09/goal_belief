# Transformer reproduction (TASK9)

Run date: 2026-09-18. 60 models: a 2-layer pre-LN causal transformer (d=64, 4
heads, MLP 128, learned positions, no dropout), with and without the final
LayerNorm, on the TASK4 HMM. 51 trained on CPU in 54 minutes (8 workers, 4000
Adam steps at 1e-3), the nine that had not reached the Bayes floor were retrained
on the GPU with 10,000 steps in 5 minutes (the transformer trains 16× faster on
the GPU: 8 ms per step against 130 ms on one CPU thread; the runner now defaults
to it). Numbers from `rounds/r10_transformer/tables.md`, per-run values in
`rounds/r10_transformer/results9.json`, the cut diagnosis in `rounds/r10_transformer/followup_cuts.json`,
weights in `rounds/r10_transformer/models/`. Reproduce with `rounds/r10_transformer/run.py --jobs 4`
then `rounds/r10_transformer/followup.py`.

## Setup

Conditions, 3 seeds each, both with and without the final LayerNorm: the core
set δ ∈ {0.1, 0.2, 0.4}, r=0 and λ=0 at δ=0.4; the allocation set at δ=0.4,
fixed unembedding gains {0.5, 2, 8} and readout learning-rate ratios {0.1, 10}.
Two cuts are measured. The "representation" is the residual stream after
block 1 (the stream that block 2's attention reads from earlier positions, the
analogue of the GRU state); the "readout interface" is the post-final-norm
vector at t*−1 that the unembedding reads. Functional measures use residual
patching: the block-1 residual of a B-branch sequence is written into an
A-branch sequence and the model is run forward from there.

Convergence: 53 of 60 models are below 0.003 nats of the Bayes floor. The six
λ=0 models are above it by design (the relevant step is never trained) and one
gain-8 model without LayerNorm stayed at 0.22 after retraining. Three seeds of
the first pass were stuck in bad basins at 4000 steps and converged at 10,000;
transformer training on this task is more seed-sensitive than the GRU's.

## 1. Availability is built, not carried

Linear branch accuracy at t*−1, where the last token is a filler for both
branches, with the final LayerNorm (chance 0.5; the init value is above chance
because untrained attention already mixes the cue token into later positions):

| condition | block-1 residual | final residual | post-norm | JS, patch at t | P_metric at t*−1 |
|---|---|---|---|---|---|
| init | 0.72 | 0.80 | 0.80 | 0.000 | 0.58 |
| r=0 (cue never matters) | 0.72 | 0.80 | 0.80 | 0.000 | 0.41 |
| λ=0 (relevant step unweighted) | 0.72 | 0.79 | 0.79 | 0.000 | 0.34 |
| δ=0.1 | 0.81 | 0.93 | 0.93 | 0.09 | 0.52 |
| δ=0.2 | 0.90 | 0.98 | 0.98 | 0.09 | 0.57 |
| δ=0.4 | 0.95 | 0.98 | 0.98 | 0.19 | 0.48 |

Without the final LayerNorm the same pattern holds: 0.80 to 0.81 for r=0 and
λ=0, 0.98 for every δ.

This is the architectural prediction, and it came out. In the GRU the branch was
99 % decodable at t*−1 in every condition, because the recurrence carries the
cue forward whether or not it is used. The transformer has to spend attention to
move the cue from t to t+1, and it does so in proportion to need: not at all
when the cue never matters (decodability stays at its initialisation value),
partly at δ=0.1, fully at δ ≥ 0.2. Availability at the point of use is a
property of the implementation class, and here the implementation class does
not supply it for free. At the cue position itself the branch is trivially
decodable (accuracy 1.000 everywhere): the token embedding is in the residual.

P_metric at t*−1 rises only weakly with use (0.34–0.41 unused, 0.48–0.57 used,
against 0.58 at init), and P_metric at the cue is dominated by the token
embedding and carries no signal. In this architecture decodability, not
distance, is the quantity that tracks use.

## 2. The required gap is met, but the split moves from scale to alignment

Achieved contrast (w_x − w_y)·Δh̄ at the post-norm interface in every converged
δ=0.4 model: 4.21 to 4.24 (required 4.39; the exact-target optimum is 4.30).
With the final LayerNorm:

| condition | ‖w_x−w_y‖ | ‖γ‖ | post-norm ‖Δh̄‖ | post-norm cos θ | pre-norm final ‖Δh̄‖ |
|---|---|---|---|---|---|
| gain 0.5 | 0.44 | 11.0 | 10.4 | 0.92 | 17.5 |
| free (base) | 0.98 | 8.6 | 7.1 | 0.62 | 15.1 |
| gain 2 | 2.56 | 7.9 | 5.0 | 0.34 | 6.1 |
| gain 8 | 10.3 | 7.8 | 3.7 | 0.11 | 5.6 |
| lr ×0.1 | 0.86 | 10.4 | 5.7 | 0.88 | 12.5 |
| lr ×10 | 1.97 | 7.7 | 6.4 | 0.34 | 7.5 |

Without the final LayerNorm:

| condition | ‖w_x−w_y‖ | ‖Δh̄‖ | cos θ |
|---|---|---|---|
| gain 0.5 | 0.61 | 11.5 | 0.62 |
| free | 0.85 | 9.4 | 0.53 |
| gain 2 | 2.40 | 5.9 | 0.31 |
| gain 8 | 10.0 | 2.6 | 0.19 |
| lr ×0.1 | 0.83 | 8.6 | 0.60 |
| lr ×10 | 1.20 | 7.8 | 0.47 |

The identity g · ‖Δh̄‖ · cos θ = Δℓ holds in every row, as it must, but the
three factors share the work differently from the GRU. In the GRU a 23× range
of gain was absorbed almost entirely by separation, with alignment near 1 at
low gain. Here, with the final LayerNorm, separation moves only 2.8× while
alignment moves from 0.92 to 0.11: the post-norm vector has a fixed norm
budget, so the network cannot grow the separation past it and instead rotates
the separation away from the readout as the readout gets stronger. The
LayerNorm gain γ absorbs a modest 1.4×. The learning-rate manipulation, which
moved the GRU's separation 2.3×, moves the post-norm separation here by 12 %
and the alignment by 2.6×. Without the final norm the separation recovers part
of its role (4.4× over the gain range, roughly g^−0.5) and alignment does the
rest. The pre-norm final residual, which is what one would measure as "the
representation" in a transformer, varies 3× across these functionally
identical models with the norm and 4× without.

So the scale-allocation result generalises in form and changes in content: the
function fixes the product, the architecture decides which factor is free, and
LayerNorm hands the freedom from norm to angle.

## 3. Invariants, and a correction to the functional object

Coefficient of variation across the functionally equivalent δ=0.4 models (18
with the final LayerNorm, 17 without):

| measure | with LN | without LN |
|---|---|---|
| ‖Δh‖ at the cue (block 1) | 0.22 | 0.18 |
| hidden norm (block 1) | 0.31 | 0.21 |
| P_metric | 0.31 | 0.17 |
| Euclidean RSA (block 1) | 0.15 | 0.10 |
| belief R² (block 1, OLS) | 0.09 | 0.07 |
| branch accuracy at t*−1 (post-norm) | 0.003 | 0.002 |
| readout gain | 1.20 | 1.30 |
| post-norm separation | 0.34 | 0.34 |
| cos θ | 0.56 | 0.36 |
| achieved contrast (w_x−w_y)·Δh̄ | 0.004 | 0.003 |
| ‖JΔh‖ (patch Jacobian) | 1.95 | 1.36 |
| D_F | 1.13 | 2.65 |
| JS, patch at position t only | 0.96 | 1.74 |
| JS, patch at t and t+1 (complete cut) | 0.07 (all 35) | |
| rank of the block-1 residual | 0 | 0 |

The readout contrast and decodability at the interface are again the stable
quantities, and raw geometry again is not. Two things are new.

**The single-position patch is not invariant.** JS after replacing the
block-1 residual at the cue position ranges from 0.000 to 0.33 across models
that compute the same function (CV 0.96). The cue reaches the prediction at
t+1 by two routes: block-2 attention from t+1 back to the residual at t, or
block-1 attention at t+1 copying the cue into t+1's own residual, which block 2
then reads locally. Block-2 attention from t+1 to t is 0.02 in some seeds and
0.30 in others; the models that route through t+1 give JS 0.00 when t is
patched and 0.35 when t+1 is patched, and the reverse for the others. Patching
both positions, which is the complete cut between the cue token and the
prediction, gives JS 0.379 ± 0.027 in all 35 δ=0.4 models (CV 0.07), against
0.130 (CV 1.28) for t alone and 0.311 (CV 0.31) for t+1 alone. The theory
note's claim that Φ descends to functional equivalence assumed the state is a
cut of the computation graph; the GRU state is, a transformer residual at one
position is not, and `rounds/r08_invariants/THEORY.md` §4 now states the condition. Which
route a model uses is an implementation choice invisible in behaviour, exactly
like the readout gain.

**Local metrics fail harder.** ‖JΔh‖ and D_F have CV 1.1 to 2.7, larger than in
the GRU (0.2 to 0.3), because the Jacobian at a single position inherits the
routing choice on top of the linearisation error.

## 4. Propagation depth after a single-position patch

JS between patched and unpatched predictions at positions t to t+4, seed means:

| model | k=0 | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| base | 0.000 | 0.19 | 0.016 | 0.000 | 0.036 |
| lr ×0.1 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 |
| lr ×10 | 0.000 | 0.21 | 0.031 | 0.000 | 0.016 |

Unlike the GRU, where the divergence was exactly zero at every step other than
the consumption step, the transformer shows small effects at k=2 and k=4 in
the models that route through position t. A B-branch residual left in an A
context stays readable by every later position, and after the relevant token
it is off-manifold for that context. The raw final-residual difference is
correspondingly position-dependent (4 to 12 at k=0–2, 0.2 at k=3, 4 to 7 at
k=4). Depth curves in a transformer are curves of the patch's reach, not of a
state's erasure.

## Conclusions

1. The core dissociation reproduces with a sharper edge. In the GRU
   decodability was free and prominence tracked use; in the transformer
   decodability itself tracks use, because carrying a cue forward costs an
   attention pattern the optimiser only builds when the cue matters.
2. The required-gap identity holds at the readout interface in every model.
   With the final LayerNorm the freedom the function leaves is spent on
   alignment rather than separation; without it, separation regains about half
   of its GRU role. Raw distances in the residual stream vary 3× to 4× across
   models with identical outputs either way.
3. The invariants carry over: the readout contrast and interface decodability
   are stable to CV 0.004; raw geometry, gain, separation and alignment are
   not; local Jacobian and Fisher metrics are the least stable of all.
4. The functional object needs a complete cut. A single position's residual
   measures the share of the computation that happens to pass through it, and
   that share is an implementation choice that varies from seed to seed.
   Patching the full set of positions between the cue and its use restores the
   invariant.
5. Training practicalities: use the GPU (16× faster per step for this model),
   expect some seeds to need more than 4000 steps at lr 1e-3, and keep the
   λ=0 control for what it is, a model that never learns the relevant step.
