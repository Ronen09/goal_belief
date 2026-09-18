# What makes information geometrically prominent? — design (TASK4)

Date: 2026-09-18. Follows round 4 (`2026-09-17-hmm-objectives-design.md`,
`results_hmm/REPORT_HMM.md`), where a GRU trained on next-token prediction kept
the delayed cue linearly decodable (R² ≈ 1) but did not amplify it (delayed pair
at 0.84 of the control distance). TASK4 asks what controls that amplification:
relevance frequency, relevance strength, loss weight, or credit-assignment delay.

## HMM family `make_hmm4(r, delta, k)` (`goalgeo/hmm4.py`)

Tokens: p q u v x y n (7). States:

| state | emits | goes to |
|---|---|---|
| P | p | FA₁ |
| Q | q | FB₁ |
| FAᵢ, i<k | x 0.5, y 0.5 | FAᵢ₊₁ (same for FB) |
| FA_k | x 0.5, y 0.5 | C with prob r, N with prob 1−r |
| FB_k | x 0.5, y 0.5 | D with prob r, N with prob 1−r |
| C | x 0.5+δ, y 0.5−δ | P, Q, U, V each 0.25 |
| D | x 0.5−δ, y 0.5+δ | same |
| N | n | same |
| U | u | A′ |
| V | v | B′ |
| A′ | x 0.9, y 0.1 | P, Q, U, V each 0.25 |
| B′ | x 0.1, y 0.9 | same |

Properties (tested):

- After p and after q, and after every filler step that follows, the one-step
  predictive is x/y 0.5 for every r, δ, k. The (k+1)-step predictive differs
  iff r > 0 and δ > 0.
- Control pair: after u versus after v the one-step predictive differs by 0.8,
  independent of r, δ, k. This is the fixed immediate-relevance control.
- At r=1, δ=0.4, k=1 the delayed branch is the round 4 "clean" chain.
- r=0 and δ=0 are the permanently irrelevant cue (Control 2).
- The relevant step t* for a cue at t is t+k+1 (token from C, D or N).

Forgetful filter: after each forward update, average the belief mass of each
FAᵢ with FBᵢ and of C with D. ΔL_forget = mean over positions of
KL(true next-token predictive ‖ forgetful predictive), estimated on 2000 sampled
sequences; also per relevant event (divide by the fraction of positions that
are relevant steps). Test: per event it equals r·(ln 2 − H(0.5+δ)) in nats.

Belief, predictive, marginals, sampling and exact sequential targets reuse the
round 4 functions; `hmm4.py` produces an `HMM` dataclass with the same fields
plus branch-mirror indices and the set of "relevant" states {C, D, N}.

## Training (`train_weighted` in `goalgeo/seqmodels.py`)

GRU `SeqNet` (64 hidden, 16 emb, 7-token vocab), sequential objective, exact
per-position targets, soft cross-entropy. T=48, batch 128, Adam lr 3e-3,
3000 steps, seeds {0, 1, 2}. Per-position weight w_t = λ where the emitting
state ∈ {C, D, N}, else 1; loss = Σ_t w_t ℓ_t / Σ_t w_t. Checkpoints (state_dict
copies) at steps {0, 50, 100, 200, 400, 700, 1000, 1500, 2000, 2500, 3000}.

## Measurements per checkpoint (`goalgeo/prominence.py`)

On 2000 held-out sequences (T=48, seed 123), using hidden states at all
positions and the true hidden-state sequence Z.

- **Prominence.** Cue states: h after p (group A) and after q (group B).
  D_delay = mean cross-group Euclidean distance; D_control the same for u vs v.
  P_metric = D_delay / D_control. Also: (i) at the pre-relevant position
  t*−1 (last filler), same denominator; (ii) D_delay / median pairwise
  distance; (iii) variance of all states along d̂ = (mean h_A − mean h_B)/‖·‖
  divided by the mean variance along 20 random unit directions (Control 3).
- **Decodability.** Ridge-CV R² of the belief from 3000 subsampled states
  (`G.ridge_cv_r2`); linear branch accuracy (A vs B) at the pre-relevant
  position, where the last token is x/y for both branches (ridge classifier to ±1, 5-fold; no sklearn in the venv).
- **RSA.** Spearman RSA of the hidden RDM (800 subsampled states) with the
  belief RDM, 1-step predictive, 2-step joint predictive, and future marginals
  over the next k+2 steps.
- **Gradients.** For each sequence take the last cue t with t* ≤ T−1; loss =
  Σ_sequences ℓ_{t*}; one backward pass gives ∇_{h_t} ℓ_{t*} per sequence
  (sequences are independent). G_t = mean ‖·‖, G_∥ = mean |∇·d̂|. Also the
  same for the immediate loss ℓ_{t+1} as a reference. Each reported at init,
  final, and integrated (mean over checkpoints). The integrated values are the
  regressors, since gradients of an exact-target loss vanish at convergence.
- **Performance.** Mean KL to the exact targets (loss minus target entropy),
  overall and at relevant steps (Control 5).

## Experiments (`scripts/run_task4.py`, results in `results4/`)

Base: r=1, δ=0.4, k=1, λ=1.

1. Frequency: r ∈ {0, 0.05, 0.1, 0.25, 0.5, 0.75, 1}.
2. Strength: δ ∈ {0, 0.02, 0.05, 0.1, 0.2, 0.4}.
3. Loss weight: λ ∈ {0, 0.1, 0.25, 0.5, 1, 2, 5}.
4. Delay: k ∈ {1, 2, 4, 8, 16}; plus a matched variant with λ_k = f(1)/f(k),
   where f(k) = π(C)+π(D)+π(N) is the stationary fraction of relevant
   positions, so the relevant loss share per sequence equals the base.
5. Grid r ∈ {0.1, 0.25, 0.5, 1} × δ ∈ {0.05, 0.1, 0.2, 0.4}, run only if
   sweeps 1 and 2 both show Spearman(condition, P_metric) > 0.5 (`--grid`
   forces it). Expected cost = r·ΔL_forget(δ), test collapse onto one curve.

Conditions shared between sweeps are run once. About 140 runs.

Main analysis (all runs, seed-level): standardised OLS of P_metric on belief
R², ΔL_forget (× λ), integrated G_t and G_∥; report betas, R², single-predictor
Spearman correlations, and leave-one-sweep-out R². Dynamics: for each run, the
checkpoint at which decodability, P_metric, G_t and KL reach 90 % of their
total change; report ordering for the base condition and across sweeps.

Figures: (1) R² and P_metric vs r; (2) P_metric vs ΔL_forget with sweep
markers; (3) P_metric vs integrated G_∥; (4) base-condition dynamics;
(5) delay sweep. Outputs: `tables4.md`, `results4.json`, `REPORT4.md`.

## Tests (`tests/test_hmm4.py`, `tests/test_prominence.py`)

HMM one-step equivalences at every filler step, control-pair difference,
reduction to the round 4 chain, analytic forgetting cost, λ=0 gives zero
gradient at t*, prominence identities (identical groups → 0, |G_∥| ≤ G_t,
random-direction ratio ≈ 1 for isotropic data), `--quick` smoke run.
