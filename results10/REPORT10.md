# Implementation freedom: two claims, two falsification attempts (TASK10)

Run date: 2026-09-18. One architecture throughout: the 2-layer pre-LN causal transformer of
`goalgeo/tfm.py` (d=64, 4 heads, MLP 128, T=48) on the TASK4 HMM. The claims and every
prediction below were written down in `docs/task10_theory.md` **before any model in this round
was trained**; this report says which of them survived.

Numbers from `results10/tables10.md`, per-run values in `results10/results10.json`, weights in
`results10/models/`, figures `fig1_routing.png`, `fig2_factorization.png`, `fig3_boundary.png`.
Reproduce with `scripts/run_task10.py --exp all --steps 25000`.

## How the models were trained

Every grid is trained as **one stacked model** (`goalgeo/tfm_batched.py`): the M models of a
grid are stacked along a leading model axis, each op is a batched einsum over M, the loss is
the sum of the per-model losses — so gradients never mix and Adam on stacked tensors is
elementwise identical to M separate Adams — and one optimiser step advances the whole grid.
Weights are initialised by stacking ordinary `CausalTransformer`s (one per seed) and unstacked
back into them for measurement, so the measured objects are the same ones rounds 9–10 handle.

This matters for more than convenience. One process per model left the A100 at 6 % of its
memory bandwidth with 99 % "utilization" — the number that only says a kernel is resident —
because at d=64 the run is kernel-launch bound. Stacking raised throughput from 355 to
**1400 run-steps/s**; TF32 supplied most of that (118 → 73 ms/step at M=102), fused attention
almost none. TF32 is switched on only inside the training loop: every identity tested below is
measured in plain fp32 on the unstacked models. `tests/test_tfm_batched.py` checks that the
batched forward matches the per-model forward to 2e-5, that 25 batched training steps reproduce
per-model training weight-for-weight, and that TF32 moves the trajectory only at the 1e-3 level.

## Experiment 1 — force the routing (Claim A)

**The claim.** Functional equivalence need not identify the effect of an individual internal
node, because equivalent implementations may route the same computation through different
parallel paths; effects defined on a *complete causal cut* do descend to functional
equivalence. Formal statement, proof and the cut enumeration for this architecture:
`docs/task10_theory.md` §2.

**The design.** The cue token at position t must reach the prediction at t+1, and it can travel
by two routes: block-2 attention at t+1 reading the residual at t (the "fetch" route), or
block-1 attention at t+1 copying the cue into t+1's own residual (the "copy-forward" route).
Five training conditions move that choice on purpose — free, block-1 attention hard-masked to
the diagonal, block-2 hard-masked to the diagonal, and soft attention penalties pushing each
way — 4 seeds each. Effects are interchange effects on minimal counterfactual pairs: the same
sequence with the cue token flipped p ↔ q, nothing else changed.

**Result.** Condition means in the table below are over the functionally equivalent models
only; `tables10.md` and `fig1_routing.png` average over all seeds of each condition, so their
l1_diag and pen_l1 rows sit lower (a non-converged seed computes a different function and its
behavioural divergence is smaller). 16 of 20 models reached the Bayes floor (the two hard-masked and the two penalised
conditions each lost a seed to a bad basin). Across those 16, behaviour is
Φ_behav = 0.3679 ± 0.0004.

| condition | node {res₁(t)} | node {res₁(t+1)} | complete cut {res₁(t), res₁(t+1)} | sum of the two nodes | attn₁(t+1→t) | attn₂(t+1→t) |
|---|---|---|---|---|---|---|
| l1_diag | 0.3678 | 0.0000 | 0.3678 | 0.368 | 0.00 | 0.22 |
| pen_l1 | 0.3674 | 0.0000 | 0.3674 | 0.367 | 0.00 | 0.39 |
| free | 0.1851 | 0.2905 | 0.3680 | 0.476 | 0.25 | 0.32 |
| pen_l2 | −0.0000 | 0.3681 | 0.3681 | 0.368 | 0.44 | 0.00 |
| l2_diag | 0.0000 | 0.3679 | 0.3679 | 0.368 | 0.21 | 0.00 |

| pre-registered prediction | outcome |
|---|---|
| P1 every complete cut reproduces behaviour to < 1e-5 | **held**, and not approximately: max deviation over all models, all three complete cuts and all 200 anchors is **0.0** |
| P2 CV(complete cut) ≤ 0.02 across equivalent models | **held**: 0.0011, which is the CV of behaviour itself |
| P3 CV(single node at t) ≥ 0.5 | **held**: 1.13 |
| P3b single node at t < 0.02 under l2_diag | **held**: 0.0000 |
| P4 CV(single node at t+1) ≥ 0.3 | **held**: 0.78 |
| P5 corr(attn₂, node effect at t) > 0.8 | **failed**: the node effect saturates, so the correlation is unstable (0.88 in an earlier process-based run of the same design, below 0.8 here) |
| P6 the sum of the two node effects is not invariant (CV ≥ 0.1) | **held here** (sum ranges 0.367 to 0.798), **failed** in the earlier run, where every seed happened to route exclusively |

Three things are worth separating.

**The theorem is exact, not statistical.** A complete cut patched with the counterfactual run's
values *is* the counterfactual run: the patched output distribution equals the flipped-input
output distribution to the last bit, at every anchor, in every model, including the ones that
never converged. Three different complete cuts — the embedding at t, the layer-1 residuals at
{t, t+1}, and the superset {t, t+1, t+2} — return the same number. That number is behaviour,
which is the deflationary half of the claim: a complete-cut interchange effect tells you
nothing that two clean forward passes would not.

**The single-node effect is the route share, and the route is free.** The same distinction,
measured at one position, ranges from exactly 0 to the full behavioural divergence across
models that compute the same function to 4 decimal places. The architecture fixes nothing here:
forcing either route by masking, or merely leaning on it with a penalty, moves the number end
to end, and the unconstrained condition scatters between the extremes by seed alone.

**The sum of the routes is not a repair.** It is tempting to add the two single-node effects
and call the total the cut. Across these models the sum runs from 0.367 (a seed that routes
exclusively) to 0.798 (a seed that duplicates the cue on both routes, so either position alone
carries it). Only the joint patch is invariant, and only because it is a cut.

One incidental confirmation of the cut enumeration: patching {res₁(t), res₁(t+2)} gives exactly
the same number as patching {res₁(t)} alone (CV 1.13, identical values), because position t+2
is not on any path to the output at t+1. Adding nodes to an incomplete cut buys nothing unless
the added nodes close it.

## Experiment 2 — remove a degree of freedom on purpose (Claim B)

**The claim.** The contrast the function requires, C = g·D·cos θ at the readout interface, is
fixed by the function; the architecture decides how C may be factorised; LayerNorm changes the
available degrees of freedom without changing the requirement. §4 of the theory note states
this, proves that C is a difference of output log-odds, and derives a ceiling: with a frozen
final LayerNorm (γ ≡ 1, β ≡ 0) the post-norm vector has norm exactly √d, so D ≤ 2√d, and with a
fixed-gain readout (row norm c) g ≤ 2c, hence C ≤ 4c√d — predicting failure below
c_crit(δ) = C*(δ)/(4√d) = 0.137 at δ=0.4 and 0.053 at δ=0.2.

**The design.** 147 models at 25 000 steps: three final-norm variants (none, frozen, learned γ)
× 11 fixed readout gains from 0.1 to 2.4 at δ=0.4, and frozen/none × 8 gains from 0.03 to 0.3
at δ=0.2, 3 seeds each. Two controls at the boundary separate "the ceiling forbids it" from
"the optimiser did not get there": a 3× learning rate and a 4× step budget.

### The product is fixed; the factors are not

Across the 77 converged δ=0.4 models — three different architectures at the interface, a 24×
range of readout budget:

| quantity | mean | CV | range |
|---|---|---|---|
| **C** (the contrast) | **4.321** | **0.006** | [4.289, 4.397] |
| D = ‖Δh̃‖ | 20.34 | 0.872 | [2.91, 107.64] |
| g = ‖w_x − w_y‖ | 0.776 | 1.048 | [0.091, 3.411] |
| cos θ | 0.681 | 0.375 | [0.124, 1.000] |

C* at δ=0.4 is 4.394 and no converged model deviates from it by more than 2.4 %. The factors
span 37×. **P7 held.**

### Which factor absorbs the budget, by architecture

Regressing each factor on the gain over the converged models (δ=0.4). The identity forces the
three slopes to sum to zero, and they do, to three decimals — which is a check that the
interface is the right cut, not an assumption:

| norm | d log g | d log D | d log cos θ | sum | d log ‖γ‖ |
|---|---|---|---|---|---|
| none | +0.993 | −0.583 | −0.412 | **−0.001** | — |
| frozen | +1.166 | −0.549 | −0.615 | **+0.002** | — |
| learned γ | +1.043 | −0.832 | −0.212 | **−0.001** | −0.296 |

The pre-registered predictions here were that one named term takes the whole change: D without
a norm (P8), cos θ under a frozen norm (P9c), ‖γ‖ with a learned one (P12). **P8 and P12
failed, P9c held.** Every condition splits the budget between separation and alignment, in
architecture-dependent proportions: the learned-γ models put the most into separation (−0.83)
and the least into alignment (−0.21), the frozen models the reverse (−0.55 / −0.62). The
ordering is predictable from the architecture, the exclusive allocation is not. ‖γ‖ does rise
as the gain falls (7.1 at c=2.4 to 17.4 at c=0.1) but it carries only a third of what was
predicted; the rest arrives as extra separation of the pre-norm states, which γ then scales.

### The ceiling is real, and it is not the one the algebra gives

The frozen-LN hard bound holds: D ≤ 2√d = 16 in every one of the 49 frozen models, largest
observed 14.19 (**P9a held**). But the failure boundary is in the wrong place, and the way it
is wrong is informative.

| frozen, δ=0.4 | c=0.1 | 0.125 | 0.15 | 0.2 | 0.3 | **0.4** | 0.5 | 0.6 | 0.8 | 1.2 | 2.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cos θ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 0.987 | 0.857 | 0.751 | 0.381 |
| D | 11.05 | 11.47 | 11.62 | 11.49 | 10.72 | 9.57 | 8.65 | 8.15 | 5.77 | 4.86 | 3.78 |
| g/c | 1.25 | 1.31 | 1.33 | 1.34 | 1.25 | 1.11 | 1.00 | 0.89 | 1.10 | 1.00 | 1.28 |
| C/C* | 0.31 | 0.43 | 0.53 | 0.70 | 0.91 | 0.97 | 0.98 | 0.98 | 0.99 | 0.98 | 0.99 |
| reached the floor | no | no | no | no | no | **yes** | yes | yes | yes | yes | yes |

Below c ≈ 0.4 the model has spent everything it can spend: cos θ is pinned at exactly 1.000 and
D has plateaued at ≈ 11.3. In that regime C is a straight line in c, which is what a ceiling
looks like. Reading the slope off the models that are *at* it (frozen, missed the floor,
cos θ > 0.99) gives an **attained ceiling** C = κ·c with

    κ = (g/c)·D = 1.29 × 11.1 = 14.4       (δ = 0.4; κ = 13.4–15.5 across gains, spread 1.16×)

against the formal 4√d·c = 32c. The shortfall factors cleanly into the two antipodality
assumptions the bound makes and training does not deliver — (g/2c) = 0.65 and (D/2√d) = 0.70,
whose product 0.45 is the measured ratio κ/32 — and *both* factors are needed: the readout-row
factor alone would put the attained ceiling at 20c, which the data exclude.

| | δ = 0.4 | δ = 0.2 |
|---|---|---|
| formal boundary C*/(4√d) | 0.137 | 0.053 |
| attained boundary C*/κ | 0.306 | 0.220 |
| observed: smallest gain where every seed reaches the floor | **0.4** | **0.3** |
| κ spread across the capped models | 1.16× | 11.3× |

The formal bound is off by 3× and 6×; the attained ceiling gets within 1.3× at both δ. It does
not close the gap, and it should not be reported as if it did. Two caveats matter. First, at
δ=0.2 there is no single attained ceiling to speak of: κ varies 11× across the sub-boundary
gains, because the small-gain models there are degenerate rather than capped (D collapses to
≈ 1 while cos θ still reads ≈ 1), so the 0.220 above uses only the two gains next to the
boundary. Second, even at δ=0.4, where κ is flat to 16 %, the models at c = 0.3 reach 0.913 C*
and KL_rel 0.018 — they fall short by a margin the ceiling accounts for only in part.
**P10, P10b and P11 failed as pre-registered.** The norm supplies a bound; what supplies the
boundary is an attained capacity that has to be measured, and even that leaves ~1.3× unexplained.

The remaining question is why D plateaus at 11.5 rather than at the geometric 16. The δ=0.2
block says the plateau is not a constant of the architecture: there it sits at ≈ 7, a different
value, with the same frozen norm and the same d. The natural reading is that the same post-norm
vector must serve every other prediction the readout makes, so the attainable separation of one
pair of states is limited by the rest of the task — a representational restriction that the
norm's algebra does not see.

The δ=0.2 block also shows a second, different failure. At c ≤ 0.07 the frozen models do not
merely fall short of the contrast, they fail to organise the branch at all: D ≈ 0.4–5.1 and
cos θ ≈ 0.3, far from the saturated ceiling signature. Those models are stuck, not capped, and
they are why the boundary at δ=0.2 (first fully converging gain 0.3) is further from its formal
c_crit (0.053) than the δ=0.4 boundary is from its own.

### Capped, not stuck: the two controls

If the frozen models below the boundary fail because the interface cannot express the contrast,
more optimisation must not help. Both controls say it does not (δ=0.4, 2 seeds each):

| frozen | baseline (25k, lr 1e-3) | 3× learning rate | 4× steps (100k) |
|---|---|---|---|
| c=0.2, KL_rel | 0.0890 | 0.0890 | — |
| c=0.3, KL_rel | 0.0183 | 0.0181 | **0.0185** |
| c=0.3, C | 4.010 | 3.984 | **4.028** |
| c=0.3, D | 10.72 | 10.65 | **10.83** |
| c=0.3, cos θ | 1.000 | 1.000 | **1.000** |
| c=0.4, KL_rel | 0.0044 | 0.0036 | **0.0037** |

Four times the steps buys 0.4 % of contrast and moves the separation by 1 %. Alignment is
exactly saturated in every one of these runs, and the two seeds agree to four decimals — a
constraint, not a search failure. The models below the boundary are capped.

The one place where the diagnosis differs is δ=0.2 at c ≤ 0.07, where cos θ ≈ 0.3 and D ≈ 0.4:
those models never found the branch structure at all, and nothing about the ceiling explains
them.

## Conclusions

1. **A complete-cut interchange effect is behaviour, exactly.** Over 20 models, 5 routing
   regimes and 3 different complete cuts, the patched prediction equals the counterfactual
   prediction bitwise. This is a theorem in this architecture, and the experiment's role was to
   check the cut enumeration, which it did.
2. **A single-node effect is a route share, and the route is free.** The same distinction
   patched at one position gives anything from 0.0000 to the full behavioural divergence across
   models with identical input–output maps; hard masks, soft penalties and plain seed variation
   all move it. Reporting a patching number at one site without saying whether the site is a
   cut reports a property of one implementation.
3. **Their sum is not a repair.** Depending on the seed, the two route effects add to the cut
   value (exclusive routing) or overshoot it by a factor 2.2 (the cue duplicated on both
   routes). Only the joint patch is invariant.
4. **The function fixes the product C = g·D·cos θ and nothing else about it.** CV 0.006 across
   77 functionally equivalent models whose factors span 37×, with the three log-slopes summing
   to zero to three decimals in every architecture.
5. **LayerNorm changes the degrees of freedom, as claimed — but the available ceiling is a
   capacity, not a norm identity.** The formal bound C ≤ 4c√d held in all 49 frozen models and
   was never close to tight: non-antipodal readout rows (×0.65) and non-antipodal states
   (×0.70) put the attained law at C = 14.4c at δ=0.4. Its boundary, 0.306 against an observed
   0.4 (and 0.220 against 0.3 at δ=0.2), is within 1.3× where the formal bound is off by 3× and
   6× — a substantial downward shift explained, and a factor ~1.3 still unexplained.
6. **The predictive question that remains is the plateau.** D saturates at 11.3 (δ=0.4) and 5.8
   (δ=0.2) with the same architecture and the same d, so what limits the separation of one pair
   of states is the rest of the task sharing the interface. That quantity — the attainable
   separation for one distinction given everything else the readout must do — is measurable
   here and is the right replacement for the geometric bound.
7. **Scorecard.** Of 12 pre-registered predictions, 8 held (P1, P2, P3, P3b, P4, P7, P9a, P9c)
   and 4 failed outright (P8, P10, P10b, P11, P12 — with P5 and P6 unstable across replications
   of the same design). Claim A survives as stated. Claim B's identity half survives; its
   "architecture predicts the factorisation" half survives only ordinally, and its quantitative
   boundary is falsified.

## Practical notes

* Training the grid as one stacked model is worth it: 4× throughput and, more usefully, the
  whole grid finishes together so a crash after training costs nothing (`task10_remeasure.py`
  rebuilds every measurement from `results10/models/` in 4 minutes).
* Stack size matters for that gain: at M=147 the step costs 108 ms, at M=4 it costs 96 ms.
  Small control runs should be folded into a large stack of seeds rather than run alone.
* TF32 is the whole speed-up on an A100 at this model size (118 → 73 ms/step); fused attention
  contributes nothing because the bottleneck is fp32 matmul throughput, not score memory.
