# What sets the split between readout gain and hidden separation? (TASK6)

Run date: 2026-09-18, 30 runs in 322 s on 8 CPU workers. Follows TASK5
(`rounds/r06_readout_scale/REPORT.md`), which showed that the required logit gap
Δℓ = (w_x − w_y)·(h̄_C − h̄_D) is met by whatever combination of readout gain g,
hidden separation and alignment the optimiser lands on, and that with a free
readout it lands mostly in the representation. This round asks what determines
that landing point. Numbers from `rounds/r07_allocation/tables.md`, per-run values and
checkpoint curves in `rounds/r07_allocation/results6.json`. Reproduce with
`rounds/r07_allocation/run.py --jobs 8`.

## Setup

HMM at r=1, δ=0.4, k=1, λ=1 (required gap 4.39 nats), the round 4 GRU with a
free linear readout, exact-target next-token training, 3000 Adam steps, 3
seeds, the TASK4/TASK5 measurements at every checkpoint. Three manipulations,
each with the shared baseline (readout lr ×1, init ×1, control 0.9):

1. **Readout learning rate** relative to the recurrent parameters: ×0.1, ×0.3,
   ×1, ×3, ×10 (Adam parameter groups, recurrent lr fixed at 3e-3).
2. **Readout initialisation scale**: the output layer's weights and bias
   multiplied by 0.1, 1, 10 at init.
3. **Control-branch contrast** P(x | A′) = P(y | B′): 0.9, 0.75, 0.6, 0.5. At 0.5
   the control cue has no consequence; the delayed cue is unchanged.

All 30 runs reach the Bayes floor (KL to the exact targets below 0.001) and the
achieved delayed gap is 4.305 in every run, so every difference below is a
difference in how an identical function is implemented.

## Results

### Readout learning rate: the split follows relative speed

| readout lr / recurrent lr | ‖w_x − w_y‖ | ‖h̄_C − h̄_D‖ at t*−1 | control ‖h̄_U − h̄_V‖ | cos θ | ‖W‖_F |
|---|---|---|---|---|---|
| ×0.1 | 0.93 | 5.89 | 5.97 | 0.79 | 2.6 |
| ×0.3 | 1.07 | 4.98 | 5.20 | 0.81 | 3.0 |
| ×1 | 1.35 | 3.95 | 4.19 | 0.81 | 4.0 |
| ×3 | 1.65 | 3.31 | 3.63 | 0.79 | 5.8 |
| ×10 | 2.12 | 2.51 | 2.72 | 0.81 | 11.0 |

A hundredfold change in the readout's speed moves the gain by 2.3× and the
separation by 2.3× the other way, alignment constant, outputs identical. The
control pair follows the delayed pair. The checkpoint curves (Figure 2) show
the mechanism: the fitting phase is over by step 50 to 100 in every condition,
and the split at that moment is set by how far each factor could travel while
the loss was still large. With the readout slow, the representation has done
almost all the work by the time the gradient vanishes; with the readout fast,
much less of it.

After the floor is reached there is a slow drift, and its direction depends on
where the fast phase ended: at ×10 the gain falls from 2.69 to 2.09 over 3000
steps while the separation rises from 1.7 to 2.4; at ×0.1 the gain creeps up
from 0.80 to 0.85. So there is an intermediate allocation the landscape prefers
(around a gain of 1.4 to 1.7 at this scale) and both sides drift toward it, but
at a rate that leaves the fitting-phase split largely intact at any practical
horizon.

### Readout initialisation: large is sticky, small washes out

| readout init scale | ‖w_x − w_y‖ init | ‖w_x − w_y‖ final | ‖h̄_C − h̄_D‖ at t*−1 | cos θ |
|---|---|---|---|---|
| ×0.1 | 0.08 | 1.09 | 4.40 | 0.90 |
| ×1 | 0.79 | 1.35 | 3.95 | 0.81 |
| ×10 | 7.93 | 7.67 | 1.15 | 0.50 |

A readout initialised ten times too small recovers to near the baseline split
(gain 1.09 versus 1.35, separation 4.40 versus 3.95), because the gain has to
grow before the model can fit at all and the fitting phase drives it. A readout
initialised ten times too large stays there (7.9 → 7.7 over 3000 steps) and the
separation stays at 1.15 with alignment 0.50: nothing in the loss requires the
gain to come down once the fit is reached, and the drift toward the preferred
allocation is far too slow to matter. The asymmetry is the same lesson as the
learning-rate sweep: the split is whatever configuration first reaches the
floor.

### Control contrast: the shared readout is negotiated only weakly

| control contrast | ‖w_x − w_y‖ | ‖h̄_C − h̄_D‖ at t*−1 | control ‖h̄_U − h̄_V‖ | control achieved gap | P_metric |
|---|---|---|---|---|---|
| 0.5 | 1.20 | 4.27 | 1.07 | 0.00 | 4.0 |
| 0.6 | 1.21 | 4.26 | 1.49 | 0.81 | 2.9 |
| 0.75 | 1.25 | 4.17 | 2.66 | 2.20 | 1.6 |
| 0.9 | 1.35 | 3.95 | 4.19 | 4.39 | 0.95 |

Removing the control branch's consequence entirely changes the free gain by
12 % (1.35 → 1.20) and the delayed separation by 8 % the other way. The
x-versus-y readout is shared by every position, but the contexts that need a
different contrast do not pull the gain much: each gets its own separation
instead, which is again the representation absorbing context-specific demands.
Two side results. The control pair, when it stops mattering, falls to 1.07,
below its initial 1.49, the same "not amplified" fate as the delayed pair at
r=0 in TASK4. And P_metric, the TASK4 prominence measure, rises to 4.0 at
contrast 0.5 with the delayed pair unchanged: the normaliser, not the quantity
of interest, moved. TASK4's delay-sweep caveat was the same effect in the other
direction.

## Answer

The optimiser's allocation of the required gap among readout gain, hidden
separation and alignment is set, in order of importance, by:

1. **Relative speed during the fitting phase.** Whichever factor can move
   further in the fifty-odd steps before the loss reaches its floor carries the
   gap. Under Adam every parameter moves about one learning rate per step, the
   separation is a function of thousands of recurrent parameters and the gain
   of 128, so by default the representation wins; changing the readout's
   learning rate changes the split in proportion.
2. **Initial scale, asymmetrically.** A gain that starts too large to need
   growth keeps its size; a gain that starts too small is grown by the fitting
   phase to roughly the default split.
3. **A weak landscape preference**, visible as a slow post-floor drift toward
   an intermediate allocation from either side, and as a slight decrease of
   the gain when fewer contexts need contrast. It is real but too slow to
   override 1 and 2 within 3000 steps.

The shared-readout negotiation across contexts, which TASK5's report offered
as an explanation for the gain's insensitivity to δ, is only a minor
contributor; the insensitivity is mostly item 1.

So the TASK4 result reads, in full: the geometric prominence of a distinction
is the hidden-state separation the required output gap forces on it, divided
by the readout gain, and the readout gain is a by-product of how fast the
readout could move before the loss was fit. Neither the frequency of use, the
expected cost of forgetting, nor the gradient magnitude enters; the
parameterisation and the learning-rate ratio do.

## Figures

- `fig1_allocation.png`: gain, separations and alignment against each of the
  three manipulations.
- `fig2_lr_ratio_dynamics.png`: gain, separation and achieved gap over
  training for the five learning-rate ratios (seed 0).
