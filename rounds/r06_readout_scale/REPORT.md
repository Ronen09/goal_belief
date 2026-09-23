# Readout scale versus hidden separation (TASK5)

Run date: 2026-09-18, 42 runs in 452 s on 8 CPU workers. Follows TASK4
(`rounds/r05_prominence/REPORT.md`), which found that the geometric prominence of the
delayed cue tracks the output separation it has to produce at t*, not the
frequency, expected cost or gradient pressure of using it. This round asks
*how* that output separation is implemented: by hidden-state distance or by
readout norm. Numbers from `rounds/r06_readout_scale/tables.md`, per-run values and
checkpoint curves in `rounds/r06_readout_scale/results5.json`. Reproduce with
`rounds/r06_readout_scale/run.py --jobs 8`.

## Setup

Same HMM family, GRU, objective and measurements as TASK4, at r=1, k=1, λ=1
and two strengths, δ=0.1 and δ=0.4. The x-versus-y logit gap that the two
branches must show at the pre-relevant position t*−1 is

Δℓ = (w_x − w_y)·(h_C − h_D) = 2·ln((0.5+δ)/(0.5−δ)),

that is 0.81 nats at δ=0.1 and 4.39 nats at δ=0.4, so the mean-state separation
must satisfy ‖h̄_C − h̄_D‖ ≥ Δℓ / (‖w_x − w_y‖ · cos θ), with θ the angle between
the separation and the readout difference.

**Manipulation.** The output layer is replaced by a weight-normalised readout
whose rows have a fixed norm c (directions learned, gains fixed, bias free),
c ∈ {0.25, 0.5, 1, 2, 4, 8}, plus the free `nn.Linear` reference. 3 seeds.

**New measurements.** The effective gain g = ‖w_x − w_y‖, the mean-state
separations of the delayed pair at t*−1 and at the cue and of the control pair,
the achieved gap (w_x − w_y)·(h̄_C − h̄_D), and cos θ. Every run is checked
against the Bayes floor (KL to the exact targets).

## Results

### Fixed gain: the hidden separation compensates

δ = 0.4, mean of 3 seeds:

| gain c | ‖w_x−w_y‖ | ‖h̄_C−h̄_D‖ at t*−1 | control ‖h̄_U−h̄_V‖ | cos θ | achieved gap | KL |
|---|---|---|---|---|---|---|
| free | 1.35 | 3.95 | 4.19 | 0.81 | 4.31 | 0.000 |
| 0.25 | 0.34 | 8.94 | 9.15 | 1.00 | 3.04 | 0.075 |
| 0.5 | 0.51 | 8.32 | 8.54 | 1.00 | 4.25 | 0.004 |
| 1 | 0.86 | 5.24 | 5.45 | 0.96 | 4.30 | 0.000 |
| 2 | 1.59 | 3.22 | 3.54 | 0.84 | 4.31 | 0.000 |
| 4 | 3.01 | 1.88 | 2.64 | 0.76 | 4.31 | 0.000 |
| 8 | 6.79 | 1.10 | 2.04 | 0.58 | 4.31 | 0.000 |

δ = 0.1:

| gain c | ‖w_x−w_y‖ | ‖h̄_C−h̄_D‖ at t*−1 | control ‖h̄_U−h̄_V‖ | cos θ | achieved gap | KL |
|---|---|---|---|---|---|---|
| free | 1.19 | 1.21 | 4.79 | 0.56 | 0.79 | 0.000 |
| 0.25 | 0.30 | 3.95 | 9.13 | 0.57 | 0.67 | 0.060 |
| 0.5 | 0.49 | 3.02 | 8.93 | 0.54 | 0.79 | 0.003 |
| 1 | 0.77 | 1.50 | 6.01 | 0.69 | 0.79 | 0.000 |
| 2 | 1.43 | 0.96 | 3.79 | 0.58 | 0.79 | 0.000 |
| 4 | 2.76 | 0.90 | 2.69 | 0.32 | 0.79 | 0.000 |
| 8 | 6.40 | 0.77 | 2.01 | 0.16 | 0.79 | 0.000 |

Outputs are identical across gains: the achieved gap is pinned at 4.31 and
0.79 (the exact-target optimum, 2 % below Δℓ because the soft targets keep the
softmax off its limit) and the KL to the targets is at the floor for every
c ≥ 0.5. What changes is where the gap lives:

- **Forcing a smaller readout norm makes the hidden separation grow.** At δ=0.4,
  halving the free gain (c=0.5, g=0.51) doubles the separation, 3.95 → 8.32,
  and the same holds for the control pair, 4.19 → 8.54. At c=0.25 the model can
  no longer fit: it would need a separation of 4.39/0.34 = 12.9, GRU states are
  bounded in (−1, 1), both pairs stall at about 9 and the loss stays 0.07 nats
  above the floor. The compensation is real up to a hard capacity limit.
- **Allowing a larger readout norm makes it shrink, with outputs unchanged.**
  At c=8 (g=6.8) the delayed separation falls to 1.10, 0.28 of the free value,
  and the control pair to 2.04, 0.49 of free. Along the readout direction the
  separation is exactly what the gap requires: g·‖h̄_C−h̄_D‖·cos θ = 4.31 for every
  converged run at δ=0.4 and 0.79 at δ=0.1 (Figure 2).
- **Slope.** Over the converged fixed-gain runs, log ‖h̄_C−h̄_D‖ against log g has
  slope −0.79 (R² 1.00) at δ=0.4 and −0.48 (R² 0.78) at δ=0.1, against the
  prediction of −1. The shortfall is alignment: as the gain rises the
  separation vector rotates away from the readout (cos θ 1.00 → 0.58 at δ=0.4,
  0.57 → 0.16 at δ=0.1), so the total separation falls more slowly than its
  readout component. At δ=0.1 the total separation floors at about 0.8 for
  c ≥ 2 while its readout component keeps shrinking: the remainder is a
  residual, readout-irrelevant separation that the recurrence carries anyway
  (the two branches saw different tokens one step earlier; the init value is
  0.67). The relation is therefore ‖h̄_C−h̄_D‖ ≈ max(Δℓ/(g·cos θ), floor), with
  the scale-allocation term dominant whenever the required gap is large relative
  to the residual.

### Free readout: the optimiser puts the scale into the representation

With the readout free, the learned gain is nearly the same at the two
strengths, 1.19 at δ=0.1 and 1.35 at δ=0.4 (ratio 1.14), while the delayed
separation is 1.21 versus 3.95 (ratio 3.3) and its alignment 0.56 versus 0.81.
The 5.4-fold difference in required gap is met almost entirely by hidden-state
distance and alignment, not by readout norm. This is the TASK4 δ sweep
explained: prominence grew with δ because the readout norm did not.

The checkpoint curves (Figure 3) show the allocation in time. At δ=0.4 the
separation jumps from 0.7 to 3.9–5.6 within the first 50 steps, when the loss at
t* collapses, while the gain moves from 0.78 to 1.2. After that the gain creeps
up (to 1.35 by step 3000, and the readout Frobenius norm from 2.6 to 4.0) and the
separation drifts down (seed 2: 5.6 → 4.3) with the achieved gap constant at
4.30: a slow reallocation from representation to readout under the usual
norm growth of cross-entropy training, but starting from a solution that was
found in the representation.

### Prominence measured the TASK4 way

P_metric (cue-position delayed distance over control distance) at δ=0.4 is
0.95 free and 0.91–0.99 for c ≥ 2, because both pairs scale together. At small
gain it drops (0.54 at c=0.5, 0.70 at c=1): the control pair, which needs its
gap one step after its cue, hits the capacity ceiling first, while the delayed
pair builds most of its separation across the filler step (at t*−1 both are at
8.3–8.5). So the normalised prominence of TASK4 is invariant to the readout
scale except where the network runs out of hidden-state range.

## Conclusion

The required logit gap is a constraint on the product of readout gain,
hidden separation and their alignment, and the optimiser solves it by moving
the representation. Constraining the readout to half its natural gain doubles
the hidden separation; letting it grow eightfold shrinks the separation to a
quarter, with identical outputs and identical loss. Left free, the readout barely
changes with the strength of the consequence and the representation absorbs it.
Representational prominence in this system is not an intrinsic mark of semantic
importance; it is one side of a scale-allocation problem set by the function to
be computed, the bounded range of the recurrent state, and the optimiser's
tendency to solve the problem in the many-parameter recurrent map before the
readout norm has time to grow.

Two limits. The compensation has a ceiling (GRU states in (−1, 1)) and a floor
(readout-irrelevant separation carried by the recurrence), and the slow late
drift toward the readout means the split between representation and readout is
a property of the training horizon as well as of the task.

## Figures

- `fig1_separation_vs_gain.png`: separations at t*−1, control, and cue against
  effective gain (log-log), free-readout runs starred, Δℓ/‖w‖ reference lines,
  unconverged runs hollow.
- `fig2_gap_accounting.png`: achieved versus required gap, and cos θ, against c.
- `fig3_free_readout_dynamics.png`: gain, separation and achieved gap over
  training with the free readout.
