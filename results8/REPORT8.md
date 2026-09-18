# Intervention equivalence, on/off-manifold local metrics, propagation depth (TASK8)

Run date: 2026-09-18, 60 s on 6 CPU workers (six new delay-4 models plus analyses
on the TASK7 models). Definitions in `docs/task7_theory.md`; numbers from
`results8/tables8.md`, arrays in `results8/results8.json`, figures
`fig1_intervention_equivalence.png`, `fig2_on_off_manifold.png`,
`fig3_propagation_depth.png`. Reproduce with `scripts/run_task8.py --jobs 6`
(needs `results7/models/`).

## Models and protocol

Functionally equivalent pairs from TASK7 at δ=0.4: readout learning rate ×0.1
(large hidden separation) against ×10 (small), seeds 0–2, plus a pair of base
seeds as the different-seed reference. All are at the Bayes floor with pairwise
output KL below 1e-4. Every measurement starts from a cue-A state h_A(x) (200
anchors on the shared evaluation set), optionally perturbed, and pushes it
through the network's own future computation F^{(k)} with the anchor's actual
continuation tokens, reading the prediction at step k. The delay-1 HMM consumes
the cue at k=1; six new delay-4 models (lr ×0.1 and ×10, 3 seeds, KL ≤ 0.0002)
consume it at k=4.

## A. Intervention equivalence: steering vectors are equivalent under their effect

Steering vector per model: v = mean h_B − mean h_A at the cue, applied as
h_A + αv and read one step later; the gradient of the y-versus-x log-odds at
that step, scaled to ‖v‖, is the second steering vector.

| pair | ‖v₁‖ | ‖v₂‖ | ‖v‖/median distance (1, 2) | alignment R² | cos(v₁ mapped, v₂) |
|---|---|---|---|---|---|
| lr ×0.1 vs ×10, seed 0 | 4.48 | 2.40 | 0.69, 0.68 | 0.997 | 0.997 |
| seed 1 | 4.07 | 2.77 | 0.66, 0.78 | 0.997 | 0.994 |
| seed 2 | 4.15 | 3.03 | 0.66, 0.88 | 0.997 | 0.997 |
| base seed 0 vs seed 1 | 4.04 | 3.45 | 0.63, 0.57 | 1.000 | 1.000 |

Cross-model divergence of the outputs after steering, mean over the three
lr pairs (the α=0 baseline is 0.00003):

| α₁ | matched α₂ | effect | cross JS, same α | cross JS, matched effect |
|---|---|---|---|---|
| 0.25 | 0.11–0.21 | 0.004–0.010 | 0.0001–0.004 | 0.00004–0.00008 |
| 0.50 | 0.45–0.63 | 0.09–0.16 | 0.0006–0.011 | 0.00004–0.00009 |
| 0.75 | 0.73–0.86 | 0.23–0.30 | 0.0004–0.011 | 0.00004–0.00005 |
| 1.00 | 1.00–1.01 | 0.368 | 0.00003 | 0.00002–0.00004 |

Three results.

- **Interventions are equivalent under their downstream effect.** Once α₂ is
  chosen so that model 2's effect equals model 1's, the two models' output
  distributions coincide to the α=0 baseline at every α, for both steering
  vectors, including the gradient vector whose effect curve overshoots the
  B-branch prediction (0.43 at α=1). Parametrising by α, that is by geometry,
  does not transfer: at the same α the cross-model divergence is 10 to 300 times
  the baseline at intermediate steering and vanishes only at α=1, where both
  land on the B cluster.
- **The vectors differ by scale, not direction.** ‖v₁‖ is 1.4 to 1.9 times ‖v₂‖,
  yet in units of each model's own median pairwise distance the two are alike
  (0.66–0.69 versus 0.68–0.88). More strongly, a single least-squares linear map
  fitted on shared states explains 99.7 % of the variance of one model's states
  in terms of the other's, and maps v₁ onto v₂ with cosine 0.994–0.997. The two
  implementations' reachable sets are, to three digits, linear images of each
  other, and so are two independent seeds (R² 1.000).
- **The gradient direction is not the steering direction.** Its cosine with the
  mean difference is 0.55–0.70 in the large-separation models and 0.84–0.91 in
  the fast-readout ones. The gradient points where the readout is steepest,
  which depends on the implementation; the mean difference points at the other
  cluster.

The second result sharpens the TASK7 puzzle: if the state clouds are linearly
related, why did ‖JΔh‖ differ by 1.7× between these models? Part B answers.

## B. Local metrics on and off the manifold: there is no on-manifold

Nine models (lr ×0.1, ×1, ×10 × 3 seeds). Perturbations of the cue state along
four direction types at scale s in units of each model's own ‖Δh‖: the mean
difference toward the other branch, the top three principal directions of the
state cloud, the bottom three, and five random directions; effects averaged
over ±direction. Finite effect JS(F(h+sv), F(h)) at one step against the
second-order prediction dzᵀF(p)dz/8 from the exact directional derivative.

| direction | linearisation error at s = 0.05 / 0.2 / 1.0 | cross-model CV of the finite effect at s = 0.05 / 0.5 / 1.0 | CV of the linear prediction (all s) |
|---|---|---|---|
| mean difference | 0.01 / 0.18 / 0.36 | 0.44 / 0.20 / 0.02 | 0.44 |
| top principal directions | 0.00 / 0.05 / 0.95 | 1.24 / 1.15 / 0.47 | 1.24 |
| bottom principal directions | 0.00 / 0.01 / 0.17 | 0.70 / 0.70 / 0.69 | 0.70 |
| random | 0.00 / 0.01 / 0.25 | 0.52 / 0.53 / 0.62 | 0.52 |

The linearisation is accurate for small perturbations in every direction
(error ≤ 0.05 at s ≤ 0.1). Yet the local effect it predicts differs across
equivalent models by CV 0.44 along the mean difference, 0.52 for random
directions, 0.70 for the bottom principal directions and 1.24 for the top ones,
at every scale. The failure of local functional metrics is therefore not
curvature at the scale probed, and it is not confined to directions the model
"never uses": the leading principal directions of the state cloud are the
worst. The finite effect agrees across models only where the perturbed state
lands on another reachable state, at s=1 along the mean difference (CV 0.02),
and disagrees again beyond it (s=1.5, CV 0.08; random directions at s=1.5,
CV 1.36).

The reachable set of this network is a set of clusters, one per belief state
and recent token, not a smooth manifold with a tangent space. The function
constrains F at the clusters and nowhere between them, so a perturbation of
any size in any direction, including straight toward the other cluster, moves
into territory where equivalent models are free to differ, and their Jacobians
do. The 99.7 % linear relation of Part A holds on the clusters; it says nothing
about the maps between them. This is the resolution of the TASK7 result:
‖JΔh‖ is exact under compensated coordinate change because that changes the
description of F everywhere, and it is not invariant across equivalent models
because they share F only on a set of points.

## C. Propagation depth: the hierarchy has a horizon axis

For the A-branch and a random B-branch state pushed through the same
continuation, four quantities at each depth k, and their CV across models.

Delay-1 HMM, 9 models (raw distance, linearised displacement, finite centred
logit displacement, JS of predictions):

| k | 0 | 1 (relevant) | 2 | 3 | 4 |
|---|---|---|---|---|---|
| raw, lr ×0.1 seed 0 | 4.51 | 6.45 | 0.06 | 0.00 | 0.00 |
| raw, lr ×10 seed 0 | 2.48 | 2.50 | 1.05 | 0.37 | 0.17 |
| CV raw | 0.19 | 0.34 | 0.80 | 1.17 | 1.31 |
| CV linearised | 0.85 | 0.28 | 0.92 | 1.18 | 1.71 |
| CV finite logits | 0.85 | 0.09 | 0.83 | 1.26 | 1.35 |
| JS, every model | 0.000 | 0.368 | 0.000 | 0.000 | 0.000 |

Delay-4 HMM, 6 models: JS is 0.000 at k = 0–3, 0.368 at k=4 in all six (CV
0.00), and 0.000 for k ≥ 5. The raw distance in the lr ×0.1 models climbs
along the fillers from 1.7–3.4 to 5.5–6.2 at k=4 and collapses to 0.1 one step
later; in the lr ×10 models it stays near 3–3.7 through the fillers and decays
over four steps after the consumption (2.2, 0.5, 0.3, 0.1). The linearised
displacement in the lr ×10 models spikes to 9–21 at k=5, the step at which the
state collapses nonlinearly with zero behavioural effect. The finite logit
displacement is invariant only at k=4 (CV 0.07); at the other depths it is
1–5 in some models and 0.2–0.5 in others while the predictions are identical,
because centred logit norms pick up differences among tokens whose probability
is zero at that position. Probability-space divergence is the only one of the
four that is invariant at every depth.

The answers to the depth questions. Equivalence of the effect does not
"emerge" with propagation: it holds exactly at every k at which the states are
compared through the output distribution, being zero where the function
predicts no difference and 0.368 at the step where the distinction is
consumed. One step is enough when the distinction is consumed in one step and
four when it is consumed in four; the horizon is the task's, not the
measure's. What does vary with depth is the residual hidden-state difference
after consumption, which one implementation erases within a step and another
carries for four, with no consequence for behaviour: the raw quantities that
looked like "memory" or "persistence" in the state are properties of the
implementation.

The hierarchy, with the measured cross-model CV at the behaviourally relevant
step: raw geometry 0.25–0.34; local functional geometry 0.28–0.82 and
unbounded at collapse steps; finite-horizon logit displacement 0.07–0.09;
behaviour, the divergence of predictions, 0.00.

## Conclusions

1. Steering interventions in functionally equivalent models are equivalent
   under their downstream effect: matched by effect, they produce the same
   output distribution to the noise floor at every strength, while matched by
   geometry they do not.
2. The "wildly different" geometries are linear images of each other on the
   reachable set (R² 0.997), differing mainly in scale; the steering vectors
   map onto one another with cosine 0.997.
3. Nevertheless local functional metrics disagree across the same models by
   CV 0.4–1.2 at arbitrarily small scale, in every direction including the
   leading principal directions. The reachable set is a set of clusters; the
   function fixes the network only there. There is no on-manifold tangent
   regime in which Jacobian or Fisher metrics become implementation-invariant.
4. The finite-propagation divergence is invariant at every horizon; raw and
   linearised quantities carry implementation-specific dynamics, in particular
   how fast an already-consumed distinction is erased from the state.

For interpretability this means: a steering vector should be reported by its
effect curve, not its norm or direction; a claim about a feature's causal
role should be tested by pushing states through the network and comparing
output distributions, not by a local metric; and "on-manifold" perturbation,
for a network whose reachable set is discrete, means moving to another
reachable state, not moving a little.
