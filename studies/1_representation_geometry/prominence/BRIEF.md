# TASK4 — What Makes Information Geometrically Prominent?

## Research Question

Why does some information become a dominant part of a neural network's representation geometry, while other information remains highly decodable but geometrically inconspicuous?

The motivating result is the HMM experiment:

- delayed belief information remains almost perfectly linearly decodable;
- the recurrent model needs that information for future prediction;
- nevertheless, hidden-state geometry is organised mainly around the immediate prediction target;
- explicitly supervising longer-horizon predictions makes the delayed distinction geometrically prominent.

This suggests a distinction between:

\[
\text{information retention}
\]

and

\[
\text{geometric prominence}.
\]

---

## Working Hypothesis

A feature becomes geometrically prominent according to the **optimization pressure associated with preserving and using it**, rather than merely because the information is present or eventually necessary.

Informally:

\[
\text{geometric prominence}(z)
\approx
f(
\text{frequency of relevance},
\text{strength of relevance},
\text{gradient pressure},
\text{optimization dynamics}
).
\]

The generative process determines **when and how strongly a latent distinction matters**.

The optimizer determines **how strongly that distinction is amplified in representation space**.

---

# Baseline Setup

Use the existing six-state HMM.

The critical pair is:

\[
b_A \neq b_B
\]

while

\[
P(x_{t+1}\mid b_A)
=
P(x_{t+1}\mid b_B).
\]

Their future predictions differ:

\[
P(x_{t+2:t+k}\mid b_A)
\neq
P(x_{t+2:t+k}\mid b_B).
\]

Use the existing GRU next-token model as the primary model.

Do not introduce new architectures until the basic effect is understood.

---

# Core Measurements

For every experiment, record the following quantities.

## 1. Decodability

Train a linear probe:

\[
\hat b = Wh.
\]

Report:

- belief-state \(R^2\);
- cue-class accuracy where appropriate.

This measures whether the information exists.

---

## 2. Pairwise Metric Prominence

For the delayed pair:

\[
D_{\text{delay}}
=
\|h_A-h_B\|.
\]

Normalise using the existing control pair:

\[
P_{\text{metric}}
=
\frac{
\|h_A-h_B\|
}{
\|h_{\text{control},1}-h_{\text{control},2}\|
}.
\]

This is the main measure of geometric prominence.

---

## 3. Global Geometry

Measure RSA between hidden-state geometry and:

- belief-state geometry;
- one-step predictive geometry;
- two-step predictive geometry;
- full-future predictive geometry where tractable.

---

## 4. Gradient Relevance

Measure the gradient associated specifically with the future prediction that depends on the delayed cue.

For example:

\[
G_t
=
\left\|
\nabla_{h_t}
L_{\text{relevant future}}
\right\|.
\]

Also measure its projection onto the delayed-information direction:

\[
G_{\parallel}
=
\left|
\nabla_{h_t}L
\cdot
\frac{h_A-h_B}{\|h_A-h_B\|}
\right|.
\]

This directly tests whether geometric prominence is related to gradient pressure.

---

# Experiment 1 — Relevance Frequency Sweep

## Goal

Test whether a distinction becomes more geometrically prominent when it matters more often.

## Manipulation

Introduce a probability

\[
r
=
P(\text{delayed cue becomes behaviorally/predictively relevant}).
\]

Sweep:

\[
r \in
\{0,\;0.05,\;0.1,\;0.25,\;0.5,\;0.75,\;1.0\}.
\]

With probability \(r\):

\[
A\rightarrow C,
\qquad
B\rightarrow D,
\]

where \(C\) and \(D\) produce different future predictions.

With probability \(1-r\), send both branches into the same future distribution.

Maintain:

\[
P(x_{t+1}\mid A)
=
P(x_{t+1}\mid B)
\]

for every value of \(r\).

Therefore the distinction never matters for the immediate prediction.

## Measure

For each \(r\):

- belief \(R^2\);
- \(P_{\text{metric}}\);
- belief RSA;
- one-step RSA;
- future-prediction RSA;
- \(G_t\);
- \(G_{\parallel}\);
- next-token loss.

Use multiple seeds.

## Competing Predictions

### Pure sufficiency hypothesis

Once \(r>0\), the distinction must be retained, so decodability and geometric prominence should rise together.

### Gradient-prominence hypothesis

Decodability should become high at relatively small \(r\), while geometric prominence should increase more gradually with relevance frequency.

Desired pattern:

\[
R^2(r)
\text{ saturates before }
P_{\text{metric}}(r).
\]

---

# Experiment 2 — Relevance Strength Sweep

## Goal

Hold relevance frequency fixed and vary how costly forgetting the distinction is.

Set:

\[
r=1.
\]

Define:

\[
P(x\mid C)=0.5+\delta
\]

and

\[
P(x\mid D)=0.5-\delta.
\]

Sweep:

\[
\delta
\in
\{0,\;0.02,\;0.05,\;0.1,\;0.2,\;0.4\}.
\]

Larger \(\delta\) means the latent distinction matters more strongly for future prediction.

## Compute Forgetting Cost

For each \(\delta\), analytically or numerically compute the increase in optimal predictive loss if the model forgets whether it was in \(A\) or \(B\):

\[
\Delta L_{\text{forget}}(\delta).
\]

## Measure

For each \(\delta\):

- belief \(R^2\);
- \(P_{\text{metric}}\);
- gradient relevance;
- belief/prediction RSA;
- \(\Delta L_{\text{forget}}\).

## Primary Test

Compare:

\[
P_{\text{metric}}
\]

against:

\[
\Delta L_{\text{forget}}.
\]

The key question is whether geometric prominence is better predicted by the **cost of forgetting** than by raw belief-state distance.

---

# Experiment 3 — Direct Loss-Weight Manipulation

## Goal

Change optimization pressure without changing the generative process.

Keep:

- the same HMM;
- the same sequences;
- the same latent distinctions;
- the same model.

Change only the weighting of the timestep at which the delayed distinction becomes useful.

Standard objective:

\[
L
=
\sum_t \ell_t.
\]

Modified objective:

\[
L_\lambda
=
\sum_{t\neq t^*}\ell_t
+
\lambda \ell_{t^*},
\]

where \(t^*\) is the prediction whose accuracy depends on remembering \(A\) versus \(B\).

Sweep:

\[
\lambda
\in
\{0,\;0.1,\;0.25,\;0.5,\;1,\;2,\;5\}.
\]

## Interpretation

Nothing about the data-generating process changes.

Only the optimization pressure associated with the latent distinction changes.

## Measure

For each \(\lambda\):

- belief \(R^2\);
- \(P_{\text{metric}}\);
- RSA;
- \(G_t\);
- \(G_{\parallel}\);
- task performance.

## Strong Result

If:

\[
\lambda\uparrow
\]

causes:

\[
P_{\text{metric}}\uparrow
\]

while decodability remains high throughout, this is direct evidence that optimization pressure controls representational scale.

---

# Experiment 4 — Delay Sweep

## Goal

Separate **importance of information** from **difficulty of assigning credit to it**.

Keep the eventual predictive consequence fixed.

Vary the number of recurrent steps between the cue and the timestep at which it becomes useful:

\[
k\in\{1,2,4,8,16\}.
\]

Ensure that the eventual prediction task and its loss contribution remain otherwise matched.

## Measure

For each delay:

- belief \(R^2\);
- \(P_{\text{metric}}\);
- relevant gradient norm;
- gradient projection;
- final prediction accuracy.

## Hypothesis

If the information remains equally useful but geometric prominence falls with delay, this would implicate optimization/credit-assignment dynamics.

Possible pattern:

\[
R^2 \approx \text{constant}
\]

while

\[
P_{\text{metric}}\downarrow
\]

as

\[
k\uparrow.
\]

---

# Experiment 5 — Combined Frequency × Strength Sweep

Only run this if Experiments 1 and 2 show clear effects.

Create a grid over:

\[
(r,\delta).
\]

For example:

\[
r\in\{0.1,0.25,0.5,1\}
\]

and

\[
\delta\in\{0.05,0.1,0.2,0.4\}.
\]

Compute:

\[
\Delta L_{\text{expected forget}}
\approx
r\,
\Delta L_{\text{forget}}(\delta).
\]

Test whether geometric prominence collapses onto a common curve when plotted against expected forgetting cost.

This would be substantially stronger than separate frequency and strength effects.

---

# Main Analysis

The central analysis should compare candidate predictors of geometric prominence.

Fit:

\[
P_{\text{metric}}
=
\beta_0
+
\beta_1 R^2
+
\beta_2 \Delta L_{\text{forget}}
+
\beta_3 G_t
+
\beta_4 G_{\parallel}
+
\epsilon.
\]

Questions:

1. Does decodability explain metric prominence?
2. Does expected forgetting cost explain it better?
3. Does measured gradient pressure explain additional variance?
4. Does the relationship remain consistent across frequency, strength, loss-weight and delay manipulations?

---

# Training-Dynamics Analysis

Save checkpoints throughout training.

For each checkpoint measure:

- belief decodability;
- delayed-pair distance;
- control-pair distance;
- relevant gradient norm;
- relevant gradient projection;
- prediction accuracy.

This allows the temporal ordering to be tested.

Important question:

> Does information become decodable before it becomes geometrically prominent?

A particularly strong result would be:

\[
\text{decodability}
\rightarrow
\text{functional use}
\rightarrow
\text{metric amplification}.
\]

---

# Essential Controls

## Control 1 — Immediate-Relevance Pair

Retain the existing pair whose distinction affects the next-token prediction immediately.

This establishes the geometric scale expected when a latent distinction receives direct supervision.

---

## Control 2 — Permanently Irrelevant Cue

Include a latent cue that never affects any future prediction.

Test whether:

\[
R^2
\]

and geometric prominence disappear during training.

---

## Control 3 — Random Direction

Compare delayed-pair amplification against arbitrary hidden-space directions.

---

## Control 4 — Initialisation

Measure every quantity before training.

All conclusions about amplification should be relative to initial geometry.

---

## Control 5 — Matched Performance

Ensure effects are not simply due to some models failing to learn.

Report predictive loss relative to the entropy/Bayes floor.

---

# Primary Figures

## Figure 1 — Information vs Prominence

Plot against relevance frequency:

\[
x=r.
\]

Show:

- belief \(R^2\);
- normalised delayed-pair distance.

The main desired observation is that decodability saturates before metric prominence.

---

## Figure 2 — Prominence vs Forgetting Cost

Plot:

\[
x=\Delta L_{\text{forget}}
\]

against:

\[
y=P_{\text{metric}}.
\]

Include frequency, strength and loss-weight interventions with distinct markers.

Test whether they fall onto approximately the same relationship.

---

## Figure 3 — Gradient Pressure vs Geometry

Plot:

\[
G_{\parallel}
\]

against:

\[
P_{\text{metric}}.
\]

This is the most direct test of the optimization-pressure hypothesis.

---

## Figure 4 — Training Dynamics

Plot over training steps:

- decodability;
- metric prominence;
- relevant gradient;
- accuracy.

This should reveal which quantities emerge first.

---

# Success Criteria

The working hypothesis is strongly supported if:

1. delayed information remains highly decodable before becoming geometrically prominent;
2. prominence increases with relevance frequency;
3. prominence increases with consequence strength;
4. changing only the loss weight changes prominence;
5. expected cost of forgetting predicts prominence across manipulations;
6. measured gradient relevance predicts prominence;
7. longer credit-assignment delays reduce prominence without necessarily destroying decodability;
8. the immediately relevant control remains more geometrically prominent than equally decodable delayed information.

---

# Possible Outcomes

## Outcome A — Gradient-pressure account works

If:

\[
P_{\text{metric}}
\approx
f(\Delta L_{\text{forget}},G)
\]

across interventions, the main claim becomes:

> **Information can be retained without dominating representation geometry; geometric prominence reflects the optimization pressure associated with using that information.**

---

## Outcome B — Generative process dominates

If frequency and strength strongly affect geometry but direct loss reweighting does not, the important variable may be the temporal/statistical structure of the generative process rather than gradient magnitude itself.

---

## Outcome C — Decodability and prominence remain coupled

If both rise and fall together, then the distinction observed in the initial GRU may be specific to that construction rather than a general phenomenon.

---

## Outcome D — Geometry is highly seed/parameterisation dependent

If information is consistently retained but its scale varies substantially across equivalent solutions, this would point toward implicit bias and parameterisation as the main object requiring explanation.

---

# Scope

Do **not** add:

- new RL environments;
- large transformers;
- natural-language tasks;
- additional interpretability methods;
- new architecture families;

until the relationship between:

\[
\boxed{\text{information retention}}
\]

and

\[
\boxed{\text{geometric prominence}}
\]

is understood in the existing HMM.

The purpose of TASK4 is to demystify one phenomenon rather than collect additional representation correlations.

---

# Target Result

The ideal result is a quantitative relationship of the form:

\[
\boxed{
\text{representational prominence of }z
\;\sim\;
\text{optimization cost of failing to preserve/use }z
}
\]

while demonstrating that:

\[
\boxed{
\text{decodable information}
\neq
\text{geometrically dominant information}.
}
\]
