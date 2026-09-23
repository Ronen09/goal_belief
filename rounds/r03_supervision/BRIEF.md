## Goal

Test the revised hypothesis:

> **Neural representation geometry preserves the distinctions contained in the training target. Hard argmax supervision quotients away predictive structure that does not affect the chosen action, while softer targets preserve more of the underlying value/advantage structure.**

The objective of TASK3 is to move from the descriptive Round-2 result to a controlled causal and theoretical account of the **supervision bottleneck**.

---

## Task 1 — Boltzmann Temperature Sweep

Replace the binary hard-vs-soft comparison with a continuous sweep.

For each temperature

$$
\tau \in
\{0.02,0.05,0.1,0.2,0.5,1,2,5\},
$$

construct teacher targets

$$
\pi_\tau(a|s,g)
=
\frac{\exp(Q(s,a,g)/\tau)}
{\sum_{a'}\exp(Q(s,a',g)/\tau)}.
$$

Train otherwise identical networks.

Measure at every layer:

* policy-table RSA,
* soft-target RSA,
* Q-value RSA,
* advantage RSA,
* occupancy RSA,
* successor-representation RSA,
* spatial RSA,
* participation ratio.

### Prediction

As \(\tau\to0\), geometry should approach the hard-policy quotient.

As \(\tau\) increases, within-action distinctions should increasingly survive.

The transition should be gradual rather than binary.

---

## Task 2 — Measure the Geometry of the Targets Themselves

For every supervision condition, construct a representational-distance matrix directly from the target vectors.

For soft targets:

$$
D_T(i,j)
=
d(\pi_\tau(\cdot|i),\pi_\tau(\cdot|j)).
$$

Compare:

$$
D_T
\quad\text{vs}\quad
D_h.
$$

Run the same RSA and regression analyses used for occupancy and policy geometry.

### Key question

Does target geometry predict hidden geometry better than the environmental quantities from which the targets were generated?

A strong result would be:

$$
\text{environment/value geometry}
\rightarrow
\boxed{\text{target geometry}}
\rightarrow
\text{hidden geometry}.
$$

---

## Task 3 — Fixed-Argmax Counterfactual Test

Construct pairs with the same optimal action but systematically different action margins.

For example:

$$
(0.51,0.49,0,0)
$$

versus

$$
(0.99,0.01,0,0).
$$

Both have the same argmax but contain very different information under soft supervision.

Train:

1. hard argmax BC;
2. soft-policy BC.

Compare the representation distance between the two examples.

### Prediction

Hard targets:

$$
d_h \approx 0
$$

relative to ordinary pairwise distances.

Soft targets:

$$
d_h > 0.
$$

Sweep the difference continuously while keeping the argmax unchanged.

This should be a central causal experiment.

---

## Task 4 — Hold Environment Fixed, Manipulate Only Labels

Remove environmental confounding entirely.

Take one fixed dataset of states and goals.

Construct several artificial teacher targets from exactly the same underlying examples:

### A. Argmax

$$
T_{\text{hard}}(Q)=\operatorname{onehot}(\arg\max Q).
$$

### B. Boltzmann

$$
T_\tau(Q)=\operatorname{softmax}(Q/\tau).
$$

### C. Advantage regression

Directly predict

$$
A(s,a,g)=Q(s,a,g)-V(s,g).
$$

### D. Q regression

Predict the full \(Q\)-vector.

### E. Occupancy regression

Predict occupancy directly as a positive control.

Use matched architecture and hidden dimensionality.

### Question

How closely does changing only

$$
T(Q)
$$

change the learned representation?

This isolates the role of the learning signal from the environment.

---

## Task 5 — Quantify Information Destroyed by Each Target

Characterise each target transformation as an information bottleneck.

Starting from the underlying value vector \(Q\),

$$
Q
\xrightarrow{T}
Y.
$$

Ask which distinctions survive in \(Y\).

Estimate, where practical:

* target rank,
* participation ratio,
* pairwise-distance preservation,
* number of distinct equivalence classes,
* recoverability of \(Q\),
* recoverability of advantage,
* recoverability of occupancy.

For argmax,

$$
Q_1\sim Q_2
\iff
\arg\max Q_1=\arg\max Q_2.
$$

For softmax at finite temperature the equivalence classes should be much finer.

Relate these properties to hidden dimensionality and geometry.

---

## Task 6 — Test a Sufficient-Statistic Hypothesis

Formalise the stronger possibility:

> The network learns a representation sufficient for predicting the supervised target rather than a faithful representation of the environment.

If

$$
Y=T(X),
$$

then seek a representation \(h(X)\) satisfying approximately

$$
p(Y|X)\approx p(Y|h(X)),
$$

while distinctions in \(X\) irrelevant to \(Y\) are compressed.

Operational tests:

* identify pairs with identical targets but different environmental structure;
* identify pairs with different targets but highly similar environmental structure;
* compare their hidden distances;
* measure whether within-target-class variance shrinks during training.

Track this across epochs.

---

## Task 7 — Track Geometry During Training

Save checkpoints throughout optimisation.

At each checkpoint measure:

* target RSA,
* policy RSA,
* advantage RSA,
* occupancy RSA,
* participation ratio,
* within-class distance,
* between-class distance.

### Question

Does geometry first reflect the input/environment and then progressively reorganise around the supervision target?

For hard BC, test whether

$$
\frac{
\text{within-policy-class distance}
}{
\text{between-policy-class distance}
}
$$

falls systematically during training.

Compare this trajectory against soft supervision.

---

## Task 8 — Generalisation Across Unseen States

Train on only a subset of the state-goal combinations.

Evaluate hidden geometry on held-out combinations.

Determine whether the supervision-induced geometry reflects:

* memorisation of target classes, or
* learned structural generalisation.

Use environments containing repeated or symmetric structures.

Test whether unseen states with identical decision structure are mapped together.

---

## Task 9 — Architecture Replication

Repeat the core experiments using at least:

1. current MLP;
2. deeper MLP;
3. small residual MLP;
4. small transformer or recurrent model if sequential input is straightforward.

Do not replicate every experiment initially.

Minimum replication:

* hard target,
* intermediate Boltzmann temperature,
* soft target,
* fixed-argmax counterfactual test.

### Success condition

The qualitative dependence of representation geometry on supervision survives architecture changes.

---

## Task 10 — Clean Up the Additive State/Goal Analysis

Do not rely on the existing interaction-subspace ablation because the estimated subspace captures 90–97% of additive variance.

Instead fit the decomposition more carefully:

$$
h(s,g)
=
\mu+h_s(s)+h_g(g)+h_{sg}(s,g).
$$

Use orthogonality/centering constraints:

$$
\sum_s h_s(s)=0,
$$

$$
\sum_g h_g(g)=0,
$$

$$
\sum_s h_{sg}(s,g)=0,
\qquad
\sum_g h_{sg}(s,g)=0.
$$

Then measure the variance genuinely unique to the interaction term.

Only perform causal ablations after validating that the decomposition isolates unique interaction structure.

---

## Task 11 — Relate Steering to Target Geometry

The current result shows that steering thresholds are almost perfectly predicted by the network's own linearised boundary:

$$
\rho \approx 0.92-1.00.
$$

Now ask whether supervision controls this geometry.

Across temperatures, compare:

* goal-steering success,
* steering threshold,
* target probability margin,
* learned logit margin,
* Q margin,
* occupancy quantities.

### Prediction

As supervision becomes softer, target probability/logit structure should increasingly explain steering behaviour.

Do not expect occupancy to contribute independently unless it survives through the target.

---

## Task 12 — Try to Derive the Quotient Theoretically

For hard classification, consider a supervised objective

$$
\mathcal L
=
\mathbb E[-\log p_\theta(y|x)].
$$

If

$$
T(x_1)=T(x_2),
$$

the loss provides no direct incentive to preserve differences between \(x_1\) and \(x_2\).

Characterise the equivalence relation

$$
x_1\sim_Tx_2
\iff
T(x_1)=T(x_2).
$$

Then ask under what assumptions optimisation or regularisation favours a representation constant, or approximately constant, on these equivalence classes.

For soft targets replace exact equality by similarity under

$$
d(T(x_1),T(x_2)).
$$

The target theoretical statement is not that networks **must** collapse to the quotient, but to identify conditions under which training pressures them toward it.

---

## Task 13 — Distinguish Target Geometry From Loss Geometry

Use the same target probabilities but change the training loss.

Candidates:

* cross entropy,
* KL divergence with teacher probabilities,
* MSE on probabilities,
* MSE on logits.

Ask whether learned geometry depends only on what information is present in the target, or also on how the loss weights differences between target vectors.

This separates:

$$
\text{target information}
$$

from

$$
\text{optimisation geometry induced by the loss}.
$$

---

## Task 14 — Final Discriminating Experiment

Construct one environment containing three pairs:

### Pair A

Large occupancy difference, identical hard target.

### Pair B

Small occupancy difference, different hard target.

### Pair C

Identical hard target but strongly different soft targets.

Predictions:

| Pair | Hard BC  | Soft BC           |
| ---- | -------- | ----------------- |
| A    | merge    | may separate      |
| B    | separate | separate          |
| C    | merge    | strongly separate |

This single environment should visually demonstrate the supervision-bottleneck result.

---

## Main Success Criteria

TASK3 succeeds if:

1. Hidden geometry changes systematically with target temperature.
2. Target geometry explains hidden geometry better than raw occupancy.
3. Same-argmax/different-confidence states merge under hard supervision but separate under soft supervision.
4. The effect appears when the environment is held completely fixed.
5. Within-target-class compression develops during training.
6. The result replicates across architectures.
7. Target/loss structure predicts intervention geometry better than occupancy.
8. A theoretical equivalence-class account explains the main empirical effects.

---

## Candidate Headline

> **Learned representations preserve the distinctions demanded by supervision rather than faithfully reproducing the geometry of the underlying predictive state.**

In this setting:

$$
\text{world structure}
\rightarrow
\text{value / occupancy structure}
\rightarrow
\boxed{\text{supervision bottleneck}}
\rightarrow
\text{representation geometry}.
$$

Hard behavioural cloning produces a policy quotient because the argmax target has already discarded within-action distinctions. Soft supervision preserves those distinctions and produces correspondingly smoother neural geometry.
