Project Goal

Test whether neural agents preserve the full geometry of future occupancy, or instead compress states into decision-relevant equivalence classes determined by the optimal policy.

Current evidence suggests:

Neural representations do not strongly preserve Euclidean occupancy geometry. Instead, they appear to preserve the distinctions in occupancy that matter for optimal action selection.

The next experiments should directly distinguish these hypotheses.

Core Hypotheses
H1 — Full Occupancy Geometry

Hidden-state similarity reflects similarity in future occupancy:

[
d(h_i,h_j)
\propto
d(\rho_i,\rho_j).
]

Representations should change smoothly as occupancy distributions change.

H2 — Policy Quotient Geometry

States are represented similarly when they induce the same optimal actions across goals, even if their occupancy vectors differ.

Define

[
s_i \sim s_j
]

when

\arg\max_a Q(s_j,a,g)
]

for the relevant goals.

The representation should primarily reflect these equivalence classes.

H3 — Smooth Value Geometry

The network may encode continuous value or successor information rather than either raw occupancy or discrete optimal actions.

Candidate alternatives include:

successor representation,
shortest-path / spatial geometry,
(Q)-value geometry,
advantage geometry,
action-margin geometry.
Task 1 — Build a Controlled Policy-Switch Environment

Create an MDP containing two competing routes to the same goal.

Example:

Route A: short but unreliable.
Route B: long but reliable.

Introduce a scalar parameter

[
\lambda
]

controlling the failure probability or cost of Route A.

Sweep (\lambda) continuously.

The environment should be designed so that:

[
\rho_\lambda
]

changes smoothly while the optimal action switches at some critical value

[
\lambda_c.
]

Task 2 — Compute Ground-Truth Quantities

For every value of (\lambda), compute:

Occupancy

[
\rho_\lambda(g|s,a).
]

Q-values

[
Q_\lambda(s,a,g).
]

Optimal policy

[
\pi_\lambda^*(a|s,g).
]

Action margin

For the best and second-best actions,

Q_\lambda(s,a_1,g)

Q_\lambda(s,a_2,g).
]

Record where

[
m_\lambda=0,
]

since these are the policy-switch boundaries.

Task 3 — Train Networks Across the Sweep

For each (\lambda):

Generate optimal behaviour.
Train the same architecture from multiple random seeds.
Train only on action prediction / behavioural cloning.
Do not explicitly supervise occupancy, Q-values, or successor representations.

Save hidden activations for all states and goals.

Use enough seeds to distinguish stable geometry from optimisation noise.

Task 4 — Measure Representational Change Across the Policy Boundary

Track

[
h_\lambda(s,g)
]

as (\lambda) changes.

Measure

[
|h_{\lambda+\Delta}(s,g)-h_\lambda(s,g)|.
]

Compare this against:

occupancy change,
successor-representation change,
Q-value change,
action-margin change,
whether the optimal action changed.

The crucial question is:

Does hidden geometry vary smoothly with the environment, or change disproportionately when the optimal action changes?

Task 5 — Test the Quotient Prediction Directly

Find pairs of states with:

Case A

Very different occupancy vectors but identical optimal policy tables.

Case B

Very similar occupancy vectors but different optimal actions.

Compare their hidden-state distances.

The quotient hypothesis predicts:

[
d_h(A) < d_h(B)
]

even when occupancy geometry predicts the reverse.

Construct these cases deliberately rather than relying on naturally occurring examples.

Task 6 — Extend the Three-Way Regression

Predict hidden-state pairwise distance using simultaneous regressors:

\beta_1D_{\text{occupancy}}
+
\beta_2D_{\text{SR}}
+
\beta_3D_{\text{policy}}
+
\beta_4D_Q
+
\beta_5D_{\text{margin}}
+
\epsilon.
]

Run this separately by layer and environment.

Main questions:

Does policy remain dominant after adding (Q)-value geometry?
Does advantage or action-margin geometry explain more than discrete policy identity?
At what layer does decision-relevant structure emerge?
Task 7 — Analyze the State–Goal Decomposition

Fit

h_s(s)
+
h_g(g)
+
h_{sg}(s,g).
]

Quantify variance attributable to:

state,
goal,
interaction.

Current result suggests interaction variance is about 11%.

Next test whether this small interaction component is causally important.

Remove or project out

[
h_{sg}
]

and measure:

action accuracy,
goal sensitivity,
steering success.

Compare against projecting out random subspaces of the same dimensionality and norm.

Task 8 — Refine the Steering Experiment

For goals (g_1,g_2), estimate

E_s[h(s,g_2)-h(s,g_1)].
]

Intervene using

h+\alpha v.
]

For each state measure:

whether the action flips,
the flip threshold (\alpha^*),
occupancy difference,
Q-value difference,
action margin.

Test which quantity best predicts

[
\alpha^*.
]

The goal is to determine whether steering acts through occupancy-like geometry, value geometry, or direct movement across an action boundary.

Task 9 — Test Stochastic Policies More Strongly

Construct an environment where deterministic and stochastic models make clearly different predictions.

Avoid ordinary grids where their occupancy vectors remain highly correlated.

Use asymmetric risk:

one short route with high transition failure,
one long route with low failure,
adjustable risk level.

Test whether hidden geometry tracks:

[
\pi^*_{\text{stochastic}}
]

rather than:

[
\pi^*_{\text{deterministic}}.
]

Task 10 — Analyze Dimensionality Separately from Metric Geometry

For each representation compute:

participation ratio,
PCA spectrum,
effective rank,
intrinsic dimensionality.

Compare:

[
PR_{\text{hidden}},
\quad
PR_{\text{occupancy}},
\quad
PR_{\text{policy}},
\quad
PR_{\text{random}}.
]

Investigate whether occupancy correctly predicts the dimensionality of the task even when it fails to predict the metric geometry.

This tests the possibility that:

Occupancy determines the available degrees of freedom, while action selection determines how those degrees of freedom are geometrically organised.

Main Desired Result

The strongest result would be evidence for a hierarchy:

[
\text{environment dynamics}
\rightarrow
\text{future occupancy/value}
\rightarrow
\text{decision equivalence classes}
\rightarrow
\text{learned neural geometry}.
]

In particular, show that the neural network systematically discards distinctions in predictive state that do not alter optimal behaviour.

Success Criteria

The policy-quotient hypothesis becomes substantially stronger if:

Policy-table geometry remains dominant after controlling for occupancy, SR, Q-values and spatial geometry.
States with different futures but identical decisions merge representationally.
States with similar futures but different required actions remain separated.
Representation changes concentrate near policy-switch boundaries.
Small goal-dependent interaction components are causally necessary for action selection.
Steering crosses the predicted decision boundary.
The effect reproduces across architectures, seeds and stochastic environments.

The resulting claim should remain scoped to the tested agents:

In small goal-conditioned neural policies, learned representations appear to compress predictive future structure into geometry organised primarily around behaviourally relevant action distinctions.
