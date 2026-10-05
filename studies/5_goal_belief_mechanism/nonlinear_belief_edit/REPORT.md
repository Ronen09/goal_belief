# A nonlinear encoding of the posterior before block 0's MLP (nonlinear belief edit)

Run date: 2026-10-02. Brief: `BRIEF.md` (the attention-belief-edit experiment's proposed next step). Encodings, edits and decision rule: `PLAN.md`,
committed (`48ad01d`) **before any nonlinear encoding was fitted on a trained model or any edit of this experiment applied to
one**. Numbers from `tables.md` and `posthoc.json`; data in `results.json`. Reproduce: `reproduce.py nonlinear_belief_edit` (6 min on one
GPU).

**Setting.** The attention-belief-edit experiment's site: the goal token's state after block 0's attention and before its MLP. The direct-belief-edit experiment's pairs,
measures and machinery. New encodings of the posterior b, fitted on the fit side under all three goals without action
labels:

* **MLP** f(b), one for every goal, with c_{g,L} offsets (primary).
* **Table** μ(b): the mean state for each of the 619 distinct posteriors, shrunk toward f where a posterior is rare.
  This is the ceiling for any encoding that is a function of the posterior alone.
* Goal-conditioned versions f(b, g) and μ_g(b) (secondary).

Edits change only the goal token's state. **Residual** is the donor's actual difference minus its table part, the part
of the route that is not a function of the posterior. Section 4 is post hoc.

## 1. The nonlinear encodings fit much better

| held-out R² within (goal, length) | shared across goals | goal-conditioned |
|---|---|---|
| linear (attention belief edit) | 0.649 | 0.672 |
| **MLP** | **0.768** (0.740–0.794) | 0.796 |
| **table** (ceiling) | 0.769 | 0.802 |

* The MLP adds 0.12 over linear (N holds). The table adds nothing beyond the MLP: the MLP already captures what the
  posterior explains. The edit vectors match the actual difference better: cosine 0.83 against 0.76; 25 % of its
  squared norm left over, against 36 %.

## 2. The decisive cases: better fit, little better control

Goal-dependent pairs (5 653; "left under G1, right under G2"); share donor-optimal under every change goal:

| edit | rate | / whole |
|---|---|---|
| none | 0.038 | 0.13 |
| linear, shared (attention belief edit) | 0.177 | 0.61 |
| **MLP f(b), shared** | **0.199** | **0.68** (0.60–0.88) |
| **table μ(b), shared** (ceiling) | 0.187 | **0.65** (0.57–0.85) |
| MLP f(b, g), goal-conditioned | 0.242 | **0.86** (0.80–1.02) |
| table μ_g(b), goal-conditioned | 0.227 | 0.81 |
| **residual** (whole minus the table's part) | 0.045 | **0.15** |
| rotated MLP edit | 0.034 | 0.12 |
| PCA-13 patch | 0.262 | 0.98 |
| whole | 0.276 | 1 |
| hybrid | 0.518 | 1.63 |

* **Nonlinearity in the posterior adds little.** The shared MLP edit gains 0.02 absolute over linear (0.68 against
  0.61 of whole; above linear in every model, p < 0.001). It stays short of 0.75 (**P fails**). The table, the best any
  shared function of the posterior can do, does no better (0.65; **T fails**).
* **What the posterior leaves out does nothing on its own.** The residual edit gives the donor's whole state minus
  its posterior part, and reaches 0.15 of whole, the same as no edit (0.13).
* **The goal-conditioned encodings clear the line.** They use the same inputs as the shared ones (the donor's posterior
  and the recipient's own goal), with a map per goal: 0.86 (MLP), 0.81 (table). So C7 fails on these pairs: the goal
  must enter the map.

## 3. Elsewhere, every posterior encoding works

| | linear | **MLP, shared** | table, shared | MLP, goal-conditioned | PCA-13 |
|---|---|---|---|---|---|
| main change cells: S (of whole) | 0.90 | **0.94** | 0.91 | 0.98 | 0.99 |
| preserve cells: harm | 0.006 | **0.008** | 0.006 | 0.007 | 0.004 |
| one-step agreeing: S | 1.08 | **0.97** | 0.95 | 1.01 | 1.00 |
| equivalent recipients, all four donor-optimal / whole | 0.93 | **0.92** | 0.90 | 0.97 | 0.99 |

C1, C3, C4, C5b and C6 hold for the shared MLP edit. The rotated MLP edit gives 0.02 of whole.

## 4. Post hoc: is the goal-conditioned advantage specific to the recipient's goal?

The same goal-conditioned encodings, but edited with *another* goal's map (both other goals, averaged); same fits
(reproduced exactly):

| goal-dependent pairs, / whole | MLP | table |
|---|---|---|
| the recipient's goal's map | **0.86** | **0.81** |
| shared map | 0.68 | 0.65 |
| another goal's map | **0.56** | **0.53** |

* **Another goal's map does worse than the shared one**, in all ten models (right against wrong p = 0.001 for both
  encodings; shared against wrong p = 0.001). The goal-specific part of the belief code is specific to the goal that
  is actually present. The policy reads it, and reading it with the wrong goal costs decisions.
* On main pairs the difference is small (S 0.98 / 0.94 / 0.91 for right / shared / wrong): it matters where the goal
  decides what the belief change means.

## 5. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **P** | goal-dependent: mlp ≥ 0.75 × whole; above rotated; above linear | 0.68 ×; p < 0.001; p < 0.001 | **no** |
| N | mlp R² within ≥ linear + 0.05 | +0.117 | yes |
| C1 | S(mlp) ≥ 0.75, gain ≥ 0.25 | 0.94; 0.46 | yes |
| C3 | harm ≤ 0.05 | 0.008 | yes |
| C4 | one-step agreeing ≥ 0.5 × whole, above rotated | 0.97 ×; p < 0.001 | yes |
| C5b | D(pca) − D(mlp) ≤ 0.05 | 0.026 | yes |
| C6 | all four ≥ 0.75 × whole | 0.92 × | yes |
| C7 | D(mlp) ≥ D(mlp_goal) − 0.05; goal-dependent ≥ mlp_goal − 0.03 | −0.023; **−0.054** | **no** |
| **T** | goal-dependent: table ≥ 0.75 × whole | 0.65 × | **no** |
| R | residual / whole (descriptive) | 0.15 × (none 0.13 ×) | |

**The registered reading for "P and T fail" is that the route carries information beyond the posterior that the policy
uses. The data contradict it.** That information, the residual, does nothing on its own. And encodings that are
functions of the posterior and the recipient's own goal, nothing more, clear the line (0.81–0.86). The plan's table
did not anticipate this case. I report the registered reading as not supported, rather than adopt it.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | N holds | yes | +0.117 |
| E2 | table ≤ mlp + 0.03 | yes | +0.001 |
| E3 | P fails | yes | 0.68 × |
| E4 | mlp above linear on goal-dependent pairs | yes | +0.02, p < 0.001 |
| E5 | T fails | yes | 0.65 × |
| E6 | R ≥ 0.2 | **no** | 0.15 (no edit 0.13) |
| E7 | C1 with S(mlp) ≥ 0.9 | yes | 0.94 |
| E8 | C3 holds | yes | 0.008 |
| E9 | C7 holds | **no** | −0.054 on goal-dependent pairs |
| E10 | C5b holds | yes | 0.026 |

8 of 10. Both misses point the same way: what is missing is not beyond the posterior, but the goal's part in it.

## Correction to the attention-belief-edit experiment

The attention-belief-edit experiment's report said the goal-specific linear edit "also falls short" on goal-dependent pairs, and that sharing cost
only "0.03 absolute". Its 0.79 of whole is above the 0.75 line, and the cost of sharing is about half of the shared
edit's shortfall. The report has been corrected (marked in its §3), and so has the summary (`FINDINGS.md`).

## Conclusions

1. **The gap on goal-dependent pairs was not the linear form.** A nonlinear encoding of the posterior fits the state
   much better (+0.12 R²) but controls these decisions only slightly better (0.61 → 0.68 of whole). The best possible
   shared function of the posterior does no better (0.65).
2. **Nor is it information beyond the posterior.** The part of the donor's state that is not a function of its
   posterior does nothing by itself (0.15, as no edit).
3. **It is the goal's part of the belief code.** At block 0's attention output, the state holds a belief code that is
   mostly shared across goals (the attention-belief-edit experiment: interaction 4.5 % of the variation) with a small goal × belief part. Encodings
   that include it reach 0.81–0.86 of whole on goal-dependent pairs. The same encodings with another goal's map fall
   to 0.53–0.56, below the shared code. That small part carries much of what turns one belief change into different
   actions under different goals.
4. **So the goal reaches the belief earlier than block 0's MLP**, a little. The goal-swap-components experiment's block 0 heads respond to the
   goal, and the goal token's query includes the goal embedding. Attention already reads the evidence slightly
   differently per goal, and the policy depends on that difference where the goal must decide. Block 0's MLP then
   strengthens it (the direct-belief-edit to attention-belief-edit experiments).

Next, if wanted: locate the goal × belief part. Patch block 0's attention pattern at the goal token from another
goal's run while keeping the values, and the reverse. That splits it into "which tokens are read" (goal-dependent
queries) and "what is read" (the goal token's own value).

Limits: the direct-belief-edit to attention-belief-edit experiments's. The table's sparse posteriors depend on the shrinkage. Section 4 is post hoc, and the
registered reading for this outcome was rejected on secondary evidence. That is stated, not hidden.
