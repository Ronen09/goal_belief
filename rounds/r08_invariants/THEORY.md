# The invariance principle for representation measures

Companion to `rounds/r08_invariants/REPORT.md` and `rounds/r09_interventions/REPORT.md`. This note fixes
the definitions those reports use, states the one result that organises them,
and classifies the candidate quantities.

## 1. Models, representations, downstream maps

A model is a triple M = (E, R, W): a token embedding E, a recurrence
h_t = R(h_{t−1}, E(x_t)) with h_t ∈ ℝ^d, and a readout z_t = W h_t + b producing
p_M(x_{t+1} | x_{1:t}) = softmax(z_t). The representation of a history x_{1:t}
is h_M(x_{1:t}), the state reached by running the recurrence on it. The set of
such states over the input distribution is the reachable set 𝓗_M ⊂ ℝ^d.

For a state h at position t and a continuation x_{t+1:t+k}, the *downstream
map at horizon k* is

F_M^{(k)}(h; x_{t+1:t+k}) = softmax(W · R^{(k)}(h; x_{t+1:t+k}) + b),

the model's prediction after consuming k further tokens from h. F^{(0)} is the
readout itself. "The actual remaining computation" means F^{(k)} with the
network's own R and W, not a surrogate.

## 2. Two equivalences

**Coordinate equivalence.** Choose a cut in the computation graph at which the
state h is passed to its consumers. An *admissible reparameterisation* is an
invertible A ∈ GL(d) applied at the cut, with every consumer compensated:
h′ = Ah, W′ = WA⁻¹, and every downstream map replaced by F′ = F ∘ A⁻¹. Two
representations are coordinate-equivalent, h₁ ∼_coord h₂, when they are related
by such a compensated change. Every quantity computed from (h, W, F) has an
image under A, and the input-output map is unchanged exactly.

Two remarks. First, for a recurrent network the cut is at the interface
between the state and its consumers (readout and next recurrence step); the
compensated F′ is a valid computation but in general not a member of the same
architecture family. A GRU's own parameter symmetries are only unit
permutations and per-unit sign flips (tanh is odd, the gates are not), so
∼_coord is a much larger group than the architecture's symmetry group, and
anything invariant under ∼_coord is invariant under the architecture's
symmetries. Second, ∼_coord is exact: it is a change of description, not of
computation, and a quantity that fails it is measuring the description.

**Functional equivalence.** M₁ ∼_func M₂ when p_{M₁}(· | x_{1:t}) =
p_{M₂}(· | x_{1:t}) for every history in the support, in practice when the
pairwise KL over the evaluation set is below a tolerance and both are at the
Bayes floor of the task. This is an equivalence of input-output maps. It says
nothing about R, W, 𝓗 or the coordinates separately.

Coordinate equivalence implies functional equivalence. The converse fails, and
the size of the gap is what TASK5 to TASK7 measured: the same input-output map
is reached by networks whose readout gains differ by 9× and whose hidden
separations differ by 6×, related by no linear map at all.

## 3. Descent

A quantity Q assigns a number to a model, or to a model and some inputs. Q
*descends to* ∼_coord when Q(M ∘ A) = Q(M) for every admissible A, and descends
to ∼_func when Q(M₁) = Q(M₂) whenever M₁ ∼_func M₂. A quantity that descends to
∼_func is a property of the computation; one that descends only to ∼_coord is a
property of the implementation stated in a coordinate-free way; one that
descends to neither is a property of the description.

Descent is necessary, not sufficient, for usefulness: a constant descends to
everything. The third requirement is sensitivity, that Q change when the
function changes in the respect one cares about. TASK7 operationalised the
three requirements as a transform ratio, a coefficient of variation across an
equivalence set, and an F-ratio across task strengths.

## 4. The functional object

For two histories x, x′ and a shared continuation c of length k, define

Φ_M^{(k)}(x, x′; c) = D( F_M^{(k)}(h_M(x); c), F_M^{(k)}(h_M(x′); c) ),

with D a divergence between output distributions (Jensen–Shannon in the
experiments). The downstream map is part of the object: Φ compares what the two
states *do*, through the network's own future computation, not where they are.

**Claim.** Φ descends to ∼_func.

*Proof.* F_M^{(k)}(h_M(x); c) is the model's prediction after the history x
followed by c, that is p_M(· | x, c). So Φ_M^{(k)}(x, x′; c) =
D(p_M(· | x, c), p_M(· | x′, c)), a functional of the input-output map alone.
If M₁ ∼_func M₂ the two predictions agree and Φ agrees. ∎

The proof uses two things. First, that the arguments are reachable states,
h_M(x) and h_M(x′): for an arbitrary point h ∉ 𝓗_M, F_M^{(k)}(h; c) is not a
prediction of the input-output map for any history, and nothing constrains it
across functionally equivalent models. This is the on-manifold restriction,
and TASK8 tests how far off the manifold the agreement survives. Second, that
h is a *complete cut*: every path from the history x to the prediction passes
through h, so that F^{(k)}(h_M(x); c) really is p_M(· | x, c). For a recurrent
network the state h_t is such a cut. For a transformer the residual stream at
a single position is not: the future also reads the past through attention to
other positions, so the cue at t reaches the prediction at t+1 both through the
residual at t and through what layer 1 at t+1 already copied into t+1's own
residual. Patching one position then measures the share of the computation
that happens to route through it, which is an implementation choice, and TASK9
found it to differ from seed to seed (patching JS 0.00 to 0.33 across
functionally identical transformers). The complete cut for a distinction made
at t and consumed at t+k is the set of residuals, at the chosen layer, at all
positions from t to t+k−1, and Φ descends to ∼_func only when computed on it.

**The linearisation does not descend.** Write Δh = h_M(x) − h_M(x′) and
J = ∂F^{(k)}/∂h at h_M(x). The local displacement JΔh, and the Fisher form
ΔhᵀJᵀ F(p) JΔh, are exactly ∼_coord-invariant: under A, J → JA⁻¹ and Δh → AΔh.
But ∼_func fixes only the composite F^{(k)} ∘ h_M, that is F^{(k)} evaluated on
𝓗_M; it fixes neither the map F^{(k)} away from the points it is evaluated at
nor the placement of those points. Two equivalent models therefore have
different J at corresponding states, and JΔh differs unless F^{(k)} is affine
along the segment between the two states. In TASK7 the coefficient of variation
of ‖JΔh‖ across 51 equivalent models was 0.27 while the finite displacement's
was 0.07, and the discrepancy grew with the size of Δh relative to the
recurrence's curvature. Local functional geometry is a coordinate-free
description of one implementation, not of the computation.

**Interventions.** An intervention on M is a map on states, h ↦ h + v, applied
at a cut and read through F^{(k)}. Two interventions (M₁, v₁) and (M₂, v₂) are
*functionally equivalent* when F₁^{(k)}(h₁(x) + v₁; c) = F₂^{(k)}(h₂(x) + v₂; c)
as distributions for every history in the relevant set and every continuation.
This is an equivalence of effects, not of vectors: v₁ and v₂ live in different
spaces, have no common norm, and are compared only through what they do. A
steering vector is then defined by its effect curve α ↦ Φ(h + αv, h), and two
vectors correspond when their effect curves can be matched. The on-manifold
question reappears here: v is admissible to the extent that h + αv stays where
F^{(k)} is constrained by the function.

**The propagation hierarchy.** Fixing the histories and continuation and
varying what is compared gives a ladder:

1. raw geometry, ‖Δh‖ and everything built from Euclidean distances in ℝ^d:
   descends to neither equivalence;
2. local functional geometry, JΔh and the Fisher form: descends to ∼_coord
   only;
3. finite-horizon functional effect, Φ^{(k)}: descends to ∼_func for every k at
   which the arguments are reachable states;
4. behaviour, Φ^{(k)} at the horizon where the distinction is consumed: the
   same object, read at the step the task cares about.

Rung 3 is already rung 4 in kind; the difference between them is which k one
reads. What separates rung 2 from rung 3 is not a matter of degree but of
which map the quantity depends on.

## 5. Classification of the candidates

| quantity | ∼_coord | ∼_func | sensitive | status |
|---|---|---|---|---|
| ‖Δh‖, hidden norm, participation ratio, effective rank, P_metric, gain, cos θ | no | no | weak | description |
| Euclidean / cosine RSA, top-k PCA fraction | no (rotations and scalars only) | empirically stable | no | description of shared coarse structure |
| unregularised linear decodability, exact rank | yes | not implied; empirically stable | no | availability |
| sample-space projection P_H, whitened distances | yes | not implied; 0.7–0.9 overlap | no | represented subspace |
| readout displacement WΔh, contrast (w_x−w_y)·Δh | yes | yes at the readout cut (Φ^{(0)} in logit form) | yes | functional |
| JΔh, Fisher form at the state | yes | no | weak | local description |
| Φ^{(k)} with finite propagation | yes | yes (Claim) | yes | functional |

"Not implied" marks quantities that ∼_func does not fix in general but that were
stable across the TASK7 equivalence set; they are properties of the class of
implementations that a given architecture and optimiser produce, and a
different architecture could break them without changing the function.

## 6. What the principle says for interpretability

Any claim of the form "feature f is represented with strength s" that is read
off Euclidean geometry is a statement about a description; it can be changed by
a change of coordinates or by training the same function with a different
readout learning rate. The claims that survive are of two kinds: availability,
that f can be read out linearly (a property of the implementation class), and
effect, that the states distinguished by f produce a given change in the
network's own downstream output (a property of the computation). The second
must be computed by running the network, not by a metric at the state, because
the function constrains the network only where the network goes.
