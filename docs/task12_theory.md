# Task 12 — why supervision induces a quotient (theoretical account)

## Setting

A network computes logits `z(x) = W h(x) + b` from a representation `h(x)` and is
trained by minimising `L = E_x[ ℓ(z(x), T(x)) ]`, where the target `T(x)` is a
deterministic function of an underlying quantity `Q(x)` (here `x = (s, g)` and
`Q(x) = Q(s,·,g)`). For behavioural cloning `ℓ` is the soft cross-entropy
`ℓ(z, p) = −Σ_a p_a log softmax(z)_a`.

## 1. The loss factors through the target

`L` depends on `x` only through the pair `(h(x), T(x))`. Hence the gradient of the
loss with respect to the representation of one input,

    ∂L/∂h(x) = Wᵀ (softmax(z(x)) − T(x)) / N,

is a function of `h(x)` and `T(x)` alone. Two inputs `x₁, x₂` with `T(x₁) = T(x₂)`
receive gradients that differ only through `h(x₁) − h(x₂)`. The loss therefore
provides **no incentive to keep `h(x₁) ≠ h(x₂)`**: any function of the
equivalence class `[x]_T = {x' : T(x') = T(x)}` that fits the targets is a global
minimiser. This is a statement about incentives, not about dynamics; it says the
quotient representation `h(x) = f([x]_T)` is always among the minimisers, and with
one-hot inputs nothing in the input distinguishes the minimisers.

## 2. Hard targets: an active pressure toward the quotient

With hard (argmax-set) targets the classes are finite and, for a network that can
fit the data, the data are separable in the last hidden layer. Two facts then
combine.

(a) **Logit growth.** Cross-entropy on separable data is minimised only as the
logits diverge; gradient descent keeps increasing the scale of `W h` along the
directions that separate the classes (the implicit max-margin bias of Soudry et
al. 2018; the terminal phase of Papyan, Han & Donoho 2020). Between-class
distances in `h` grow throughout training.

(b) **Vanishing within-class gradient.** Once a point is correctly classified with
confidence, `softmax(z(x)) − T(x) → 0`, so `∂L/∂h(x) → 0`. The residual
within-class differences `h(x₁) − h(x₂)` stop being pushed by the loss and, with
no regulariser, simply stop changing. (With weight decay they would additionally
shrink; our experiments use none.)

Together these give the prediction tested in Task 7: the between-class distance
grows much faster than the within-class distance, so the ratio `within / between`
falls monotonically. The within-class distance need not stay at its initial
scale: in an unnormalised ReLU network the whole representation grows as the
logits grow (Task 7 measures a 13× increase within class against 28× between),
because same-class inputs also differ along the label-carrying directions. What
the argument predicts is a *relative* collapse: the network never actively merges
same-class points, it inflates the directions that carry the label and lets the
others ride along at a smaller rate. Deeper layers
inherit the same pressure through the chain rule, weighted by the downstream
Jacobian, which is why the collapse is strongest at the last layer.

Consequently, under hard targets, any distinction in `Q` (and hence in
occupancy) that does not change the argmax set is invisible to the loss and is
represented, if at all, only at initialisation scale.

## 3. Soft targets: the readout must reproduce the target logits

For a Boltzmann target `T_τ(x) = softmax(Q(x)/τ)` the cross-entropy is minimised
iff `softmax(z(x)) = T_τ(x)`, i.e. iff

    W h(x) + b = Q(x)/τ + c(x)·1

for some per-input constant `c(x)`. The readout is linear, so `h` must contain a
linear image of the centred vector `Q(x)/τ`. Two inputs with different centred
`Q` cannot be mapped to the same `h`; the required separation scales with
`‖Q(x₁) − Q(x₂)‖ / τ` (in logit units) and, through the softmax, with
`‖T_τ(x₁) − T_τ(x₂)‖` in probability units. Distinctions in `Q` that do not
change the argmax therefore *must* survive, with a strength set by τ. In the
limit τ → 0 the required logit differences within an argmax class diverge while
their effect on the probabilities vanishes, and the situation reverts to §2.

The same argument shows why the hidden geometry follows the **advantage** rather
than raw `Q`: the readout is invariant to the per-input constant `c(x)`, so only
`Q − max_a Q` (or any centring) is constrained. This is the pattern observed in
round 2 with Boltzmann targets.

## 4. Conditions

The quotient (or its soft counterpart) is favoured when:

1. the loss depends on `x` only through `T(x)` (any supervised objective);
2. the input carries no geometry of its own that the network would have to
   actively destroy (one-hot inputs; with coordinate inputs the input geometry
   persists, as in round 1);
3. training runs into the regime where the fitted points' gradients vanish
   (long training, no label noise), so between-class scale grows while
   within-class differences are frozen;
4. optionally, an explicit regulariser (weight decay, bottleneck width) that
   turns the frozen within-class differences into shrinking ones.

It is *not* favoured, or only weakly, when targets are graded (soft), when the
loss weights target differences non-uniformly (Task 13 tests whether the loss
matters beyond the target), or when the architecture normalises activations in a
way that fixes the between-class scale (LayerNorm in the residual and transformer
models, Task 9).

## 5. What would falsify the account

- Within/between ratio not falling under hard targets, or falling equally under
  soft targets (Task 7).
- Same-argmax probes staying separated under hard targets at a scale above
  initialisation (Task 3).
- Hidden geometry tracking `Q` rather than the centred advantage under soft
  targets (Tasks 1, 4).
- Target geometry failing to explain hidden geometry once environment quantities
  are controlled (Task 2).
