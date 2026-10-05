# Belief-encoding edit: can the transferable representation support a selective causal belief edit?

(Follows the predictive-transfer experiment; its backbones and goal-conditioned heads, frozen. No training.) The brief as sent:

For the next experiment, I would freeze training and test whether the transferable representation can support a
selective causal belief edit. The predictive-transfer experiment finally gives you a promising setting for that test.

Fit a simple encoding model on separate data:

h ≈ c + E b,

then intervene on held-out pairs using

h' = h_A + E(b_B − b_A).

Unlike your earlier decoder-direction edit, this attempts to change activations according to how the representation
naturally varies with belief. It is still only a candidate intervention, so compare it with whole-state replacement
and matched generic subspace edits.

Feed the edited state to the already trained, frozen goal-conditioned head, under each goal. Ask whether it produces
the donor-belief action when required, preserves the recipient action when appropriate, and works across multiple
histories sharing the same belief. Include cases where one-step predictions agree but optimal actions differ.

A successful result would establish causal use of belief-associated information by the transferred controller—not
yet by the original RL policy. That is a meaningful bridge toward your main question: you would have identified
information learned without goals that is subsequently combined with a goal to select an action.
