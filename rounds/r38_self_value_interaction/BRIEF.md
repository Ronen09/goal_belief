# Round 38: does the self-value swap move the goal × belief interaction at the MLP input?

(Follows round 37; round 23's reward models, frozen. No training.) The brief as sent:

For example, define the interaction component at the MLP input:

    I(b,g) = f(b,g) − f(b).

Then swap only the goal self-value and ask whether I(b,g₁) moves toward I(b,g₂), while the shared belief component
f(b) remains approximately fixed.
