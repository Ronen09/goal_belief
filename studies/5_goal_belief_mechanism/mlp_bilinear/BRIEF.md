# MLP bilinear: how does block 0's MLP compute J(b, g)?

(Follows the self-value-interaction experiment; the observation-prediction experiment's reward models, frozen. No training.) The brief as sent:

A very natural next experiment is to ask how the MLP computes J(b,g).
For example, fit a local second-order model to the MLP input-output map:

    y ≈ A B + C G + Q(B,G),

where Q captures bilinear or multiplicative terms. If a low-rank bilinear interaction explains most of the goal×belief
component, then you'd have a concrete computational description:

    J(b,g) ≈ Σ_{k=1}^r (u_kᵀ B(b)) (v_kᵀ G(g)) w_k.
