# History mediator: what is the smallest history-derived variable that carries the history's effect on the policy?

(Follows the additive-ablation experiment; the observation-prediction experiment's reward models, frozen. No training.)

The brief as sent (two messages, condensed; the formulas are the sender's):

> I'd treat this as a mediation problem: what is the smallest history-derived variable Z(h) such that, once Z is
> fixed, changing the rest of the history no longer changes the policy? Compare candidates of increasing richness:
> the posterior b(h) in R^14, H(h) in R^4, the two-part H-mixture code, and compressed versions of H (its top action,
> top-two ranking, a 2–3D projection).
>
> Decompose the history-derived neural state as x(h) = x_Z(h) + r(h), x_Z the component predicted by Z, r everything
> else. Remove x_Z: does performance collapse to the history-blind ceiling? Remove only r: does it stay unchanged?
> Swap x_Z between histories: does the decision transfer? Swap r while holding x_Z fixed: does almost nothing happen?
>
> Success, two independent criteria:
>     swap effect(Z) / swap effect(all history) >= 0.9     and     swap effect(r | Z) ~ 0.
> Minimal: the smallest Z for which adding any other history-derived information does not improve causal control.
