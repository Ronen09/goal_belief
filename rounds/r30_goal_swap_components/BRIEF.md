# Round 30: a goal swap with the history held fixed — which downstream components carry the switch?

(Follows round 29; round 23's reward models, frozen. No training.) The brief as sent:

I would make the next experiment a goal swap with the history held fixed, targeting the downstream computation:

1. Run the same history under two goals that require different actions.
2. Patch individual attention-head or MLP outputs between those runs, beginning after the two-route interface.
3. Identify components that transfer the donor-goal decision.
4. Test those components across other histories where the donor goal requires a different action.
