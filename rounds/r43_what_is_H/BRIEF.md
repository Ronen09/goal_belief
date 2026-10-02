# Round 43: what is the history profile H?

(Follows round 42; round 23's reward models, frozen. No training.) The brief as sent:

I'd test candidate interpretations directly. For each posterior b, compute quantities like:

    max_g Q*(b,g,a),
    (1/|G|) Σ_g Q*(b,g,a),

probability that a is optimal under a randomly chosen goal, or perhaps reachability/occupancy-like quantities
independent of the current goal. Then ask which one linearly/nonlinearly predicts H best and, crucially, which one
supports causal edits.
If H turns out to approximate something like

    H_a(b) ≈ max_g Q*_g(b,a),

then the model's learned algorithm would be extremely interpretable: infer state uncertainty → score actions by general
usefulness across possible goals → add a goal-specific preference → choose.

That would also explain why the model performs very well without implementing the exact Bayes-optimal mapping Q(b,g,a),
and why it systematically fails on cases requiring genuinely non-additive goal–belief reasoning.
