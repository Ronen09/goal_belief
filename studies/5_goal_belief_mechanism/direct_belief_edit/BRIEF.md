# Direct belief edit: a shared belief edit at the goal token's state after block 0

(Follows the cross-history-MLP experiment; the observation-prediction experiment's reward models, frozen. No training.) The brief as sent:

The target is the goal token's state after block 0, entering block 1. The goal-route-selection experiment identified this as the route carrying
most of the history-dependent behavioural difference.

1. Fit a candidate belief encoding without action labels.
On separate fitting histories, collect that state under all three goals. Start with:

    z(h,g) ≈ c_g + E b(h).

Here c_g captures a goal-specific offset, while the same E maps location belief into activations across goals.
Check held-out fit before interpreting edits. If this shared linear encoding fits poorly, that limits this particular
test; it does not establish that belief is absent.

2. Change the belief-associated component while keeping the recipient goal.
For recipient history A and donor belief B:

    z' = z(A,g) + E (b_B − b_A).

Keep the recipient's prefix states unchanged and continue the original network forward. The edit uses the donor's
posterior, not its action or full activation.
Apply the same edit vector under all three goals. Compare the output with both:
- The model naturally run on history B under the recipient goal.
- The solver's optimal actions for belief B and that goal.
The first measures faithful transfer of the model's computation; the second measures correctness.

3. Make the decisive cases goal-dependent.
Select pairs using the solver, before examining patch outcomes:

| Pair type | Required behaviour |
|---|---|
| Belief change requires left under G1 and right under G2 | The same edit produces different, appropriate actions |
| Belief changes but the optimal action stays unchanged | The edit preserves correct decisions |
| One-step observation predictions agree, but optimal actions differ | The edit transfers information beyond immediate prediction |

Repeat each belief change using multiple histories with the same recipient posterior. That tests whether the edit's
effect depends primarily on belief or on the particular token sequence.

4. Include controls that address your previous failures.
- Whole direct-route replacement, same goal: how much effect is available through this route?
- Rotated edits with matched norms: is the effect direction-specific?
- Equal-rank PCA patches: does generic evidence replacement work equally well?
- A goal-specific encoding as a secondary comparison: does success require fitting a different map for every goal?

Report absolute donor-optimal action rates, preservation harm, and stable aggregate transfer—not just per-cell ratios
with tiny denominators.
The strongest result would be a shared belief edit that transfers donor behaviour across goals and equivalent
histories, with little preservation harm and clear advantages over generic controls. That would support the native
policy using a manipulable belief-like variable at its dominant interface.
