# Round 20: does the policy use the decoded belief?

(Follows rounds 18 and 19; same task, same trained models, no new training.) The brief as sent:

Run one focused causal round on whether the policy uses the decoded belief. The question: if you change the
represented belief while keeping the recipient's goal, do decisions change as predicted for that new belief?
Use the three successful reward-trained seeds first, with the supervised model as a comparison. The failed seeds
can come afterward.

1. **Choose matched donor–recipient cases.** Different evidence, initially the same goal, where the solver predicts
   different action preferences. Match prefix length and intervention position. Include cases where uncertainty
   matters rather than only the most likely location.
2. **Change the belief-associated component of prefix states.** At the goal-free prefix interface, use the posterior
   decoder to construct a donor-directed intervention while preserving the recipient's component outside the
   decoder's row space. Compare it with a whole-prefix-state patch. Treat this as an intervention on a
   probe-defined subspace, not automatically a clean belief edit.
3. **Test the same edit under different recipient goals.** A reusable belief interface predicts different,
   appropriate action changes for different goals, all consistent with the same donor belief. This is stronger
   than merely making the model take the donor's action.
4. **Check intervention validity and alternative routes.** Verify that the decoded posterior actually moves toward
   the donor's. Include matched-norm random edits and donors with similar posteriors but different histories.
   Inspect whether later access to untouched raw evidence restores the recipient's belief; otherwise a null effect
   could simply reflect recomputation.

| result | what it supports |
|---|---|
| belief-directed edits cause appropriate changes across goals | a causally used, reusable belief-associated representation |
| whole-state patches work, but belief-directed edits do not | evidence is causally useful; the decoded belief subspace is not established as its interface |
| edits initially work, then their effect disappears | possible recomputation or correction through another route |
| edits change decoding but produce arbitrary behaviour | the intervention may disrupt the representation; no clean mechanistic conclusion |

Start with same-goal interventions, then test cross-goal reuse. The previous cross-goal patches mixed two
questions: whether a route carries useful evidence and whether that evidence transfers independently of the goal.
