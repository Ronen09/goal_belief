# Round 13: the full Bayes filter state

(Follows round 12, `rounds/r12_hidden_goal/BRIEF.md`.)

Test the full Bayes filter state explicitly in the channel environment. Decode jointly

$$ \big(P(G\mid h_t), P(R_t\mid h_t)\big) $$

or whatever minimal sufficient statistic the generative model implies. Then perform matched interventions in both coordinates. You could ask whether two histories with the same full filter state—not merely the same goal posterior—become exactly future-equivalent.

If yes, then you have something quite deep:

The recurrent network has discovered a representation functionally equivalent to the environment's minimal predictive state, even when supervision exposes only a projection of that state.

That would move the project beyond "goal belief geometry" into something much broader about learned sufficient statistics and architectural bottlenecks.

And I think that is now the right direction. The new conceptual hierarchy is:

(The message ended here; the hierarchy was not included.)
