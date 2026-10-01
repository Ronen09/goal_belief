# Round 29: does the goal change which evidence route the policy relies on?

(Follows round 28; round 23's reward models, frozen. No training.) The brief as sent:

The one-step-agreeing pairs are the most interesting clue for your goal question. Whole-interface transfer rises from
0.21 overall to 0.48 there, and the belief edit follows it to 0.40. This suggests the policy may rely on different
evidence routes depending on what the decision requires. But agreement of one-step predictions does not itself mean
recent symbols are uninformative, so keep that explanation provisional.

I would now investigate that route selection, without further training or trying to increase transfer. Hold each
history pair fixed and compare all three goals:

Does the policy become more sensitive to the pre-goal interface specifically when the goal requires distinctions that
the alternative route handles poorly?

Use the existing whole-interface and encoding edits, alongside targeted interventions on the direct route. Evaluate
goal-by-goal, controlling for the unedited donor–recipient behavioural difference and oracle action gap, so transfer
normalization and decision difficulty do not create the pattern.

If route dependence changes appropriately with the goal, you would have a concrete answer about goal understanding:
the goal influences which evidence contributes to the decision. If it stays similar across goals, the next question
becomes how a shared evidence estimate is transformed into different action preferences.
