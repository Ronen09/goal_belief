# Observation prediction: reward only against reward plus observation prediction

(Follows the pair-types experiment; same task, same architecture, same evaluation pairs. New training.) The brief as sent:

For the next training experiment, I would prioritize reward-only versus reward-plus-observation-prediction, keeping
the architecture and evaluation pairs fixed. Test whether the auxiliary objective reduces:

- behavioural differences between identical-posterior histories;
- incorrect action changes when evidence changes but the optimal action does not;
- regret across all three goals.

Keep the causal claim separate: stronger patch transfer is useful only alongside competent behaviour and controls
for conflicting evidence routes.
