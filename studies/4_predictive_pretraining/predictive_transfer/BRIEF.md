# Predictive transfer: does learning to predict the maze produce a representation from which different goals can be solved efficiently?

(Follows the observation-prediction to balanced-prediction experiments; same maze, observation noise, backbone and context length. New training.) The brief as sent:

Keep the maze, observation noise, backbone and context length fixed. Use three conditions:

| Condition | Training |
|---|---|
| Prediction-only | Predict the next symbol after a supplied move, on goal-free exploratory trajectories |
| Reward-only | Your existing goal-directed RL baseline |
| Random backbone | No backbone training; a control for information already available at initialization |

First, compare their representations on exactly the same held-out histories. Measure posterior decodability and
identical-posterior consistency. For the predictor, also measure prediction error across all candidate moves. This
establishes what prediction training actually learned before interpreting transfer.

Then freeze each backbone and train the same small goal-conditioned policy head. Give it the representation at a
fixed history-summary position plus the goal, with no separate access to raw history. Use identical optimal-action
training examples and budgets across conditions, and evaluate on held-out histories.

Using supervised actions here makes the comparison easier to interpret: you are measuring how accessible the
information needed for decisions is, without introducing differences in RL exploration. Keep the head small so it
cannot readily learn the entire inference problem itself.

The central question becomes:

Does learning to predict the maze produce a representation from which different goals can be solved efficiently?

Measure decision regret and the number of action-labelled examples needed. A small learning curve is more
informative than one final accuracy.

Possible outcomes:

- The prediction-trained representation supports accurate decisions with little training: predictive learning
  creates information reusable for control.
- Belief decodes well, but the policy head struggles: recoverable belief information is not necessarily easy to use
  for goal-dependent decisions.
- The reward-trained representation transfers better: control training makes decision-relevant information more
  accessible.
- The random backbone performs similarly: the task or policy head may be doing too much of the work to establish a
  learned representation advantage.

One precaution: your one-step observation objective may not distinguish every relevant posterior. Check whether
different beliefs can predict the same observations for all four moves yet require different optimal actions. If so,
a predictor's failure on those cases would not show that prediction training failed to learn what its objective
required.

I'd make transfer to the frozen, goal-conditioned head the primary result, with probes as supporting evidence. This
tests a concrete link between learning about the world and acting toward goals, without spending another round
trying to eliminate small consistency errors.

**Added in conversation, before the plan was written.** The precaution check (`checks.md`) found that one-step
predictions leave the optimal action undetermined in most decisions and two-step predictions in few. Agreed: a
two-step predictor is added as a fourth, secondary backbone; the brief's three conditions remain the primary
comparison.
