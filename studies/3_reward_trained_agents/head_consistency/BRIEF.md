# Head consistency: does the prediction head agree on identical-posterior histories?

(A diagnostic on the observation-prediction experiment's models; no training.) The brief as sent:

There is one useful diagnostic before further training: does the auxiliary prediction head itself give the same
predictions for identical-posterior histories? Match the candidate action, horizon, and other relevant variables.

- If its predictions differ substantially, the auxiliary task itself is being solved with history-dependent
  approximations.
- If its predictions agree while actions differ, the model can produce belief-consistent observation predictions,
  but that consistency does not extend to its policy.
