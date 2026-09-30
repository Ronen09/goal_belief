# Round 25: selected-action against balanced candidate-action supervision of the prediction head

(Follows rounds 23 and 24; same task, architecture and evaluation pairs. New training.) The brief as sent:

Compare the current selected-action supervision with balanced candidate-action supervision, keeping the total
auxiliary weight and training budget comparable. Your simulator can supply counterfactual next observations without
supplying posterior labels.

Register two main questions:

1. Does balanced supervision reduce held-out prediction error and identical-posterior inconsistency?
2. If it does, does policy inconsistency also decrease?

If prediction becomes more consistent while the policy does not, you finally obtain the dissociation we were looking
for. If both improve, that supports a connection, with causal mediation still to establish.
