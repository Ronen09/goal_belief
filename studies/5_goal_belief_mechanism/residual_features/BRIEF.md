# Residual features: which history features, beyond the posterior, carry the residual's effect?

(Follows the belief-error experiment; the observation-prediction experiment's reward models, frozen. No training.)

The belief-error experiment's proposed next step: decode history features from r outside the belief subspace (the
prefix actions, the last symbol, whether the landmark was seen), and test which one, swapped alone, carries its effect.

The brief as sent: "sure find out, but im just thinking those features would be absorbed by the posterior itself no?
thats exactly why it exists"

On that point: the posterior is sufficient for the optimal decision, so no feature should matter to the Bayes-optimal
policy once it is known. r is by construction the variation left once the posterior (and prefix length) is fixed. Any
feature it encodes varies among histories with the same posterior, so it is information the posterior has made
redundant. The question is therefore which superseded features the network still carries and acts on. Removing r costs
nothing (history-mediator experiment), consistent with that. 10 203 distinct prefix sequences reach 619 posteriors; 66 %
of histories share their (posterior, length) with other sequences.
