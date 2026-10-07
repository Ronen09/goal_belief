# How the probability-space belief is formed

(Follows the reward-bandit experiment; its trained transformers, no training.)

The reward-bandit report found that the transformer's final state extrapolates the posterior's probabilities (0.85)
where the untrained network and the GRU extrapolate only the log-odds, which are affine in the token counts; the
post-hoc edit follow-up found the heads read nothing beyond the belief and that a probability encoder steers better
than a log-odds one.

The brief as sent: "we need to check how the probability space representation is actually formed".
