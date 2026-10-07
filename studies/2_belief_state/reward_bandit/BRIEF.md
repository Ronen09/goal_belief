# Reward-trained hidden goal: does reward alone produce a Bayesian belief?

(A new task in the belief-state study; follows the hidden-goal experiment, whose models were trained with supervised
objectives, and the reward-trained agents of studies 3 and 7.)

The brief as sent, pasted:

> At the start of each episode, sample a hidden goal g ~ p(g) from K possible goals/reward functions. Each g defines
> a different mapping from actions to reward. The model never receives a goal token. Instead, during the episode it
> sees evidence correlated with g: noisy cue tokens, contextual observations, and possibly rewards from previous
> actions. So after history h_t, the normatively relevant latent state is b_t(g) = P(g | h_t). The agent is trained
> only to maximize reward. It is not supervised on g, b_t, or Bayes-optimal actions. That makes the central question:
> does reward optimization cause the model to spontaneously infer and represent a Bayesian belief over the hidden goal
> because that belief is useful for choosing rewarding actions?
>
> A minimal version could have 3 hidden goals and 4 actions. Each hidden goal has a different reward vector, e.g.
> r_g1 = (1, .6, .1, 0), r_g2 = (.1, 1, .4, 0), r_g3 = (0, .2, .7, 1). The model sees a few noisy cues generated
> according to P(o | g), then chooses an action and gets reward according to r_g(a). With several decisions per
> episode, previous rewards can themselves become evidence about g. The exact Bayesian solution is available offline
> for analysis: b_t(g) = P(g | h_t), and the Bayes-optimal expected value of action a is Q(b_t, a) = sum_g b_t(g) r_g(a).

Then: "let training run, lets do this next".
