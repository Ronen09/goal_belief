# Round 19: is goal-conditioned occupancy represented, beyond the posterior and the action values?

(Follows round 18; same task, same trained models, no new training.) The brief as sent:

Occupancy could explain that middle ground: a prediction of where the agent will go under a goal-conditioned
policy, rather than just where it currently believes it is. Keep the existing task and trained models, and make
this a focused comparison:

| candidate representation | what it describes |
|---|---|
| location posterior b(s) | where am I now? |
| goal-conditioned occupancy d_g(s) | which states will I visit while pursuing this goal? |
| action values Q(b, g, a) | how useful is each available action? |

Including action values matters: otherwise an occupancy probe might simply recover information associated with
the imminent decision, and that could be mistaken for a representation of future visitation.

Define occupancy carefully first, for example
d_g^π(s | b) = E_{π,g,b}[ Σ_{k=1..H} γ^{k−1} 1{S_{t+k} = s} ]:
expected future visitation from the current belief under a specified policy. Starting at k = 1 keeps the
current-state posterior from being included directly. Distinguish occupancy under the solver's policy from
occupancy under the model's policy, especially for the two seeds that failed G2.

Three tests, in order of priority:

1. Where does each quantity become accessible? Compare posterior, occupancy and action-value decoding across
   prefix and reveal layers, using held-out evidence–goal combinations.
2. Does occupancy explain more than the current decision? Test cases with the same optimal first action but
   different future visitation. Successful discrimination is stronger evidence than aggregate probe scores.
3. Does occupancy have a selective causal effect? If decoding looks convincing, intervene on an
   occupancy-associated subspace and test predicted behavioural changes, with posterior and action-value controls.

Caveat: occupancy is derived from belief, goal, dynamics and policy. Finding it decodable does not establish that
the model first computes a Bayesian belief and then transforms it into occupancy.
