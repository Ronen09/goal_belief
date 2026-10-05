# Maze belief: a location belief in a fixed maze, and goal-dependent decisions

Put together from the conversation (three messages). In short:

* A fixed maze of 10–15 cells; the hidden state is the agent's cell; a named goal cell varies across episodes;
  observations are noisy local symbols that several cells share; actions N, S, E, W, deterministic, walls block;
  +1 on reaching the goal, discounted by delay. The model sees the goal and the history of observations and
  actions, never its location, the posterior or the simulator's tables. The map is fixed, so its dynamics can be
  learned; the location must be inferred.
* The task should contain decisions where different beliefs change the optimal action, and acting as though the
  most likely location were certain should sometimes lose substantial reward. Pure information-seeking detours are
  not required.
* The posterior b_{t+1}(s') ∝ P(o_{t+1} | s') Σ_s P(s' | s, a_t) b_t(s) is computed exactly for analysis and is not
  supplied in training. The network learns from reward.
* Central test: the same location belief under different goals. Does an internal belief representation transfer
  across goals while downstream quantities change with the destination? Patch the belief at an intermediate layer
  and test whether the decision moves to the counterfactual solver prediction. Reverse test: change the goal with
  the history fixed; the posterior should not move, the preferred route can.
* Before freezing: measure the effective dimension of the reachable posterior; match evidence by a shared
  observation–action prefix followed by the goal's revelation; treat the corridor-end goal as a control for action
  relevance, not predictive relevance; define each baseline separately (most likely state, averaged fully observed
  values, planning without future observations).
* Occupancy: distinguish the optimal solver's from the network's own. Whether occupancy mediates the computation
  is a separate hypothesis and is not part of this experiment's claim.
