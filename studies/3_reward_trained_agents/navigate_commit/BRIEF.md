# Navigate–commit: navigate, investigate, commit

The brief as sent:

## Research question

How does a full-attention transformer learn to construct and use a belief over a hidden reward location as its decision policy improves?

Separate three possible outcomes: improved inference from evidence; improved use of an already available belief; and learning a small set of action-relevant distinctions without a reusable posterior representation. Optimal behavior alone does not distinguish these outcomes.

The experiment studies ordinary full causal attention. Do not introduce carry connections, attention gates, or a forced memory bottleneck. Recomputing a belief from the history is an admissible mechanism to investigate.

## Task: navigate, investigate, commit

Use an empty 5×5 grid. One of the four corners contains a hidden reward, sampled uniformly each episode. The agent observes its position, remaining horizon, station locations, station reliabilities, and whether each station has been used. It never observes the hidden reward location before termination.

Two distinct non-corner cells contain clue stations. The horizontal station reports left versus right; the vertical station reports top versus bottom. A report matches the true coordinate with probability q, independently across stations conditional on the reward location. Each station can be queried once, using an explicit QUERY action while occupying its cell. Revisiting it supplies no new evidence.

Actions are UP, DOWN, LEFT, RIGHT, QUERY, and COMMIT. COMMIT is legal only at a corner and terminates the episode, awarding 1 if correct and 0 otherwise. Merely visiting a corner reveals nothing. Each move or query costs c; committing has no additional cost. Mask illegal actions, including moves outside the grid and unavailable queries. A horizon of H actions ends the episode with zero terminal reward if the agent has not committed; accumulated costs remain.

Pilot defaults: H=20, c=0.02, q=0.8. Sample starts and station placements independently of the hidden goal. All layout information is public. Tune the task with the exact solver before model training, then freeze the configuration. Require meaningful cases in which zero, one, and two queries are optimal, and cases in which a clue changes the preferred route. If these regimes do not occur, change costs, horizon, or station placement distribution before proceeding.

## Exact belief and policy benchmark

For goal g and history h, compute b(g)=P(g|h). Movement and the agent's own actions supply no additional goal evidence in this environment. On receiving report y at station j:

    b'(g) = P(y | g, j, q_j) b(g) / Σ_g' P(y | g', j, q_j) b(g')

Solve the finite-horizon belief-state problem by backward induction. The state is (position, belief, used-station flags, remaining actions, public layout and reliabilities). The value of committing at corner g is b(g); moving or querying has value −c plus the expected next-state value. Terminal value at zero remaining actions is zero. Queries branch over their two possible reports. Retain all tied optimal actions and exact action values, not just one arbitrarily chosen label.

Check posterior normalization and query probabilities, and compare the solver to exhaustive policy enumeration on a smaller grid/horizon. Use expected return regret as the main behavior metric. Also report local decision regret Q*(s,a*)−Q*(s,a), distinguishing it from episode-level regret, and separate query, route, and commitment errors.

## Model and training

Use a small causal transformer: initially four layers, width 128, four heads. Encode public episode configuration and the full sequence of observation/action records with a consistent field format. Predict the next action at a designated decision token. Include all past reports; never expose the exact posterior, hidden goal, or solver values as model inputs.

Primary training uses episodic reward optimization, with a standard on-policy actor–critic method such as PPO. Supply no auxiliary posterior loss or expert action labels in the primary condition. Fix the training budget after an exploratory pilot; run five independent seeds for the main study. Save initialization and approximately 20 log-spaced checkpoints, with denser checkpoints around any performance transition identified in the pilot.

Use two interpretive controls if the primary model learns successfully:

- Frozen initialized backbone, trained policy/value heads: tests how far readout learning on initial features can go. Failure is not proof that the primary model learns Bayesian inference.
- Solver-supervised policy, same transformer: tests whether the architecture and task encoding support good decisions when exploration is removed. This is a diagnostic control, not evidence about reward-driven learning.

If reward training fails but supervision succeeds, diagnose optimization/exploration before making claims about belief formation. Record return, update count, and environment interactions separately.

## Fixed evaluation histories

Create an independent evaluation bank before main training, using legal exploratory trajectories with diverse clue outcomes, orders, detours, and positions. Replay the identical histories at every checkpoint, regardless of what that checkpoint's policy would have visited. Label each decision with its exact belief and optimal action values. Report on-policy performance separately.

Split probe fitting and evaluation by whole histories and held-out combinations of layouts, starts, and reports. Do not randomly split tokens from the same trajectories across train and test. Include matched cases with:

- Same position, time, layout, and station-use flags, but different reports and beliefs.
- Same evidence and belief, but different current positions and optimal moves.
- Same immediate optimal action, but different beliefs.
- Same final belief reached through different clue orders or irrelevant movement histories, matched on public decision state where feasible.

The first comparison tests evidence sensitivity; the second separates belief from action; the third challenges action-label explanations; the fourth tests history-specific encoding.

## Trace the computation across layers and training

At each checkpoint collect activations before and after attention and MLP sublayers at decision positions. Fit held-out affine decoders for the four goal probabilities; report error and invalid probability outputs rather than silently projecting all outputs into the simplex. Add a small nonlinear decoder as a secondary diagnostic, with matched fitting budgets. Include initialization and raw-input-feature baselines.

Because independent left/right clues and a uniform prior produce a factorized posterior, also decode left/right and top/bottom marginals. This task does not require a generic three-dimensional belief simplex: its reachable beliefs have additional structure. Report that structure explicitly.

Inspect which token contributions construct the decoded belief. For each head, analyze attention weight times its value/output contribution, not attention maps alone. Test whether a horizontal report changes horizontal belief contrast while preserving the vertical marginal, and vice versa. Compare pre-attention, post-attention, and post-MLP errors, allowing intermediate representations to implement partial computations rather than complete posteriors.

Use one shared affine decoder across a block's input and output when interpreting residual additions as belief-space displacements. Independently fitted decoders at different layers do not define comparable geometric arrows. Across checkpoints, emphasize predictive and causal function; raw directions can rotate or rescale during training.

## Causal tests

Select mechanisms on discovery histories, then evaluate them on held-out matched histories. Patch a head output or MLP contribution from a donor with different clue evidence but matched public decision state. Predict the direction of the belief and action-value change from the exact counterfactual report. Measure both the change in decoded belief and the action distribution's movement toward the counterfactual optimal choices.

Include same-belief donor patches and norm-matched random perturbations. Compare candidate component effects to a broader patch at the same layer. A null single-head effect may reflect distributed computation or compensation, rather than absence of belief use. Refit decoders and identify corresponding functional components within each checkpoint; do not assume that a head index retains the same role throughout training.

Whole-vector patches can import action information. Stronger evidence comes from interventions that behave appropriately across different recipient positions, where the same belief change requires different actions. No individual patch establishes the complete inference algorithm.

## Challenge lookup-table explanations

The fixed-q, two-clue task has only nine possible posterior patterns. Use it as the first mechanistic pilot, not as evidence of a general Bayesian algorithm.

For the main generalization test, vary each station's publicly observed reliability over a finite grid, for example 0.60–0.95 in steps of 0.05. Hold out some reliability values and combinations from policy training, and evaluate belief accuracy and policy regret on those cases. Reliabilities remain fixed within an episode and are supplied numerically in the same input format. Solve each evaluation configuration exactly.

Success would challenge memorization of the nine fixed-q posteriors, but would still not prove general inference. A later extension could use nonfactorizing clue likelihoods to test whether the mechanism extends beyond two independent binary beliefs. Keep that extension out of the initial claim.

## Interpretation and deliverables

| Observation | Supported interpretation |
|---|---|
| Belief decoding and causal evidence integration improve with regret | Policy training develops an inference computation used for decisions |
| Belief is recoverable early, but decision effects develop later | Learning increasingly uses available information; test whether upstream computation also changes |
| Action-relevant contrasts improve while full posterior recovery stays poor | Consistent with a policy-specific representation, subject to decoder limitations |
| Changes occur only on on-policy histories | Visitation changes remain an alternative explanation |
| Fixed-q success fails under held-out reliability | The simple task may be solved by discrete shortcuts |

Deliver the frozen environment specification and validated solver; per-seed learning curves; layer-by-checkpoint belief errors on fixed histories; two or three worked computational traces; held-out causal intervention results; and reliability-generalization results. Show per-seed values and uncertainty across seeds. Register concrete thresholds after the pilot and before the main runs.

The target claim is that identifiable changes in evidence processing or belief use accompany and help explain policy improvement in this task. Do not infer that optimal agents must have identical internal beliefs, that decoding establishes causal use, or that the learned computation must match a particular Bayesian implementation.
