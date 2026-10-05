Objective

Investigate whether neural agents trained for goal-directed behaviour learn internal representations that reflect the geometry of goal-conditioned future occupancy.

The core hypothesis is:

If two states afford similar future access to possible goals, their internal representations should become similar, even when they are spatially or symbolically different.

The aim is to derive a theoretically defined occupancy representation from a small MDP and test whether this geometry emerges in a neural network trained only to act optimally.

Phase 1: Construct a Controlled MDP

Create a small deterministic gridworld, initially around (8\times8).

The environment should contain:

four movement actions,
walls or obstacles,
approximately 10–20 possible goal states,
a goal supplied as part of the agent's input,
reward (1) for reaching the current goal and (0) otherwise.

Keep the transition dynamics fully known so that optimal behaviour and occupancy can be computed exactly.

Phase 2: Compute Optimal Goal-Directed Behaviour

For every possible goal (g), compute the optimal policy

[
\pi_g^*(a|s)
]

using value iteration or dynamic programming.

Generate optimal trajectories and state-action labels from these policies.

Phase 3: Compute Ground-Truth Goal Occupancy

For each state-action pair and possible goal, compute the discounted future occupancy

(1-\gamma)
\sum_{t=0}^{\infty}
\gamma^t
P(s_t=g|s_0=s,a_0=a,\pi_g^*).
]

Construct an occupancy vector for each state-action pair:

[\rho(g_1|s,a),\dots,\rho(g_k|s,a)].
]

Treat (z(s,a)) as the theoretically predicted representation of the future possibilities available from that state-action pair.

Characterize its geometry using:

pairwise distances,
cosine similarity,
PCA/SVD,
effective rank,
clustering.
Phase 4: Train a Neural Agent

Train a small neural network to imitate the optimal policies.

Input:

[
(s,g)
]

Output:

[
a^*.
]

Use behavioural cloning initially rather than reinforcement learning so that optimization instability does not confound the representation analysis.

A simple model is sufficient:

state embedding,
goal embedding,
3–4 layer MLP,
hidden width around 128,
action classification output.

Do not train the network to predict occupancy directly.

Phase 5: Compare Learned and Predicted Geometry

Extract hidden activations

[
h_l(s,g)
]

from every layer.

Test whether activation geometry corresponds to occupancy geometry.

Primary analyses:

Compare pairwise activation distances with pairwise occupancy distances.
Measure representational-similarity correlations.
Train linear maps from activations to occupancy vectors.
Compare the dimensionality and principal components of learned and theoretical representations.
Measure how the relationship changes across layers.

Do not treat linear decodability alone as evidence that the network genuinely uses occupancy.

Phase 6: Create Discriminating Environments

Construct environments where competing representations make different predictions.

For example:

states that are spatially distant but have almost identical future reachability;
adjacent states separated by one-way transitions or barriers;
duplicated maze regions with identical transition structure but different absolute coordinates.

Compare two hypotheses:

[
H_{\text{spatial}}:
d(h_i,h_j)\propto d_{\text{physical}}(s_i,s_j)
]

versus

[
H_{\text{occupancy}}:
d(h_i,h_j)\propto d_{\text{occupancy}}(s_i,s_j).
]

The key test is whether neural representations follow occupancy geometry when it conflicts with simple spatial geometry.

Phase 7: Test Goal Dependence

Hold the environment fixed while changing the requested goal.

Investigate whether the representation decomposes into components resembling

h_{\text{world}}(s)
+
h_{\text{goal}}(g)
+
h_{\text{interaction}}(s,g).
]

Test whether early representations remain relatively goal-independent while later representations encode goal-specific action relevance or future occupancy.

Phase 8: Causal Intervention

Estimate activation differences associated with changing goals:

E_s[h(s,g_2)-h(s,g_1)].
]

Intervene on hidden activations:

h(s,g_1)+\alpha v_{g_1\rightarrow g_2}.
]

Test whether the agent's behaviour shifts toward actions appropriate for (g_2).

Compare the behavioural change against the predictions made by the goal-occupancy model.

Phase 9: Extensions

If the deterministic experiment succeeds:

Introduce stochastic transitions.
Test whether representations encode probabilistic rather than shortest-path reachability.
Move to partially observable environments.
Compute exact belief states

[
b_t(s)=P(s_t=s|h_t)
]

and investigate whether belief-state representations and goal-occupancy representations factorize.
5. Eventually replace the MLP with a recurrent network or transformer.

Main Research Question

The project should ultimately answer:

To what extent can the internal geometry of a neural agent be predicted from the goal-conditioned occupancy structure of the decision problem it is trained to solve?

A stronger result would show not only that occupancy is decodable, but that:

occupancy geometry predicts representational similarity,
it wins against plausible competing geometries,
its structure emerges systematically during training,
and manipulating the corresponding representation causally changes goal-directed behaviour.
