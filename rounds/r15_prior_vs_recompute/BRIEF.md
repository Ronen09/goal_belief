# Round 15: prior or recomputation in a next-token transformer

(Follows rounds 12–14.) The brief as sent:

Train a normal full-attention transformer on next-token prediction in the hidden-channel environment. No artificial carry.

At position t, decode the model's belief b_t. Then append the same next observation x_{t+1} under several interventions. The critical one is b_t^A → b_t^B while leaving all raw tokens x_{1:t} unchanged. Don't just edit the residual at t, because that representation only reaches the next position through its cached keys/values. Edit the belief-carrying component of the K/V information exported by position t, ideally across the layers where belief is represented. Then ask whether the new representation at t+1 moves toward F(b_t^B, x_{t+1}), the exact Bayesian update obtained by treating the edited belief as the prior.

The converse intervention: preserve b_t at the immediately preceding position but corrupt/remove the older evidence that attention can reread. A 2×2:

| Previous belief | Earlier evidence | What it tells you |
|---|---|---|
| genuine | genuine | baseline |
| edited | genuine | does the represented belief act as a prior? |
| genuine | edited/masked | how much can it propagate recursively without rereading? |
| edited | edited consistently | positive-control counterfactual |

Decode the resulting belief at t+1 and compare it against exact Bayesian counterfactuals: B⁺ = BayesUpdate(B, x) means b_t acts as a prior; A⁺ = BayesUpdate(A, x) means the model recomputed from the prefix; b_{t+1} ≈ λB⁺ + (1−λ)A⁺ means redundant/hybrid inference.

Contradictory-information version: keep history A's raw tokens, transplant position t's exported representation/KVs from B, and see what t+1 does — the relative weight of the previous inferred state vs recomputation from evidence. Sweep context length. Hypothesis (the brief's): the recursive contribution may become stronger when recomputation is expensive or difficult, and weaker when raw evidence is short/easy to aggregate.

The question: when a next-token transformer has already inferred a predictive belief at position t, does position t+1 treat that representation as a prior, recompute the belief from the token history, or combine both?
