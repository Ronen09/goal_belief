# Round 24: measures, criterion and expectation

Written before the head's outputs on these pairs were computed. Code: `rounds/r24_head_consistency/run.py`.

**Models**: round 23's twenty models with a trained head (coefficient 1 and 0.1, seeds 0–9), final checkpoint.
**Pairs**: round 22's, unchanged. Type A: the same node of the belief graph (identical posterior), different tokens,
the same prefix length. Type C (posteriors ≥ 0.25 apart, different optimal actions under every goal) gives the scale.

**Where the head is read, and what is matched.** The head gives a distribution over the next symbol for each of the
four candidate moves.

* *Goal token* (primary): the two histories with the same goal, for each of G1, G2, G3. Matched: posterior, goal,
  candidate move, prefix length and token position, 12 moves left. The exact predictive distribution (conditional on
  the episode continuing) is then identical for the two histories. Moves that enter the goal with certainty have no
  next symbol and are left out.
* *Last prefix token* (goal-free): prefix lengths 2 and 3 only. Attention is causal, so this token's output is what
  it would be inside a longer prefix, where the head is trained. At length 4 the head has never had a target there;
  it is reported apart.

**Measures**, per (pair, goal, candidate move), in total variation over the three symbols, averaged:

* **D_same**: between the head's predictions for the two histories of a type A pair. The exact value is 0.
* **error**: between the head's prediction and the exact one, per history.
* **D_diff**: D for type C pairs, and the exact predictive's own D there.
* **policy**: TV between the two action distributions for the same cells (round 23's O1 for type A), and for type C.
* **relative inconsistency**: D_same / D_diff for the head, and the same ratio for the policy. The two outputs have
  different scales; the ratio puts each against its own response to a real change of posterior.
* Within type A cells: D_same where the policy's greedy action differs between the two histories and where it
  does not.

**Criterion**, fixed here, on the medians over seeds at coefficient 1, goal token:

| result | reading |
|---|---|
| the head's relative inconsistency is at least half the policy's | the auxiliary task is itself solved with history-dependent approximations |
| it is below a quarter of the policy's, and D_same is no larger than the head's error | the head is belief-consistent to within its accuracy; the consistency does not extend to the policy |
| between | not decided |

**Expectation.** The head is within 0.001 nats of exact on average, so its error in total variation should be about
0.01–0.02 and D_same no larger; the policy's relative inconsistency was about 0.08. I expect the second row. I also
expect D_same to be no larger in the cells where the greedy action differs than where it does not.
