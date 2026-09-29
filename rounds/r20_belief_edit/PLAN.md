# Round 20: interventions, measures and expectations

Written before any edit was run. Code: `goalgeo/mazeedit.py`, `rounds/r20_belief_edit/run.py`.

## The interface

The residual stream at the prefix tokens (the first symbol and the imposed moves), entering block l, l = 1 or 2.
These tokens precede the goal token, so their states cannot depend on the goal. Round 18: replacing them whole at
l = 1 moved the decision 0.27 of the way (same goal); at l = 2, 0.04. The decision token also reads the raw prefix
tokens through its own first attention layer; that route is untouched by any edit at the interface.

## Pairs

Recipient (evidence A) and donor (evidence B) from round 18's held-out histories, the same prefix length (≥ 2), so
every position matches. Goal g the same for both. The exact optimal action sets for (A, g) and (B, g) are disjoint.
* **mode**: the most likely cell differs between A and B and acting on it as if certain already gives different actions;
* **uncertainty**: acting on the most likely cell as if certain gives the *same* action for A and B; only the rest
  of the posterior separates them.
* **similar**: donors whose posterior is within 0.1 (L1) of the recipient's, from different token histories (control).
Reference for every edit: the model on the **hybrid** sequence (B's prefix, recipient's goal).

## Edits at the interface (every prefix token, each from the donor's state at the same position)

Posterior decoder: ridge, from the prefix-token state entering block l to the exact goal-free posterior after that
token, fitted on the fitting histories, all positions pooled. W [d, 14]; P = projector on the column space of W
(rank 13: the decoded probabilities sum to one).

| edit | new state |
|---|---|
| whole | x_B |
| belief | x_A + P (x_B − x_A): the donor's component inside the decoder's row space, the recipient's outside |
| belief, rank 7 | the same with the 7 leading directions of the decoded posterior (its effective dimension is 6.7) |
| complement | x_B − P (x_B − x_A): everything but the row space |
| random subspace | the same with 13 random orthonormal directions |
| random, same norm | x_A + a random vector of the norm of P (x_B − x_A) |

## Measures

* **validity**: the posterior decoded at the interface after the edit (by construction the donor's for `belief`);
  the posterior decoded at the goal token's output, as a share of its displacement under the hybrid; the share of
  ‖x_B − x_A‖² inside the row space.
* **behaviour at the reveal**: moved-to-hybrid = 1 − TV(edited, hybrid) / TV(base, hybrid) on pairs where the model
  itself responds to the evidence (TV(base, hybrid) > 0.3); and **solver gain**: the gain in probability of the
  actions that are optimal for (B, g) and not for (A, g), as a share of the hybrid's gain.
* **arbitrary behaviour**: probability moved onto actions that are optimal for neither A nor B.
* **raw route**: the same edits with the decision token's first attention output replaced by its mean for that goal
  (its evidence-specific part removed), references recomputed under the same ablation.
* **persistence**: the edit is applied at every forward pass; the measures are repeated at the decisions 1, 2 and 4
  moves after the reveal, on the recipient's own continuation.
* **cross-goal reuse**: the same edited prefix states, recipient's goal set to each of G1, G2, G3. Among pairs where
  the hybrid's greedy action differs between two goals and both differ from the base's: how often the edited
  model's greedy action is the hybrid's under both goals.

Models: reward-trained seeds 1, 3, 4 and the supervised model; then seeds 0 and 2.

## Expectations

| | expectation |
|---|---|
| E1 | whole, l = 1: moved-to-hybrid 0.2–0.4 (round 18: 0.27); l = 2: < 0.1 |
| E2 | belief, l = 1: at least half of the whole patch's effect, and more than the complement's |
| E3 | random subspace and same-norm random: < 0.05 |
| E4 | similar-posterior donors: whole and belief both < 0.05 of the base–hybrid distance of ordinary pairs |
| E5 | with the raw route ablated the whole patch's share rises above 0.5 |
| E6 | uncertainty pairs respond less than mode pairs to the belief edit |
| E7 | the effect does not vanish at later decisions (it shrinks in absolute size as evidence accumulates, not as a share of the hybrid's) |
| E8 | cross-goal: the edited model takes the hybrid's action under both goals in more than half of the cases in which the whole patch does |

I hold E2 and E8 weakly. Rounds 17–19 found decodable quantities that were not the interface.
Which row of the brief's table is supported is decided by E2 (row 1 or 2), E7 (row 3) and the arbitrary-behaviour
measure (row 4).
