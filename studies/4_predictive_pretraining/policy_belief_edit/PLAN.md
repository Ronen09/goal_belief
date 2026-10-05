# Policy belief edit: interface, edits, readers, measures, decision rule and expectations

Written before any edit of this experiment was applied to a trained model. Code: `run.py` (smoke-tested on an untrained
network with untrained heads only). Brief: `BRIEF.md`.

## Models (frozen)

The observation-prediction experiment's ten reward-only PPO models (the *reward* backbones of the predictive-transfer to belief-encoding-edit experiments), final checkpoint. For each, the predictive-transfer experiment's
three 16-unit heads trained on 100 000 examples on its last-prefix-token state after the last block (rebuilt and
reproduced exactly in the belief-encoding-edit experiment).

## One edit, two readers

**Interface**: the residual stream entering block 1 at every prefix token, the belief-edit to pair-types experiments's pre-goal interface. Prefix
tokens never see the goal, and the maze-belief experiment found that nothing is read from prefix states after block 2. The same edited
states feed both readers in one forward pass:

* **the original policy**: its action distribution at the goal token, under each goal. *Natural access* (primary):
  nothing else changed, so the goal token's first attention layer still reads the recipient's raw tokens. *Direct
  route removed* (secondary diagnostic): that layer's output at the goal token is replaced by its mean over reference
  histories for that goal and prefix length (the pair-types experiment's ablation).
* **the new head**: the propagated state after the last block at the last prefix token, through the predictive-transfer experiment's frozen head.

**Encoding**: h ≈ c_position + E b at the interface, fitted on all prefix tokens of the fit side of the pair-types experiment's bank (b:
the exact goal-free posterior at that token). Its held-out R² is reported.

**Edits**, at every prefix position i of a held-out pair (recipient A, donor B, same prefix length 2–4):

| edit | interface state at position i | |
|---|---|---|
| none | h_A,i | |
| whole | h_B,i | whole-state replacement |
| **encoding** | h_A,i + E (b_B,i − b_A,i) | **the candidate: the donor's posteriors only** |
| rotated | h_A,i + Q E (b_B,i − b_A,i), Q a random rotation (5 draws) | generic directions, same edit vectors |
| random patch | h_A,i + P_R (h_B,i − h_A,i), random rank 13 (5 draws) | generic subspace, the donor's state |
| PCA patch | top 13 principal components (secondary) | |

**References**: for the policy, the *hybrid*: the model on the donor's own prefix with the recipient's goal, natural
(or with the same ablation in the secondary condition). For the head, the head on the donor's own state, which
equals `whole` by construction (checked: 2·10⁻⁴).

## Pairs and cells

The belief-encoding-edit experiment's pairs, unchanged: main (30 000 pairs; 47 861 change cells, where the two posteriors' optimal sets are
disjoint under that goal, and 42 042 preserve cells, where they are identical) and one-step agreeing (20 000 pairs;
34 799 change cells). Every pair under all three goals.

## Measures, per reader, on main change cells unless stated

* D(k): share of cells whose greedy action after edit k is in the donor's optimal set (the head: mean of its three
  heads).
* **Transfer** T(k) = (D(k) − D(none)) / (D(reference) − D(none)): the share of the way to what that reader does with
  the donor's actual evidence.
* **Interface share** S(k) = (D(k) − D(none)) / (D(whole) − D(none)): the share of the whole interface patch's
  effect. For the head S = T.
* **Preservation**: on preserve cells, the share still optimal, against no edit.
* Also: moved (TV toward the reference), greedy changes, regret at the donor's posterior; the same on one-step
  agreeing pairs; edit statistics at the interface.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests over the ten, p < 0.05 (one-sided where a direction is stated).

| | criterion |
|---|---|
| H | prerequisite, the edit controls the head: T_head(encoding) ≥ 0.5, above rotated and random patch |
| **same change controls both** | H, T_policy(encoding) ≥ 0.5 × T_head(encoding), and the policy's encoding above rotated and random patch |
| **dissociation** | H, T_policy(encoding) < 0.25 × T_head(encoding), and T_head above T_policy (one-sided) |
| between | partial; reported as such |

**Where a dissociation comes from**, read from natural access:

| | reading |
|---|---|
| T_policy(whole) small | the policy takes little from the interface at all: access, not form |
| S_policy(encoding) ≥ 0.75 × S_head(encoding) | of what the policy does take from the interface, the belief-encoding edit carries the same share as for the head: the same form, less access |
| S_policy(encoding) < 0.5 × S_head(encoding), significantly | the policy does not read belief-associated directions at the interface the way the head does |

The direct-route-removed condition is reported with the same measures and no criterion.

## Expectations

| | expectation |
|---|---|
| E1 | H holds: T_head(encoding) ≥ 0.5 (the belief-encoding-edit experiment's edit at the head's own site gave 0.84 of whole for reward) |
| E2 | natural access: T_policy(whole) between 0.15 and 0.45 (the pair-types experiment: 0.21–0.36) |
| E3 | the dissociation holds under natural access |
| E4 | S_policy(encoding) ≥ 0.75 × S_head(encoding): the same form, less access |
| E5 | direct route removed: T_policy(whole) ≥ 0.8 and T_policy(encoding) ≥ 0.5 |
| E6 | rotated and random patch: T ≤ 0.15 for every reader and condition |
| E7 | preserve cells: the policy (natural) loses less than the head |

## Limits known in advance

Reward models only (the predictors have no policy). One interface (entering block 1, all prefix tokens), the first
decision after the reveal. Under natural access the policy's goal token reads the raw tokens directly, so no
interface edit can reach the hybrid. The edit of the original policy is measured against its own behaviour on the
donor's evidence, which is not always optimal.
