# One belief-encoding edit at the pre-goal interface, read by the original policy and by the new head (policy belief edit)

Run date: 2026-10-01. Brief: `BRIEF.md`. Interface, edits, readers, measures and decision rule: `PLAN.md`, committed
(`1764da8`) **before any edit of this experiment was applied to a trained model**. Numbers from `tables.md`; data in
`results.json`. Reproduce: `reproduce.py policy_belief_edit` (3 min on one GPU; needs the predictive-transfer to belief-encoding-edit experiments).

**Setting.** The observation-prediction experiment's ten reward-only PPO models, frozen. Interface: the residual stream entering block 1 at every
prefix token (the belief-edit to pair-types experiments). There, h ≈ c_position + E b is fitted on the fit side and each prefix state of a
held-out recipient is replaced by h_A,i + E (b_B,i − b_A,i). One forward pass then gives two readings:

* **the original policy** at the goal token, under each goal. Natural access is primary: the goal token still reads
  the recipient's raw tokens directly.
* **the new head** (the predictive-transfer experiment's, trained on 100 000 examples) on the propagated state at the last prefix token.

Transfer T is the share of the way from no edit to what that reader does with the donor's own evidence (policy: the
hybrid, the model on the donor's prefix; head: the donor's state). Interface share S is the share of the whole
interface patch's effect. Main pairs, change cells (the two posteriors' optimal sets are disjoint under that goal),
all three goals; median over ten models.

## 1. The key comparison (natural access)

| | new head | original policy |
|---|---|---|
| no edit: donor-optimal | 0.048 | 0.108 |
| **encoding edit**: donor-optimal | 0.751 | 0.243 |
| **encoding edit: T** | **0.81** (0.77–0.86) | **0.19** (0.11–0.22) |
| whole interface replacement: T | 1 (by construction) | **0.21** (0.14–0.25) |
| **encoding edit: S** (share of the whole interface effect) | 0.81 | **0.89** (0.81–0.93) |
| rotated encoding: T | 0.08 | 0.02 |
| random rank-13 patch: T | 0.04 | 0.01 |
| PCA-13 patch: T | 0.93 | 0.20 |
| preserve cells: still optimal, none → encoding | 0.960 → 0.875 | 0.916 → 0.911 |

* **The same edit controls the new head and moves the original policy a fifth of the way.** T_policy / T_head =
  0.22 (0.13–0.28); T_head above T_policy in all ten models (p < 0.001).
* **But that fifth is all the interface gives.** Replacing every prefix state by the donor's moves the policy only
  0.21: under natural access, the goal token's first attention layer reads the raw tokens and recomputes the
  evidence itself (the pair-types experiment: 0.21–0.36).
* **Of what the policy does take from the interface, the belief-encoding edit carries 0.89**, more than its share
  for the head (0.81; S_policy above S_head, p 0.002). The generic controls carry 0.02–0.11. The policy reads the
  interface along the same belief-associated directions as the head, and somewhat more exclusively.
* **Preservation**: the edit leaves the policy's correct decisions alone (−0.005), because it moves the policy
  little. The head loses 0.085, as in the belief-encoding-edit experiment.

## 2. Decision rule

| | criterion | value | held |
|---|---|---|---|
| H | T_head(encoding) ≥ 0.5, above rotated and random patch | 0.81; p < 0.001, < 0.001 | **yes** |
| same change controls both | T_policy ≥ 0.5 × T_head, policy's encoding above controls | 0.22 × | no |
| **dissociation** | T_policy < 0.25 × T_head, T_head above T_policy | 0.22 ×; p < 0.001 | **yes** (median; 3 of 10 models are at 0.26–0.28) |
| source of the dissociation | T_policy(whole) small; S_policy ≥ 0.75 × S_head | 0.21; 1.09 × | **access, not form** |

**By the registered rule: a dissociation, and its source is access.** The representation supports belief-based
control through the head. The original policy takes most of its evidence elsewhere, from the raw tokens through the
goal token's own first attention layer. Where it does read the interface, it reads it in the belief-associated form.

## 3. Secondary diagnostics

**Direct route removed** (the goal token's first attention output replaced by its mean). The whole interface patch
then equals the hybrid, so T_policy(whole) = 1 by construction:

| | donor-optimal: none / encoding / hybrid | T(encoding) | rotated / random patch T |
|---|---|---|---|
| policy, route removed | 0.270 / 0.456 / 0.484 | **0.89** (0.79–0.93) | 0.12 / 0.07 |

With the direct route gone, the edit moves the policy as far as it moves the head (0.89 against 0.81). But the
ablated policy is much less competent: on the donor's own evidence it chooses the donor-optimal action in 0.48 of
change cells, against 0.81 with natural access. This is the weakening the brief warned of.

**One-step-agreeing pairs** (identical one-step predictions, different optimal actions), natural access:

| | none | encoding | whole | hybrid | T(encoding) | S(encoding) |
|---|---|---|---|---|---|---|
| new head | 0.098 | 0.643 | 0.859 | | 0.71 | 0.71 |
| original policy | 0.179 | 0.408 | 0.451 | 0.745 | **0.40** | 0.81 |

Here the interface carries much more of the policy's decision (whole 0.48 against 0.21 on main pairs), and the
encoding edit transfers 0.40. Where recent symbols do not separate the beliefs, the direct route has less to work
with, and the policy relies more on the prefix states (post hoc reading).

**The encoding at the interface**: held-out R² 0.79 (0.65 within position); the edit's norm is 0.70 of the actual
difference, cosine 0.68. The interface after block 0 is less belief-linear than the head's site (the belief-encoding-edit experiment: R² 0.65,
norm 0.79 for reward at site 4). Even so, through three more blocks the propagated edit reaches the head with T
0.81, the same as the belief-encoding-edit experiment's direct edit at the head's own site (0.84 of whole).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | H holds | yes | 0.81 |
| E2 | natural: T_policy(whole) between 0.15 and 0.45 | yes | 0.21 |
| E3 | the dissociation holds | yes, at the median | 0.22 × (0.13–0.28) |
| E4 | S_policy ≥ 0.75 × S_head | yes | 1.09 × |
| E5 | route removed: T_policy(whole) ≥ 0.8, T_policy(encoding) ≥ 0.5 | yes (the first by construction) | 1; 0.89 |
| E6 | rotated and random patch T ≤ 0.15 (main pairs) | yes | head 0.08 / 0.04; policy natural 0.02 / 0.01; removed 0.12 / 0.07 |
| E7 | preserve: the policy loses less than the head | yes | −0.005 against −0.085 |

7 of 7. E5's first part was guaranteed by the ablation's construction (the pair-types experiment says so); I did not notice that
when registering it.

## Conclusions

1. **Only the new head is controlled by the same belief-associated change, under natural access.** It moves the
   head 0.81 of the way to the donor's behaviour and the original policy 0.19.
2. **The dissociation is in access, not in representation.** The original policy takes only 0.21 of its decision
   from the pre-goal interface at all; it recomputes the evidence from the raw tokens at the goal token. Of what it
   takes from the interface, 0.89 is carried by the belief-encoding directions, at least as much as for the head,
   and generic directions carry almost none. Remove the direct route, and the same edit moves the policy 0.89 of the
   way.
3. **So the original policy uses this interface the way the head does, only less.** The question the brief leaves
   open has a partial answer: the policy does recompute belief elsewhere, from the raw tokens through the goal
   token's first attention layer. Where that route is less informative (one-step-agreeing histories), it leans more
   on the interface (0.48), and the edit transfers further (0.40).

Next, if wanted: the same encoding edit at the goal token's own first-attention output, the route that carries the
remaining 0.8, would test whether the recomputed evidence is also belief-linear there and complete the account of
the policy's goal-seeking computation.

Limits: reward models only; one interface; the first decision after the reveal; the dissociation verdict is at the
median (three models are just above the 0.25 line); the direct-route ablation lowers competence, so its numbers are
a diagnostic only.
