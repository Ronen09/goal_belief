# Does the policy use the decoded belief? Edits at the goal-free prefix interface (belief edit)

Run date: 2026-09-29. Brief: `BRIEF.md`. Interventions, measures and expectations: `PLAN.md`, committed (`f3b68c4`)
**before any edit was run**. No model was trained: the maze-belief experiment's six trained models (reward-trained seeds 1, 3, 4;
supervised; then seeds 0, 2, which had not learned G2). Numbers from `tables.md`; data in `results.json`
(`results_registered.json` holds the registered edits alone). Reproduce: `reproduce.py belief_edit` (20 min on one GPU).

**Registered and post hoc.** The plan's edits are whole, belief, belief rank 7, complement, random subspace and random
of the same norm, with the raw route intact or removed, at the reveal and 1, 2, 4 moves later, and cross-goal with
the raw route intact. After they were read, I added three edits (pattern, pattern shift, pattern complement) and
cross-goal reuse with the raw route removed. Everything in sections 3 and 5 is post hoc.

## Setup

**Interface**: the residual stream at the prefix tokens, entering block 1. These tokens come before the goal token;
their states cannot depend on the goal. **Pairs**: recipient A and donor B from held-out histories, the same prefix
length and goal, exact optimal action sets disjoint. 7 025 *mode* pairs (the most likely cell differs, and acting on
it as certain already gives different actions), 3 529 *uncertainty* pairs (acting on the most likely cell gives the
same action for both), 351 *similar* pairs (posteriors within 0.1, different histories). **Reference**: the model on
the hybrid sequence (B's prefix, the recipient's goal). **Moved** = 1 − TV(edited, hybrid) / TV(base, hybrid).
**Raw route removed**: the decision token's own first attention output is set to its mean for that goal, which
removes what it reads from the raw prefix tokens.

## 1. The registered edit of the decoder's row space does nothing

Moved to the hybrid at the reveal (mode / uncertainty pairs):

| model | whole prefix state | belief (decoder's row space) | complement | random subspace |
|---|---|---|---|---|
| reward, seed 1 | 0.12 / 0.17 | 0.00 / 0.00 | 0.12 / 0.17 | 0.01 / 0.01 |
| reward, seed 3 | 0.20 / 0.34 | 0.00 / 0.00 | 0.20 / 0.33 | 0.01 / 0.01 |
| reward, seed 4 | 0.17 / 0.26 | 0.00 / 0.00 | 0.17 / 0.26 | 0.01 / 0.01 |
| supervised | 0.19 / 0.27 | 0.00 / 0.00 | 0.19 / 0.27 | 0.02 / 0.04 |
| reward, seed 0 | 0.25 / 0.45 | 0.00 / 0.00 | 0.25 / 0.45 | 0.00 / 0.00 |
| reward, seed 2 | 0.24 / 0.42 | 0.00 / 0.00 | 0.24 / 0.42 | 0.01 / 0.01 |

* **The edit is valid where it is made and inert everywhere else.** The decoder (held-out R² 0.90–0.92) reads the
  donor's posterior exactly after the belief edit. The action distribution does not move, the posterior decoded at
  the goal token's output moves 0.00–0.03 of the way, and no probability goes to unrelated actions (≤ 0.00).
* **It is not recomputation.** With the raw route removed the whole patch moves the decision all the way (1.00) and
  the belief edit 0.00–0.04, less than a random subspace of the same dimension (0.03–0.14).
* **The complement carries the whole effect** (0.92–0.99 with the raw route removed).
* **Why.** The decoder's row space (rank 14) holds 2–3 % of the squared difference between the donor's and the
  recipient's states; a random subspace of that dimension holds 9–16 %. A regression decoder reads along directions
  in which the state varies little. Moving the state there changes what the decoder says and not what later layers
  receive.

By the brief's table this is row 2: whole-state patches work, the belief-directed edit does not, and the decoder's
row space is not established as the interface.

## 2. The remaining registered checks

* **The effect of the whole patch does not fade.** 1, 2 and 4 moves after the reveal it moves the decision 0.12–0.40,
  as at the reveal (E7 held). The belief edit stays at 0.00.
* **Entering block 2 it is too late**: the whole patch moves 0.01–0.12 (E1 held).
* **Uncertainty pairs respond more than mode pairs** in all six models (0.17–0.45 against 0.12–0.25). E6 expected the
  opposite. When the most likely cell is the same and the rest of the posterior differs, more of the decision is
  read from the prefix states.
* **Similar-posterior donors are not inert.** A whole patch from a donor with nearly the recipient's posterior and
  another history changes the action distribution by 0.02–0.06 (total variation), and by 0.05–0.15 with the raw
  route removed, against 0.36–0.48 between base and hybrid for ordinary pairs there. The prefix states hold
  something about the history that the posterior does not, and the decision uses it. E4 held only with the raw
  route intact.
* **The two seeds that had not learned G2 lean most on the prefix states**, above all for G2 (per goal: 0.53–0.63
  against 0.13–0.21 in the seeds that learned it).

## 3. Post hoc: the directions along which the state varies with the posterior

Let Σ be the covariance of the prefix states and W the decoder. The columns of ΣW are the directions along which
the state co-varies with the decoded posterior (14 of 128). `pattern` replaces the donor's component in their span;
`pattern shift` moves the state along them, by the oblique map ΣW (WᵀΣW)⁺ Wᵀ, until the decoder reads the donor's
posterior; `pattern complement` replaces everything outside their span.

| edit | share of ‖x_B − x_A‖² | moved, raw route intact (mode / uncertainty, range over 6 models) | raw route removed |
|---|---|---|---|
| whole | 1.00 | 0.12–0.25 / 0.17–0.45 | 1.00 |
| pattern | 0.90–0.92 | 0.11–0.25 / 0.16–0.44 | 0.84–0.93 / 0.76–0.91 |
| pattern shift | 0.79–0.96 | 0.10–0.22 / 0.14–0.41 | 0.67–0.81 / 0.60–0.76 |
| pattern complement | 0.08–0.10 | 0.01 / 0.01–0.04 | 0.03–0.07 / 0.04–0.12 |
| belief (registered) | 0.02–0.03 | 0.00 / 0.00 | 0.00–0.04 |

* Fourteen directions hold nine tenths of what distinguishes two prefix histories and carry nearly all of the
  effect of the whole state. What lies outside them carries almost none.
* Moving the state along them just far enough for the decoder to read the donor's posterior gives 60–81 % of the
  full effect with the raw route removed, and the gain on the solver's optimal set is 0.67–1.07 of the hybrid's.
* These edits add no more probability to unrelated actions than the whole patch does.
* **This edit is not selective.** Its subspace is nine tenths of the evidence-dependent variation. It shows that the
  part of the prefix state that depends on the evidence is low-dimensional and varies with the posterior. It cannot
  separate the posterior from whatever else varies with the evidence, and section 2's similar-posterior donors show
  that something else does.

## 4. Cross-goal reuse, registered (raw route intact)

The same edited prefix states, with the recipient's goal set to each of the three. Per goal the whole patch moves
the decision 0.13–0.34 of the way to that goal's hybrid (seeds 1, 3, 4 and supervised; one cell at 0.03), and the
belief edit 0.00. The registered count, how often the edited model takes the hybrid's action under *both* of two
goals when those actions differ, is 0.00–0.22 for the whole patch: with the raw route intact an edit that moves the
distribution a fifth of the way seldom changes the greedy action. E8 cannot be assessed from it.

## 5. Post hoc: cross-goal reuse with the raw route removed

| model | goals | pairs | whole | pattern | pattern shift | pattern complement | random subspace | belief |
|---|---|---|---|---|---|---|---|---|
| reward, seed 1 | G1, G2 | 1 364 | 1.00 | 0.78 | 0.58 | 0.00 | 0.00 | 0.00 |
| reward, seed 3 | G1, G2 | 270 | 1.00 | 0.71 | 0.50 | 0.14 | 0.19 | 0.00 |
| reward, seed 4 | G1, G2 | 475 | 1.00 | 0.68 | 0.25 | 0.05 | 0.04 | 0.00 |
| reward, seed 4 | G1, G3 | 822 | 1.00 | 0.55 | 0.40 | 0.01 | 0.02 | 0.00 |
| reward, seed 4 | G2, G3 | 710 | 1.00 | 0.47 | 0.41 | 0.29 | 0.19 | 0.01 |
| supervised | G1, G2 | 2 282 | 1.00 | 0.65 | 0.43 | 0.00 | 0.03 | 0.00 |
| reward, seed 0 | G1, G2 | 2 040 | 1.00 | 0.88 | 0.66 | 0.00 | 0.00 | 0.00 |
| reward, seed 2 | G1, G2 | 847 | 1.00 | 0.91 | 0.47 | 0.01 | 0.00 | 0.00 |
| reward, seed 2 | G1, G3 | 371 | 1.00 | 0.74 | 0.33 | 0.10 | 0.01 | 0.00 |

Share of pairs in which the edited model takes the action appropriate to the donor's evidence under both goals,
where those two actions differ from each other and from the unedited ones. The whole patch is 1.00 by construction
(with the raw route removed the prefix states are all the evidence there is). One edit of the goal-free prefix
states produces different, appropriate actions under different goals in 47–91 % of cases for the pattern edit and
25–66 % for the shift. With G3, most models have no responsive pairs once the raw route is removed: for that goal
the action does not depend on the evidence.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | whole at block 1: 0.2–0.4; at block 2: < 0.1 | in part | 0.12–0.45; 0.01–0.12 |
| E2 | belief edit ≥ half of the whole patch, and above the complement | **no** | 0.00 against 0.12–0.45; complement = whole |
| E3 | random edits < 0.05 | yes | ≤ 0.04 (raw route intact) |
| E4 | similar-posterior donors inert | in part | 0.02–0.06 intact; 0.05–0.15 with the raw route removed |
| E5 | raw route removed: whole > 0.5 | yes | 1.00 |
| E6 | uncertainty pairs respond less | **no** | more, in all six models |
| E7 | the effect does not vanish later | yes | 0.12–0.40 at 1–4 moves |
| E8 | cross-goal: belief edit ≥ half of the whole patch's rate | not assessable as registered; **no** for the belief edit in any condition | section 4 |

## Conclusions, by the brief's table

1. **Registered result: row 2.** Evidence held in the prefix states is causally used; the decoder's row space is not
   its interface. A held-out R² of 0.9 and a causal effect of 0.00 come from the same decoder.
2. **Not row 3.** The whole patch's effect persists, and with the raw route removed the prefix states are the whole
   of the evidence. The null is not recomputation.
3. **Not row 4.** No edit produced arbitrary behaviour.
4. **Post hoc, a weaker form of row 1.** A 14-dimensional part of the goal-free prefix state that varies with the
   posterior carries the evidence to the decision, and one edit of it yields different, appropriate actions under
   different goals. It is a belief-*associated* interface, not shown to be a belief: it is most of the state's
   evidence-dependent variation, and histories with the same posterior still differ in what they cause.
5. **The decision has two routes to the evidence**: its own first attention layer reading the raw tokens, and the
   prefix states. With both intact the prefix states account for 0.12–0.45 of the dependence on evidence.

For method: a decoder's row space is where the read-out looks, not where the variable lives. The navigate-commit to maze-occupancy experiments used
decoder directions to measure displacement, which is sound, and the maze-belief experiment's group and whole-component patches are
unaffected. An intervention should move the state along ΣW.

Limits: the post hoc edits were chosen after seeing a null and need replication on new models; one maze; edits at
one interface and one layer; mean ablation of the raw route is itself an intervention off the training
distribution.
