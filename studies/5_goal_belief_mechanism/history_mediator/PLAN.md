# History mediator: candidates, arms, decision rule and expectations

Draft for review; to be committed before `run.py` is run on any trained model. Brief: `BRIEF.md`.

## Site and the history part

The additive-ablation experiment found that the history enters the goal token's decision through the **four attention
outputs** at the goal token. Edits there with vectors measured in natural runs behaved cleanly: removing the history part
there collapsed the policy to the history-blind ceiling (0.613, history dependence 0.068), and swapping it took the
donor's action in 0.927 of pairs. Edits at all eight components with natural vectors overshot. **All arms here edit
only the four attention outputs**, so every candidate and the all-history reference are measured at the same site in the
same way.

For attention output k at the goal token: x_k(h, g) natural; history part xbar_k(h) = mean over the three goals;
Xbar_k(L) its fit-side mean over histories of length L. The history's deviation is d_k(h) = xbar_k(h) − Xbar_k(L).

## Candidates Z(h) and x_Z

For each Z, x_Z,k(h) = E[d_k | Z(h), L], estimated on **fit-side histories only**. Every measurement is on held-out
cells, so r(h) = d_k(h) − x_Z,k(h) includes everything Z fails to predict, fit error included.

| Z | what | size | estimator of E[d \| Z, L] |
|---|---|---|---|
| none | L only (history-blind) | 0 | x_Z = 0 |
| top | H's top action | 4 values | table |
| rank2 | H's top two actions, ordered | 12 values | table |
| H2 | H projected on its top two fit-side principal components | 2 reals | MLP |
| H | the goal-averaged logit profile (additive-code experiment), centred | 3 reals | MLP |
| mix | the H-mixture experiment's code: P(optimal \| random goal) and mean reachability per action | 8 reals | table over its distinct values |
| b | the exact posterior | 13 reals (619 distinct) | table over posteriors |
| all | d itself | the state | x_Z = d (reference) |

* Tables are per (Z value, L); a cell whose (Z, L) value has fewer than 5 fit-side histories is excluded from **every**
  candidate, so all candidates share one cell set. (Posteriors: 97.8 % of cells have a fit-side posterior; median 564
  fit histories each.)
* MLP: input Z and a one-hot of L; one hidden layer of 256; output the four components' d_k; trained on 90 % of
  fit-side histories, early stopping on the other 10 %. Held-out R² of x_Z for d is reported per Z.
* H is computed from the network's own logits under all three goals. It is a readout of the network, not an
  independent hypothesis about the computation: Z = H asks whether a 4-number summary of the state suffices.

## Arms (each added to the four attention outputs at the goal token)

| arm | edit | question |
|---|---|---|
| remove_Z | −x_Z(h) | collapse to the history-blind ceiling? |
| remove_r | −r(h) | unchanged? |
| **swap_Z** | x_Z(h′) − x_Z(h) | does the decision transfer to the donor's? |
| **swap_r** | r(h′) − r(h) | does almost nothing happen? |
| swap_r_matched | r(h″) − r(h), h″ another history with the same Z value (table candidates) | sufficiency: does history beyond Z matter when Z agrees? |
| swap_all | d(h′) − d(h) | the all-history reference |
| swap_Z_rot | swap_Z's vectors rotated at random, same norm | control |

Donors h′: another held-out history of the same length in the same batch, the same for every candidate (fixed seed).

## Cells and measures

The query-swap experiment's cells under each goal, restricted to the shared cell set. **Informative pairs**: the
recipient's and donor's natural actions under g differ.

* **swap effect(Z)** = on informative pairs, the share whose action under swap_Z is the donor's natural action. The
  same measure for swap_all is the reference; swap_all at this site took the donor's action in 0.927 of pairs in the
  additive-ablation experiment.
* **swap effect(r | Z)** = on the same informative pairs, the share whose action under swap_r differs from the
  recipient's natural action (any change, not just towards the donor: a disruption counts).
* remove arms: optimal under g on all cells; history dependence (1 − share taking the modal action of the cell's
  (g, L) group); the history-blind ceiling as in the additive-ablation experiment.

## Decision rule

Ten models; medians; exact one-sided Wilcoxon signed-rank tests, p < 0.05. For each candidate:

| | criterion |
|---|---|
| **S90** | swap effect(Z) / swap effect(all) ≥ 0.9 |
| **R0** | swap effect(r \| Z) ≤ 0.05, and ≤ 0.1 × swap_all's change rate |

**Z mediates the history** if S90 and R0 both hold. **Minimal**: the first mediating candidate in the order
none < top < rank2 < H2 < H < mix < b. If none mediates, the result is the best ratio and the smallest r effect
reached, and the reading is that no tested summary carries the history alone.

Secondary (reported, not part of the definition): remove_Z within 0.05 of the history-blind ceiling with history
dependence ≤ 0.25 × natural's; remove_r drop ≤ 0.02; swap_r_matched change ≤ 0.05; swap_Z above swap_Z_rot.

## Expectations

| | expectation |
|---|---|
| E1 | swap_all reproduces the additive-ablation experiment's attention swap within 0.01 (0.927) |
| E2 | b fails S90 (ratio 0.7–0.9): in the H-mixture experiment the posterior table moved decisions 0.68 of the way to whole replacement |
| E3 | b fails R0: histories with the same posterior differ in the policy's actions by 0.04–0.09 (pair-types experiment), and r(h) carries that |
| E4 | H reaches S90; H2 does not |
| E5 | no candidate below H passes R0 |
| E6 | top's swap effect ≥ 0.5: the top action alone carries much of the transfer, but not the margins that decide near-ties |

## Limits known in advance

One site (the four attention outputs, natural vectors); first decision at the goal token only; reward models only; the
MLP estimator for continuous Z may understate what a better estimator would extract; H is the network's own readout.
