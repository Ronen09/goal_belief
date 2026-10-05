# Direct belief edit: site, encoding, edits, pairs, measures, decision rule and expectations

Written before any edit of this experiment was applied to a trained model, and before any encoding was fitted on one. Code:
`run.py`. It was smoke-tested on the untrained initial checkpoint of seed 0 only: the no-edit patch reproduces the
natural run (9·10⁻⁸). Brief: `BRIEF.md`.

## Models and site

The observation-prediction experiment's ten reward-only models, final checkpoint, frozen. **Site**: z(h, g), the residual stream entering block 1
at the goal token of history h under goal g. Block 0 computes it from the raw tokens and the goal. This is round
29's direct route, which carries about 0.7 of the history-dependent behavioural difference. The first decision after
the reveal.

## Encoding (no action labels)

The site is fitted on the fit side of the pair-types experiment's bank (144 306 histories), each under all three goals. b is the exact
goal-free posterior over the 14 cells at the last prefix token. Held-out R² on the 55 694 test histories × 3 goals,
overall and within (goal, prefix length):

| encoding | form | role |
|---|---|---|
| **shared** | z ≈ c_{g,L} + E b | **primary**: one E for every goal |
| brief offset | z ≈ c_g + E b | the brief's literal form (secondary) |
| goal-specific | z ≈ c_{g,L} + E_g b | a different map per goal (secondary comparison) |
| offsets only | z ≈ c_{g,L} | reference |

**Why c_{g,L}, not c_g, is primary.** The goal token sits at position 1 + L, so its state carries a positional term
that varies with prefix length L. On the untrained network, c_g + E b leaves that term to E (R² 0.47, against 0.87
with c_{g,L}). Recipient and donor have the same L, so the offset cancels in every edit; it only changes how E is
fitted. The brief's literal form is reported and edited alongside.

## Edits

Per held-out pair (recipient A, donor B, same prefix length 2–4) and goal g, only the goal token's state entering
block 1 is replaced. The recipient's prefix states are kept, and the network runs on.

| kind | new state | |
|---|---|---|
| none | z(A,g) | |
| whole | z(B,g) | whole direct-route replacement, same goal: the effect available through this route |
| **encoding** | z(A,g) + E (b_B − b_A) | **the donor's posterior only; the same vector under all three goals** |
| enc_brief | z(A,g) + E′ (b_B − b_A) | E′ from the brief's offset form |
| enc_goal | z(A,g) + E_g (b_B − b_A) | goal-specific map |
| rotated | z(A,g) + Q E (b_B − b_A) | Q a random rotation: identical norms, generic directions (5 draws, probabilities averaged) |
| pca | z(A,g) + P Pᵀ (z(B,g) − z(A,g)) | the donor's actual state in the top 13 principal components of the site (centred within goal and length); 13 = rank of E (posteriors sum to one) |
| random | the same in a random 13-dimensional subspace | 5 draws |

**References**: the *hybrid* (the model run naturally on B under g) for faithful transfer, and the solver's optimal
set for B's posterior under g for correctness.

## Pairs (solver tables only, fixed before any edit)

The belief-encoding-edit experiment's pairs, unchanged (seed 27; held-out histories; posteriors at least 0.25 apart in L1):

| set | selection | size |
|---|---|---|
| main | random same-length pairs | 30 000 pairs |
| change cells | (pair, goal) where A's and B's optimal sets are disjoint | 47 861 |
| **goal-dependent pairs** | the belief change matters under two goals, and B's optimal sets under those two goals are disjoint ("left under G1, right under G2") | 5 653 |
| preserve cells | the posteriors differ, the optimal sets are identical | 42 042 |
| one-step agreeing | identical exact one-step observation predictions, different posteriors, disjoint optimal sets under at least one goal | 20 000 pairs, 34 799 change cells |
| **equivalent recipients** | for each main pair, three further held-out histories with the same posterior node and prefix length and different token sequences (four recipients, all distinct; seed 32) | 13 165 pairs with four recipients |

## Measures

On change cells unless stated. Absolute rates are pooled over all cells of a model, then the median over models is
taken.

* **Donor-optimal** D(k): share of cells whose greedy action after edit k is in B's optimal set (correctness).
* **As hybrid**: share whose greedy action equals the hybrid's (faithful transfer). Also recipient-optimal, unchanged,
  and regret at B's posterior.
* **Aggregate transfer**, from pooled rates, never per-cell ratios: T(k) = (D(k) − D(none)) / (D(hybrid) − D(none)),
  and the **share of whole** S(k) = (D(k) − D(none)) / (D(whole) − D(none)). Both denominators are differences of
  rates over about 48 000 cells.
* **Pooled logit projection**: Σ⟨Δℓ_k, Δℓ_hyb⟩ / Σ|Δℓ_hyb|², with centred log-probabilities, summed over cells.
* **Preservation harm**: on preserve cells, the optimal rate with no edit minus that after edit k.
* **Goal-dependent pairs**: the share of pairs where edit k is donor-optimal under every change goal. Also the share
  where it is donor-optimal under two goals that require different actions, with different actions.
* **Equivalent recipients**: the same edit vector on the four recipients. Measured: the share of change cells where
  all four are donor-optimal; the mean pairwise agreement of their greedy actions; and the share of the variance of
  donor-optimality that lies between cells (the belief change) rather than across recipients.
* Per goal: the same change-cell measures. Edit statistics: the norm of E Δb over the actual direct-route difference,
  the cosine, and the residual share.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests over the ten, one-sided, p < 0.05.

| | criterion |
|---|---|
| F | prerequisite for interpretation: shared encoding, held-out R² within (goal, length) ≥ 0.5. If it fails, the edits are reported, read as limited by fit, and not as evidence that belief is absent |
| C1 transfer | S(encoding) ≥ 0.75, and D(encoding) − D(none) ≥ 0.25 absolute |
| C2 goal-dependent | right under every change goal: encoding ≥ 0.75 × whole, and above rotated |
| C3 preservation | harm(encoding) ≤ 0.05 |
| C4 beyond one-step | one-step agreeing change cells: D(encoding) − D(none) ≥ 0.5 × (D(whole) − D(none)), and above rotated |
| C5 specificity | D(encoding) above rotated and random (p < 0.05); rotated's gain ≤ half of encoding's; and generic replacement does not do better: D(pca) − D(encoding) ≤ 0.05 at the median |
| C6 equivalent histories | all four donor-optimal: encoding ≥ 0.75 × whole; agreement(encoding) ≥ agreement(none) − 0.05 |
| C7 shared map suffices | D(encoding) ≥ D(enc_goal) − 0.05 |

| result | reading |
|---|---|
| F and C1–C7 | **the native policy uses a manipulable, shared belief-like variable at its dominant interface** |
| F, C1, C2, C5 (first part); C3 fails | the shared belief direction controls the decision across goals, but the edit is not selective |
| C1 fails while enc_goal meets C1 | belief is encoded at the site, but in a goal-specific way: the goal is already mixed in after block 0 |
| C1 fails for both encodings, whole large | the route carries the decision, not in a belief-linear form |
| C5's PCA part fails only | the belief edit works, and so does generic replacement of equal rank: no advantage over generic evidence |
| otherwise | partial; reported criterion by criterion |

## Expectations

| | expectation |
|---|---|
| E1 | F holds |
| E2 | goal-specific R² exceeds shared by ≥ 0.03 (the cross-history-MLP experiment: block 0's MLP mixes goal and posterior) |
| E3 | whole: T between 0.5 and 0.9 (the goal-route-selection experiment: a_direct ≈ 0.7) |
| E4 | S(encoding) ≥ 0.6 (C1's 0.75 is held weakly) |
| E5 | C3 holds |
| E6 | rotated and random: T ≤ 0.15 |
| E7 | D(pca) ≥ D(encoding): C5's PCA part fails (it uses the donor's whole state, as in the belief-encoding-edit to policy-belief-edit experiments) |
| E8 | one-step agreeing: whole's gain smaller than on main pairs (the goal-route-selection experiment: the direct route carries less there) |
| E9 | C6 holds |
| E10 | C7 holds |

## Limits known in advance

One site (the goal token entering block 1); the interface (prefix tokens) is left unedited, so the edit can reach at
most the direct route's share. The first decision after the reveal; reward models only. The equivalent-recipient set
is restricted to posteriors with at least four distinct held-out token sequences of that length.
