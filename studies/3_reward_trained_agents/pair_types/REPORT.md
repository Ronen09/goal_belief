# Three types of donor–recipient pair (pair types)

Run date: 2026-09-30. Brief: `BRIEF.md`. Pairs, patches, measures and expectations: `PLAN.md`, committed (`10529a3`)
**before any patch was run**. No model was trained: the maze-belief experiment's six models, the pattern-specificity experiment's saved rank-13 PCA subspaces.
Numbers from `tables.md`; data in `results.json`. Reproduce: `reproduce.py pair_types` (2 min on one GPU).

**Registered and post hoc.** Pairs are drawn symmetrically (each history is as likely to be donor as recipient), so
the signed change in probability on the optimal set averages to zero wherever the two optimal sets are the same. I
saw that after the first run and added the unsigned measure (*crossing*: the probability that moves across the
boundary of the optimal set) and the split of greedy changes into within the set, out of it and into it. The
per-prefix-length table was added because type A could not be balanced over lengths. Everything else is as planned.

## Pairs

| type | pairs | prefix length 2 / 3 / 4 | posterior L1 | cells (pair × goal): same / disjoint |
|---|---|---|---|---|
| A: same posterior, different history | 7 243 | 400 / 2 843 / 4 000 | 0 | 21 729 / 0 |
| B: same optimal set under one goal, disjoint under another | 12 600 | 4 200 each | 1.56 | 18 427 / 19 359 |
| C: disjoint under all three goals | 12 000 | 4 000 each | 1.84 | 0 / 36 000 |

Prefix length, every token position and the horizon after the reveal are matched within each pair. Type A is short
of pairs at lengths 2 and 3 (few posteriors recur there within the cap of 50 per node), so it is weighted towards
length 4; section 1 gives it per length. `hybrid` is the model on the donor's prefix with no patch: the natural
swap of the evidence. Ranges are over the six models.

## 1. Same posterior, different history: a small effect that is not nothing

TV from base, all goals pooled:

| | natural access: hybrid | whole patch | pca | pca complement | direct route removed: hybrid (= whole) | pca | pca complement |
|---|---|---|---|---|---|---|---|
| type A | 0.04–0.09 | 0.05–0.09 | 0.05–0.08 | 0.01–0.02 | 0.08–0.14 | 0.08–0.14 | 0.01–0.05 |
| type C, for scale | 0.82–0.98 | 0.17–0.39 | 0.17–0.39 | 0.01–0.04 | 0.34–0.49 | 0.34–0.47 | 0.02–0.06 |

* **History matters beyond the posterior, a little.** With nothing patched and nothing removed, swapping one
  history for another with the identical posterior changes the action distribution by 0.04–0.09, 4–10 % of what a
  different posterior with a different optimal action does (E3 held here). The greedy action changes in 3–9 % of
  cells.
* **The change is between right and wrong, not between equally good actions.** Post hoc: no greedy change stays
  inside the optimal set (0.00 in every model); 0.03–0.07 of the probability crosses its boundary. Which of two
  same-posterior histories the model gets right depends on the history. The model is on the optimal set with
  probability 0.66–0.96 in these cells, so its errors are for the most part shared by histories with the same
  posterior.
* **It grows with prefix length**: 0.00–0.04, 0.02–0.08, 0.05–0.11 at lengths 2, 3, 4 (type C: 0.80–1.00
  throughout). Longer prefixes give more ways of reaching one posterior.
* **With the direct route removed** the effect is 0.08–0.14, which is 0.22–0.38 of type C's there, because the
  ablated model responds less to type C (0.34–0.49). E3 failed in one model (0.38).
* **The patches carry it.** The pca patch equals the whole patch to 0.02. Relative to the whole patch, the pca
  complement carries 0.15–0.37 for type A against 0.03–0.13 for type C, in all six models (E7 held): a little more
  of the history-specific part lies outside the top 13 components.

Plan's table: TV > 0 with a net change of zero in optimality — history is carried beyond the posterior. It is not
comparable to type C's effect under natural access.

## 2. Same optimal action under one goal, different under another

Type B, natural access, hybrid (the natural swap), all goals pooled; the two classes of cell come from the same
pairs:

| model | TV, same cells | TV, disjoint cells | ratio | greedy changed, same cells | Δ optimal, disjoint cells |
|---|---|---|---|---|---|
| reward, seed 1 | 0.23 | 0.69 | 0.33 | 0.22 | 0.64 |
| reward, seed 3 | 0.23 | 0.62 | 0.37 | 0.22 | 0.56 |
| reward, seed 4 | 0.25 | 0.75 | 0.34 | 0.25 | 0.68 |
| supervised | 0.07 | 0.95 | 0.07 | 0.05 | 0.94 |
| reward, seed 0 | 0.34 | 0.64 | 0.53 | 0.26 | 0.52 |
| reward, seed 2 | 0.31 | 0.50 | 0.62 | 0.25 | 0.43 |

* **The models separate the two kinds of cell, the supervised one nearly perfectly and the reward-trained ones in
  part.** Where the solver says the evidence changes the action, the natural swap changes it (Δ optimal 0.43–0.94).
  Where the solver says it does not, the supervised model changes 0.07; the reward-trained models change 0.23–0.34
  and the greedy action in 22–26 % of cells. E4 (ratio below a half) held in four models and failed in the two that
  had not learned G2; its bound on greedy changes (< 10 %) held only for the supervised model.
* **Those changes in same cells are errors moving, not indifference** (post hoc): none is within the optimal set;
  0.11–0.13 of cells go from an optimal greedy action to a non-optimal one and as many the other way. The
  reward-trained models are on the optimal set with probability 0.75–0.83 for G1 and G2 in these cells (0.30 for G2
  in seeds 0, 2), and which cells they miss depends on the evidence. For G3 (0.94–0.99) same cells change 0.01–0.10.
* **One edit, read differently by each goal.** The prefix states are goal-free, so the same patched states are used
  for both classes of cell. With the direct route removed the pca patch gives 0.93–0.98 of the hybrid's gain in the
  disjoint cells (E5 held) and reproduces the hybrid in the same cells (moved 0.82–0.95). What it reproduces there
  is the model's own behaviour on the donor's evidence, errors included: TV 0.20–0.33 in same cells against
  0.31–0.45 in disjoint cells (ratio 0.60–0.83; E4 failed in this condition). The ablated model is on the optimal
  set with probability 0.59–0.80 in same cells.
* **Under natural access the patch does harm in same cells in the supervised model.** There the natural swap
  changes 0.07, the whole patch 0.24, and 0.23 of cells go from an optimal greedy action to a non-optimal one
  (0.01 the other way; Δ optimal −0.20). The donor's prefix states and the recipient's raw tokens disagree, and the
  model has never seen that. In the reward-trained models the patch moves same cells 0.14–0.26, less than or about
  as much as the natural swap, with losses and gains of similar size (Δ optimal −0.05 to 0.03).
* In disjoint cells under natural access the whole and pca patches give 0.21–0.36 and 0.20–0.35 of the hybrid's
  gain and agree within 0.02 (E5 held).

Plan's table: between its two rows. The representation keeps evidence beyond what one goal's action needs — the
same goal-free states yield the solver's predicted change under the goal where the optimal action differs. "Same
cells unchanged" holds for the supervised model under the natural swap and nowhere else.

## 3. Different posteriors, different optimal actions

Type C, all goals pooled:

| | natural access: hybrid | whole | pca | direct route removed: hybrid (= whole) | pca |
|---|---|---|---|---|---|
| Δ optimal | 0.73–0.97 | 0.09–0.17 | 0.08–0.16 | 0.09–0.24 | 0.09–0.22 |
| as a share of the hybrid's | 1 | 0.11–0.23 | 0.10–0.22 | 1 | 0.92–1.00 |
| moved | 1 | 0.10–0.23 | 0.09–0.21 | 1 | 0.87–0.96 |
| greedy action in the donor's optimal set (base 0.00–0.09; 0.24–0.30 removed) | 0.78–0.99 | 0.15–0.23 | 0.14–0.21 | 0.38–0.50 | 0.38–0.49 |

* **Replacing the evidence produces the predicted change, to the extent the model on the donor's evidence does.**
  Under natural access the model itself follows the solver (0.78–0.99); the patch gets 0.10–0.23 of the way,
  because the goal token still reads the recipient's raw tokens. With that route removed the pca patch gives
  0.92–1.00 of the hybrid's gain, but the hybrid's gain is 0.09–0.24: the ablated model puts its greedy action in
  the donor's optimal set in 0.38–0.50 of cells.
* **Type B's disjoint cells move further than type C's under natural access**: moved 0.19–0.35 against 0.10–0.23
  for the whole patch, in every model (E6's equality failed; with the direct route removed they agree within
  0.04). Pairs that share an optimal action under some goal have closer posteriors (L1 1.56 against 1.84) and rely
  more on the prefix states. This is the belief-edit experiment's and 21's uncertainty–mode difference seen in another split.
* Per goal, natural access, Δ optimal for the pca patch: 0.09–0.15 (G1), 0.06–0.14 (G2), 0.00–0.23 (G3). With the
  direct route removed the models that do not respond to evidence under G3 (hybrid TV 0.00) show no change under
  G3 for any patch.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | type A, natural: TV(hybrid, base) 0.02–0.06; Δ optimal within ± 0.01 | in part | 0.04–0.09; Δ optimal 0.00 (zero by the symmetry of the pairs: uninformative) |
| E2 | type A, direct route removed: TV 0.03–0.14 | yes | 0.08–0.14 |
| E3 | type A's TV less than a third of type C's | in part | natural 0.04–0.10; removed 0.22–0.38 |
| E4 | type B same cells: TV less than half the disjoint cells'; greedy changes < 10 % | in part | natural: ratio 0.07–0.37 in four models, 0.53 and 0.62 in two; removed: 0.60–0.83; greedy < 10 % in the supervised model only |
| E5 | type B disjoint cells: pca ≥ 0.85 of the hybrid's gain (removed); whole and pca 0.1–0.45, within 0.03 (natural) | yes | 0.93–0.98; 0.21–0.36 and 0.20–0.35 |
| E6 | type C as E5; moved equal to type B's disjoint cells within 0.05 | in part | 0.92–1.00; 0.11–0.23; equal with the route removed, not under natural access (0.09–0.18 apart) |
| E7 | pca complement's share larger for type A than for type C in ≥ 5 models | yes | six of six: 0.15–0.37 against 0.03–0.13 |

## Conclusions, by the brief's table

1. **Does history matter beyond belief? Yes, by a small amount.** Two histories with the identical posterior give
   action distributions 0.04–0.09 apart under natural access, a tenth or less of the effect of a different
   posterior, growing with prefix length. The difference is in which cases the model gets wrong.
2. **Does the representation preserve information beyond one goal's action? Yes.** Prefix states from a donor whose
   optimal action under one goal equals the recipient's still carry what another goal needs: the same patched
   states give 0.93–0.98 of the model's own change under the goal where the optimal action differs (direct route
   removed), 0.20–0.36 under natural access.
3. **Does replacing evidence produce the predicted change? As far as the model on the donor's evidence does, and
   with it the model's errors.** The patch reproduces the hybrid, not the solver. Where the solver predicts no
   change, the reward-trained models change their greedy action in about a quarter of cells under the natural swap,
   and the patch reproduces that as well.
4. **The two conditions answer different questions and should stay apart.** Natural access shows how much of the
   decision runs through the prefix states (a tenth to a third) and that a patch conflicting with the raw tokens
   can harm (the supervised model in same cells). Direct route removed shows that the prefix states suffice for
   the ablated model's whole response, and that the ablated model is a poor policy (on the optimal set 0.56–0.80
   where the intact one is 0.66–0.96).
5. **The rank-13 PCA patch equals the whole patch** in every type and class of cell to within 0.02 under natural
   access (0.03 per goal), and moves 0.64–0.96 of the way to it with the route removed (lowest for type A).

Limits: the same six models and one maze; type A's posteriors are the 297 that recur with different histories,
weighted to prefix length 4; optimal sets are the solver's, and the reward-trained models are off them in 4–34 % of
these cells before any patch, which is what the same cells mostly measure; mean ablation is off the training
distribution.
