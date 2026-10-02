# Round 33: site, measures, decision rule and expectations

Written before any encoding was fitted on a trained model or any edit applied to one. Code: `run.py`, which reuses
round 32's `run.py` with the site changed. That file was parametrised by its site, and round 32 rerun with it gives a
byte-identical `results.json`. `run.py` was smoke-tested on the untrained initial checkpoint of seed 0 only: the no-edit
patch reproduces the natural run (9·10⁻⁸). Brief: `BRIEF.md`.

## Site

m(h, g) is the residual stream at the goal token after block 0's attention and before its MLP: block 0's MLP input,
before its layer norm. It equals the goal token's embedding plus block 0's attention output. The embedding is a function
of (goal, prefix length) only, so c_{g,L} + E b fitted at m is the same fit as at the attention output. Round 32's site
is z = m + MLP₀(m).

Block 0's attention can still depend on the goal in two ways: the goal token's query includes the goal embedding, and
the goal token attends to its own value. Round 30 found that patching single block 0 heads across goals moves the
decision, so some goal-dependence at m is possible.

## Unchanged from round 32

The models, the encodings (shared c_{g,L} + E b primary; the brief's c_g + E b; goal-specific E_g), the edits (none,
whole, encoding, enc_brief, enc_goal, rotated ×5, PCA-13, random rank-13 ×5), the references (hybrid; solver), the pairs
(main 30 000; goal-dependent 5 653; preserve 42 042 cells; one-step agreeing 20 000; equivalent recipients 13 165 × 4),
and every measure. **Check**: MLP₀ acts on each token separately, so `none`, `whole` and `hybrid` must reproduce round
32's numbers.

## Added

* **Goal × history interaction** at m and at round 32's site, on held-out histories. Take the state minus its (goal,
  length) mean, then remove each history's mean over the three goals; the share of the squared variation that remains.
  It is 0 if the site holds a goal-free evidence code plus a goal offset.
* **Propagation**: the shared edit at m, carried through block 0's MLP. It induces a change at round 32's site. Its
  cosine is reported against the actual difference z(B,g) − z(A,g), against round 32's shared edit vector and against
  its goal-specific one, with norm ratios (first 8 192 main pairs, all goals).

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05. C1–C7 are round 32's, with the same
thresholds.

| | criterion |
|---|---|
| F | shared R² within (goal, length) ≥ 0.5 |
| **G** shared map fits | goal-specific R² − shared R² (within) ≤ 0.03 |
| C1 transfer | S(encoding) ≥ 0.75, gain ≥ 0.25 |
| **C2 goal-dependent** | every change goal right: encoding ≥ 0.75 × whole, above rotated |
| C3 preservation | harm ≤ 0.05 |
| C4 beyond one-step | gain ≥ 0.5 × whole's, above rotated |
| C5 specificity | (a) above rotated and random, rotated's gain ≤ half; (b) D(pca) − D(encoding) ≤ 0.05 |
| C6 equivalent histories | all four ≥ 0.75 × whole; agreement ≥ none − 0.05 |
| **C7** shared map suffices | D(encoding) ≥ D(enc_goal) − 0.05 |
| **U** upstream gain (paired with round 32, per model) | the shared edit's goal-dependent ratio (encoding / whole) is higher here, and the gap D(enc_goal) − D(encoding) is smaller here; both p < 0.05 |

| result | reading |
|---|---|
| F, G, C1, C2, C7 | **a shared belief variable at block 0's attention output; round 32's goal-specificity is created by block 0's MLP** |
| U, but G or C7 fails | moving upstream reduces the goal-specificity, but part of it is already in block 0's attention output |
| U fails | no gain upstream: the goal-specificity is already at block 0's attention output |
| F fails | limited by fit at this site |

## Expectations

| | expectation |
|---|---|
| E1 | F holds |
| E2 | G fails, but the gap is smaller than round 32's (0.12): between 0.03 and 0.08 |
| E3 | interaction smaller at m than at round 32's site, in every model |
| E4 | check: none, whole and hybrid equal round 32's (donor-optimal within 10⁻³) |
| E5 | U holds |
| E6 | C1 holds |
| E7 | C2 fails, but the goal-dependent ratio is above round 32's 0.36 |
| E8 | C3 holds |
| E9 | C5b fails (PCA above encoding by > 0.05) |
| E10 | propagation: the induced change is closer to round 32's goal-specific vector than to its shared one |

## Limits known in advance

The same as round 32's. In addition: m is before a layer norm, so an edit's effect on MLP₀ depends on its norm relative
to the state's.
