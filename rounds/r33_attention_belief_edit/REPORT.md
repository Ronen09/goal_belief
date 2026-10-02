# The belief edit one step earlier: before block 0's MLP (round 33)

Run date: 2026-10-02. Brief: `BRIEF.md` (round 32's proposed next step). Site, added measures and decision rule:
`PLAN.md`, committed (`f46882a`) **before any encoding was fitted on a trained model or any edit applied to one**.
Numbers from `tables.md`; data in `results.json`. Reproduce: `reproduce.py r33` (2 min on one GPU).

**Setting.** Round 32's design exactly (same models, encodings, edits, pairs, measures), with one change: the site. Here
it is m(h, g), the goal token's state after block 0's attention and before its MLP. That is the goal embedding, a
function of (goal, prefix length) only, plus block 0's attention output; the c_{g,L} offset absorbs the embedding.
Round 32's site is z = m + MLP₀(m). The no-edit, whole-replacement and hybrid runs are therefore the same runs in both
rounds. They match exactly (largest difference 0), which checks that the two rounds are comparable cell by cell.

## 1. Before block 0's MLP, one belief map serves all three goals

| held-out, within (goal, length) | round 32's site (after block 0's MLP) | **this site (before it)** |
|---|---|---|
| shared map c_{g,L} + E b: R² | 0.555 | **0.649** (0.617–0.685) |
| goal-specific maps E_g: R² | 0.674 | 0.672 |
| goal-specific − shared | 0.114 | **0.023** (0.012–0.041) |
| goal-specific maps' distance from the shared one | 13–28 % | 3–4 % |
| goal × history interaction share | 0.166 | **0.045** (0.028–0.070) |

* **G holds**: a different map per goal adds only 0.02 R² here, against 0.11 after the MLP.
* **The goal hardly interacts with the history at this site.** For a fixed history, only 4.5 % of the state's
  variation changes with the goal beyond a goal offset, against 17 % after the MLP. It is smaller in every model.
* Goal-specific maps fit equally well at the two sites (0.672 / 0.674); only the shared one gains. The belief content
  is the same; what changes is whether the goal has been mixed into it.

## 2. The shared edit here does almost all the route can do

Main pairs, change cells; median over ten models:

| edit | donor-optimal | S (of whole) | round 32's site: donor-optimal / S |
|---|---|---|---|
| none | 0.108 | 0 | same |
| whole | 0.595 | 1 | same |
| **shared encoding** | **0.540** (0.447–0.618) | **0.90** (0.83–0.97) | 0.424 / 0.66 |
| goal-specific encoding | 0.570 | 0.95 | 0.563 / 0.94 |
| rotated (same norms) | 0.115 | 0.02 | 0.109 / 0.00 |
| PCA-13 of the donor's state | 0.589 | 0.99 | 0.585 / 0.98 |
| hybrid | 0.812 | | same |

* **The same E (b_B − b_A), under all three goals, carries 0.90 of the direct route's effect** (C1 holds). It raises
  donor-optimal actions from 0.11 to 0.54 using only the donor's posterior.
* **It is within 0.02 of the goal-specific edit** (C7 holds; round 32: 0.12) and within 0.05 of the PCA-13 patch of the
  donor's whole state (C5b holds, just; round 32: 0.15).
* Direction-specific as before: rotated 0.02 of whole, random rank-13 0.02 (p < 0.001, all models).
* Per goal, shared: 0.55 / 0.50 / 0.69 for G1 / G2 / G3. Whole: 0.59 / 0.60 / 0.69.

## 3. The decisive cases

| | shared encoding | goal-specific | whole | round 32, shared |
|---|---|---|---|---|
| goal-dependent pairs: right under every change goal | **0.177** (0.61 × whole) | 0.205 (0.79 ×, post hoc) | 0.276 | 0.105 (0.36 ×) |
| preserve cells: harm | **0.006** | 0.010 | 0.003 | 0.001 |
| one-step agreeing: S (of whole) | **1.08** | 1.09 | 1 | 0.85 |
| equivalent recipients: all four donor-optimal | **0.445** (0.93 × whole) | 0.467 | 0.473 | 0.347 (0.71 ×) |
| equivalent recipients: agreement − none | −0.097 | −0.077 | −0.030 | −0.091 |

* **Goal-dependent pairs ("left under G1, right under G2"): much improved, short of the criterion.** The same edit
  gets B's different actions right under both goals in 0.18 of pairs, 0.61 of whole (registered 0.75; round 32 0.36).
  The goal-specific linear edit reaches 0.79 of whole here, above the 0.75 line (post hoc). Measured against
  whole, the shared edit falls 0.39 short and the goal-specific one 0.21. So about half of the shared edit's shortfall
  is the cost of one map for all goals, and half is common to both linear edits.
  *Corrected 2026-10-02, after round 34: this bullet first said that the goal-specific edit "also falls short" and
  that sharing cost "a smaller part (0.03 absolute)". 0.79 is above 0.75, and 0.03 absolute is 0.1 of whole's rate
  on these pairs. Round 34 confirms that the goal × belief interaction is what the policy reads here.*
* **Preservation**: harm 0.006 (C3 holds).
* **Beyond one-step prediction**: where one-step predictions are identical, the shared edit does as well as whole
  replacement (S 1.08; C4 holds).
* **Equivalent histories**: on four histories with the same posterior and different tokens, the edit makes all four
  donor-optimal 0.93 as often as whole replacement does (round 32: 0.71). Agreement among them still falls by 0.10, so
  C6 fails on its second part. As noted in round 32, a partial edit lowers agreement mechanically: the no-edit
  baseline agrees because all four keep the recipient's action.

## 4. Where the goal enters: propagation through block 0's MLP

The shared edit at m, carried through block 0's MLP, induces a change at round 32's site:

| induced change against | cosine |
|---|---|
| round 32's **goal-specific** edit vector | **0.92** (norm 1.03 ×) |
| round 32's shared edit vector | 0.87 |
| the actual difference z(B,g) − z(A,g) | 0.76 (round 32's own shared edit: 0.69) |

**Block 0's MLP turns one shared belief change into the goal-specific one.** The same input vector, under each goal,
comes out aligned with that goal's own belief map at the next site (0.92). It matches the actual state difference
better than any linear edit made after the MLP.

## 5. Decision rule

| | criterion | value | held |
|---|---|---|---|
| F | shared R² within ≥ 0.5 | 0.649 | **yes** |
| **G** | goal-specific − shared R² ≤ 0.03 | 0.023 | **yes** |
| C1 | S ≥ 0.75, gain ≥ 0.25 | 0.90; 0.44 | **yes** |
| **C2** | goal-dependent: ≥ 0.75 × whole, above rotated | 0.61 ×; p < 0.001 | no |
| C3 | harm ≤ 0.05 | 0.006 | **yes** |
| C4 | one-step agreeing: ≥ 0.5 × whole, above rotated | 1.08 ×; p < 0.001 | **yes** |
| C5a | above rotated and random; rotated ≤ half | p < 0.001, < 0.001 | **yes** |
| C5b | D(pca) − D(encoding) ≤ 0.05 | 0.048 | **yes** |
| C6 | equivalent: ≥ 0.75 × whole; agreement ≥ none − 0.05 | 0.93 ×; −0.097 | no (agreement) |
| **C7** | D(encoding) ≥ D(enc_goal) − 0.05 | −0.018 | **yes** |
| **U** | goal-dependent ratio higher, enc_goal gap smaller than round 32 | 0.61 against 0.36, p 0.002; 0.018 against 0.122, p < 0.001 | **yes** |

**Reading.** The registered table's first row needs F, G, C1, C2 and C7. All hold except C2. Its second row needs U with
G or C7 failing; U holds, but G and C7 do not fail. The plan has no row for this combination, and I did not add one.
Stated directly: **a shared belief variable sits at block 0's attention output, and it is turned goal-specific by block 0's
MLP (G, C7, U, propagation). On the goal-dependent pairs, a shared linear edit at this site reaches 0.61 of whole.**
That is short of the registered 0.75, and the goal-specific edit is short there too (0.79).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | F holds | yes | 0.649 |
| E2 | G fails, gap between 0.03 and 0.08 | **no**: G holds | 0.023 |
| E3 | interaction smaller here, every model | yes | 0.045 against 0.166 |
| E4 | none, whole, hybrid equal round 32's | yes | identical |
| E5 | U holds | yes | p 0.002, < 0.001 |
| E6 | C1 holds | yes | 0.90 |
| E7 | C2 fails, ratio above 0.36 | yes | 0.61 |
| E8 | C3 holds | yes | 0.006 |
| E9 | C5b fails | **no**: holds, by 0.002 | 0.048 |
| E10 | propagation closer to round 32's goal-specific vector than its shared one | yes | 0.92 against 0.87 |

8 of 10. Both misses were in the same direction: the encoding at this site is more shared than I expected. Round 30's
goal-sensitive block 0 heads had led me to expect goal-specificity in the attention output.

## Conclusions

1. **The goal-specificity found in round 32 is created by block 0's MLP.** Before it, one map from the posterior fits
   as well as three (within 0.02 R²), and the goal barely interacts with the history (4.5 % of the variation). After
   it, a different map per goal is needed (0.11 R²), and the interaction is 17 %.
2. **At block 0's attention output, a shared belief edit controls the native policy.** The same vector E (b_B − b_A)
   under every goal carries 0.90 of what the whole route carries. It does not harm decisions that should stay (0.006).
   It works fully where one-step predictions agree (1.08), and works across histories with the same posterior (0.93).
   Rotated directions of the same norm do nothing. It is within 0.05 of a generic patch of the donor's whole state of
   the same rank.
3. **Block 0's MLP is the evidence–goal combiner.** A shared belief change entering it comes out as the goal-specific
   change (cosine 0.92 with that goal's own map). With rounds 29–31, the goal token's computation reads:
   attention gathers a goal-free belief code → block 0's MLP combines it with the goal → blocks 1–3's MLPs turn that
   into an action preference.
4. **What remains open: goal-dependent pairs.** Where one belief change must give different actions under two goals,
   the shared edit reaches 0.61 of whole replacement and the goal-specific one 0.79 (post hoc). The small goal × history
   interaction left at this site (4.5 %) matters for exactly these pairs. (Corrected after round 34; see §3.)

Next, if wanted: on goal-dependent pairs, measure what the whole-replacement difference contains beyond E Δb at this
site, with a nonlinear (e.g. small MLP) encoding of the posterior. Or test whether adding round 28's interface edit
closes the gap, since together the two routes carry the full hybrid.

Limits: as round 32's (one route; linear encodings; first decision; reward models). The site is before a layer norm, so
edits act on MLP₀ through their norm relative to the state. The goal-specific-encoding ratios on goal-dependent pairs in
§3 are post hoc.
