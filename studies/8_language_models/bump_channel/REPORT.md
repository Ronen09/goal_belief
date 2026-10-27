# Bump channel: the belief reaches the decision through the goal's four directions

Run date: 2026-10-27. Models, sites, pairs and decision rule: `PLAN.md`, committed (`88e6f4d`) **before any measurement
was run**. Numbers from `tables.md`; data in `subspace_<kind>.json` (part 0), `channel_<kind>.json` (parts 1–3), logs
in `logs/`. Reproduce: `reproduce.py bump_channel` (about 15 min on one GPU, given the goal study's models and the
coordinates study's recorded states; the plan's 2–3 h estimate was wrong by an order of magnitude).

**Disclosures.** (1) Every script was smoke-tested on 16 pairs (`--quick`, outputs under `_smoke/`, not used) before
the run; the smoke found that cuDNN's GRU backward refuses eval mode, so the small models' Jacobians run with cuDNN
disabled (numerically the same kernels). (2) The registered rank is the median over the twelve goal pairs of the
pooled subspace's causal rank, rounded **up** to the grid, as the plan fixed it; for the fine-tuned model that is 4 at
23 and 8 at 21 from `../bump_coords/rank.json`; at 19 and 15 the plan's 64 and 128 are that file's per-pair medians,
not the pooled ones (128 at both), so the registered rule would give 128 at 19, where the sweep reads 0.82 instead
of 0.77: not a decision-bearing site, and the plan's numbers were kept. For the from-scratch models part 0's numbers. (3) A scorecard whose registered rank reaches half the state's width is flagged vacuous in `channel.py`; no
model triggered it (ranks 4, 16, 32). (4) The upstream-swap part is decision-bearing for the fine-tuned model only, as
the plan says: in the from-scratch models the goal is the start token and every state carries it; accordingly their
COMP and CTRL are judged on part 2 alone and part 1's values are reported beside them (under a "both parts" reading
the transformer's COMP would fail on part 1: 0.31, 0.25–0.37). (5) The run took 12 minutes, not the plan's 2–3 h.

**The question.** The coordinates study found that the goal's effect on the fine-tuned model's decision is read from
four directions of the decision token's 896-dimensional residual at layer 23 (one pooled subspace, 0.935 of the full
effect for every pair), and the goal-swap study that the belief arrives in the token at layers 13–19 and the goal at
20–23. The hypothesis, in the author's words: the belief set combines with the goal to output a four-dimensional
subspace, so intervening on that subspace should change actions. The test: two held-out histories at the same step
with different belief sets under the same goal; the difference between their states (or the change that one's
belief induces when swapped into the other upstream of the goal) restricted to the goal's directions, to their
complement, to bottom, random, the logit readout's and the belief effect's own directions; the share of the full
swap's decision switches each reproduces. The same on the from-scratch four-goal transformer and GRU (120 000
episodes), whose goal subspaces part 0 finds first.

## 0. The from-scratch models' goal subspaces are not four-dimensional

The coordinates study's rank analysis on `tfm_120000` and `gru_120000` (1 979 fit histories, 660 test histories, the
pool of the coordinates study; the causal sweep on the test histories' goal-dependent decisions):

| model | site | natural goal swap | full map | pooled causal rank per pair | registered k | pooled rank-k / full |
|---|---|---|---|---|---|---|
| transformer | block 1 / 2 / 3 entrance | 0.19 / 0.27 / 0.48 | 0.18 / 0.25 / 0.40 | 1–16 / 4–16 / 8–64 | 4 / 4 / **16** | 1.00 / 0.94 / 0.92 |
| transformer | final residual (linear) | 1.00 | 0.61 | 2 in every pair | 2 | 1.02 |
| GRU | h_t−2 / h_t−1 | 1.00 / 1.00 | 0.76 / 0.72 | 32–128 / 16–128 | 64 / **32** | 0.87 / 0.93 |
| GRU | h_t (linear) | 1.00 | 0.65 | 2–4 | 2 | 0.91 |

The registered sites (the non-linear site with the largest natural goal swap) are the transformer's block 3 entrance
and the GRU's h_t−1. At the linear sites the causal rank is 2, the rank of the head's centred rows, by construction.
Away from them the goal's effect occupies 16 of 128 directions in the transformer and 32 of 128 in the GRU: an eighth
and a quarter of the state, where the fine-tuned model's four are one in 224. E7 (median k_c ≤ 8) fails for the GRU
and, after the rounding, for the transformer. The small models' states are dense, and the "subspace" test on them is
a test of a large subspace.

## 1. In the fine-tuned model the goal's four directions carry the belief's effect

512 same-step pairs of held-out histories with different goal-free sets, each under the four goals; 999 of the 2 048
cells are history-dependent (the two histories' natural decisions differ), 388 of them at known states.

**Part 2, the natural history difference** x(h2) − x(h1) at each site, restricted and installed in h1's run:

| site | k | natural swap | goal subspace | complement | bottom-k | random-k | readout top-k | history top-k | floor | ‖Δ‖² share in the goal subspace |
|---|---|---|---|---|---|---|---|---|---|---|
| layer 15 | 128 | 0.368 | 0.62 | 0.15 | 0.03 | 0.10–0.15 | 0.63 | 0.93 | 0.23 | 0.59 |
| layer 19 | 64 | 0.650 | 0.77 | 0.12 | 0.02 | 0.05–0.06 | 0.87 | 0.94 | 0.09 | 0.50 |
| layer 21 | 8 | 0.920 | **0.92** (0.90–0.94) | 0.07 | 0.01 | 0.01–0.02 | 0.85 | 0.93 | 0.05 | 0.40 |
| layer 23 | 4 | 0.972 | **0.94** (0.92–0.96) | 0.07 (0.05–0.09) | 0.00 | 0.00–0.01 | 0.92 | 0.95 | 0.05 | 0.37 |

* **THRU-D holds at 23 and 21.** Four directions of 896 carry 0.94 of what the whole difference carries, eight at 21
  carry 0.92; the complement, 0.63 of the difference's squared norm, moves 0.07 of the decisions the full difference
  moves, the bottom four nothing, four random directions nothing, a random vector of the same norm 0.05.
* **The sweep is the goal's own.** Through 1 / 2 / 4 / 8 directions at 23: 0.34 / 0.65 / 0.94 / 0.96; the
  coordinates study's goal sweep at the same site was 0.35 / 0.56 / 0.73 of a full 0.79, that is 0.44 / 0.71 / 0.92.
  The belief's effect needs the same two-to-four directions as the goal's.
* **Downstream of the goal only.** At 19 and 15, where the registered subspace has 64 and 128 directions, the
  through ratio is 0.77 and 0.62 and random directions of the same count already give 0.05–0.15: the goal subspace
  there is diffuse and the belief's effect is not yet in it.

**Part 1, the belief swapped in upstream** (h2's state installed at 15 or 19 in h1's run, the change it induces at
21 or 23 restricted the same way; the reference is that upstream swap's own switch):

| upstream → downstream | upstream swap | goal subspace | complement | bottom-k | random-k | readout | history |
|---|---|---|---|---|---|---|---|
| 15 → 21 | 0.369 | 0.83 (0.78–0.89) | 0.14 | 0.01 | 0.01–0.02 | 0.67 | 0.92 |
| 15 → 23 | 0.369 | 0.84 (0.78–0.90) | 0.13 | 0.01 | 0.01 | 0.82 | 0.89 |
| 19 → 21 | 0.646 | 0.88 (0.85–0.91) | 0.11 | 0.01 | 0.01 | 0.82 | 0.94 |
| 19 → 23 | 0.646 | **0.91** (0.88–0.94) | 0.09 (0.06–0.12) | 0.01 | 0.00–0.01 | 0.91 | 0.93 |

* **THRU-U holds.** A belief installed at 19, before the goal is read, and re-read by layers 20–22 under h1's goal
  and context changes the decision at 23 through the four goal directions in 0.91 of the cases the swap changes it;
  installed at 15 (where the swap itself works in only 0.37 of cells) 0.84. The induced change puts 0.23 of its
  squared norm into the four directions and 0.77 outside, and the outside part moves 0.09.
* The full induced change re-installed at the downstream site reproduces the upstream swap in 1.00 of cells: the
  patch touches the decision position only, so the measurement is exact.

**What the four directions are.** The belief effect's own top-4 directions (the singular directions of the
centred history differences over fit pairs) overlap the goal's at 0.76 (random 0.004) and carry 0.95; the logit
readout's top-4 directions (from the Jacobians of the centred logits at 23 over 2 048 fit decisions) overlap the
goal's at 0.20 (random 0.004) and carry 0.92. **DISTINCT holds** by its threshold, and the goal subspace and the
readout's principal directions are different four-dimensional objects; but two subspaces that share a fifth of
their content both carry over nine tenths of the effect. A switch needs only the logit pushed past the decision
boundary, so two saturated ratios do not locate the effect in the subspaces' common part; what they show is that
neither subspace is necessary (the direct test, the difference restricted to the goal subspace's part outside the
readout's, was not registered and is left as a post hoc). The description the numbers support is a shared decision
subspace: a few directions the last layer is sensitive to, into which the goal layers write both the goal and the
belief's consequence for the action, rather than a channel that belongs to the goal.

**Part 3, known and uncertain states.** At 23 the through ratio is 0.94 at known states (388 cells) and 0.94 at
uncertain ones (611); at 21 0.95 and 0.91 (difference −0.00 to 0.08); in part 1 0.93 and 0.90. **KNOWN fails**: the
conversion is as complete where the belief is a singleton as where it is a set, and E6's picture (the
information-seeking part of the history's effect passing outside the goal's directions) is not seen.

## 2. The from-scratch models: the same in a large subspace

**Transformer** (779 history-dependent cells; registered: block 3 entrance, k = 16):

| site | k | natural swap | goal subspace | complement | bottom-k | random-k | readout | history | ‖Δ‖² share goal |
|---|---|---|---|---|---|---|---|---|---|
| block 1 entrance | 4 | 0.321 | 0.32 | 0.94 | 0.02 | 0.04–0.11 | 0.74 | 0.91 | 0.30 |
| block 2 entrance | 4 | 0.504 | 0.46 | 0.70 | 0.02 | 0.05–0.09 | 0.70 | 0.83 | 0.28 |
| block 3 entrance | 16 | 0.647 | **0.83** (0.76–0.88) | 0.24 (0.19–0.29) | 0.03 | 0.11–0.20 | 0.94 | 0.94 | 0.63 |
| final residual (linear) | 2 | 1.000 | 0.93 | 0.04 | 0.00 | 0.02–0.05 | 0.96 | 0.83 | 0.51 |

THRU-D holds at 0.83 (expected 0.65–0.7), upstream swaps from block 2 into block 3 through the same subspace 0.83.
But the subspace is 16 of 128 directions and holds 0.63 of the difference's norm; the complement still moves 0.24
(0.19–0.29; 0.31 in part 1, reported) and 16 random directions 0.11–0.24: COMP holds on part 2 by its interval while
**CTRL fails**: the belief's effect is concentrated in the goal's directions, not confined to them. At the first two entrances, where the goal's own effect is small
(natural goal swap 0.19 and 0.27), the four goal directions carry 0.32 and 0.46 of the belief effect and the
complement 0.94 and 0.70: before the goal is written in, the belief acts elsewhere, as in the fine-tuned model at
15. DISTINCT holds (0.34 at block 3; 0.82 at the final residual, the readout by construction). KNOWN fails at the
registered site (0.83 against 0.82); at the first two entrances the ratio is higher at known states (0.47 against
0.24, 0.58 against 0.38, intervals excluding 0), where the subspace is weak.

**GRU** (791 cells; registered: h_t−1, k = 32):

| site | k | natural swap | goal subspace | complement | bottom-k | random-k | readout | history | ‖Δ‖² share goal |
|---|---|---|---|---|---|---|---|---|---|
| h_t−2 | 64 | 0.430 | 0.92 | 0.11 | 0.11 | 0.47–0.61 | 0.99 | 0.97 | 0.82 |
| h_t−1 | 32 | 0.649 | **0.89** (0.85–0.93) | 0.13 (0.09–0.18) | 0.03 | 0.16–0.27 | 0.98 | 0.93 | 0.67 |
| h_t (linear) | 2 | 1.000 | 0.64 | 0.28 | 0.00 | 0.01–0.02 | 1.00 | 0.52 | 0.20 |

THRU-D holds at 0.89 (the author's 0.8); the upstream swap from h_t−2 passes through the same 32 directions at
0.89; COMP holds (0.13, 0.19). CTRL fails: 32 random directions of 128 carry 0.16–0.27 (0.36 in part 1), a quarter
of the state being a quarter of the effect. At the decision state, through the linear head, the goal's two
directions carry 0.64 of the belief effect and the belief's own two 0.52, and the two pairs overlap at 0.16 (random
0.01): at the point of readout the GRU moves the decision along different directions for a change of goal and for
a change of history (the goal's two overlap the head's two rows at 0.50, the history's at 0.32; the readout
projection itself carries 1.00, by construction). The sweep at h_t−1 reaches 0.58 at eight
directions and 0.89 at 32: no four-direction channel exists in the GRU.

## Post hoc (after the review, not registered): the goal subspace outside the readout

Proposed in conversation after the report, to settle whether the goal subspace C carries the belief's effect
through its own directions or through its overlap with the logit readout R (the top-k_R right-singular directions
of the Jacobian of the centred logits): C⊥R = orth((I − P_R) C), the part of C outside R, against C∥R = orth(P_R C),
its projection into R, R itself, C⊥R ∪ R, and random subspaces of C⊥R's dimension anywhere and inside R's
complement (`posthoc_readout.py`; the same pairs, cells and bootstraps as the main run):

| part | site | k, k_R | overlap C / R | C | R | C⊥R | C∥R | C⊥R ∪ R | random (dim of C⊥R) | ‖Δ‖² share: C⊥R / C∥R |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 23 | 4, 4 | 0.20 | 0.94 | 0.92 | **0.06** (0.05–0.08) | **0.92** (0.90–0.94) | 0.98 | 0.00–0.01 | 0.27 / 0.10 |
| 2 | 23 | 4, 8 | 0.29 | 0.94 | 0.97 | 0.03 | 0.95 | 0.99 | 0.00–0.01 | 0.23 / 0.15 |
| 2 | 23 | 4, 16 | 0.34 | 0.94 | 0.98 | 0.02 | 0.95 | 1.00 | 0.00–0.01 | 0.20 / 0.17 |
| 2 | 21 | 8, 4 / 8 / 16 | 0.11–0.17 | 0.92 | 0.81 / 0.84 / 0.91 | 0.12 / 0.10 / 0.07 | 0.81 / 0.84 / 0.90 | 0.94–0.97 | 0.00–0.01 | 0.30 / 0.11 |
| 1 (19 →) | 23 | 4, 4 / 8 / 16 | 0.20–0.34 | 0.91 | 0.91 / 0.97 / 0.97 | 0.07 / 0.02 / 0.01 | 0.91 / 0.93 / 0.94 | 0.97–0.99 | 0.00–0.01 | 0.17 / 0.07 |

The goal subspace's part outside the readout carries nothing: four directions holding 0.27 of the difference's
squared norm switch 0.06 of what the whole difference switches, as four random directions do (0.01), and the share
falls further as the readout is taken wider (0.03, 0.02). Its projection into the readout, 0.10 of the norm, carries
all of the effect the subspace carries (0.92 against 0.94), as much as the readout's own directions (0.92); the two
together 0.98. The same for the belief swapped in at 19 and, with eight directions, at layer 21 (C⊥R 0.12 → 0.07,
C∥R 0.81 → 0.90). With one block between the site and the logits the readout's directions are, to first order, the
only ones a change can act through, so the finding is less that the readout wins than that nothing else does: the
block after layer 23 does not turn the goal subspace's other directions into a decision. The four goal directions
matter through their projection onto the directions the last layer reads, and the belief's effect reaches the
decision through the same projection; "combined into one subspace" is true of the readout's input coordinates,
which the goal layers fill with the belief's consequence under the goal.

## Scorecard (the registered site of each model)

| | criterion | fine-tuned 0.5B, layer 23, k = 4 | transformer, block 3, k = 16 | GRU, h_t−1, k = 32 |
|---|---|---|---|---|
| THRU-D | the natural difference through the goal subspace ≥ 0.65 × the natural swap | 0.94 (0.92–0.96): **yes** | 0.83 (0.76–0.88): **yes** | 0.89 (0.85–0.93): **yes** |
| THRU-U | the upstream swap's induced change through the goal subspace ≥ 0.65 (fine-tuned model) | 0.91 (0.88–0.94): **yes** | reported: 0.83 | reported: 0.89 |
| COMP | the complement ≤ 0.35 (both parts for the fine-tuned model; part 2 for the others, part 1 reported) | 0.07, 0.09: yes | 0.24 (0.19–0.29): yes; part 1 0.31 | 0.13 (0.09–0.18): yes; part 1 0.19 |
| CTRL | bottom-k and every random-k ≤ 0.20 (the same parts) | 0.00–0.01: yes | random up to 0.20 (0.24 in part 1): no | random up to 0.27 (0.36 in part 1): no |
| DISTINCT | goal / readout overlap ≤ 0.50 | 0.20: yes | 0.34: yes | 0.44: yes |
| KNOWN | higher at known states | 0.94 / 0.94: no | 0.83 / 0.82: no | 0.92 / 0.87 (−0.03–0.14): no |
| rejected (< 0.40) | | no | no | no |

| | expectation | held |
|---|---|---|
| E1 | THRU-D for every model: GRU ≈ 0.8, the others 0.65–0.7; the sweep saturates by k = 8 (fine-tuned) / 16 (small) | yes, above every expectation (0.94 / 0.83 / 0.89); the fine-tuned sweep saturates at 4; the small models' only at 32–64 |
| E2 | THRU-U at THRU-D's level | yes: 0.91 against 0.94 |
| E3 | COMP and CTRL at every registered site | COMP yes everywhere, CTRL for the fine-tuned model only: the small models' subspaces are an eighth and a quarter of the state and random subspaces of that size carry 0.1–0.4 |
| E4 | DISTINCT at 23 / 21 (0.2–0.5) and at the transformer's entrance; ≥ 0.8 at the linear sites | 0.20 / 0.13 and 0.34: yes; the transformer's final residual 0.82: yes; the GRU's h_t 0.50: no (the goal's top-2 at the decision state are half the head's rows) |
| E5 | history / goal overlap ≥ 0.5 at the registered sites | yes: 0.76 / 0.58 / 0.65 (random 0.004 / 0.12 / 0.25) |
| E6 | KNOWN: the conversion is complete at known states and partial at uncertain ones | no: equal everywhere at the registered sites |
| E7 | the from-scratch goal subspaces low-rank (median k_c ≤ 8) | no: 16 and 32 |

5 of 6 criteria hold for the fine-tuned model (KNOWN fails), 3 of their 5 for the transformer and the GRU (THRU-D,
COMP, DISTINCT); no model is rejected.

## Reading

In the fine-tuned model the hypothesis holds in its strong form. The four directions of layer 23 that the
coordinates study found to carry the goal's effect on the decision carry the belief's effect as well: the
difference between two histories' states restricted to them switches 0.94 of what the whole difference switches,
a belief installed before the goal is read and re-read by the goal layers 0.91, while the rest of the state, two
thirds of the change's norm, switches 0.07 and random or bottom directions nothing, and the belief effect's own
principal directions are the goal's (overlap 0.76 against 0.004). Two or four directions are enough for both, in
the same proportion (0.65 at two, 0.94 at four). The goal layers do not add a goal term to a belief code; they write
the consequence of the belief under the goal into the few directions the last layer reads, the same ones under every
goal and every belief. Those directions are not the readout's principal axes (overlap 0.20), but the post hoc below shows that only
their projection into the readout's directions carries the effect (0.92 of the 0.94; the part of the goal subspace
outside the readout, 0.27 of the change's norm, carries 0.06), so the better name for the subspace is the decision's
input: the model's last computation reads a few coordinates linearly, and belief and goal are both converted into
them by layer 21. Known and
uncertain states convert alike.

In the from-scratch models the same picture holds at lower resolution. Their goal subspaces are an eighth (the
transformer) and a quarter (the GRU) of a 128-dimensional state, and through those the belief's effect passes at
0.83 and 0.89 while the complement moves 0.1–0.3 and random subspaces of the same size 0.1–0.4: the belief's effect
is concentrated where the goal's is, not confined to a few directions, because the goal's own effect is not
confined to a few directions there. The GRU's decision state is the exception that shows the convergence is a fact
about the layers before the readout: at h_t, through the linear head, the goal moves the decision along two
directions and the history along two others (overlap 0.16), and the goal's two carry 0.64 of the history's effect.
One token earlier, where the GRU cell still has to act, the two are already in the same 32 directions.

## Limits

One seed per model; linear subspaces of one site's state, fitted on the optimum's trajectories, tested on held-out
histories of the same pool (the coordinates study found these subspaces do not transfer to shortest histories); the
registered rank is a rounded median over goal pairs, so the small models' subspaces are coarse and their COMP and
CTRL failures partly reflect their size; the readout subspace is a pooled local linearisation; the through ratio is
a ratio of switch rates over history-dependent cells, 999 / 779 / 791 of 2 048, and known states have fewer of them;
the upstream swap at 15 switches only 0.37 of cells, so its through ratios rest on about 370 switches; part 0 was
registered as a procedure, not a prediction, and its ranks are this run's.
