# Navigate, investigate, commit: how a full-attention transformer comes to use a belief (round 17)

Run date: 2026-09-29. Brief: `BRIEF.md`. Frozen environment and solver validation: `SPEC.md`. Design, pilot and 15
predictions: `THEORY.md`, committed (`0fba5e1`) after the exploratory pilot and **before the main runs**. Numbers
from `tables.md`; per-seed data in `runs/<condition>/seed*/{log,measures,qgen}.json`; worked traces in `traces.md`.
Reproduce: `reproduce.py r17` (about 4 h on one GPU), `reproduce.py r17 --tables` for the tables and figures.

## Setup in brief

5 × 5 grid, hidden reward in one corner, two clue stations (left/right and top/bottom, reliability q), actions
UP / DOWN / LEFT / RIGHT / QUERY / COMMIT. The brief's defaults (H = 20, c = 0.02) make two queries optimal in every
case, so the task was retuned with the exact solver to **H = 12, c = 0.025**: zero, one and two queries are optimal
in 19 %, 23 % and 58 % of layout × start cases, and in 38 % the number depends on a report. The optimal return is
0.311; never querying loses 0.101.

Model: 4 layers, width 128, 4 heads, ordinary full causal attention. Two station tokens, then a decision token
and an event token per step; a report appears once, in its event token. Primary condition: PPO on episodic reward
only, 5 seeds, 3 000 updates (1.0–1.1 × 10⁸ interactions each). All representation measurements are made on a bank
of 72 000 fixed histories generated without any model and replayed at every checkpoint.

**Deviations from the registration.** (1) TF32 matrix multiplication was switched on for the main runs (the pilot
ran without). (2) The dense checkpoints were placed at 600–1 800 updates from the pilot's on-policy transition.
The change in how evidence is used on fixed histories happens at 80–270 updates, where there are only the
log-spaced checkpoints (82, 110, 149, 201, 272). (3) Patches and Adam-trained decoders were run at 9 checkpoints,
everything else at all 33.

## 1. Behaviour: two transitions, and only the first is about the evidence

| update | greedy regret (5 seeds) | queries / episode | use of evidence on fixed histories |
|---|---|---|---|
| 0 | 0.17–0.61 | 0.12 | 0.00 |
| 110 | 0.102–0.106 | 0.00 | 0.06 |
| 201 | 0.0855–0.0861 | 0.16 | 0.67 |
| 903 | 0.077–0.083 | 0.21 | 0.82 |
| 1 500 | 0.015–0.058 | 0.98 | 0.87 |
| 3 000 | **0.009–0.013** | 1.31 (optimal: 1.25) | 0.93 |

* **First transition, updates 80–270.** The models learn to query a station they happen to start on (0.16 queries
  per episode is exactly the chance of starting on one) and to act on the report. On the fixed histories, use of
  evidence (probability of the actions that are optimal given the reports, minus that of the actions that would be
  optimal had a report been the other one) goes from 0.01 to 0.74. On-policy regret falls only from 0.10 to 0.085,
  because the policy rarely has evidence.
* **Second transition, updates 1 000–1 800.** The models learn to walk to stations. Queries rise from 0.2 to 1.2
  per episode and regret falls to about 0.01. The regret lost to route errors falls from 0.06 to 0.008. Use of
  evidence on the fixed histories changes little (0.81 → 0.90).
* Most of the on-policy improvement is therefore a change in **which histories the policy visits**, made possible by
  a use of evidence that was in place 800 updates earlier. Measured on-policy only, the two would have been
  confounded (the last-but-one row of the brief's table).
* All 5 seeds end at greedy regret 0.009–0.013; residual regret is mostly route (0.008).

Controls. **Solver-supervised**, same network: greedy regret 0.0020, so the architecture and encoding are not the
limit. **Frozen backbone**, trained heads: regret 0.117, 0.00 queries, use 0.01. It does not reach even the
no-query plateau's level, although its first attention layer holds the marginals at R² 0.77 (below).

## 2. The belief is decodable before training, and training does not make it more so

Held-out affine decoding of P(right) and P(bottom) at decision tokens (mean R² of the two, mean of 5 seeds; layouts
held out from the fit):

| update | L0.pre | L0.mid | L1.pre | L1.mid | L2.pre | L3.pre | L3.post |
|---|---|---|---|---|---|---|---|
| 0 | 0.01 | **0.76** | 0.67 | 0.65 | 0.63 | 0.59 | 0.57 |
| 201 | 0.01 | 0.84 | 0.91 | **0.91** | 0.91 | 0.91 | 0.91 |
| 3 000 | 0.01 | 0.73 | 0.75 | 0.76 | 0.74 | 0.73 | 0.73 |

* **At initialisation** the raw tokens give 0.01 and the output of the first attention layer 0.76. Untrained
  attention averages the report embeddings into the decision token. Nothing acts on it: use is 0.000.
* **Decodability rises and then falls.** It peaks at 0.91 around the first transition and returns to the initial
  level while the policy keeps improving. Over checkpoints it does not track regret (Spearman +0.35 ± 0.32 across
  seeds; use: −0.97 ± 0.01).
* **The joint posterior is not affinely represented.** The posterior is the two marginals plus their product
  (`SPEC.md`). The product's R² is 0.00 at initialisation, 0.55 at the peak and 0.16–0.19 at the end (best seed and
  site: 0.30). The four goal probabilities decode at 0.69–0.72 (initialisation: 0.71 at L0.mid), and 2–3 % of
  decoded beliefs have a component outside [0, 1].
* **A 64-unit decoder** with the same optimiser budget reaches 0.80–0.86 for the marginals and at most 0.11 for the
  product (negative in 4 of 5 seeds). An affine decoder trained with that budget is worse than ridge (0.55–0.66),
  so the budget-matched numbers are lower bounds.

## 3. What changes upstream

The read-out is not the only thing that learns. The difference the first attention layer's output shows between
two histories that differ in one report, relative to the size of that output, grows from 0.05 at initialisation to
1.25 at update 200 and settles at 0.87. Through the block's shared decoder the attention layer's contribution
moves the decoded marginal 0.44 of the exact displacement at initialisation and 0.71 at the end, and the other
marginal by a tenth as much. After training a report is most of what the first attention layer writes; before, it
was 5 % of it.

Per head (`tables.md` §5, `traces.md`): in every seed two or three layer-0 heads carry it, with 0.26–0.55 of their
attention on report tokens and a decoded displacement of 0.05–0.11 per report in the reported coordinate (other
coordinate: ≤ 0.007). In the traced model different heads read RIGHT and LEFT reports.

## 4. Interventions

A component's output at the decision token is replaced by its value in a matched donor history. cf = gain in
probability of the actions that are optimal under the donor's evidence, as a share of the model's own gain when
given that evidence. Final checkpoint, layouts held out from component selection, mean ± sd over 5 seeds:

| component | report flipped: cf | same belief, other history | random, same norm | donor at another cell: to own cell with donor's belief | to donor's behaviour |
|---|---|---|---|---|---|
| L0 attention | **1.00 ± 0.01** | 0.00 | 0.01 | 0.12 ± 0.02 | **0.88 ± 0.02** |
| L0 MLP | 0.79 ± 0.17 | 0.00 | 0.01 | 0.12 ± 0.05 | 0.67 ± 0.17 |
| L1 attention | 0.35 ± 0.41 | 0.00 | 0.00 | 0.05 ± 0.05 | 0.27 ± 0.33 |
| L1 MLP | 0.41 ± 0.22 | 0.00 | 0.00 | 0.12 ± 0.06 | 0.34 ± 0.10 |
| L2, L3 attention | 0.00 | 0.00 | 0.00 | 0.00 | ≤ 0.01 |
| best single head (L0.H1) | 0.41 ± 0.19 | 0.00 | 0.00 | 0.19 ± 0.07 | 0.13 ± 0.09 |
| whole residual, final | 1.00 | 0.00 | 0.01 | 0.07 ± 0.03 | 1.00 |

* **Evidence reaches the decision through the first attention layer in every seed** (1.00), on discovery and
  held-out layouts alike. Same-belief donors and random moves of the same size do nothing (≤ 0.017 for every
  component in every seed).
* **Two solutions.** In seeds 0, 1 and 3 later attention carries nothing (L1: 0.00–0.05) and the layer-0 and
  layer-1 MLPs do the rest (0.87–0.96 and 0.45–0.65). In seeds 2 and 4 layer-1 attention also carries it (0.78,
  0.93) and the MLPs less (0.56–0.60, 0.13–0.17). Layer-1 attention is recruited late: its mean cf is 0.05 at
  update 900 and 0.38 at 1 500. This is why N8 failed.
* **No component holds the belief apart from the position.** With a donor at another cell, the first attention
  layer moves the recipient 0.88 of the way to what the model does *at the donor's cell* and 0.12 of the way to
  what it does at its own cell with the donor's belief. No head or sublayer passes the registered test for a
  position-free carrier. Single heads come closest (0.19 vs 0.13) but move little. Caveat: only 51–71 pairs per
  seed satisfy the separation criterion.
* Decoded belief moves with the action: the first attention layer's patch moves the decoded marginals 0.61 of the
  exact displacement and the other marginal 0.04.

## 5. Reliability

| model | trained pairs | held-out values (0.70, 0.90) | held-out combinations | (0.6, 0.6) | (0.95, 0.95) |
|---|---|---|---|---|---|
| PPO on the grid, 3 seeds | 0.0110 ± 0.0006 | 0.0125 ± 0.0007 | 0.0129 ± 0.0009 | 0.003 | 0.022 |
| supervised on the grid | 0.0065 | 0.0076 | 0.0065 | 0.000 | 0.027 |
| PPO at (0.8, 0.8) only, 5 seeds | 0.025 ± 0.002 | 0.021 ± 0.002 | 0.025 ± 0.003 | **0.061 ± 0.007** | 0.026 |

* Models trained over the grid do as well on reliabilities and combinations they never saw as on trained ones
  (difference ≤ 0.002), and affine decoding of the marginals transfers (held-out pairs 0.66, trained pairs 0.65).
  The nine-posterior lookup table of the fixed-q task is not what they learned.
* Models trained at 0.8 only behave as if every station were 0.8: at (0.6, 0.6), where the optimal policy almost
  never queries, they lose 0.061.
* Decodability in the grid models is 0.65, the same as at their initialisation (0.64).
* Scope: two independent binary clues. Non-factorising likelihoods were not tested.

## Registered predictions

| | prediction | held | numbers |
|---|---|---|---|
| N1 | reward training: greedy regret ≤ 0.02 in ≥ 4 of 5 seeds | **yes** | 0.0094–0.0129, 5 of 5 |
| N2 | supervised ≤ 0.005 and below every PPO seed | **yes** | 0.0020 |
| N3 | frozen backbone: regret ≥ 0.05, use ≤ 0.2 | **yes** | 0.117, 0.01 |
| N4 | marginals decodable at initialisation (≥ 0.6; raw ≤ 0.05) | **yes** | 0.763 ± 0.010; 0.012 |
| N5 | final − initial decodability within ±0.15; final < 0.9 | **yes** | −0.001; 0.763 ± 0.026 |
| N6 | product term R² < 0.4 at every site | **yes** | max 0.30 |
| N7 | use tracks regret (ρ ≤ −0.8), decodability does not (ρ > −0.5) | **yes** | −0.97; +0.35 |
| N8 | L0 attention cf ≥ 0.7, later attention ≤ 0.2 | no | L0 1.00; L1 0.78 and 0.93 in seeds 2, 4 |
| N9 | controls ≤ 0.05 | **yes** | ≤ 0.017 |
| N10 | other / own marginal ≤ 0.2 | **yes** | 0.110 ± 0.012 |
| N11 | no position-free carrier | **yes** | L0 attention to donor 0.88; none |
| N12 | L0 attention's twin difference grows ≥ 3× | **yes** | 0.053 → 0.868 |
| N13 | grid: trained ≤ 0.03; held-out within 0.01 | **yes** | 0.011; 0.0125, 0.0129 |
| N14 | decoding on held-out reliabilities ≥ trained − 0.1 | **yes** | 0.661 vs 0.648 |
| N15 | fixed-q models lose ≥ 0.02 more at (0.6, 0.6) | **yes** | 0.061 vs 0.003 |

14 of 15 held. Several thresholds were set from one pilot model (N4, N8, N11, N12), so their holding is a
replication over seeds more than a test. N5 held on its endpoints and missed the rise to 0.91 in between.

## Conclusions

1. **Row 2 of the brief's table, with row 3's qualification.** The marginals of the belief are affinely
   recoverable from the first attention layer of an untrained network (0.76), and no more recoverable at the end of
   training (0.76). What training builds is their use: 0.00 → 0.93, tracking regret at ρ = −0.97. A frozen backbone
   with the same recoverable marginals cannot use them.
2. **The upstream computation changes as well.** The brief asked for this to be tested when row 2 appears. The
   first attention layer's response to a report grows 16-fold relative to its output, and specific heads come to
   read report tokens. "Already available" describes what a decoder can find, not what the network was computing.
3. **The trained representation is organised for the action at the current cell.** It carries the two marginals
   but not their product, and no component transfers the evidence to another position without importing that
   position's behaviour. This is consistent with a policy-specific representation. It does not show that there is
   no reusable posterior: the decoders are affine or small, and the cross-position test has few pairs.
4. **Policy improvement has two sources, and they are 800 updates apart.** Acting on evidence that is present
   (updates 80–270), then going to get it (1 000–1 800). The second accounts for most of the on-policy regret and
   is a change in visitation, not in evidence processing. Fixed histories were needed to separate them.
5. **Reliability is used when it varies**, including values and combinations never trained; trained at one value,
   it is ignored.

Not claimed: that decoding shows use (at initialisation it does not), that optimal agents must share this
representation (two solutions appeared in five seeds), or that the computation is Bayesian beyond two independent
binary clues.

## Figures

`fig1_learning.png` regret, queries and error kinds along training · `fig2_layers.png` decodability by site and
checkpoint · `fig3_timeline.png` regret, use, decodability and the first attention layer's response on the fixed
histories · `fig4_causal.png` patches · `fig5_reliability.png` regret over the reliability grid.
