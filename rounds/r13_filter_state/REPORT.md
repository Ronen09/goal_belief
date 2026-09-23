# The full filter state: does a recurrent network learn the minimal predictive state? (TASK12, round 13)

Run date: 2026-09-23. Brief: `rounds/r13_filter_state/BRIEF.md`. Theory and 10 predictions: `rounds/r13_filter_state/THEORY.md`,
committed (`84994af`) **before any measurement or training of this round**. Models: the round-12
channel-env GRUs and transformers (K = 4, 4 seeds), plus 60 new GRUs with hidden size 2–32
(`rounds/r13_filter_state/models/`). Numbers from `rounds/r13_filter_state/tables.md`, per-model values in `results12.json`,
figures `fig1_interventions.png`, `fig2_hierarchy.png`, `fig3_bottleneck.png`. Reproduce:
`rounds/r13_filter_state/run.py` (7 min on 96 CPU workers), then `rounds/r13_filter_state/tables.py`.

**One deviation from the pre-registration.** The cross-length equal-state pairs (Ex) could not
be found at the registered tolerance of 0.02: the nearest cross-length match among 300k × 300k
histories is 0.025 apart. They were built at 0.05 (228 pairs). Because Ex is judged against the
Bayes divergence of the same pairs, the looser tolerance is absorbed by the reference.

## The minimal statistic, and what supervision exposes

The channel environment has hidden state (G, c_t), 2K = 8 values. Its observability matrix has
rank 8 over two-step futures, so the joint filter α(g, c) is the **minimal** predictive state,
with 7 degrees of freedom: 3 goal log-odds y and 4 channel coordinates ℓ_g = logit P(on | g).

* **The marginals suggested in the brief are not sufficient.** P(on | g) differs across goals by
  a median of 0.56, and (b, P(on)) predicts it with R² 0.56.
* **No objective's target is the minimal state.** `goal` and `act_soft` targets determine
  3 coordinates, and `next_obs` (the one-step predictive, rank 5) determines 5.

## 1. The full state is decodable, but decodability does not discriminate for the channel block

| GRU | goal block IID | channel block IID | channel TIME | channel EXT |
|---|---|---|---|---|
| `goal` | 0.996 | 0.997 | 0.988 | 0.984 |
| `next_obs` | 0.973 | 0.992 | 0.982 | 0.959 |
| `act_soft` | 0.964 | 0.988 | 0.975 | 0.948 |
| untrained | 0.701 | **0.978** | 0.970 | 0.963 |

Every trained GRU carries all 7 coordinates affinely, and the probe generalises to later times
and to extreme channel beliefs. But an untrained GRU decodes the channel block almost as well.
The channel belief depends mostly on the last few tokens, and a random recurrent network keeps a
linear fading memory of them. Only the goal block, which integrates evidence over the whole
history, separates trained from untrained networks (0.996 vs 0.701). This is the round-12
lesson again: decodability shows what is available, and the causal tests below show what is
used.

## 2. Matched interventions in both coordinates

A's GRU state at t is moved along the encoder directions so that its decoded coordinates hit a
target exactly. F is the fraction of the gap closed toward that target's own Bayes future
(k ≥ 1, 1500 pairs), for `goal` models.

| edit | t = 6 | t = 12 | size of the gap it must close (KL) |
|---|---|---|---|
| goal block → B, channel block kept | 0.962 | 0.965 | 1.01 / 1.86 |
| channel block → B, goal block kept | 0.544 | 0.629 | **0.036 / 0.025** |
| both blocks → B | **0.994** | **0.993** | 0.90 / 1.85 |
| B's whole state (SWAP) | 1.000 | 1.000 | |
| random direction, the size of "both" | −0.12 | −0.05 | |

* **Controlling the whole state makes the transplant complete.** Toward the genuine-B run,
  "both" closes 0.991 (t = 6) and 0.988 (t = 12), against 0.950 and 0.968 for round 12's
  goal-only probe. The remaining gap in round 12 was the channel belief it did not control.
* **The goal block alone follows its own counterfactual** (goal belief moved, channel belief
  given the goal kept) at 0.96.
* **The channel-only edit is directionally right but imprecise** (0.54–0.63). The future it has
  to produce differs from A's by a KL of only 0.03, about 1/40 of the goal edit's gap. So the
  channel coordinates have little leverage over the goal output, and a small error in the edit
  is a large fraction of a small gap.
* For `act_soft` and `next_obs` the channel-only edit moves the future the wrong way (−0.33 to
  −1.09). "Both" still reaches 0.92.

## 3. The equivalence hierarchy: exactly as much divergence as the filter state allows

Pairs of histories matched on more and more of the filter state. The divergence is JS between
the model's outputs over 8 continuations and k ≥ 1, compared with the Bayes divergence of the
same pairs (fig. 2).

| pairs matched on | `goal`: model | Bayes | model ÷ Bayes | Spearman over pairs | ÷ random |
|---|---|---|---|---|---|
| goal posterior b | 4.4 × 10⁻³ | 4.4 × 10⁻³ | 1.01 | 1.00 | 0.023 |
| marginals (b, P(on)) | 8.6 × 10⁻⁴ | 8.5 × 10⁻⁴ | 1.01 | 1.00 | 0.0045 |
| full state, same length | 1.7 × 10⁻⁷ | 2 × 10⁻⁸ | (8.1) | 0.17 | 9 × 10⁻⁷ |
| full state, ≥ 3 positions differ beyond relabelling the two neutral tokens | 7 × 10⁻⁷ | 4 × 10⁻⁷ | 1.9 | 0.74 | 4 × 10⁻⁶ |
| **full state, lengths 8 vs 16** | 5.1 × 10⁻⁵ | 4.5 × 10⁻⁵ | **1.13** | 0.88 | 2 × 10⁻⁴ |
| random pairs | 0.19 | | | | 1 |

* **At every level, the network's histories diverge by the Bayes amount.** Equal b: the future
  still differs, by exactly as much as Bayes says (1.01, ρ = 1.00). Equal marginals: less, again
  exactly (1.01, ρ = 1.00). Equal full state: nothing is left at the level of 10⁻⁷, six orders of
  magnitude below random pairs. `act_soft` and `next_obs` show the same pattern (tables.md §3).
* **Histories of different lengths with the same full state are future-equivalent** (model 1.13×
  Bayes, 2 × 10⁻⁴ of random). Nothing in the filter state records time, and nothing in the
  network's state that the future reads does either.
* **The same-length ratio of 8.1 is a floor, not a history effect.** When the Bayes divergence
  is ~10⁻⁸, the ratio measures the model's own precision. The two runs on equal-state histories
  agree with each other to 1–3 × 10⁻⁷ JS, while each disagrees with the exact Bayes future by
  0.5–1.7 × 10⁻⁵. The model's error is a function of the filter state, shared by both histories,
  and is 50–65× larger than the difference between them (7–20× for the ≥ 3-position set).

## 4. The bottleneck: the minimal dimension shows up in the hidden size

| hidden | KL iid | KL channel | goal R² | channel R² | Ex model ÷ Bayes |
|---|---|---|---|---|---|
| 2 | 0.0275 | 0.0913 | 0.636 | 0.439 | 176 |
| **3** | **0.0001** | 0.0314 | 0.913 | 0.563 | 105 |
| 4 | 0.0001 | 0.0128 | 0.938 | 0.671 | 32 |
| 5 | 0.0000 | 0.0082 | 0.941 | 0.730 | 25 |
| 6 | 0.0000 | 0.0052 | 0.959 | 0.847 | 21 |
| **7** | 0.0000 | **0.0016** | 0.982 | **0.944** | 10 |
| 8 | 0.0000 | 0.0012 | 0.983 | 0.956 | 7.8 |
| 12 | 0.0000 | 0.0003 | 0.989 | 0.965 | 2.4 |
| 32 | 0.0000 | 0.0001 | 0.994 | 0.994 | 1.2 |
| 64 | | 0.0001 | 0.996 | 0.997 | 1.1 |

* **iid: the network becomes exact at the minimal dimension.** The minimal state has 3
  coordinates. At n = 2 the KL is 0.028; at n = 3 it falls 250-fold to 10⁻⁴ and stays there.
* **Channel: no sharp threshold, but the minimal dimension is visible.** The KL falls steadily
  with n. Its largest single-step drop (3.3×, 0.0052 → 0.0016) comes at n = 7, the minimal
  state's dimension, where the channel block first passes R² 0.94. Future-equivalence at the
  Bayes level (Ex within 2.4× of Bayes) needs n ≥ 12, so the network uses spare dimensions to
  hold the 7 coordinates precisely.
* **Under a bottleneck, the supervised projection is kept first.** At every n ≤ 6 the goal block
  is decoded better than the channel block (n = 3: 0.91 vs 0.56). The network gives up the
  unsupervised part of the state first, and pays for it in exactly the target it was trained
  on, since the goal posterior's future depends on the channel.

## Pre-registered predictions

| | prediction | held | numbers |
|---|---|---|---|
| F1 | full state recoverable (channel ≥ 0.85, goal ≥ 0.98 for `goal`, `next_obs`), beats marginals; untrained channel ≤ 0.5 | no | trained ✓ (0.997 / 0.992; goal 0.996, **0.973**); untrained channel **0.978** ✗ |
| F2 | `goal` channel TIME ≥ 0.8, EXT ≥ 0.6 | **yes** | 0.988, 0.984 |
| F3 | G-only, R-only, both ≥ 0.9 of own Bayes gap; random ≤ 0.1 | no | G-only 0.96 ✓, both 0.99 ✓, random ≤ −0.05 ✓; **R-only 0.54 / 0.63** ✗ |
| F4 | both ≥ 0.97 toward genuine-B, above round 12's goal-only probe | **yes** | 0.991 / 0.988 vs 0.950 / 0.968 |
| F5 | Eb, Em model ÷ Bayes in [0.5, 2]; Ej ≤ 0.002 of random and ≤ 3× Bayes | no | Eb 1.01, Em 1.01 ✓; Ej ÷ random 9 × 10⁻⁷ ✓, **÷ Bayes 8.1** ✗ |
| F6 | cross-length equal state ≤ 0.01 of random, ≤ 3× Bayes | **yes** | 2 × 10⁻⁴, 1.13 (tolerance 0.05) |
| F7 | Ej and Ex also for `next_obs`, `act_soft` | no | Ex ✓ (1.35, 1.38); Ej ÷ Bayes 9.9, 13 ✗ (same floor as F5) |
| F8 | KL < 0.01 at n ≤ 4 in iid, needs n ≥ 6 in channel | no | iid n = 3 ✓; channel **n = 5** ✗ |
| F9 | channel, n ∈ {3, 4, 5}: goal ≥ 0.9, channel ≤ 0.6 | no | goal 0.91–0.94 ✓; channel 0.56, **0.67, 0.73** ✗ |
| F10 | transformer: channel R² at res2 ≥ at u | **yes** | 0.984 vs 0.979 |

4 of 10 held. The failures:
* **F5 and F7 share one cause.** The "≤ 3× Bayes" criterion used the wrong reference for pairs
  whose Bayes divergence is ~10⁻⁸. The model's between-history divergence for those pairs is 7–65×
  below its own error against Bayes (§3).
* **F1**: the prediction for the untrained network was wrong. The channel block is a
  short-memory quantity any recurrent network holds.
* **F3**: the channel-only edit.
* **F8 and F9** are threshold misses on curves whose shape was as predicted: the goal block is
  kept before the channel block at every n, and exactness arrives at the minimal dimension in iid.

## Conclusions

1. **The recurrent network's future is a function of the environment's minimal predictive
   state, not of its training target.** Histories matched on the goal posterior or on the
   marginals diverge by exactly the Bayes amount (ρ = 1.00). Histories matched on the full
   7-coordinate filter state are future-equivalent, at the same length (to 10⁻⁷, below the
   model's own error) and across different lengths (1.13× Bayes). This holds for `goal`,
   `act_soft` and `next_obs` models, whose targets expose only 3 or 5 of the 7 coordinates.
2. **Both blocks are causal, and controlling both completes the transplant.** Editing the whole
   decoded state reaches 0.99 of the genuine-B future. The goal-only edit of round 12 stopped at
   0.95–0.97, and the difference is the unsupervised channel belief.
3. **Decodability alone would have misled here.** Untrained networks decode the channel block
   at 0.98. What distinguishes a network that has learned the minimal state is that its future
   depends on the history only through that state. That is a causal and functional property,
   measured by the equivalence hierarchy, not by a probe.
4. **The minimal dimension is an architectural bottleneck the network respects.** In iid the
   GRU becomes exact at 3 hidden units, the minimal state's dimension. In the channel env the
   largest single-step gain in precision comes at 7, and under tighter bottlenecks the supervised projection is kept at
   the expense of the rest of the state, which the supervised quantity's own future needs.
