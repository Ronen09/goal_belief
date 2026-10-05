# Belief-encoding edits of the frozen prediction-trained state, read by the frozen goal-conditioned head (belief-encoding edit)

Run date: 2026-10-01. Brief: `BRIEF.md`. Edits, pairs, measures and decision rule: `PLAN.md`, committed (`42a6eec`)
**before any edited state was passed through a trained head**. Numbers from `tables.md`; data in `results.json`.
Reproduce: `reproduce.py belief_encoding_edit` (5 min on one GPU; needs the predictive-transfer experiment).

**Setting.** The predictive-transfer experiment's backbones and 16-unit goal-conditioned heads, all frozen. The heads were rebuilt with round
26's code and seeds and reproduce its held-out regret exactly (`heads.json`); primary are those trained on 100 000
examples. The state is the residual stream at the last prefix token, which the head reads through layer norm plus a
one-hot goal. Per backbone, h ≈ c + E b is fitted on the fit side (b: the exact posterior over 14 cells). On
held-out pairs, the recipient's state is replaced by h_A + E (b_B − b_A) and given to the head under each goal.
Primary backbone: *predict1* (next-symbol prediction on goal-free walks, never given a goal or reward).

## 1. Primary: predict1, main pairs (30 000 pairs, every goal)

Share of cells where the head's greedy action is in the donor's optimal set; median (range) over ten backbones, each
the mean of its three heads. *Change*: the two posteriors have disjoint optimal sets under that goal (47 861 cells).
*Preserve*: identical optimal sets (42 042 cells).

| edit | uses | change: donor-optimal | change: moved | preserve: still optimal | selective pairs |
|---|---|---|---|---|---|
| none | | 0.051 | 0 | 0.963 | 0.040 |
| whole (h_B) | donor's state | 0.902 | 1 | 0.962 | 0.821 |
| **encoding: h_A + E(b_B − b_A)** | **donor's posterior only** | **0.765** (0.734–0.783) | 0.703 | **0.872** (0.850–0.886) | 0.554 |
| rotated encoding (generic directions) | | 0.150 | 0.124 | 0.914 | 0.083 |
| random rank-13 patch | donor's state | 0.102 | 0.079 | 0.960 | 0.085 |
| encoding-subspace patch | donor's state | 0.782 | 0.776 | 0.902 | 0.633 |
| PCA-13 patch | donor's state | 0.786 | 0.785 | 0.925 | 0.657 |
| decoder patch (the belief-edit experiment's) | donor's state | 0.149 | 0.103 | 0.950 | 0.144 |
| one-step encoding: h_A + F(p_B − p_A) | donor's one-step prediction | 0.597 | 0.518 | 0.865 | 0.395 |

*Selective pairs*: pairs with both kinds of cell (17 914), where the same edit is right under every goal.

* **The encoding edit produces the donor-belief action** in 0.77 of change cells, 0.85 of whole-state replacement.
  It does this from the donor's posterior alone: nothing from the donor's history or state enters.
* **It is specific to the directions along which the state varies with belief.** The same edit vectors rotated
  into generic directions move 0.15; a random rank-13 patch of the donor's actual state moves 0.10. All ten
  backbones, p < 0.001 for both.
* **It is not fully selective.** On preserve cells it loses 9 % of optimal actions (0.963 → 0.872), where whole
  replacement loses none. The registered margin was 0.05. Pairs on which one edit is right under every goal: 0.55,
  against 0.82 for whole replacement.
* The edit's size matches the actual state difference (norm ratio 0.97) but its direction does not entirely (cosine
  0.75): 27 % of the squared difference is outside it (history-specific). Patching the donor's own state into the same
  13-dimensional subspace does a little better on both counts (0.782 / 0.902), and so does PCA-13 (0.786 / 0.925).
  The 13 dimensions hold 96–97 % of the difference. The loss on preserve cells is partly the edit's form and partly
  any rank-13 partial state.
* **The belief-edit experiment's decoder direction fails again** (0.149): its subspace holds 7 % of the state difference. The encoding
  subspace, which follows how the state varies with belief, holds 96 %.
* **A third posterior gives its own action.** With b_C in place of b_B, the head chooses C's optimal action in 0.73
  of cells where C's set is disjoint from A's and B's, and B's in 0.13.

## 2. One-step-agreeing pairs and shared beliefs

| | none | whole | **encoding** | rotated | PCA-13 patch | one-step encoding |
|---|---|---|---|---|---|---|
| one-step agreeing, change cells: donor-optimal (34 799 cells) | 0.151 | 0.801 | **0.723** | 0.163 | 0.573 | 0.151 (identically zero edit) |
| shared belief: both recipients donor-optimal (10 986 cells) | 0.041 | 0.875 | **0.708** | 0.117 | 0.721 | 0.553 |
| shared belief: the two recipients agree | 0.920 | 1 | 0.903 | 0.913 | 0.919 | 0.903 |

* **The edit works where the one-step predictions are identical and the optimal actions differ.** There it gives
  0.72 donor-optimal, 0.90 of whole replacement, against 0.15 with no edit and with rotated vectors. An edit built
  from the one-step prediction is exactly zero there. So the head uses belief information that the prediction
  objective did not require, and the encoding finds it.
* **The same edit vector works on different histories with the same belief**: both recipients reach the donor's
  action in 0.71 of cells (0.81 of whole), and they agree with each other as often as without the edit (0.90
  against 0.92).

## 3. Decision rule

| | criterion | predict1 | held |
|---|---|---|---|
| S1 | change cells: encoding ≥ 0.75 × whole | 0.765 / 0.902 = 0.85 | **yes** |
| S2 | preserve cells: encoding ≥ none − 0.05 | 0.872 against 0.963 (−0.091) | **no** |
| S3 | encoding above rotated and random patch (p < 0.05); rotated ≤ half of encoding | p < 0.001, < 0.001; 0.20 | **yes** |
| S4 | shared: both donor-optimal ≥ 0.75 × whole; agreement ≥ none − 0.05 | 0.81; −0.017 | **yes** |
| S5 | one-step agreeing: encoding ≥ 0.5 and above rotated | 0.723, p < 0.001 | **yes** |

**By the registered rule the edit is not a selective causal belief edit**: S2 failed (the plan's reading: "the
edit is not selective: it moves decisions that should stay"). S1, S3, S4 and S5 hold. With the 1 000-example heads
the pattern is the same (S1 0.87, S2 −0.08, S5 0.69).

## 4. Other backbones: the edit is not specific to a prediction-trained state

Same heads and measures (100 000 examples; random at its site 1):

| | predict1 | predict2 | reward | random |
|---|---|---|---|---|
| encoding R² (state from posterior, held-out) | 0.737 | 0.770 | 0.652 | 0.400 |
| change cells: encoding / whole | 0.765 / 0.902 | 0.786 / 0.918 | 0.761 / 0.908 | 0.675 / 0.789 |
| change cells: moved | **0.703** | 0.734 | 0.676 | 0.475 |
| preserve cells: encoding − none | −0.091 | −0.079 | −0.072 | −0.092 |
| one-step agreeing: encoding (none) | **0.723** (0.151) | 0.747 (0.113) | 0.680 (0.098) | 0.602 (0.253) |

* **Every backbone passes S1, S3, S4 and S5 and fails S2.** Even the random backbone, which learned nothing, passes
  S1 at 0.86 of its own whole replacement. Its head was trained on its features, and 40 % of those features vary
  linearly with belief. An encoding edit can steer a head on any state from which that head learned to read belief.
  So the edit's success shows causal use of belief-associated information by the head; it does not by itself show
  that the information was learned.
* **What learning adds is size, not form.** Against random, predict1's edit reaches the donor's action more often
  (0.765 against 0.675; the ten-backbone ranges do not overlap) and moves the head's distribution further (0.70
  against 0.48). On one-step-agreeing pairs it gains 0.57 over no edit against random's 0.35. Its state is more
  belief-linear (R² 0.74 against 0.40), so the edit reaches what the head reads more completely.
* The reward-trained state behaves like the predictors' (0.76 / 0.91); its encoding is less complete (R² 0.65, edit
  norm 0.79 of the difference).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | encoding R²: predict1 above reward and random | yes | 0.737; 0.652; 0.400 |
| E2 | S1 holds | yes | 0.85 |
| E3 | S2 holds | **no** | −0.091 |
| E4 | rotated and random patch below 0.2 on change cells | yes | 0.150; 0.102 |
| E5 | S4 holds | yes | 0.81; −0.017 |
| E6 | S5 holds | yes | 0.723 |
| E7 | PCA patch ≥ 0.9 × whole | **no** | 0.87 |
| E8 | decoder patch below encoding | yes | 0.149 against 0.765 |
| E9 | one-step edit below encoding on main change cells | yes | 0.597 against 0.765 |
| E10 | reward's encoding / whole below predict1's | yes, barely | 0.839 against 0.848 |
| E11 | wrong: C's action ≥ 0.75 × encoding's rate, B's below 0.1 | **no** | 0.731 (yes); 0.131 (no) |

8 of 11 held.

## Conclusions

1. **The frozen goal-conditioned head uses belief-associated information causally.** Changing the
   prediction-trained state only along how it varies with belief makes the head choose the donor-belief action in
   77 % of the cells where that action must change (85 % of whole replacement). This holds under every goal, on
   different histories sharing a belief, and with a third belief. Generic edits of the same size and rank do almost
   nothing. The information was learned without goals and is combined with the goal only by the head.
2. **Including belief information beyond the one-step prediction.** Where one-step predictions are identical but
   the actions differ, the edit works (0.72, against an exactly zero one-step edit).
3. **But the edit is not selective enough to call it a clean belief edit.** It changes 9 % of decisions that should
   stay. The cause is the edit's form: it moves the belief component of a state whose remaining 27 % is
   history-specific, so the combination is not a state the backbone produces. A rank-13 patch of the donor's own
   state also loses 4–6 %.
4. **The edit works on any backbone whose head learned to read belief, the random one included.** It establishes
   causal use by the head, not that the state's belief information was learned. Learning makes the effect larger:
   the prediction-trained state is more belief-linear, so the edit carries more of what the head reads.

Next, if wanted: an edit that keeps states on the backbone's manifold should be tested on preserve cells, which
this one fails. Examples are a nonlinear encoding, or an edit through the input: replacing the history by one with
the donor's belief and a matched history-specific part. The same edits through the original reward policy's
decision would test the bridge to the main question.

Limits: one site, one head width, the first decision after the reveal; linear encoding; the rebuilt heads equal
the predictive-transfer experiment's (exactly reproduced) but are not the RL policy. The preserve criterion's margin (0.05) was set without a
pilot; the loss is 0.07–0.09 in every backbone.
