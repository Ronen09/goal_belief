# Is the decoded posterior the causal state? (TASK11, round 12, part 2)

Run date: 2026-09-23. Brief: `rounds/r12_hidden_goal/BRIEF.md` (claim 2 and the three tests that followed it). Theory,
cut analysis and the 14 predictions are in `rounds/r12_hidden_goal/causal/THEORY.md`, committed (`56b0d1f`)
**before any intervention was run**. This part uses the round-12 models (K = 4, main grid, 4
seeds per cell, 64 models); nothing is retrained. Numbers from `rounds/r12_hidden_goal/causal/tables.md`,
per-model series in `causal.json`, pairs in `pairs_*.npz`, figures `fig4_transplant.png`,
`fig5_equal_belief.png`, `fig6_ladder.png`. Reproduce: `rounds/r12_hidden_goal/causal_run.py` (4 min on
64 CPU workers), then `rounds/r12_hidden_goal/causal_tables.py`.

## The one fact that organises the results

A belief can only be *the state* at a complete cut, meaning a site every path from the history
to the future passes through.

* **The GRU state h_t is a complete cut.** Swapping it is exactly equivalent to running the
  other history: SWAP matches the genuine-B run with max |Δp| = 0.0 in every model. If the
  posterior is the causal state anywhere, it is here.
* **No single transformer position is a cut.** Every future position re-reads the raw tokens.
  Nothing after position t reads res2(t) or u(t), and res1(t) is read only through block-2
  attention. The transformer recomputes its belief at every step and carries none forward.

## Test 1 — posterior transplant

Pairs (A, B) with ‖y_A − y_B‖ ≥ 2 nats, 1500 per t ∈ {6, 12}. A's state at t is replaced, and A's
own continuation is run to t = 24. F = fraction of the gap to the genuine-B run that the
intervention closes (1 = indistinguishable from having seen B's evidence). k0 is the immediate
output; "future" is the mean over k ≥ 1. Values are for t = 6 (t = 12 agrees to ±0.02 unless
stated).

| GRU `goal`, complete cut | iid, k0 | iid, future | channel, k0 | channel, future |
|---|---|---|---|---|
| SWAP (B's whole state) | 1.000 | 1.000 | 1.000 | 1.000 |
| PROBE-E: set decoded belief to y_B along the encoder directions | 1.000 | **1.000** | 0.998 | **0.950** |
| PROBE-D: same decoded belief, minimum-norm move | 0.839 | 0.778 | 0.537 | 0.460 |
| random direction, PROBE-E's norm | −0.017 | −0.023 | −0.000 | −0.017 |

* **In iid, moving only the decoded belief is the whole transplant.** The entire 18-step future
  becomes indistinguishable from the genuine-B run (F 1.000, KL to the Bayes target 0.0002). A
  half move to y_A + ½(y_B − y_A) closes 0.999 of the gap to its own Bayes target, so the future
  follows the log-odds line, not just its end point (fig. 4).
* **In the channel env, the transplant follows the transplant, not genuine-B.** PROBE-E leaves
  the output closer to the Bayes future of "b_B with A's channel belief" than to the genuine-B
  Bayes future (KL 0.021 vs 0.030 at t = 6; 0.026 vs 0.034 at t = 12). It closes 0.977 of the
  gap to that target. The 5% gap to genuine-B is the channel belief the probe does not move, and
  SWAP, which moves it, closes the rest.
* **The direction matters even though both moves set the probe exactly.** The minimum-norm move
  steps into directions the states barely vary in. The least-squares probe is unconstrained
  there, and the network's own dynamics disagree with it. The move is off by 16% at the immediate
  output and by 22–54% in the future. The encoder direction stays where the states actually lie,
  as round 9 found for steering vectors.
* **Other objectives** (iid, PROBE-E future): `next_obs` 0.952, `act_soft` 0.890, `act_hard`
  0.613 (0.706 at t = 12). The hard-target GRU's belief code is only approximately affine; a SWAP
  still gives 1.000.

**Transformer, one position** (`goal`, t = 6):

| site | iid k0 | iid future | channel k0 | channel future |
|---|---|---|---|---|
| res1(t) SWAP | 0.989 | 0.031 | 0.791 | 0.156 |
| res1(t) PROBE-E | 0.719 | 0.014 | 0.451 | 0.054 |
| res2(t) SWAP | 1.000 | 0.000 | 1.000 | 0.000 |

A single-position edit switches the immediate output and leaves 84–100% of the future
untouched. This is the brief's "merely switching the immediate action", and here it is not a
failure of the representation but a property of the architecture. There is no transformer
belief state to transplant. The part of the future that does move (3–16% at res1) is the share
of block-2 attention that happens to read position t.

## Test 2 — equal-belief history equivalence

Pairs at t = 12 with 8 continuations each, drawn from H₁'s posterior predictive. At the GRU cut,
the patch effect is the JS between H₁⊕c and H₂⊕c.

| GRU `goal` | effect (JS, future) | ÷ random pairs | Bayes divergence | model ÷ Bayes | Spearman over pairs |
|---|---|---|---|---|---|
| iid, exactly equal b, counts differing in ≥ 8 of 12 tokens | 6 × 10⁻⁷ | < 10⁻⁵ | 0 | — | — |
| channel, equal b, channel belief differing by ≥ 0.15 | 0.0044 | 0.021 | 0.0044 | **1.01** | **1.00** |
| channel, equal joint (G, c) state | 1 × 10⁻⁵ | 0.0001 | 1 × 10⁻⁵ | 1.12 | 0.91 |
| random pairs | 0.20 | 1 | | | |

* **Where the posterior is sufficient, patching one history for another does nothing.**
  Histories with 8 of 12 tokens different and the same posterior give downstream outputs equal
  to within 10⁻⁶ JS. All iid objectives agree: `act_hard` 0.0024 of random, `act_soft` 0.0001,
  `next_obs` < 0.0001.
* **Where the posterior is not sufficient, the network knows it.** Equal-b pairs that differ in
  channel belief diverge downstream by exactly the Bayes amount (ratio 1.01, pair-level
  Spearman 1.00). Pairs equal in the joint (G, c) state do not diverge. The GRU's state is the
  sufficient statistic of the environment, the joint filter, and not the quantity it was trained
  to output. The transformer's behaviour shows the same numbers (1.02, 0.99).
* **Equal belief does not mean equal state** (descriptive, not pre-registered). The GRU `goal`
  states of iid equal-belief pairs are 0.20 of the random-pair distance apart, but only 0.8% of
  that difference lies in the decoder's span, and none of it reaches the future. The history
  detail is kept in directions the dynamics ignore. In the hard-action GRU, equal-belief states
  decode 0.3 nats apart (6.7% in the decoder span), and downstream they still agree to 0.0024 of
  random.

## Test 3 — same action, different belief

Pairs at t = 12 with the same untied optimal action, ‖Δy‖ ≥ 1.5, and a continuation under which
the Bayes action later differs between the two histories (1500 pairs; 8288 differing future
steps in iid).

**Causally distinguishable?** Flip rate is how often the model's actions differ on the future
steps where the Bayes actions differ.

| | goal | act_soft | act_hard |
|---|---|---|---|
| GRU, iid | 0.999 | 0.995 | **0.991** |
| transformer, iid | 0.999 | 0.999 | **0.999** |
| GRU, channel | 0.974 | 0.964 | [0.912, not converged] |
| transformer, channel | 0.963 | 0.958 | [0.930, not converged] |

No network merges two beliefs that later evidence will separate, including those trained only
on the action. Correct future behaviour requires this, and the networks meet the requirement in
architecture-specific ways. In the GRU the state at the cut is the only carrier of the future, so
the distinction has to survive there. In the transformer the future never reads the sites where
compression happens (below).

**The quotient across depth** (transformer, iid, evaluation set; fig. 6):

| site | `goal`: belief within action | action | neutral count | `act_hard`: belief within action | action | neutral count |
|---|---|---|---|---|---|---|
| res0 (token + position) | 0.143 | 0.439 | 0.081 | 0.143 | 0.439 | 0.081 |
| res1 (read by the future) | 0.992 | 0.897 | 0.911 | 0.973 | 0.949 | 0.798 |
| res2 | 0.995 | 0.878 | 0.899 | 0.971 | **0.997** | **0.557** |
| u (interface) | 1.000 | 0.861 | 0.975 | **0.932** | **1.000** | **0.433** |

The ladder the brief described is visible in the hard-action transformer:
* **history**: res0 holds only the current token;
* **belief**: res1 holds the posterior (0.97) plus history detail with no evidential value
  (0.80);
* **task-relevant quotient**: res2 and u make the action exactly linear (0.95 → 1.000), discard
  most of the history detail (0.80 → 0.43) and start to compress the belief within each action
  class (0.973 → 0.932);
* **output**.

The compression happens only in the sites nothing downstream reads. The posterior-trained
transformer has no such step: its within-action belief rises to 1.000 at u, and its history
detail rises too (0.975), because nothing in a posterior target penalises detail that has no
effect.

The belief is quotiented much less than the history detail. The within-action drop is 0.04, not
the 0.2 pre-registered. Hard targets at finite training still scale the logits by confidence, so
the interface keeps most of the within-class ordering.

## Pre-registered predictions

| | prediction | held | numbers |
|---|---|---|---|
| A1 | GRU iid `goal`: PROBE-E ≥ 0.9 future, ≥ 0.95 k0; PROBE-D ≥ 0.8 future | no | PROBE-E 1.000 / 1.000 ✓; PROBE-D 0.778, 0.765 ✗ |
| A2 | GRU channel `goal`: PROBE-E nearer the transplant than genuine-B; F_T ≥ 0.8 | **yes** | KL 0.021 vs 0.030; F_T 0.977, 0.986 |
| A3 | halfway PROBE-E ≥ 0.8 of its own gap | **yes** | 0.999, 0.999 |
| A4 | random direction ≤ 0.2, every GRU cell | **yes** | max 0.016 |
| A5 | GRU iid `act_soft` / `act_hard` / `next_obs` PROBE-E ≥ 0.7 future | no | 0.890 / **0.613**, 0.706 / 0.952 |
| A6 | transformer res1 single position ≤ 0.3 future; res2 SWAP ≥ 0.95 at k0 | **yes** | max 0.156; 1.000 |
| B1 | GRU iid `goal` equal-belief ≤ 0.05 of random at every k | **yes** | < 10⁻⁵ |
| B2 | GRU channel `goal` equal-b: model ÷ Bayes in [0.5, 2], Spearman ≥ 0.5 | **yes** | 1.01, 1.00 |
| B3 | GRU channel `goal` equal-joint ≤ 0.1 of random | **yes** | 0.0001 |
| B4 | GRU iid other objectives ≤ 0.1 of random | **yes** | ≤ 0.0024 |
| C1 | GRU iid `act_hard` flip rate ≥ 0.9 | **yes** | 0.991 |
| C2 | transformer iid within-action R²_y at u: `goal` − `act_hard` ≥ 0.2; res1 ≥ 0.8 | no | 0.067; res1 0.992, 0.973 |
| C3 | transformer iid `goal` within-action ≥ 0.95 at res2 and u | **yes** | 0.995, 1.000 |
| C4 | transformer iid neutral-count R² lower at u than res1, every objective | no | `act_hard` 0.80 → 0.43 ✓, `next_obs` 0.99 → 0.96 ✓, `act_soft` 0.84 → 0.81 ✓, `goal` 0.91 → 0.975 ✗ |

10 of 14 held. Four failures:
* **A1**: the minimum-norm move underperforms (0.77 against 0.8) while the encoder move is
  exact. Which direction is used matters.
* **A5**: the hard-target GRU is only 0.61–0.71 transplantable along an affine code, though
  fully by SWAP.
* **C2**: the belief quotient at the last block is 0.067, not 0.2. The action and the history
  detail are quotiented far more than the belief.
* **C4**: the posterior-trained transformer does not discard history detail with depth. It
  holds more at u than at res1.

## Conclusions

1. **At a complete cut, the decoded posterior is the causal state.** Moving only the decoded
   log-odds of a GRU's state, along the directions its states actually vary in, makes the whole
   future indistinguishable from having seen the other evidence (F 1.000 over 18 steps), and
   half a move gives the Bayes future of half the evidence.
2. **Histories with equal posteriors are causally interchangeable exactly when the posterior is
   sufficient.** In iid the downstream effect is below 10⁻⁵ of that of random pairs, despite
   states that differ (in directions the future ignores). In the channel env the network
   separates equal-b histories by exactly the Bayes amount: its state is the environment's
   sufficient statistic, not its training target.
3. **Where the question is ill-posed, the answer shows it.** With a latent channel, "as though it
   had observed evidence producing b′" has two readings. A probe edit follows the reading that
   moves only b (transplant), and a state swap follows the other (genuine-B).
4. **A transformer has no belief state to transplant.** Single-position edits switch the
   immediate output and leave the future essentially untouched (≤ 16%). The recursive-belief
   picture is a property of recurrent architectures, not of every network that computes a
   posterior.
5. **The quotient forms, and only where it is free.** In the hard-action transformer the action
   becomes exactly linear and the history detail is discarded in the last block, which the
   future never reads. At the GRU cut and in the transformer's future-read layer, no network
   merges beliefs that later evidence would separate.
