# A windowed transformer: does restricting attention force a steerable belief state? (TASK13, round 14)

Run date: 2026-09-23. Request: restrict a transformer's attention to the last l tokens so that it
has to form a stronger, more steerable belief representation. Theory and 7 predictions:
`docs/task13_window_theory.md`, committed (`1b3fcf1`) **before any model was trained**. Model:
`goalgeo/wtfm.py`. Numbers from `results13/tables13.md`, per-model values in `results13.json`,
weights in `results13/models/`, figure `fig1_window.png`. Reproduce: `scripts/run_task13.py`
(31 min on CPU workers), then `scripts/task13_tables.py`.

## Design

2-layer pre-LN transformer (d = 64, 4 heads), attention in both layers restricted to the last l
positions, with a learned relative-position bias and no absolute positions. Two variants:
* **no carry**: the window is the only restriction;
* **carry**: the previous position's readout interface u_{t−1} = LN_f(x_top) is projected and
  added to the next position's input, a recurrent path alongside the window.

l ∈ {1, 2, 4, 8, full} × {no carry, carry} × {iid, channel}, 3 seeds, trained to output the exact
posterior (the round-12 `goal` objective), 15k steps. Steerability is measured with round 12's
transplant pairs: one edit at position t (swap u_t or the block-1 residual res1(t) with
history B's, or move u_t to B's decoded state with the round-12/13 probe). F is the fraction of
the gap to the model's own future on B's history that the edit closes, over k ≥ 1.

## 1. A window alone does not create a belief state; it removes one

| no carry | KL, iid | KL, channel | KL at positions 17–24 (channel) |
|---|---|---|---|
| l = 1 | 0.430 | 0.451 | 0.658 |
| l = 2 | 0.340 | 0.317 | 0.514 |
| l = 4 | 0.204 | 0.165 | 0.326 |
| l = 8 | 0.049 | 0.039 | 0.110 |
| full | 0.0001 | 0.0002 | 0.0003 |

A 2-layer model with window l sees at most 2l − 1 tokens, and every windowed model fails exactly
where its receptive field runs out (fig. 1, left). The error is small while position t is inside
the field and rises steeply after it. There is nowhere to keep evidence older than the window, so
the network cannot form a belief over the history at all.

## 2. With a recurrent carry, every window works, and a small window makes the carry the state

With the carry, every window converges (KL 0.0001–0.0003, both environments), and the carried
vector holds the belief (goal-block R² 0.985–0.999 from u; channel block 0.99). What changes with
the window is how much of the future runs through that vector:

| carry | SWAP u_t, iid | SWAP u_t, channel | PROBE-E u_t, iid | PROBE-E u_t, channel | SWAP res1(t), channel |
|---|---|---|---|---|---|
| **l = 1** | **1.000** | **1.000** | **0.996** | **0.961** | 1.000 |
| l = 2 | 0.806 | 0.837 | 0.731 | 0.699 | 0.844 |
| l = 4 | 0.464 | 0.423 | 0.414 | 0.360 | 0.490 |
| l = 8 | 0.288 | 0.265 | 0.256 | 0.221 | 0.340 |
| full | −0.005 | 0.174 | −0.009 | 0.135 | 0.340 |

(t = 6; t = 12 agrees within 0.08.)

* **Window 1 with a carry is as steerable as the GRU.** Swapping the single carried vector
  transfers the whole future (1.000), and moving it to another history's decoded belief with the
  affine probe closes 0.996 (iid) and 0.961 (channel). u_t is a complete cut here, as the GRU
  state is.
* **Steerability falls monotonically as the window widens.** At l = 2 one edit controls 80% of
  the future. By l = 8 it controls a quarter, because the future re-reads the tokens directly
  through attention.
* **Given full attention, the network ignores the carry.** In iid, swapping u_t moves nothing
  (−0.005), even though the carry path exists and u holds the posterior. When the tokens are
  available, the network recomputes from them rather than relying on a recurrent state.
* The immediate output always follows the edit (k = 0, 0.99–1.00). As in round 12, what differs
  across architectures is whether the future follows it.

## 3. The carried state is a filter state

With carry and l = 1 in the channel env, histories of lengths 8 and 16 with the same full filter
state give futures that differ by 1.63× the Bayes divergence of the same pairs (3 × 10⁻⁴ of random
cross-length pairs). The GRU of round 13 gives 1.13×. The carried vector keeps no record of
elapsed time that the future reads. Without the carry, windowed models fail this test by 25–105×,
because they have forgotten the history.

## Pre-registered predictions

| | prediction | held | numbers |
|---|---|---|---|
| W1 | no carry: l ≤ 8 fail (KL ≥ 0.01), full window converges | **yes** | KL 0.039–0.451 for l ≤ 8; 0.0001–0.0002 full |
| W2 | carry: every window converges | **yes** | KL ≤ 0.0003 |
| W3 | carry, l = 1: SWAP u_t ≥ 0.95, PROBE-E ≥ 0.9 | **yes** | SWAP 1.000; PROBE-E 0.996 / 0.961 (t = 6), 0.996 / 0.924 (t = 12) |
| W4 | carry: SWAP u_t decreases with l; ≤ 0.5 at full | **yes** | 1.000 > 0.81–0.84 > 0.35–0.46 > 0.18–0.29 > −0.005–0.17 |
| W5 | no carry: editing u_t moves nothing; SWAP res1 ≤ 0.3 for converged models | **yes** | exactly 0; 0.05–0.17 |
| W6 | carry, l = 1, channel: cross-length equal state ≤ 3× Bayes | **yes** | 1.63× |
| W7 | carry, l = 1, channel: goal R² ≥ 0.98, channel ≥ 0.9 from u | **yes** | 0.987, 0.992 |

7 of 7 held.

## Conclusions

1. **Restricting attention alone does not force a belief representation.** It removes access to
   evidence the posterior needs, and every windowed model fails beyond its receptive field.
2. **A belief state has to be given a path.** With a recurrent carry, the windowed transformer
   computes the exact posterior at every window. With window 1, the carried vector is a complete
   cut: one edit of it steers the entire future (1.000 by swap, 0.96–1.00 by the affine probe),
   and it is time-invariant like the GRU state.
3. **Steerability is set by how many routes the future has around the state.** Each doubling
   of the window roughly halves the share of the future one state edit controls, and at full
   attention a network that could use the carry does not.
