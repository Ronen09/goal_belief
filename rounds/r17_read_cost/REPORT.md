# Pricing recomputation: a learned, costed read of the history (TASK16, round 17)

Run date: 2026-09-24. Request (given in conversation): charge a next-token transformer for every
read of the history and let a learned gate decide when a read is worth paying for. Theory and 8
predictions: `rounds/r17_read_cost/THEORY.md`, committed (`cdcfd93`) **before any model was trained**.
Numbers from `rounds/r17_read_cost/tables.md`; per-seed, per-layer and per-position values from
`results16.json`; weights in `rounds/r17_read_cost/models/` (36 checkpoints); figure `fig1_price.png`;
timings in `run_log.txt`. Reproduce: `.venv/bin/python reproduce.py r17` (2 h 53 min), or
`reproduce.py r17 --tables` to rebuild the tables and figure from the saved data.

## Design

Round 16's two families on the channel environment (K = 4, T = 24, sampled next tokens, 30k steps,
3 seeds), with one binary read gate per block and query position instead of random dropout:
* **plain**: a 4-layer full-attention transformer. A closed gate hides keys s < u − 1, so the query
  still sees itself and the previous position.
* **carry**: round 14's 2-layer transformer with full attention plus a carried copy of the previous
  position's readout interface. A closed gate hides every key s < u; the carry is always present.

The gate logit is affine in the block's normalised input at u, so in the carry model it can consult
the carried belief. Gates are straight-through Gumbel-sigmoid samples in training, deterministic
(open iff logit > 0) at test, and initialised open. The loss adds c × (mean open probability over
blocks and positions u ≥ 2), so reading everywhere costs c nats per token; c ∈ {0, 0.003, 0.01,
0.03, 0.1, 0.3}. The read-out is round 16's: calibrated prior weight λ (position t's exports taken
from another history, A's tokens kept) at t = 4, 8, 16, 23, measured with every gate forced open and
with the learned gates, plus the read rate, KL to the exact predictive, and the entropy / movement
splits of the exact filter at open vs closed reads (u ≥ 8).

The plain family trained in 1922 s on the GPU with the stacked trainer. The carry family ran on 12
CPU workers alongside it and finished at 10338 s: about 1 h 53 min for each of the first 12 models,
run concurrently, then about 50 min more for the remaining 6. The gate doubles the attention work
inside the position-by-position loop, which is what makes the carry family slow; a stacked GPU
trainer for it, as `tfm_batched` is for the plain family, is the fix for a follow-up.

## 1. A price induces recurrence, and the carry model chooses the complete cut

Seed means (fig. 1, left and middle):

| price c | 0 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|---|---|
| carry: read rate | 1.000 | 0.500 | 0.095 | 0.000 | 0.000 | 0.000 |
| carry: prior weight t = 16 (forced open) | 0.401 | 0.474 | 0.931 | 1.000 | 1.000 | 1.000 |
| carry: weight on older evidence t = 16 (R3, open) | 0.524 | 0.438 | 0.052 | -0.000 | 0.000 | 0.000 |
| carry: KL under own gates | 0.0058 | 0.0038 | 0.0025 | 0.0020 | 0.0019 | 0.0019 |
| plain: read rate | 1.000 | 0.380 | 0.250 | 0.110 | 0.000 | 0.000 |
| plain: prior weight t = 16 (forced open) | 0.021 | 0.049 | 0.026 | 0.190 | 0.176 | 0.145 |
| plain: KL under own gates | 0.0080 | 0.0052 | 0.0049 | 0.0052 | 0.0215 | 0.0214 |

* **From c = 0.03 up, every carry model closes every read and is fully recurrent.** The prior weight
  is 1.000 under its own gates and also when every gate is forced open. The network does not use the
  history even when it is handed back, like round 16's p = 1 model.
* **It is the most accurate model in the family.** KL falls from 0.0058 (reading everywhere) to
  0.0019–0.0020 (reading nowhere). Once the carry is trained, a read of the history is worth nothing
  to the carry model, as §1 of THEORY.md argued, and any price the optimiser registers closes the
  reads.
* **The transition is abrupt, not a continuum.** Round 16's dropout gave a smooth, logit-linear path
  from 0.36 to 1.000. The price gives three states: read everything (c = 0, prior weight 0.401,
  close to round 16's 0.36), close one block (c = 0.003, 0.474), close everything (c ≥ 0.03, 1.000).
  c = 0.01 is a mix of seeds, not an intermediate model: seeds 0 and 2 read nothing (read rate 0.000,
  prior weight 1.000 at every t), and seed 1 reads at rate 0.285 (prior weight 0.792 at t = 16,
  forced open, calibrated per seed as in the tables). The seed that still reads is the least
  accurate of the three (KL 0.0035 vs 0.0021 and 0.0019).

## 2. The reading is not selective: the price picks which block reads, not when

R4 and R5 were to be tested at the one carry price whose read rate lay in [0.1, 0.6], c = 0.003
(0.500). Its splits are exactly 0.000 and its early and late read rates are both 0.500. The
per-layer data explain why (`results16.json`, `rate_layer` and `rate_pos`):

* **At c = 0.003 the gate decides per block.** In all three seeds block 0 is open at every position
  of every sequence and block 1 is closed at every gated position. (Block 1's rate of 0.04 over the
  25 positions is position 0, where no gate is evaluated.) Every position therefore has one open and
  one closed read, so the exact filter's entropy and movement are the same on both sides and the
  splits are zero by construction. The network made no position- or belief-dependent read decision.
* **The one partially open gate is keyed on the current token, not on the carried belief.** Seed 1
  at c = 0.01 keeps block 0 open at a fraction 0.570 of positions u ≥ 2 (twice its read rate 0.285;
  block 1 is closed), flat over position (0.28–0.29 at every u ≥ 2; fig. 1, right). Its gate decision
  at position t+1 is identical in all four transplant cells R1–R4 at every t (0.565, 0.578, 0.566,
  0.531 at t = 4, 8, 16, 23), although those cells change the carried state, the older history, or
  both. Only the token x_{t+1} is common to all four cells. Where the gate opens, the exact filter is
  *more* certain than where it is closed (entropy split −0.074), and the belief has moved slightly
  more (+0.013). Both are short of the registered thresholds, and the first has the wrong sign.
* **No early bootstrapping reads.** No carry model reads more at u = 2–4 than at u ≥ 16 (0.500 vs
  0.500 at c = 0.003; 0.095 vs 0.096 at c = 0.01). Whatever the reads contribute to building the
  carry happens during training, not at the start of each test sequence.

**What this means for the selectivity hypothesis.** The hypothesis was not refuted by a gate that
reads uniformly. It was not tested, because the network never made a state-dependent read decision.
The hypothesis assumed that a read is worth more at some positions than at others, because the
carried belief is less reliable there. In this environment, once the carry is trained, a read is
worth nothing anywhere: every step up in price that closes more reads also lowers the KL (§1). A
belief-dependent gate has nothing to gain, so the price acts on the cheapest thing it can change, a
block's gate bias, and switches whole blocks off. A test of selectivity needs an environment in which
the carry is genuinely unreliable at identifiable positions, so that some reads are worth paying for.

## 3. The plain model's ceiling survives, and its reads split into worthless and worth paying for

* **The prior's weight stays low (R6).** Under forced-open gates it is at most 0.190 at t = 16
  (c = 0.03), with 0.176 and 0.145 at c = 0.1 and 0.3. That is below round 16's plateau of
  0.228–0.250 (p = 0.25–0.97), and the closed model at c = 0.3 matches round 16's p = 1 model (0.145
  vs 0.140). The largest value at any t and c is 0.355 (t = 4, c = 0.3). A price does not buy a plain
  transformer more recursion than dropout did.
* **Which reads the plain model keeps.** Per layer (`rate_layer`, over all 25 positions): at
  c = 0.003, block 0 reads everywhere, block 1 at 0.434–0.486 of positions, blocks 2–3 (almost) never.
  At c = 0.01, block 0 reads at every position and blocks 1–3 never, in all three seeds, which is the
  read rate 0.250 (1 of 4 blocks). At c = 0.03 only block 0 reads, at 0.362–0.584 of positions, flat
  over position. At c ≥ 0.1 nothing reads. Block 0's keys are the token and position embeddings, so
  its read is a read of the raw tokens. Blocks 1–3 read what earlier positions inferred.
* **The inferred-state reads are worthless and the token reads are not.** Closing blocks 1–3 costs
  nothing: KL is 0.0049 at c = 0.01, against 0.0080 with every read open at c = 0. This is round 15's
  result seen from the other side. Position t+1 recomputes from the tokens and gives position t's
  exported state 0.4–8 % of the weight, so a price removes those reads first. The token read in block
  0 is kept until c = 0.1, and closing it raises KL to 0.0215. The plain model does what §1 of
  THEORY.md called the structural outcome: it pays for the reads that buy accuracy until the price
  exceeds what they buy, then loses accuracy rather than becoming recurrent.
* **At c = 0.003, block 1's reads grow with position.** The total read rate is 0.251 at u = 2–4 and
  0.447 at u ≥ 16. In seed 0, block 1 is open at t+1 = 5 for 0.028 of pairs and at t+1 = 17 for
  0.747. The entropy split is +0.174 (seeds 0.164–0.193). This is the only position-dependent gate in
  the round, and it is in the plain family. I have not separated it from the position dependence of
  the filter's entropy.

**Why R7 and R8 failed, and what that does to round 16's explanation.** R7's threshold (KL ≥ 0.03
once the reads close) came from THEORY.md §2's figure of 0.05 nats lost by round 16's plain model at
p = 1. That figure is the p = 1 model's KL with full attention at test (0.0515). Under its own
dropout, the regime R7 measures, round 16's number is 0.0228. The closed plain models here reach
0.0215 and 0.0214, and 0.0452 and 0.0527 under forced-open gates, so they are essentially round 16's
p = 1 model. The threshold was mis-transcribed. R7 fails for the same reason round 16's I6 did: a
4-layer model infers more from the last two positions than I expected. R8 assumed the plain model
would keep most of its reads at c = 0.01 because it needs the history. It needs the history, but only
through one block's read of the tokens, and the registered read rate weights all four blocks equally.

Neither failure contradicts round 16's depth-ceiling explanation; both refine it. The ceiling on the
prior's weight holds at every price (R6). The accuracy cost of the ceiling is about 0.016 nats per
token above the reading models, not 0.05. And most of a plain transformer's reads, those of inferred
states in blocks 1–3, can be given up cheaply because the model does not use them as a prior in the
first place.

Two numbers I cannot explain. First, under its own gates the closed plain model's prior weight is
0.803–0.818, above round 16's p = 1 model under its own dropout (0.621). Second, in all three seeds
the carry model at c = 0.003 pays 0.0015 nats per token for block 0's reads, which it does not need
(KL 0.0038, against 0.0019–0.0020 for the closed models). The gates start open, and a small price may
not close a block that was useful early in training.

## Pre-registered predictions

| prediction | held | numbers |
|---|---|---|
| R1 carry: read rate non-increasing in c, ≥ 0.5 at c=0, ≤ 0.05 at c=0.3 | yes | c=0.0: 1.000 → c=0.003: 0.500 → c=0.01: 0.095 → c=0.03: 0.000 → c=0.1: 0.000 → c=0.3: 0.000 |
| R2 carry: KL under own gates < 0.01 at every c | yes | c=0.0 0.0058; c=0.003 0.0038; c=0.01 0.0025; c=0.03 0.0020; c=0.1 0.0019; c=0.3 0.0019 |
| R3 carry, read rate ≤ 0.05: prior weight own ≥ 0.95 and open ≥ 0.8 | yes | c=0.03: own 1.000, open 1.000; c=0.1: own 1.000, open 1.000; c=0.3: own 1.000, open 1.000 |
| R4 carry, intermediate c: entropy split ≥ +0.05, movement split ≥ +0.02 | **no** | c=0.003: entropy 0.000, movement 0.000 |
| R5 carry, intermediate c: read rate at u=2–4 exceeds u≥16 by ≥ 0.2 | **no** | c=0.003: 0.500 vs 0.500 |
| R6 plain: prior weight under forced-open ≤ 0.4 at every c | yes | c=0.0 0.021; c=0.003 0.049; c=0.01 0.026; c=0.03 0.190; c=0.1 0.176; c=0.3 0.145 |
| R7 plain, read rate ≤ 0.1: KL under own gates ≥ 0.03 | **no** | c=0.1: 0.0215; c=0.3: 0.0214 |
| R8 c*_carry < c*_plain; plain read rate ≥ 0.5 at c=0.01 | **no** | c* carry 0.003, plain 0.003; plain rate at 0.01: 0.250 |

4 of 8 held. What the failures show:
* **R4 and R5** were not tested in the sense intended. The qualifying price produced a per-block gate
  (one of two blocks open everywhere). For such a gate both splits are zero and both position rates
  are equal by construction. No carry model made a belief-dependent read decision (§2).
* **R7** was mis-specified. Its threshold came from round 16's full-access KL (0.05), not the
  own-dropout KL (0.023) that the closed plain models reproduce.
* **R8** failed on substance. Both families cross at c* = 0.003, because the plain model gives up its
  reads of inferred states as cheaply as the carry model gives up its reads. What the plain model
  keeps is its read of the raw tokens (§3).

## Conclusions

1. **A price on reading the history induces full recurrence in a transformer that has a recurrent
   path.** From c = 0.03 nats per token the carry model closes every read and puts weight 1.000 on
   its carried state, even when the history is handed back. It is also more accurate than when it
   reads (KL 0.0019–0.0020 vs 0.0058). The recurrence is chosen, not imposed.
2. **The choice is made per block, not per position or per belief state.** At the intermediate price
   one of the two blocks reads everywhere and the other nowhere. The only partially open carry gate
   is keyed on the current token and is blind to the carried belief. Selective, belief-dependent
   reading did not appear, because in this environment a trained carry leaves no read worth paying
   for.
3. **Without a recurrent path, the price removes the reads the model was not using and keeps the
   token reads.** The plain transformer drops its reads of inferred states (blocks 1–3) at the
   smallest price with no loss. It keeps its read of the raw tokens until c = 0.1, then loses accuracy
   (KL 0.0215) while its prior weight stays at or below 0.190. This is round 16's depth ceiling
   again, with a smaller accuracy cost than registered, and it is round 15's recomputation seen as a
   choice.
