# A hidden goal: is the Bayesian posterior affinely recoverable? (TASK11, round 12)

Run date: 2026-09-23. Brief: `rounds/r12_hidden_goal/BRIEF.md` (claim 1 of three; the brief did not include claims 2
and 3). Theory and the eleven predictions in `rounds/r12_hidden_goal/THEORY.md` were committed (`9cdcad4`)
**before any model of this round was trained**. Numbers from `rounds/r12_hidden_goal/tables.md`, per-model
values in `rounds/r12_hidden_goal/results11.json`, weights in `rounds/r12_hidden_goal/models/`, figures
`fig1_interface.png`, `fig2_sites.png`, `fig3_extrapolation.png`. The follow-up in the last
section is not pre-registered (`rounds/r12_hidden_goal/long/`).

Reproduce: `rounds/r12_hidden_goal/run.py` (160 models, 15 min training + 11 min measurement on one GPU
and 96 CPU workers), `rounds/r12_hidden_goal/followup.py` (10 min), `rounds/r12_hidden_goal/tables.py`.

## Setup

A goal G ∈ {1..K} (K = 4 unless stated, uniform prior) is drawn per episode, and 24 noisy
evidence tokens follow. There are two evidence processes, both at MAP accuracy ≈ 0.78 by t = 24:

* **iid**: o_t ~ L[G]. The posterior log-odds are then an *affine function of the token
  counts* (best count-affine R² = 1.000), so recovering y cannot distinguish a belief from a
  tally.
* **channel**: a hidden sticky reliability channel (on: informative, off: uniform noise). The
  exact filter runs over (G, c) jointly and the best count-affine predictor of y reaches only
  R² 0.869. The remaining 13% is the part of the posterior that no running tally carries,
  measured by **G_count** = 1 − SSE(probe)/SSE(count-affine predictor).

Four objectives, all with exact per-position targets from the filter: `goal` (b_t itself),
`act_soft` (softmax of the action values over τ = 0.1), `act_hard` (the Bayes-optimal action:
commit to the MAP goal once its posterior exceeds ½, otherwise stay safe) and `next_obs` (the
posterior predictive). Two architectures: the round-10/11 2-layer pre-LN transformer (sites res0,
res1, res2, post-LN interface u) and the round-4 GRU (h). Probe: unregularised least squares
[h, 1] → y = (log b(g_k)/b(g_K))_{k<K}, exactly invariant to invertible affine maps of h.

Generalisation splits (fit on one set, score on a disjoint one): **IID** (by sequence),
**EXT** (fit on max|y| < 2, score on ≥ 3), **CONF** (fit outside a region where two goals are
both strongly supported, score inside it), **TIME** (fit t ≤ 12, score t ≥ 13), and **NETHO**
(the CONF split on models *trained* on pools that never enter that region).

Convergence (mean KL to target < 0.01 nats, or ≥ 99% agreement with the optimal action):
every `goal`, `act_soft` and `next_obs` model and every iid `act_hard` model converged. **No
channel-env `act_hard` model at K = 4 or 5 converged** (98.4–98.8% agreement), and 1 of 3 did
at K = 3. By the pre-registered rule those models are excluded from the prediction tests. The
follow-up shows that they are capped, not undertrained.

## 1. Where the objective forces it, the posterior is exactly affine — and it generalises

`goal` models, readout interface:

| | IID R²_y | EXT R²_y | CONF R²_y | TIME R²_y | NETHO CONF R²_y | G_count |
|---|---|---|---|---|---|---|
| transformer, iid | 1.000 | 0.999 | 1.000 | 0.999 | 1.000 | — |
| transformer, channel | 0.992 | 0.982 | 0.993 | 0.983 | 0.994 | 0.940 |
| GRU, iid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | — |
| GRU, channel | 0.996 | 0.991 | 0.996 | 0.992 | 0.996 | 0.973 |

A probe fit on |y| < 2 predicts log-odds out to ±10 nats on the line (fig. 3, left). In the
channel env the probe removes 94–97% of the error the best tally makes, so the representation
carries the nonlinear filter, not the counts. Models that never saw a sequence enter the
conflict region decode it as well as models that did (0.994 vs 0.993, 0.996 vs 0.996). Their
own output KL there is unchanged: 0.0004 vs 0.0006 nats for the transformer, 0.0005 vs 0.0004
for the GRU. Seed CV of interface IID R²_y is ≤ 0.0007 for these models.

## 2. The coordinates follow the target, and only held-out splits show it

In-distribution, almost everything decodes y well. The iid transformer reaches R²_y ≥ 0.969 at
its best site under every objective, and even the untrained transformer reaches 0.82. Held-out
extrapolation separates the objectives:

| interface | IID R²_y | IID R²_b | EXT R²_y | EXT R²_b |
|---|---|---|---|---|
| transformer channel, `goal` | 0.992 | 0.954 | **0.982** | **−0.972** |
| transformer channel, `act_soft` | 0.920 | 0.999 | **0.553** | **0.999** |
| transformer channel, `next_obs` | 0.935 | 0.991 | 0.667 | 0.965 |
| GRU channel, `goal` | 0.996 | 0.970 | **0.991** | **−1.644** |
| GRU channel, `act_soft` | 0.964 | 0.998 | **0.757** | **0.993** |

The objective with log b as its minimiser gives an interface affine in log-odds. The one with
b/τ as its minimiser gives an interface affine in probabilities. Each fails to extrapolate in
the other's coordinates (fig. 3): the `act_soft` and `next_obs` probes saturate at ±3 nats
because their targets stop changing there. This is the exact statement of §2.4 of the theory,
and in-distribution R² hides it (0.92 vs 0.99).

## 3. The iid environment does not discriminate; the channel environment does

In iid, every trained objective gives R²_y ≥ 0.969 at the transformer's best site, and y is
exactly affine in counts. Gain over counts at the interface, channel env:

| | goal | act_soft | next_obs | act_hard (not converged) | untrained |
|---|---|---|---|---|---|
| transformer u | 0.940 | 0.390 | 0.506 | −0.568 | −0.782 |
| transformer, best site | 0.940 (u) | 0.731 (res2) | 0.696 (res2) | 0.42 (res2, follow-up) | −0.752 |
| GRU h | 0.973 | 0.728 | 0.794 | 0.187 | −1.275 |

Every objective's minimiser is a function of the belief, and every trained model carries more
than a tally somewhere. Only `goal` puts the full nonlinear posterior in affine form at the
interface. In the transformer the nonlinear part appears in block 2 (goal: G_count 0.25 at res1,
0.94 at res2; fig. 2).

The first block does more than average. In iid, res1 of the `goal` transformer decodes y at
0.992, where the affine-in-(counts/t, t) baseline, which is what uniform attention would supply,
reaches 0.560. Its TIME drop is 0.063, so block 1 already codes something close to the
time-invariant sum, not a running mean.

## 4. Hard targets: the posterior upstream, the decision at the interface

In the transformer, iid `act_hard` models decode y at 0.969 at res1 and 0.878 at u, where the
optimal action is decodable at 1.000. The last block compresses the belief towards the decision
regions, the round 1–3 policy-quotient pattern. The iid GRU does not compress (0.982), because
its state is also the only carrier of the sufficient statistic to the next step.

Trained without the conflict region, `act_hard` models fail there behaviourally, and `goal`
models do not. KL inside the region is 2.71 nats (transformer) and 1.69 (GRU), against 0.012
and 0.097 for the same objective trained on the full pool. The unseen region contains a
decision boundary (commit to g₁ vs commit to g₂) that the no-conflict pool never shows. The
decision targets carry no information about where it lies, whereas the posterior targets
determine it.

## 5. Pre-registered predictions

| | prediction | held | the numbers |
|---|---|---|---|
| P1 | `goal` interface R²_y ≥ 0.99 IID, ≥ 0.97 on EXT, CONF, TIME, every cell | **yes** | lowest: tfm channel EXT 0.982 |
| P2 | `goal`: R²_y > R²_b; `act_soft`: R²_b > R²_y, every cell | **yes** | e.g. tfm channel act_soft 0.920 vs 0.999 |
| P3 | `act_soft` EXT R²_y ≥ 0.2 below `goal`'s, every cell | no | 0.31, 0.43, 0.23 — but GRU iid only 0.023 |
| P4 | `act_hard` IID R²_y ≥ 0.1 below `goal`'s; action ≥ 95% decodable | no | tfm iid 0.122 ✓; GRU iid 0.018 ✗; channel untestable (not converged) |
| P5 | iid transformer: best site R²_y ≥ 0.95 for all objectives | **yes** | lowest act_hard 0.969 |
| P6 | channel: `goal`, `next_obs` G_count ≥ 0.7; `act_hard` < `goal` | no | goal 0.940 / 0.973 ✓; next_obs tfm 0.696 ✗, GRU 0.794 ✓; act_hard untestable |
| P7 | untrained: R²_y ≤ 0.8 everywhere; channel G_count ≤ 0.3 | no | tfm iid 0.819 ✗; all G_count ≤ −0.75 ✓ |
| P8 | NETHO `goal`: CONF R²_y ≥ 0.9; KL in region ≤ 3× outside | no | R²_y 0.994–1.000 ✓; KL ratio 4.0 / 4.8 in channel ✗ |
| P9 | tfm `goal` iid res1: TIME R²_y ≥ 0.1 below IID | no | 0.063 |
| P10 | seed CV of interface R²_y ≤ 0.02 | **yes** | max 0.016 |
| P11 | P1 and `goal` P6 at K = 3, 5 | no | K = 5 tfm channel IID 0.988 (< 0.99) |

4 of 11 held. The failures are of three kinds:

* **Near-misses on a threshold, with the claimed effect present:** P6 (tfm `next_obs` 0.696
  against 0.7), P7 (0.819 against 0.8) and P11 (0.988 against 0.99; EXT, CONF and TIME all
  ≥ 0.975).
* **A badly designed criterion:** P8. The KL ratio inside vs outside the conflict region is
  just as high in models trained on the full pool (3.75, 3.50) as in NETHO models (4.00, 4.79).
  Conflict states are harder, not unseen. The comparison that tests generalisation, KL inside
  the region with vs without training exposure, shows no cost (§1). The recoverability half of
  P8 held.
* **Substantive failures:**
  * P3 and P4 for the GRU in iid. The GRU state carries log-odds affinely even when the target
    does not require it (act_soft EXT R²_y 0.977, act_hard IID 0.982), so "coordinates follow
    the target" holds for the transformer's interface but not for a recurrent state that must
    also carry the statistic forward.
  * P9. The transformer's first block does not code a running mean.
  * The `act_hard` halves of P4 and P6 in the channel env could not be tested because none of
    those models converged.

## 6. Follow-up (not pre-registered): the channel `act_hard` models are capped

At 4× the steps (80k transformer, 60k GRU), agreement with the optimal action goes *down*,
from 0.988 to 0.985 (transformer) and from 0.987 to 0.985 (GRU). The probes do not move:
interface IID R²_y 0.794 → 0.789 and 0.893 → 0.885. More training does not close the gap.

The models are also not tally policies. A policy that acts on the best count-affine posterior
agrees with the optimal action on only 83.3% of channel states (100% in iid), and the `act_hard`
models reach 98.5%. The transformer's res2 carries the posterior with G_count 0.40–0.46, and the
last block compresses it (u: G_count −0.4 to −0.8, action decodable at 0.98). The models compute
most of the nonlinear filter and stop short of the part that only matters for the 1.5% of states
nearest a decision boundary.

## Conclusions

1. **Claim 1 holds where the objective asks for the posterior, and it holds out of
   distribution.** An affine map of the readout interface recovers the K−1 log-odds with R²
   ≥ 0.975 on every held-out split, in both architectures and at K = 3, 4, 5. This includes
   posterior values neither the network nor the probe saw in training, and an environment
   where the posterior is a nonlinear filter no tally reproduces (94–97% of the tally's error
   removed).
2. **In-distribution R² does not distinguish the hypotheses.** Under i.i.d. evidence the
   log-odds are affine in counts, and every trained model, plus an untrained transformer at
   0.82, decodes them well.
3. **Which coordinates are affine is set by the objective, and extrapolation is what reveals
   them.** `goal` gives an interface affine in log b, `act_soft` one affine in b. Each fails in
   the other's coordinates on the EXT split (R²_b −0.97 for `goal`; R²_y 0.55 for `act_soft`),
   while their in-distribution R² values differ by at most 0.07.
4. **Upstream, the posterior is present under every objective, but only partially as a
   nonlinear filter** (transformer res2 G_count 0.40–0.94 across objectives). Hard decision
   targets compress it at the last block, and models trained on them fail in an unseen region
   where posterior-trained models do not.
5. **Scorecard:** 4 of 11 pre-registered predictions held. Of the 7 failures, 3 are
   near-misses on a threshold, 1 is a mis-specified criterion and 3 are substantive (the GRU
   carries log-odds regardless of target; the first transformer block is not a running mean;
   channel hard-target models are capped below the convergence criterion).
