# H on the belief simplex: the history profile is a logit of the belief, not a belief-weighted table (H simplex)

Run date: 2026-10-08. Plan, measures and decision rule: `PLAN.md`, committed (`8819944`) **before any trained model
was measured**. Models: the maze10 experiment's six reward-trained transformers (55-cell aliased maze, four goals).
Numbers from `tables.md` (registered) and `posthoc.md` (post hoc, after the registered result was seen); data in
`results.json`, `posthoc.json`, per-cell tables in `fits.pt`. Reproduce: `reproduce.py H_simplex` (15 min on one GPU).

**Setting.** H(h) is the goal-averaged, action-centred logit profile at every decision of each model's own greedy
episodes (4 096 fit episodes, 4 096 separate test episodes; about 80 000 decisions each). Each decision carries the
exact posterior b over the 55 cells, the step, and the prefix tokens. Every model of H is fitted on the fit episodes
(penalties and early stopping chosen on a tenth of them, by episode) and scored by held-out R² on the test episodes,
pooled over the four centred numbers. G(g, step) is the maze10 experiment's goal bias, refitted on the fit episodes.

## 1. H is not affine in the belief; it is a nonlinear function of it, and of little else

| model of H | held-out R² (median, min–max over 6 models) |
|---|---|
| step only | 0.122 (0.096–0.233) |
| **affine in b, the QMDP form Σ_s b(s) h_s** | **0.445 (0.436–0.546)** |
| belief-weighted optimal share popt, 4 × 4 map | 0.260 (0.177–0.356) |
| belief-weighted reachability, 4 × 4 map | 0.258 (0.194–0.349) |
| both | 0.300 (0.224–0.385) |
| table at the most likely cell | 0.426 (0.408–0.468) |
| affine + sharpened belief b² | 0.549 (0.533–0.640) |
| affine + entropy-scaled profile | 0.624 (0.582–0.677) |
| **MLP on b** | **0.938 (0.911–0.943)** |
| MLP on b + prefix features | 0.983 (0.977–0.985) |
| ridge on b + prefix features | 0.702 (0.655–0.717) |

* **QMDP fails** (0.445 against the registered 0.75). The small maze's 0.82 does not scale: on the 55-cell simplex
  the best affine function of the posterior explains less than half of H.
* **NONLIN fails, by a wide margin**: a two-layer MLP on the same posterior explains 0.94. The missing half is a
  **nonlinear function of the belief**, not information beyond it: the prefix record adds 0.05 on top of the MLP
  (**BEYOND fails**: the beyond-b gain is a tenth of the nonlinear-in-b gain). Both of the plan's named nonlinearities
  (sharpening, an entropy-scaled profile) move in the right direction and fall far short (0.55, 0.62).
* **MIX fails**: the two solver-free per-cell candidates, mixed under the belief, explain 0.30 (the small maze: 0.82).
  The three-number version is at 0.05.

## 2. The per-cell profiles of the affine fit are weak

| measure | value |
|---|---|
| cells whose top action in h_s is a shortest-path move for some goal | 0.80 (0.75–0.93) |
| correlation of h_s with popt(s, ·) / reach(s, ·) / their best mix | 0.25 / 0.08 / 0.28 |

* **PROF fails.** With the affine form explaining under half of H, its per-cell table is poorly determined (figure
  `profiles.png`: a few large entries, most near zero). The post hoc section replaces it with the table of the form
  that does fit.

## 3. Decisions: the nonlinear part of H matters; the affine part keeps two thirds

Offline, on test decisions where the model's move depends on the goal (0.93 of decisions), the share where
argmax(Ĥ + G) is the model's move; online, the policy argmax(Ĥ(b_t) + G(g, t)) run in the environment, a function of
(b, g, t) only.

| Ĥ | offline agreement | online return (success) | recovery of the goal-directed return |
|---|---|---|---|
| H itself (the additive code) | 0.867 | 0.482 (0.76) | 0.81 (0.73–0.86) |
| affine in b | 0.748 | 0.413 (0.63) | 0.63 (0.41–0.78) |
| mix (popt + reach) | 0.704 | 0.398 (0.65) | 0.56 |
| mix3 (three numbers) | 0.661 | 0.311 (0.50) | 0.30 |
| MLP on b | 0.851 | 0.491 (0.76) | **0.80 (0.76–0.85)** |
| goal-blind argmax H | 0.550 | 0.189 (0.32) | 0 |
| natural policy | | 0.548 (0.84) | 1 |

* **DEC fails** (offline 0.748 against 0.9 × 0.867 = 0.78; recovery 0.63 against 0.9 × 0.81 = 0.73): dropping the
  nonlinear part of H costs a fifth of the additive code's recovery. **NONET fails** (0.30).
* **The MLP on the posterior recovers the additive code in full** (0.80 against 0.81; offline 0.85 against 0.87): a
  function of the exact belief alone, plus the goal bias, is as good a policy as the additive code read from the
  network. So the history enters the decision through the posterior, as in the small maze; what differs is the shape
  of the map from posterior to profile.

## 4. Post hoc: what the nonlinearity is (after the registered result was seen; `posthoc.md`, `posthoc2.md`)

**Not a logit of a belief-weighted action probability.** The natural reading of a logit profile, H ≈ log Σ_s b(s)
p_s(a) with a free probability table per cell (linear in b in probability space, then the log), explains 0.455
(0.424–0.529): no more than the affine form. Fitted as a policy it recovers 0.49 of the goal-directed return.

**Not a gain on a fixed direction.** The affine fit of H's unit direction is 0.39 (the MLP's 0.93): the belief changes
*which* actions H prefers, not only by how much. ‖H‖ is uncorrelated with the posterior's entropy (0.01) and with the
most likely cell's probability (0.05).

**The map changes with uncertainty.** A separate per-cell table per entropy bin (eight bins) explains 0.78, and the
eight tables are nearly unrelated to one another (correlations 0.1–0.5 between neighbouring bins, ≈ 0 between distant
ones): at low entropy H reads one table, at high entropy another.

**It is close to quadratic in the belief.**

| features (ridge, held-out R²) | R² |
|---|---|
| affine in b | 0.445 (0.436–0.546) |
| affine in log b | 0.619 (0.567–0.673) |
| affine + expected information gain of each move + entropy | 0.521 |
| affine + the two most likely cells (one-hot) and their probabilities | 0.624 |
| log b + the two most likely cells | 0.726 |
| log b, a table per entropy bin | 0.849 (0.810–0.881) |
| **b and all pairwise products b(s) b(s′)** | **0.861 (0.832–0.875)** |
| MLP on b / on log b | 0.938 / 0.952 (0.924–0.956) |

* Pairwise products of the belief carry most of what the affine form misses: 0.86 against 0.45, within 0.08 of the
  MLP. The expected information gain of the moves, the obvious nonlinear quantity a policy that seeks information
  would compute, adds 0.08 to the affine form and is 0.23 alone: the nonlinearity is not (mainly) a value of
  information term.
* log b is a better coordinate than b (0.62 against 0.45 affine; 0.95 against 0.94 for the MLP), and the beliefs along
  the episodes are often concentrated (most likely cell's probability: quartiles 0.14, 0.45, 0.78).

**Reading.** In the 55-cell maze H is a function of the posterior (an MLP on b recovers the additive code's return in
full, 0.80 against 0.81, and the prefix record adds 0.05 of variance), but not the belief-weighted average of a
per-cell table that the small maze suggested. It is, to a good approximation, a **second-order** function of the
belief: the profile for a belief split between cells s and s′ is not the mixture of the two cells' profiles. What the
pair terms encode, whether a route that serves both candidate cells, a commitment to one, or the disambiguating move,
is the next question; the first test is to compare the fitted pair table h_{ss′} on pairs of aliased cells with the
actions that are optimal for both, for the likelier, and for information.

## 5. Scorecard

| | criterion | value | held |
|---|---|---|---|
| **QMDP** | R²(affine) ≥ 0.75 | 0.445 | **no** |
| **MIX** | R²(mix) ≥ 0.9 × R²(affine) | 0.300 against 0.400 | **no** |
| **NONLIN** | nonlinear-in-b gain ≤ 0.05 | 0.494 | **no** |
| **BEYOND** | beyond-b gain > nonlinear-in-b gain | 0.046 against 0.494 | **no** |
| **PROF** | top action optimal for some goal ≥ 0.85 and corr with mix ≥ 0.7 | 0.80, 0.28 | **no** |
| **DEC** | affine ≥ 0.9 × H offline and online | 0.748 against 0.780; 0.63 against 0.73 | **no** |
| **NONET** | mix3 recovery ≥ 0.8 × additive | 0.30 against 0.65 | **no** |

| | expectation | held |
|---|---|---|
| E1 | QMDP holds (0.75–0.85) | no (0.445) |
| E2 | MIX holds | no |
| E3 | NONLIN holds; mls well below affine | no; mls 0.43 against affine 0.45 |
| E4 | BEYOND holds: the prefix record adds 0.05–0.10, more than any nonlinearity in b | no: it adds 0.05, a tenth of the nonlinearity |
| E5 | PROF holds | no |
| E6 | DEC holds | no |
| E7 | NONET holds | no |

0 of 7 expectations held. The registered reading for "not QMDP, NONLIN fails" applies: **H is a nonlinear function of
the belief; the small-maze result does not scale.** The post hoc analysis locates the nonlinearity in pairwise
interactions between cells of the belief (0.86 quadratic against 0.45 affine; MLP 0.94) and not in the value of
information, the magnitude of H, or history beyond the belief.

## Limits

Decisions are the models' own, so the fitted region of the simplex is the one their policies visit; beliefs there are
often concentrated, which favours low-order fits. R² is pooled over the four centred numbers and weights every
decision equally. No causal edit is made here; the maze10 experiment's removals are the causal evidence for the
history part, and the online runs show that a function of the exact posterior plus G reproduces the additive code's
return. The post hoc models were chosen after the registered result and after the user suggested the logit reading.

## 6. Post hoc: H is computed through the encoded posterior, not in parallel to it (`causal.md`, `causal2.md`)

Exploratory, no registration. Matched pairs of decisions (recipient A, donor B; the same step; posteriors differing by
1.6 of mass in L1; 4 096 pairs per model); the decision token's residual stream edited at the input of block l (site
l), the same vector under every goal; read-outs: the transfer of H toward the donor's (projection on H_B − H_A) and,
among pairs whose decisions differ, the share now taking the donor's.

**Where H is determined.** Swapping the decision token's whole state (per goal) at site 2, after block 1, transfers
0.96 of H and 0.89 of decisions; at site 1, 0.76; at site 4, 1.00. Blocks 2 and 3 compute H from that token's own
state; the history is not re-read.

**How the posterior is encoded there.** From the goal-averaged state the posterior decodes linearly at 0.49, in log
coordinates at 0.76, by an MLP at 0.81, from site 1 on and unchanged after: a nonlinear code made by block 0. The
state is predicted *from* the posterior at 0.57–0.65 linearly and **0.93–0.97 by an MLP** (inputs b and log b): all but
3–7 % of the goal-averaged state's variance is a function of the posterior.

**The decomposition test.** x̄ = f(b) + r at each site, f the MLP encoder; swap one part at a time:

| site | edit | H transfer | decision to donor |
|---|---|---|---|
| 2 (after block 1) | **posterior part f(b_B) − f(b_A)** | **0.75 (0.72–0.80)** | **0.74 (0.70–0.80)** |
| 2 | residual part r_B − r_A | 0.07 (0.05–0.10) | 0.06 (0.04–0.10) |
| 2 | both parts (the goal-averaged difference) | 0.83 | 0.80 |
| 2 | linear belief edit | 0.34 | 0.33 |
| 2 | random direction, the posterior part's norm | 0.05 | 0.06 |
| 2 | whole per-goal swap | 0.96 | 0.89 |
| 4 (final) | posterior part | 0.82 (0.80–0.84) | 0.82 (0.75–0.84) |
| 4 | residual part | 0.09 | 0.08 |
| 4 | both parts | 0.92 | 0.90 |

* **The posterior part carries H; the residual does not.** Moving the part of the state that the posterior predicts
  moves H three quarters of the way to the donor's and switches three quarters of the differing decisions; moving the
  part it does not predict (the 5 % of the state that is history beyond the belief) does what a random direction does.
  Together they reach 0.83–0.92; the rest, up to the whole per-goal swap, is the goal-specific part of the state that a
  goal-free vector cannot carry.
* The linear belief edit's 0.35 was the instrument, not the mechanism: the posterior code is nonlinear (log-like), and
  the nonlinear encoder's edit recovers what the linear one missed.

**Reading.** The network computes its action response through its encoded posterior: the decision token holds a
nonlinear code of the belief, formed by block 0, from which blocks 2–3 compute H, including its pair terms; the
posterior and the response are not parallel consequences of the history. This is the causal counterpart of the
behavioural result that an MLP on the exact posterior recovers the additive code's return in full.

## 7. Post hoc: the encoded posterior is reusable for new decisions, and the posterior part transfers them (`readout.md`)

Exploratory. Seven new decision tasks per belief: the QMDP-optimal move under the exact posterior for six new goal
cells (the interior-goals experiment's junction goals, none a trained goal) and for a new rule (toward the nearest
landmark). Readouts (linear, or an MLP of 64 units) from the frozen, goal-averaged decision-token state, fitted on
24 576 fit decisions, scored by hit rate on 8 192 test decisions. **Matched-preference subset**: test decisions paired
with another of the same step, the same own-goal move and nearly the same H (within the 25th percentile of H
distances), whose optimal moves for the task are disjoint, so the current preferences do not tell them apart (7–17 %
of decisions per task; chance there 0.44).

| readout from | all decisions | matched-preference subset |
|---|---|---|
| majority move | 0.58 | 0.44 (chance) |
| H (the current preferences), MLP | 0.85 | 0.69 |
| all 16 current logits, MLP | 0.92 | 0.81 |
| **state after block 1, MLP** | **0.96** | **0.91** |
| state after block 1, linear | 0.89 | 0.76 |
| final residual, MLP | 0.95 | 0.90 |
| the exact posterior, MLP (ceiling) | 0.99 | 0.99 |

* **A small readout of the state supports the new decisions** (0.96 of decisions; 0.91 where current preferences
  cannot distinguish the beliefs, against 0.69 from H and 0.81 from every current logit). The representation carries
  the belief beyond what the trained goals' decisions use; a linear readout is weaker (0.76 on the matched subset),
  as expected of a nonlinear code.

**Transfer.** On the matched pairs of section 6 whose optimal moves for a task are disjoint, the swap at the input
of block 2, propagated through blocks 2–3 to the final residual and read by the new readouts there:

| edit at the input of block 2 | new decision to the donor's | kept the recipient's |
|---|---|---|
| **posterior part f(b_B) − f(b_A)** | **0.74 (0.72–0.77)** | 0.16 |
| residual part | 0.06 | 0.90 |
| random direction, the posterior part's norm | 0.18 | 0.66 |
| whole swap | 0.76 | 0.15 |
| no edit | 0.04 | 0.93 |

* **Swapping the posterior-predicted component transfers the new decisions as completely as swapping the whole
  state** (0.74 against 0.76), and the residual transfers nothing (0.06). The same component that carries H (section
  6: 0.75) carries decisions the network was never trained to make.

**Reading.** The decision token holds a reusable belief: a nonlinear code of the posterior, formed by block 0, that a
small new readout can turn into new decisions, and whose causal content for those decisions is the same
posterior-predicted component that drives the trained policy. This strengthens the interpretation of section 6: the
posterior is represented as such, not merely as whatever the current action preferences happen to need.

## 8. Post hoc: the goal changes what is read from the belief, from block 0 on (`goal_belief.md`)

Exploratory. History fixed, the goal token changed: the decision token's state x(h, g) = hist(h) + M(g) + I(h, g) at
every residual site and component output; the posterior-predictable part compared between one shared encoder plus a
goal offset, f(b) + M(g), and per-goal encoders f_g(b) (MLPs, 12 288 fit decisions, 6 144 test).

| site / component | hist / goal / interaction (variance shares) | R² shared + offset | R² per-goal | interaction from b |
|---|---|---|---|---|
| after block 0 (resid1) | 0.67 / 0.16 / 0.17 | 0.78 | 0.94 | 0.96 |
| after block 1 (resid2) | 0.55 / 0.25 / 0.21 | 0.76 | 0.96 | 0.95 |
| final residual | 0.50 / 0.23 / 0.25 | 0.72 | 0.97 | 0.96 |
| block 0 attention / MLP | 0.68 / 0.17 / 0.15 and 0.68 / 0.14 / 0.17 | 0.79 / 0.78 | 0.93 / 0.94 | 0.96 / 0.95 |
| block 1 attention / MLP | 0.50 / 0.37 / 0.13 and 0.50 / 0.23 / 0.27 | 0.84 / 0.70 | 0.96 / 0.96 | 0.94 / 0.95 |
| blocks 2–3 MLPs | interaction 0.32, 0.33 | 0.66, 0.65 | 0.96 | 0.95 |

* **The goal × history interaction is a function of the belief (0.95 from b per goal) and is there from block 0's
  attention onward** (0.15 of that output's variance; 0.17 of the residual after block 0). A shared belief code plus a
  goal offset explains 0.72–0.78 of the state; letting the goal choose the encoder explains 0.94–0.97. The gap (0.16
  after block 0, 0.24 at the end) is the goal-dependent reading of the belief; the MLPs add most of it (block 1's MLP
  0.27, blocks 2–3's MLPs 0.32–0.33 interaction shares), the attention outputs carry more of the pure goal offset.
* So the computation is not b → H(b), L = H(b) + G(g) at the level of the state: the goal changes which features of
  the belief the state holds, already at the first component, and increasingly through the MLPs.
* **Correction (section 15).** The decomposition counts the step as part of the history, so a goal offset that
  depends on the step is counted as interaction. With the offset allowed to depend on the step the goal × belief
  interaction proper is 0.06 / 0.09 / 0.13 at the input of block 1 / of block 2 / at the end, and the encoder gap
  0.05 / 0.08 / 0.12: the coding is goal-specific, but by less than the figures above say.

**One donor belief component under every recipient goal** (the posterior-part edit at the input of block 2, logits
read under each of the four goals):

| | goal-specific share of the logit change | mean cosine between the goals' changes | transfer of the natural change |
|---|---|---|---|
| the edit | 0.16 | 0.78 | shared part 0.87, goal-specific part 0.26 |
| the natural change (donor − recipient) | 0.44 | 0.41 | |

* The edited belief component produces a change that is mostly **shared across goals** (0.84 of its variance; cosine
  0.78 between goals), and it reproduces 0.87 of the shared part of the natural change but only 0.26 of its
  goal-specific part, which is 0.44 of the natural change. Decisions switch to the donor's in 0.55 under the four
  goals (0.70 under one of them), 0.45 under the episode's own goal.
* So what blocks 2–3 compute from a goal-free belief vector is largely goal-independent; the goal-specific part of the
  response is made earlier, in the goal-dependent reading of the history that the belief component (goal-averaged by
  construction) does not carry.

**Localisation by removal** (one component's interaction at a time replaced by its history and goal parts, offline):
no single component matters much (own-goal decisions changed 0.02–0.10; the two late MLPs most), all eight together
change 0.20 of decisions and leave 0.77 of the goal-dependence (natural 0.93). The goal-dependent reading is
distributed over the components, as the maze10 removals found online.

**Reading.** At the level of the output the additive code holds to first order, but the state is not additive: from
block 0's attention, each goal extracts its own function of the belief, the MLPs deepen it, and the goal-free belief
component carries the shared part of the response while the goal-specific part rides on the goal-dependent reading.
A belief-encoding edit therefore transfers the shared response (and new readouts, section 7), not the full
goal-conditioned one; the whole per-goal swap (0.96) does both.

## 9. Post hoc: beliefs equivalent for one goal, different for another (`equiv.md`)

Exploratory. Pairs of test decisions at the same step whose horizon-free QMDP profiles (Σ_s b(s) γ^d, in effective
steps) are within 0.5 step of each other with the same optimal set under goal A, with A within reach (value ≥ 0.4),
and at least 1 step apart with disjoint optimal sets under goal B; pooled over ordered goal pairs, about 5 300 pairs
per model, their posteriors 1.9 apart in L1. The model's own decision is the same for the two beliefs under A in 0.60
and under B in 0.39. (A first selection on horizon-limited values was degenerate: late decisions with goal A out of
reach, every action worth 0; it was discarded.)

**The representation distinguishes them equally at first, then more under B.** Distance between the two states,
over the typical same-step distance under that goal: after block 0, 0.87 under A and 0.88 under B; after block 1,
0.84 and 0.93; at the final residual, 0.66 and 0.94 (logits 0.72 and 1.11). The late blocks amplify the distinction
under the goal it matters to and attenuate it under the goal it does not.

**The distinction is fully decodable under A.** An MLP decoder of B's optimal move from the state under A hits 0.94
on these pairs (from the state under B 0.94; chance 0.43), at both sites. What is irrelevant to A's decision is
retained in A's state.

**Patched, its effect depends on how it is represented** (added at the input of block 2 under the recipient goal only):

| vector added to b1's state | under B: to b2's decision | under A: decision changed |
|---|---|---|
| the distinction as represented under B, x2(B) − x1(B) | **0.91** | 0.39 |
| the distinction as represented under A, x2(A) − x1(A) | 0.31 | 0.40 |
| the goal-free posterior part f(b2) − f(b1) | 0.48 | 0.35 |
| random, the B-distinction's norm | 0.04 | 0.08 |

* The same information, written as goal A writes it, moves B's decision a third as often as written as goal B writes
  it (0.31 against 0.91). The goal-free belief component is in between (0.48). So what changes across goals is not
  whether the distinction is kept, it is kept and decodable either way, but **how it is represented**: each goal
  holds the belief in its own coordinates from block 0 onward (section 8), and blocks 2–3 read those coordinates.
* Every non-random vector also changes A's decision in 0.35–0.40 of pairs, although the beliefs are A-equivalent: the
  recipient's A-decision is not invariant to a change of belief that leaves A's values unchanged. (The model's own
  A-decisions agree on only 0.60 of these pairs, so A-equivalence by exact values does not imply equivalence for the
  network's policy.)

**Reading.** Across goals the information is the same and the format differs: a goal-specific encoding of the belief
that the late blocks read, with the goal-free posterior component as its common part. This is the mechanism behind
the earlier pattern: belief edits transfer the shared response (0.87 of the natural change's shared part) but not the
goal-specific one (0.26), while a change written in the recipient goal's own coordinates transfers almost fully.

## 10. Post hoc: the change of representation across goals is systematic and largely linear (`equiv2.md`)

Exploratory. For each ordered goal pair (A, B), a map from the A-written difference x2(A) − x1(A) to the B-written
difference at the input of block 2, fitted on 12 288 random same-step pairs of fit decisions (posteriors ≥ 0.5
apart), applied to the held-out A-equivalent, B-different pairs of section 9 (test episodes, about 450 per goal
pair): x1(B) + T(Δ_A), logits read under B.

| vector added under B | B's decision to b2's | held-out R² for the B-written difference |
|---|---|---|
| direct B-written difference | 0.91 (0.89–0.96) | 1 |
| **linear map of the A-written difference** | **0.86 (0.80–0.89)** | 0.95 (0.94–0.96) |
| MLP on differences | 0.89 | 0.96 |
| MLP from the A-written state to the B-written state | 0.91 | 0.99 |
| untransformed A-written difference | 0.33 (0.24–0.40) | 0.39 |
| random, the same norm | 0.05 | |

* **Strong transfer.** A single linear map per goal pair, fitted on unrelated pairs of beliefs, turns the A-written
  difference into one that moves B's decision 0.86 of the time on entirely held-out pairs, against 0.33 untransformed
  and 0.91 for the B-written difference itself; it predicts the B-written difference at R² 0.95. Nonlinear maps add
  little (0.89 on differences, 0.91 from state to state).
* So the goal-specific representation of the belief (sections 8–9) is a **systematic, reusable change of coordinates**
  between goals, close to linear, not a belief-specific recoding. Together with the shared posterior component
  (section 6) this gives the structure of the decision token's state: a common nonlinear code of the posterior, read
  through a goal-dependent, approximately linear transformation that each goal applies from block 0 onward.

## 11. Post hoc: the pairwise maps form one consistent system (`equiv3.md`)

Exploratory. The 12 linear maps of section 10 composed and inverted, evaluated on actual held-out state differences
(4 096 random same-step pairs of test decisions, posteriors ≥ 0.5 apart), not as matrices; the effect of a composition
on the A-equivalent, C-different pairs of section 9, logits read under C.

| | held-out R² for the target difference | R² against the direct map's prediction | norm / direct's | decision to b2's |
|---|---|---|---|---|
| direct map M(A→C) Δ_A | 0.95 (0.94–0.96) | 1 | 1 | 0.86 (0.80–0.89) |
| **composed M(B→C) M(A→B) Δ_A** | 0.94 (0.93–0.95) | **0.99 (0.98–0.99)** | 0.98 | **0.85 (0.79–0.89)** |
| untransformed Δ_A | 0.36 | | | 0.33 |
| **round trip M(B→A) M(A→B) Δ_A** (target Δ_A) | **0.98 (0.97–0.98)** | | 0.96 | **0.86 (0.82–0.89)** (Δ_A itself 0.90) |
| three-cycle M(C→A) M(B→C) M(A→B) Δ_A (target Δ_A) | 0.96 (0.96–0.97) | | 0.94 | 0.84 |
| random, the target's norm | | | | 0.04–0.06 |

* **Composition commutes with the direct map.** Going A → B → C predicts the same B-independent vector as A → C
  (R² 0.99 between the two predictions, cosine 0.99, in every one of the 144 model × triple cases 0.97–0.99), fits
  Δ_C as well (0.94 against 0.95) and moves C's decision as often (0.85 against 0.86; the difference per case
  −0.01 on average, −0.13 to +0.08).
* **The round trip returns the difference.** A → B → A recovers Δ_A at R² 0.98 (every model and pair 0.95–0.99) with
  4 % shrinkage, and switches A's decision in 0.86 of the pairs whose A-decisions differ, against 0.90 for Δ_A itself
  and 0.06 for random; a three-goal cycle 0.96 and 0.84. The losses are the ridge's shrinkage and the 5 % the maps
  miss per hop, not a change of content.

**Reading.** The goal-specific representations are related by one system of invertible, composable, close-to-linear
transformations: a change of coordinates between goals, not a set of unrelated encodings with a pairwise fit between
each two. With section 6 this gives the decision token's state the structure of a common nonlinear code of the
posterior carried in goal-specific coordinates, where the coordinates for any goal follow from those of any other by a
fixed linear map.

## 12. Post hoc: in block 0 the goal does not change what is retrieved; it adds gated content, which the MLP then transforms (`block0.md`)

Exploratory. The goal is one token (position 1). For every decision after the first the decision token's embedding
is goal-free, so in block 0 its query is goal-free and the goal reaches it by two routes only: the goal token's
**key**, which reshapes the attention pattern over the history, and the goal token's **value**. History fixed, goal
A → B over the 12 ordered pairs (8 192 test decisions after step 0), each route swapped alone and block 0's MLP applied
to the result; the state measured against the natural change x(h, B) − x(h, A), against the fitted change of
coordinates at the input of block 1, and at the decision.

**Structure.** With a goal-free query, changing the goal rescales every history weight by a single factor per head
(an identity, checked to 1e-7): the pattern route changes *how much* of the history aggregate enters each head and
nothing about *which* tokens. The decision token puts 0.23 of its block-0 attention on the goal token. By the same
algebra the value route's belief-dependent part is exactly four history-dependent scalars (each head's attention mass
on the goal token) times four fixed goal-difference vectors: a per-head gate on the goal's content.

| route swapped A → B | attention output: projection on the natural change | its interaction part, R² | state after block 0: projection | its interaction part, R² | difference on held-out pairs vs Δ_B, R² (Δ_A: 0.66; M(A→B) Δ_A: 0.98) | decision switches to B's (ceiling 0.37) |
|---|---|---|---|---|---|---|
| pattern (goal token's key) | 0.21 (0.18–0.30) | 0.22 | 0.21 | 0.23 | 0.76 | 0.09 |
| value (goal token's value) | **0.79 (0.70–0.82)** | **0.73** | **0.79** | **0.78** | **0.90** | **0.24** |
| both (= the natural change) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 |

* **The value route carries four fifths of the goal's effect on the block-0 state, and the same share of its
  belief-dependent part** (0.73–0.78 of the goal × history interaction's change; the pattern route 0.22–0.23). The two
  routes are additive (projections sum to 1.00). Measured against the change of coordinates, the value-route states'
  difference reaches R² 0.90 for the B-written difference (untransformed 0.66, the fitted map 0.98; pattern route
  0.76). At the decision (the token's block-0 state patched under A), the value route switches 0.24 of the decisions
  that differ between the goals, the pattern route 0.09, both 0.37, which is the ceiling for this token's block-0
  state (the goal also reaches blocks 1–3 through the goal token and the history tokens; random 0.03).
* **What the MLP adds.** The goal change at the MLP's input is a large step (0.71 of the state's norm), and the MLP's
  response is curved: its first-order response at A explains 0.27 of its change (the interaction part 0.39), at the
  midpoint of the step 0.85, while a single linear map of the attention change fitted over decisions explains 0.92.
  The MLP amplifies the goal change 1.55× (the value route's 1.53×, the pattern route's 1.42×) and raises the
  interaction share from 0.13 of the attention output to 0.16 of its own output; the value route's MLP change matches
  the natural MLP change at R² 0.72, the pattern route's not at all (−0.48).

**Reading.** At block 0 the sharper question has a definite answer: the goal does not change what is retrieved from the
history, which is structurally impossible here and, through the one lever it has (the goal token's share of
attention), accounts for a fifth of the effect. The goal's content enters through its value, gated per head by how
much attention the goal token wins against the history, and block 0's MLP transforms that gated mixture nonlinearly,
amplifying it and adding interaction. The goal-specific coordinates of sections 8–11 begin as gated goal content plus
a nonlinear MLP response to it; goal-dependent retrieval can only start in block 1, where the decision token's query
first carries the goal.

## 13. Post hoc: block 1 does not select the history by goal; it reads goal content the history tokens already carry (`block1.md`)

Exploratory. At the input of block 1 every token but the first carries the goal (each attended to the goal token in
block 0): the decision token's own state, the history tokens' and the goal token's. Block 1's attention at the
decision token is recomputed with one source of the goal switched A → B and the rest held at A (8 192 test decisions
after step 0, 12 ordered goal pairs), block 1's MLP applied to the result. The query route is measured as a
redistribution over the history tokens (total variation of the per-head attention renormalised over history tokens,
goal token excluded) and causally; the state patched at the input of block 2 under A for the decision.

| goal source switched A → B | TV over history tokens (natural 0.24) | block-1 attention output: share of the natural change | state after block 1: share | its belief-dependent part, R² | decisions switched to B's (all 0.86, random 0.03) |
|---|---|---|---|---|---|
| decision token's **query** | 0.14 | 0.04 | 0.03 | 0.09 | 0.03 |
| history tokens' **keys** only | 0.19 | 0.02 | 0.03 | 0.06 | 0.04 |
| history tokens' **values** only | 0 | **0.63** | **0.44** | 0.20 | **0.29** |
| decision token's own key and value | 0.08 | 0.12 | 0.05 | 0.04 | 0.03 |
| goal token's key only / value only | 0 / 0 | 0.02 / 0.15 | 0.01 / 0.08 | 0.04 / 0.06 | 0.02 / 0.04 |
| residual only (block-0 content carried, no attention change) | 0 | 0 | 0.32 | **0.58** | 0.25 |
| token's own block-0 state (query + self + residual) | 0.17 | 0.17 | 0.41 | **0.70** | 0.37 |
| attention sources (history + goal tokens) | 0.19 | 0.83 | 0.59 | 0.37 | 0.43 |

* **Selection changes, and does not matter.** The goal-laden query redistributes the decision token's attention over
  the history tokens by 0.14 of total variation (the natural change 0.24), and the history tokens' goal-laden keys by
  0.19, as much as keys and values together. Neither has causal weight: the query route carries 0.04 of block 1's
  attention-output change and switches decisions at the random rate (0.03); the keys-only route 0.02 and 0.04. The
  decision token's block-1 attention on the goal token is 0.10, and the goal token's key moves it by 0.02.
* **Content matters.** The history tokens' values alone carry 0.63 of the attention-output change, 0.44 of the state
  change and switch 0.29 of goal-dependent decisions: the history has been rewritten goal-dependently by block 0
  (each history token gated the goal's content as the decision token did, section 12) and block 1 reads that content
  with an unchanged selection. The goal token's own value adds 0.15 at the attention output and 0.04 of decisions.
* **Continuation and new content, in about equal parts, with different roles.** The token's own block-0 state (the
  residual carried forward, with the query and self key/value it feeds) accounts for 0.41 of the state change after
  block 1 and **0.70 of its belief-dependent part**; the attention sources account for 0.59 of the change and 0.37 of
  the belief-dependent part (the two shares sum to 1.00). What block 1 retrieves from the history is mostly the goal's
  offset; the goal × history interaction is mostly what block 0 already combined at the token, which block 1's MLP
  then deepens: the interaction share rises from 0.13–0.15 at the MLP's input to 0.27 at its output (the state after
  block 1: 0.19). The MLP's goal response is far from its local Jacobian (first order 0.07, midpoint 0.64) but is a
  single linear map of its input change across decisions (R² 0.97), with gain 0.71.

**Reading.** Block 1 introduces no goal-dependent selection of the history: the goal does change where the decision
token attends, but what is attended differently carries nothing; the effect comes through content, from the token's
own block-0 state (most of the belief-dependent part) and from the history tokens' values (most of the goal offset),
which carry the goal by the same gated block-0 mechanism. Through blocks 0 and 1 the goal × belief interaction is
therefore made by gating goal content into token states and transforming the mixtures through the MLPs, not by goal-
dependent retrieval.

## 14. Post hoc: the MLPs are approximately affine on the tested inputs; affine replacements preserve most of the policy (`mlp_affine.md`)

Exploratory. Each MLP sublayer's best affine approximation, fitted by ridge from LN(mid) to the MLP output on every
live token of the fit episodes under every goal (shared: one map; per goal: one map per goal; constant: the mean
output, the MLP removed; one variant affine in the residual itself, the layer norm removed too), substituted in the
network with attention intact. **Scope:** unless marked "decision token only", the replacement is at every live token of
the sequence, history tokens included, so no nonlinear MLP processing remains upstream in the replaced block. Behaviour: the greedy return on the test
spawns (recovery of the goal-directed return above the goal-blind H policy), agreement with the natural model's
own-goal decisions on its test decisions. Transport: section 10's test at the input of block 2 refitted on the variant's
states (held-out R² of M(A→B); on the A-equivalent, B-different pairs, B's decision switched by the direct B-written
difference and by M(A→B) Δ_A).

| MLP(s) replaced | fit R² at the decision token (shared / per goal) | recovery | agreement (goal-dependent decisions) | map R² | switch: direct Δ_B / M(A→B) Δ_A |
|---|---|---|---|---|---|
| none (natural) | | 1 | 1 | 0.95 | 0.92 / 0.85 |
| **block 1, shared affine** | 0.96 / 0.99 | **1.00** | **0.98** | 0.95 | 0.91 / 0.85 |
| block 1, per-goal affine | | 1.00 | 0.99 | 0.95 | 0.92 / 0.86 |
| block 1, shared, decision token only; LN removed too | | 1.00; 1.00 | 0.98; 0.98 | 0.95 | 0.91 / 0.85 |
| block 1, constant (removed) | | 0.94 | 0.95 | 0.96 | 0.83 / 0.78 |
| block 0, shared / per goal | 0.94 / 0.97 | 0.97 / 0.99 | 0.97 / 0.98 | 0.95 | 0.91 / 0.87; 0.92 / 0.85 |
| blocks 2, 3, each or both, shared | 0.98, 0.99 | 1.00 | 0.99 | 0.95 | 0.92 / 0.85 |
| **all four, shared affine, every token** | | **0.95 (0.87–0.98)** | **0.93 (0.87–0.96)** | 0.96 | 0.89 / 0.82 |
| all four, shared affine, decision token only | | 0.94 (0.91–0.97) | 0.94 (0.88–0.96) | 0.95 | 0.89 / 0.82 |
| all four, shared affine in the residual (layer norms removed too) | | 0.93 (0.86–0.96) | 0.91 (0.87–0.95) | 0.96 | 0.89 / 0.84 |
| all four, per-goal affine | | 0.97 (0.96–0.99) | 0.96 (0.91–0.98) | 0.95 | 0.91 / 0.85 |
| all four, constant (removed) | | −0.15 | 0.45 | 0.98 | 0.56 / 0.56 |

* **Block 1's MLP is approximately affine on the inputs it meets.** A single affine map of LN(mid) explains 0.96 of its output on
  held-out decision tokens; substituted for it, the return is unchanged (recovery 1.00), 0.98 of the natural decisions
  stand, and cross-goal transport is intact (the map's R² 0.95, the direct difference switches 0.91, the mapped one
  0.85, as in the natural model). Per-goal maps fit better (0.99) and add 0.01 of agreement; replacing at the decision
  token only, or removing the layer norm as well, changes nothing. **Normalisation is not doing the work**: on the same
  held-out inputs an affine map of the raw residual fits as well as one of LN(mid) (block 0: 0.92 against 0.94; block 1:
  0.96 against 0.96; blocks 2–3: 0.98, 0.99 against 0.98, 0.99), and the no-norm replacement behaves identically at
  block 1; with all four MLPs and their norms replaced at once it costs a further 0.02 of recovery and of decisions
  (0.93 / 0.91 against 0.95 / 0.93). Replacing all four at the decision token only, leaving the history tokens their
  MLPs, gives the same result as at every token (0.94 / 0.94): no nonlinear processing upstream is propping it up. Removing the MLP (its mean output) costs 0.06 of the
  recovery and 0.05 of decisions, so the MLP does something, and what it does is affine.
* **So, approximately, are the other three.** Blocks 2 and 3 are affine to 0.98–0.99 and replaceable without loss; block 0 is the least
  affine (0.94) and its replacement costs 0.03 of the recovery. **All four MLPs replaced by fixed affine maps keep
  0.95 of the goal-directed return and 0.93 of the decisions** (per-goal maps 0.97 and 0.96), while removing all four
  drops the policy below goal-blind (−0.15, agreement 0.45). The nonlinear part of the MLPs, the 1–6 % of their output
  the affine fits miss, is worth about 0.05 of the goal-directed return and 0.07 of decisions: small, but room for
  meaningful exceptions on particular decisions.
* **The Jacobian puzzle of sections 12–13 resolves.** The MLP's response to the goal change was a poor fit to its
  local Jacobian (R² 0.07 at block 1) and an excellent fit to one linear map across decisions (0.97): the function is
  affine to 0.96 on the natural manifold, with a small-amplitude nonlinear ripple (GELU pre-activations sit near the
  kink, |h| ≈ 0.5, 46 % positive) whose derivative is large but whose finite differences are small. The secant across
  the goal step is the affine part; the derivative at a point is the ripple.

**Reading.** Of the two alternatives posed, neither holds on the tested inputs: the MLPs do not perform general
nonlinear computation on the goal–history mixture there, and the goal does not select among linear responses to the
history (per-goal maps add 0.01–0.03). Each MLP is close to one fixed affine transformation on the inputs the trained
policy meets, and the four together, replaced as such at every token, carry 0.95 of the goal-directed return. Most of
the nonlinearity that makes H a nonlinear function of the posterior (sections 1 and 4) and the posterior's nonlinear
code (section 6) is then made in attention, by the softmax over gated goal content and history (sections 12–13), with
the MLPs acting mostly as linear maps on what attention assembles; the layer norms are not where the mixing happens.

**Limit.** The affine maps are fitted and evaluated on the models' own episodes and the four trained goals; off that
manifold the MLPs are not affine, and the 5 % of return and 7 % of decisions lost when all four are replaced are a real,
if small, nonlinear contribution whose location (which decisions, which block) is not identified here.

## 15. Post hoc: block 0's gates on the goal token make block 0's share of the goal-specific coordinates, and half of the final; the rest is rebuilt downstream (`gate0.md`)

Exploratory. In block 0 the goal reaches a goal-free decision token through the goal token's key and value only
(section 12): per head, the attention mass on the goal token is a gate on fixed goal content, and the gate depends on
the history through the softmax competition with the history tokens' keys. Here that history dependence is removed:
each head's mass on the goal token is replaced by its mean for that goal and query position (step), the other tokens'
weights rescaled to keep their relative weights (their total is then 1 − mean, fixed per goal and step). Goal content
is kept; its gating by the history is not. The test is run in section 14's simplified network (all four MLPs replaced
by shared affine maps at every token, so that attention softmaxes and layer norms are the only nonlinearities left) and,
as a control, in the natural network. Measured at the decision token under every goal: the goal × belief interaction
of the state, the goal-specific coding of the belief (shared encoder plus offset against per-goal encoders), the
cross-goal coordinate change on the A-equivalent, B-different pairs (R² of the untransformed Δ_A for Δ_B; section 10's
transport test), and the policy.

**A correction to section 8 first.** The decomposition x = hist(h) + M(g) + I(h, g) there counts the step as part
of the history, so a goal offset that depends on the step lands in I. Letting the offset depend on the step,
x = hist(h) + M(g, step) + I'(h, g), the goal × step offset is 0.27 / 0.36 / 0.37 of the state's variance at the
input of block 1 / of block 2 / at the end, and the interaction beyond it, the goal × belief part proper, is
**0.06 / 0.09 / 0.13** (section 8's 0.17 / 0.21 / 0.26). Likewise a shared encoder f(step, b) plus a goal × step
offset explains 0.89 / 0.87 / 0.84 of the state against per-goal encoders' 0.94 / 0.96 / 0.97: the goal-specific
coding of the belief is a gap of **0.05 / 0.08 / 0.12**, not 0.16–0.24. The same-step pair results of sections 9–11
are unaffected (the offset cancels in a same-step difference), and so is the conclusion that the coding is
goal-specific; its size was overstated.

The gates: mass on the goal token 0.23 (0.22–0.28) at decision tokens after the first step; its spread across
histories within one (goal, head, step) is a std of 0.10, the spread of its means across goals 0.08.

| variant | recovery | agreement | goal × belief interaction I′: site 1 / 2 / 4 | encoder gap: 1 / 2 / 4 | R² of Δ_A for Δ_B: 1 / 2 / 4 | switch: direct Δ_B / M(A→B) Δ_A / Δ_A |
|---|---|---|---|---|---|---|
| natural | 1 | 1 | 0.06 / 0.09 / 0.13 | 0.05 / 0.08 / 0.12 | 0.74 / 0.39 / 0.12 | 0.92 / 0.85 / 0.33 |
| simplified (section 14) | 0.95 | 0.93 | 0.04 / 0.07 / 0.09 | 0.04 / 0.06 / 0.08 | 0.80 / 0.52 / 0.26 | 0.89 / 0.82 / 0.39 |
| **simplified, block-0 gates fixed per goal and step** | **0.78 (0.68–0.85)** | **0.86** | **0.01 / 0.03 / 0.05** | **0.00 / 0.02 / 0.04** | **0.95 / 0.71 / 0.47** | 0.88 / 0.81 / **0.53** |
| simplified, gates fixed at the decision token only | 0.77 | 0.87 | 0.01 / 0.04 / 0.06 | 0.00 / 0.03 / 0.05 | 0.95 / 0.65 / 0.42 | 0.86 / 0.80 / 0.51 |
| natural, block-0 gates fixed | 0.89 (0.74–0.93) | 0.95 | 0.03 / 0.06 / 0.10 | 0.02 / 0.05 / 0.09 | 0.85 / 0.49 / 0.20 | 0.91 / 0.85 / 0.42 |
| simplified, gates fixed to the goal-averaged mean | 0.67 | 0.82 | 0.00 / 0.02 / 0.04 | 0.00 / 0.02 / 0.03 | 0.98 / 0.77 / 0.54 | 0.87 / 0.85 / 0.58 |

The map's R² for Δ_B stays 0.92–1.00 at every site in every variant.

* **The gates make block 0's share of the goal-specific coordinates.** With them fixed, the goal × belief interaction
  at block 0's output falls from 0.04 to 0.01 (natural network 0.06 to 0.03), the encoder gap there to 0.00, and the
  untransformed A-written difference predicts the B-written one at 0.95 (from 0.80): at the input of block 1 the two
  goals' coordinates are then all but shared. With the gate at its goal-averaged value block 0 adds exactly
  hist(h) + M(g, step) (I′ 0.00, Δ_A for Δ_B 0.98).
* **About half of the final goal-specific coordinates survive them, rebuilt downstream from step-constant goal content.**
  At the input of block 2 the interaction is 0.03 and the gap 0.02 (from 0.07 and 0.06), at the end 0.05 and 0.04 (from
  0.09 and 0.08); Δ_A predicts Δ_B at 0.71 and 0.47 (from 0.52 and 0.26), and switches B's decision in 0.53 of the
  differing pairs (from 0.39) against 0.88 for the direct B-written difference. Even when block 0 contributes no
  interaction at all (the goal-averaged gate) blocks 1–3 make 0.02–0.04 of it and the untransformed Δ_A still only
  reaches 0.58 of the direct difference's effect. In the simplified network the only places left to do this are the
  attention softmaxes of blocks 1–3 (and the layer norms): with the gate fixed, the goal enters those blocks as a
  constant per goal and step at every token, and a history-dependent attention pattern over tokens that carry a
  goal-dependent constant is itself a goal × history interaction.
* **The gating matters for the policy, and the two simplifications compound.** Fixing the gates costs the natural
  network 0.11 of its goal-directed return (0.89; 0.74–0.93) and 0.05 of its decisions; in the simplified network the
  same change costs 0.17 of recovery (0.78 against 0.95) and 0.07 of decisions. The affine maps were fitted on natural
  inputs, and the gate-fixed states are off that distribution, so the simplified network is not a faithful substrate
  for interventions that leave the natural manifold; its own losses must be read against the natural control. Fixing
  the gate at the decision token only costs the same as at every token (0.77 against 0.78): it is the decision token's
  own gate, not the history tokens', that the policy uses. Fixing it to the goal-averaged mean costs more (0.67), so the
  goal-specific mean level of the gate is used too.

**Reading.** The history-dependent gating of goal content in block 0 is where block 0's goal-specific coding of the
belief is made, as section 12 implied, and it is behaviourally necessary; but it does not explain the goal-specific
coordinates at the decision. Half of those are made after block 0 by attention in blocks 1–3 from goal content that,
with the gate fixed, is a constant per goal and step. The goal-specific coding is smaller than section 8 reported
(0.05–0.12 of the state's variance beyond a goal × step offset, not 0.16–0.24), and it is distributed: block 0's gate
for the first half, the later blocks' attention for the second.

**Limit.** The gate is fixed to a mean over all histories at that goal and step, so the per-head rescaling of the
history (1 − mean) still differs across goals; the goal-averaged variant removes that too and gives the same picture
downstream. Which attention routes in blocks 1–3 rebuild the coordinates (goal-dependent queries, or a history-dependent
pattern over goal-laden values) is not separated here.

## 16. Post hoc: with block 0's gates fixed, block 1's attention adds the goal × belief interaction in two additive halves, goal-dependent weights and goal-conditioned values, and its MLP transforms the sum linearly (`gate1.md`)

Exploratory. The natural network with block 0's gates fixed per goal and step (section 15's control; no affine MLPs,
so the inputs are the network's own), decision tokens after the first step, goals A and B with the gates fixed in both.
Same-step pairs of decisions (posteriors ≥ 0.5 apart), so goal × step offsets cancel: the **belief-dependent goal
difference** of a quantity y is (y(h2, B) − y(h2, A)) − (y(h1, B) − y(h1, A)). Block 1's attention at the decision
token is recomputed with the incoming decision-token residual held at A's and B's attention weights (B's query and
every key) with A's values, B's values of the other tokens (history and goal tokens) with A's weights, or both; a
route's belief-dependent difference is measured against the natural one at block 1's attention output and, block 1's
own MLP applied, at the state after block 1 (projection, R²), then propagated to decisions: on the A-equivalent,
B-different pairs the route's pair difference added under B at the input of block 2 (B's decision switched to the
donor's; ceiling the direct B-written difference 0.93, floor the A-written one 0.41, random 0.05), and the route's whole
state patched under A (A's decision switched to B's; the whole state under B 0.85, random 0.03).

**What block 1 receives and what it adds.** With the gates fixed the goal × belief interaction (beyond a goal × step
offset) is 0.03 of the state's variance at block 1's input (natural 0.06), 0.05 at block 1's attention output, 0.11 at
its MLP output and 0.06 after block 1 (natural 0.07 / 0.15 / 0.10). On the same-step pairs the belief-dependent goal
difference is 0.39 of the A-written difference's norm at block 1's input and 0.67 after block 1; carried unchanged,
the input's difference explains 0.23 of the output's (R²), the attention output's projects 0.31, input plus attention
(the MLP's input) explain 0.54, and the MLP, whose belief-dependent response is a linear map of its input difference
across pairs (R² 0.95, gain 1.10), supplies the rest.

| goal source switched A → B at block 1 (residual at A) | pattern change (TV over history) | attention output: projection / R² | after block 1's MLP: projection / R² | B's decision switched by the route's pair difference | A's decision switched by the route's state |
|---|---|---|---|---|---|
| **B's weights (query and every key), A's values** | 0.23 | **0.41 / 0.37** | 0.22 / 0.23 | **0.51** | 0.08 |
| **B's values of the other tokens, A's weights** | 0 | **0.50 / 0.44** | 0.38 / 0.39 | **0.51** | 0.37 |
| **both** | 0.23 | **0.93 / 0.93** | **0.60 / 0.64** | **0.74** | 0.50 |
| decision token's own key and value from B | 0.08 | 0.12 / 0.06 | 0.04 / 0.02 | 0.42 | 0.03 |
| the whole attention sublayer from B | 0.23 | 1.00 / 1.00 | 0.64 / 0.68 | 0.74 | 0.57 |
| residual only (the token's block-0 state from B) | 0 | 0 / 0 | 0.36 / 0.43 | 0.54 | 0.24 |
| all (= B) | 0.23 | 1 / 1 | 1 / 1 | 0.93 | 0.85 |

* **Two additive halves, not a coordination.** At block 1's attention output B's weights over A's values carry 0.41 of
  the natural belief-dependent difference and B's other-token values under A's weights 0.50; together 0.93, the sum of
  the parts. Of the three readings posed (weights, values, or mostly the joint swap) the answer is the first two at
  once and in about equal measure, with no excess for the joint swap: goal-dependent weighting recombines the history
  features A already supplies, and goal-conditioned content arrives from the history and goal tokens, independently.
  The decision token's own key and value add little (0.12).
* **The weights do matter once the gates are fixed.** Section 13 found the query route causally null in the natural
  network (0.04 of the total attention-output change). Here, on the belief-dependent part and with block 0's
  history-dependent gating removed, goal-dependent weights (query and keys, TV 0.23 over the history) carry 0.41 of
  block 1's attention contribution and switch 0.51 of B's decisions against the 0.41 floor, as much as the values
  (0.51). The two are not in conflict: in the natural network the goal × belief content that the values carry from the
  gates dominated the total change; with that removed the reweighting's share becomes visible. The weights carry no
  goal offset, so in the whole-state patch under A (which includes the goal × step offset) they move the decision
  little (0.08) while the values move it 0.37.
* **After the MLP, the split is values 0.38, weights 0.22, residual 0.36.** The residual route, zero at the attention
  output by construction, carries block 0's remaining interaction (0.03 of variance) through block 1's MLP to 0.36 of
  the output's belief-dependent difference; attention's 0.64 and the residual's 0.36 add to one. Both attention routes
  together switch 0.74 of B's decisions, the residual alone 0.54, everything 0.93.

**Reading.** Beyond what block 0 supplies, the operation that adds goal × belief interaction is block 1's attention,
in two independent ways: the goal content now constant per goal and step in the decision token's query (and the
tokens' keys) reweights the history tokens, and the history tokens' values bring goal-conditioned content made at
their own positions by block 0 (its nonlinear MLP on the goal offset, and the per-head rescaling). Block 1's MLP then
applies a near-linear map to the sum and amplifies it (gain 1.10), doubling the interaction's variance share through the
block as in the natural network. Nothing here requires coordinating weights with their value content.

**Limit.** The weights route switches every key as well as the query, so "goal-dependent weighting" includes the
history tokens' goal-dependent keys; section 13 found the history keys' share small in the natural network. The
decision test under A includes the goal × step offset and so favours the values; the pair test under B is the
belief-dependent one.
