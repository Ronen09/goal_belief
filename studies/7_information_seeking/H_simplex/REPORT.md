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
