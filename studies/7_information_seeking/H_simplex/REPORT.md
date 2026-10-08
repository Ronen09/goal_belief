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
