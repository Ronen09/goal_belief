# Pair terms: on a two-cell belief, H is not about the two cells' routes (pair terms)

Run date: 2026-10-08. Plan, measures and decision rule: `PLAN.md`, committed (`cb2b8a3`) **before any trained model was
measured**. Models: the maze10 experiment's six transformers; decisions, H and the single-cell profiles from the
H-simplex experiment's machinery. Numbers from `tables.md` (registered) and `posthoc.md` (post hoc); data in
`results.json`, `posthoc.json`. Reproduce: `reproduce.py pair_terms` (6 min on one GPU).

**Setting.** Two-cell decisions: the posterior's two most likely cells hold ≥ 0.85 of the mass and the minor one
≥ 0.15; 0.12 (0.07–0.13) of each model's decisions, about 10 000 per model. Five readings of the pair term, each a
goal-averaged four-number profile computed from the pair, the weight w and shortest paths: the **mixture** of the
model's own single-cell profiles (its mean H where b(s) ≥ 0.9; defined for both cells in 0.42 of the two-cell
decisions), **robust** (a shortest-path move from both cells), **commit** (the likelier cell's profile), **disambiguate**
(expected information gain of each move) and **lookahead** (goal-averaged one-step-lookahead value). Each is fitted to
H through a free 4 × 4 map plus step intercepts on the fit episodes and scored by held-out R² on the test episodes.

## 1. None of the readings describes H on two-cell decisions

| model of H on two-cell decisions | held-out R² |
|---|---|
| step only | 0.13 |
| mixture of the model's single-cell profiles | 0.31 (0.23–0.33) |
| robust / commit / disambiguate / lookahead | 0.19 / 0.20 / 0.22 / 0.16 |
| mixture + each of the four | 0.33–0.34 |
| mixture + all four | 0.45 (0.39–0.50) |
| the mixture in H's own units, no map | −0.03 |
| **MLP on the full belief b, these decisions only** | **0.88 (0.62–0.92)** |

* **GAIN fails**: no single rule adds anything to the mixture (the best, disambiguate, is 0.10 *below* it); all four
  together add 0.14. **WHICH and TOP fail** with it: where a rule's top action differs from the mixture's, H's top
  action is the mixture's about twice as often as the rule's (0.41–0.43 against 0.18–0.25), for every rule.
* The mixture of what H is when the model is nearly certain of either cell, taken literally (no refitting), has no
  explanatory power at all (R² −0.03): on a split belief H is not an average of the two certain profiles, and the
  departure is as large when the two cells agree on every goal's move as when they conflict (**CONF fails**: 0.70 at
  conflict 0 against 0.86 at conflict > 0.5, not ordered in every model).
* Yet H on these decisions is still a function of the posterior: an MLP on b, fitted on the two-cell decisions alone,
  reaches 0.88.

## 2. The model's own moves on two-cell beliefs are not shortest-path moves from either cell

| decisions | move is QMDP-optimal | move is a shortest-path move from the true cell | from the likelier cell |
|---|---|---|---|
| near-certain (max b ≥ 0.9; 0.17 of decisions) | 0.96 | 0.84 | — |
| two-cell (0.12 of decisions) | 0.59 | 0.28 | 0.30 |
| all | 0.55 | 0.44 | 0.41 |

* Where no common shortest-path move exists, the model takes a likelier-cell move in 0.43, QMDP's in 0.44, QMDP +
  lookahead's in 0.42 and the most informative move in 0.20. **BEH** as registered was empty by construction (a move
  optimal from both cells is a move of the likelier cell); posed properly (post hoc), where a common move exists and
  the likelier cell also has non-common moves, the model's move is a likelier-cell move in 0.25, and given that, a
  common one at chance rate.
* So on a split belief the models neither commit, nor hedge, nor average: in most such decisions they move in a way
  that is a shortest-path move for neither candidate cell, although when they are nearly certain they are QMDP-optimal
  0.96 of the time. (The maze10 experiment's deviation rate from QMDP over all decisions, 0.40–0.46, is this: the
  deviations concentrate on uncertain beliefs.)

## 3. Post hoc: a function of the pair, not of the pair's routes (`posthoc.md`)

| model of H on two-cell decisions | held-out R² |
|---|---|
| MLP on the pair alone: s1, s2 (one-hot), w, step | 0.77 (0.50–0.80) |
| MLP on the pair + the tail of the belief (mass and entropy outside the two cells) | 0.80 |
| MLP on the full belief b | 0.88 (0.73–0.93) |
| a table by ordered pair ⊗ [1, w] + step (pairs seen ≥ 30 times, 0.78 of decisions) | 0.66 (0.41–0.85) |

* **H on a split belief is mostly a function of which two cells and of w** (0.77 from the pair alone, 0.80 with the
  tail of the belief, against 0.88 from the full posterior): a learned response per pair, not an operation on the two
  cells' shortest-path moves (section 1) and not an average of the two certain profiles.
* **Not landmark seeking**: the move is a shortest-path move toward the nearest landmark in 0.26 of two-cell decisions,
  below the 0.32 a random move would give (near-certain decisions: 0.16 against 0.30).
* **Not hedging**: given a likelier-cell move, it is a common move at the chance rate (0.46 against 0.50).

**Reading.** The pair term of H, the part that made H quadratic rather than affine in the belief, is a per-pair
response the network has learned for the aliased pairs its policy actually meets, and it is not reducible to the five
readings: not an average of certain-cell profiles, not a robust or committed shortest-path choice, not a one-step
information gain, not a one-step lookahead value, not a landmark heading. Since the models are QMDP-optimal 0.96 of
the time when nearly certain and 0.59 when split between two cells, the pair term is where their deviations from
QMDP live. Whether those deviations are *good* (optimal for the belief-MDP over several steps, which no reading here
computes) or habits of the training distribution is the next question; a short-horizon exact expectimax on two-cell
beliefs (a small tree) would settle it pair by pair.

## 4. Scorecard

| | criterion | value | held |
|---|---|---|---|
| **CONF** | departure at conflict > 0.5 > at conflict 0 in every model | 0.86 against 0.70, not in every model | **no** |
| **GAIN** | best single rule ≥ mixture + 0.1 | disambiguate 0.22 against mixture 0.31 | **no** |
| **WHICH** | best rule ≥ runner-up + 0.03 | 0.22 against 0.20 | **no** |
| **TOP** | H's top is the best rule's more often than the mixture's | 0.25 against 0.43 | **no** |
| **BEH** | takes the common move ≥ 0.6 | the set is empty by construction; posed properly, no hedging | **no** |

| | expectation | held |
|---|---|---|
| E1 | CONF holds | no |
| E2 | GAIN holds | no |
| E3 | the best rule is commit | no (disambiguate, by 0.02, and below the mixture) |
| E4 | disambiguate is the weakest | no (lookahead is) |
| E5 | TOP holds for commit | no |
| E6 | BEH fails: the model follows the likelier cell | the set is empty; posed properly the model follows neither (likelier-cell move 0.23) |
| E7 | mixture + commit ≥ 0.9 × MLP | no (0.34 against 0.88) |

0 of 7 expectations held. The registered reading "not GAIN: the pair term is none of these; it is not about the two
candidate cells' routes" applies.

## Limits

Two-cell decisions are 0.12 of decisions and the model's own; the single-cell profiles exist for both cells in 0.42
of them (the departure measure uses those). The rules are goal-averaged shortest-path quantities and one-step
quantities; the belief-MDP optimum over several steps was not computed. BEH was mis-specified in the plan (the set it
names is empty) and is replaced post hoc. The landmark and hedging tests are post hoc.
