# Which superseded history features the residual carries (residual features)

Run date: 2026-10-06. Brief: `BRIEF.md` (the belief-error experiment's proposed next step). Features, arms and decision
rule: `PLAN.md`, committed (`94efdfe`) **before `run.py` was run on any trained model**. Numbers from `tables.md`; data
in `results.json` (registered) and `explore.json` (post hoc). Reproduce: `reproduce.py residual_features` (2 min on one
GPU).

**Setting.** As in the belief-error experiment: at the goal token's four attention outputs, d(h) = x_b(h) + r(h),
x_b the posterior's table. The posterior is sufficient for the optimal decision (the point raised in the brief), so
each history feature is centred within its (posterior, prefix length) group. What remains is exactly the part of the
feature the posterior has made redundant. x_F = ridge regression of r on the centred feature. On informative pairs,
x_F(h′) − x_F(h) is swapped against the same vector rotated. **Share** = the feature's excess flip / r's (r's excess:
0.077).

The prefix here is up to 4 moves. 10 203 distinct prefix sequences reach 619 posteriors, so most posteriors are reached
by several histories. Whether the landmark was seen is fully determined by the posterior, since the landmark is never
noisy; it has nothing left after centring.

## 1. A linear code of superseded token features carries r's effect

| group (centred within the posterior) | R² of r | share of r's excess |
|---|---|---|
| tokens by absolute position | 0.61 | 0.77 |
| tokens by recency (k events before the goal) | 0.56 | 0.81 |
| counts (symbols, moves) and landmark | 0.32 | 0.62 |
| **all** | 0.64 | **0.91** |

## 2. The moves, more than the symbols

| single feature | R² of r | share of r's excess | best in models |
|---|---|---|---|
| **counts** (each symbol and each move) | 0.32 | **0.61** | 7 |
| **last move** (act-0) | 0.18 | **0.56** | 3 |
| move at positions 3, 4 | 0.07–0.09 | 0.26–0.29 | 0 |
| first symbol | 0.08 | 0.16 | 0 |
| last symbol | 0.06 | 0.12 | 0 |
| every other position | ≤ 0.12 | ≤ 0.11 | 0 |

Post hoc (`explore.json`), counts split in two:

| | share of r's excess |
|---|---|
| counts (both) | 0.63 |
| move counts | 0.37 |
| symbol counts | 0.25 |

* **The features that carry r are about the moves the agent was made to take.** The last move alone carries 0.56,
  and the count of each move 0.37. The symbols, which carry the evidence, matter less: the last symbol 0.12, symbol
  counts 0.25.
* The registered LOC criterion names "counts", but that feature is a composite. Split, neither half reaches 0.5, and
  together they combine more than additively (0.37 + 0.25 against 0.63). The last move overlaps with the move counts.
* The moves are not evidence about where the agent is beyond what the posterior already contains: given the
  posterior, they are redundant. The network keeps a record of them and acts on it.

## 3. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **TOK** | all features together ≥ 0.75 of r's excess | 0.91 | **yes** |
| **LOC** | best single feature ≥ 0.5, best in ≥ 7 of 10 models | counts 0.61, best in 7 | **yes** (a composite: see section 2) |

**Registered reading: r is mostly one superseded token feature, the counts.** Post hoc, that feature is two: the move
counts (0.37) and the symbol counts (0.25). Together with the last move (0.56), the superseded record of the prefix
moves is the main content of r.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | r's excess flip ≈ 0.078 | yes | 0.077 |
| E2 | TOK holds | yes | 0.91 |
| E3 | LOC fails; recency carries more than absolute position | in part: LOC holds on the composite; recency 0.81 against 0.77 | |
| E4 | the best single feature is the last symbol | **no** | the last symbol carries 0.12; the last move 0.56 |

2 of 4, one in part.

## Conclusions

1. **Yes, the posterior absorbs these features for the optimal policy. The network has not dropped them.** Once the
   posterior and prefix length are fixed, a linear code of the prefix tokens carries 0.91 of the residual's effect on
   the decision.
2. **The superseded information is mainly about the moves, not the observations.** The last move and how often each
   move was made carry most of it; the observed symbols, which carry the evidence, little. A plausible reading, not
   tested here: a policy trained by reward keeps a direct, heuristic record of recent moves alongside the belief. That
   record should be redundant, and it tips decisions where the posterior leaves them close, as the residual-trace
   experiment found.
3. Across the last four experiments: the history acts through a posterior-level code (0.94–0.96 of the transfer). The
   remainder is not a belief error (belief-error experiment). It is a record of the prefix moves, which the posterior
   should have made redundant.

Next, if wanted: test whether that record is harmful by swapping in the move features of a history whose optimal action
agrees, and comparing regret; or check whether the predictive-pretraining backbone (study 4), trained without reward,
keeps the same record.

Limits: linear in hand-chosen features; the counts split is post hoc; the rotated control differs between the
registered and post hoc runs (counts 0.61 against 0.63); one draw of donors; the goal token's first decision only.
