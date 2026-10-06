# Residual features: features, arms, decision rule and expectations

Written before `run.py` was run on any trained model. Brief: `BRIEF.md`.

## Setting

As in the belief-error experiment: at the goal token's four attention outputs, d(h) = x_b(h) + r(h), x_b the table
over (posterior, L). The prefix is up to 4 moves: [OBS o_0] [EVT a_1 o_1] … [EVT a_L o_L]. Informative pairs, random
same-length donor (new draw, fixed seed).

## Features, centred within the posterior

Each feature F(h) is a one-hot or count vector of the history's tokens, centred within its (posterior, L) group on
fit-side histories: F̃ = F − E[F | posterior, L]. F̃ is therefore exactly the part of the feature the posterior does not
determine.

| feature | |
|---|---|
| sym@j, act@j | symbol / action at absolute position j (j = 0–4; act from 1) |
| sym-k, act-k | symbol / action k events before the goal token (k = 0: the last; k = 0–4 / 0–3) |
| counts | number of a, b and landmark symbols; number of each action |
| landmark seen | 1 if the landmark symbol appears |
| **groups** | absolute (all sym@j, act@j); recency (all sym-k, act-k); counts (counts and landmark); **all** (every feature above) |

x_F(h) = F̃(h) β_F, ridge regression of r on F̃ on fit-side histories (penalty chosen on a 10 % hold-out). Held-out R²
of r is reported per feature.

## Arms

For each feature or group: **swap** x_F(h′) − x_F(h) at the four attention outputs, and **rot**, the same vector
rotated to a random direction of the same size. Excess flip = swap − rot. r's own swap and rotation are rerun on the
same pairs as the reference. **Share** = a feature's excess flip / r's.

## Decision rule

Ten models; medians.

| | criterion |
|---|---|
| **TOK** | the group "all" carries ≥ 0.75 of r's excess |
| **LOC** | the best single feature carries ≥ 0.5 of r's excess (the same feature best in ≥ 7 of 10 models) |

| result | reading |
|---|---|
| TOK, LOC | **r is mostly one superseded token feature**, named by LOC |
| TOK, not LOC | r is a distributed linear code of superseded token features; the groups say which kind (absolute, recency, counts) |
| not TOK | r is not a linear code of these features: a conjunctive or nonlinear history code |

## Expectations

| | expectation |
|---|---|
| E1 | r's own excess flip ≈ 0.078 (belief-error experiment) |
| E2 | TOK holds |
| E3 | LOC fails; the recency group carries more than the absolute group |
| E4 | the best single feature is the last symbol (sym-0) |

## Limits known in advance

Linear in one-hot features; features chosen by hand; one draw of donors; goal token's first decision only.
