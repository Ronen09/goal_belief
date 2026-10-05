# Navigate–commit: frozen environment specification

Frozen after the solver sweep (`tune.py`, `tune.md`) and before any model was trained. Code: `goalgeo/navcommit.py`
(environment and solver), `goalgeo/navppo.py` (the same environment vectorised, for training).

| | |
|---|---|
| grid | 5 × 5, empty; cell = 5·row + col, row 0 at the top |
| hidden reward | one of the four corners, uniform; g = 2·tb + lr (lr: 0 left, 1 right; tb: 0 top, 1 bottom) |
| stations | two distinct non-corner cells, uniform over the 420 ordered pairs (horizontal station, vertical station) |
| start | uniform over the 25 cells, independent of the stations and of the goal |
| reports | horizontal station: LEFT / RIGHT; vertical station: TOP / BOTTOM; correct with probability q of that station, independently given the goal; one report per station |
| actions | UP, DOWN, LEFT, RIGHT, QUERY (on an unused station's cell), COMMIT (on a corner); illegal actions are masked |
| reward | COMMIT ends the episode and pays 1 on the reward corner, 0 elsewhere; every move or query costs c |
| **horizon H** | **12** actions (brief's pilot default: 20); no terminal reward if the agent has not committed |
| **cost c** | **0.025** (brief's pilot default: 0.02) |
| reliability, fixed-q pilot | q_h = q_v = 0.8 |
| reliability, generalisation study | each station independently from {0.60, 0.65, …, 0.95}. Held out of training: the values 0.70 and 0.90 (either station) and the combinations (0.60, 0.95), (0.95, 0.60), (0.75, 0.85), (0.85, 0.75). 32 of 64 pairs are trained. |

## Why the brief's defaults were changed

With H = 20, c = 0.02, q = 0.8 two queries are optimal from **every** layout and start: there is time to visit both
stations and the information is always worth its price. The brief requires meaningful cases with zero, one and
two optimal queries. Shares over 420 layouts × 25 starts (`tune.md` has the whole sweep):

| H | c | q | zero | one | two | number of queries depends on a report | report changes the next move |
|---|---|---|---|---|---|---|---|
| 20 | 0.02 | 0.8 | 0.000 | 0.000 | 1.000 | 0.006 | 0.109 |
| **12** | **0.025** | **0.8** | **0.191** | **0.225** | **0.584** | **0.376** | **0.376** |

In the frozen configuration a wrong number of queries costs at least 0.02 of expected return in 12 % (zero),
22 % (one) and 58 % (two) of cases. The optimal value averages 0.311, the best policy that never queries 0.210.
Over the reliability grid the share of zero-query cases runs from 94 % (0.6, 0.6) to 3 % (0.95, 0.95).

## Belief structure

Uniform prior and reports that are independent given the goal make the posterior factorise at every history:
b(g) = P(lr) · P(tb), and each factor is 1/2 (station unused), 1 − q or q. For fixed reliabilities there are
9 reachable beliefs. They lie on a two-parameter surface in the simplex:

    b(g) = 1/4 + s_lr·(P(right) − 1/2)/2 + s_tb·(P(bottom) − 1/2)/2 + s_lr·s_tb·(P(right) − 1/2)(P(bottom) − 1/2)

(s = ±1 by the corner's side). The marginal probability of either report is exactly 1/2 whatever the other
station said. Movement carries no evidence.

## Solver

Backward induction over (remaining actions, cell, state of the horizontal station, state of the vertical station),
station state ∈ {unused, reported 0, reported 1}, for a batch of layouts and reliabilities. Every action value
is kept (−∞ if illegal); ties are actions within 10⁻⁹ of the maximum.

Validation (`tests/test_navcommit.py`, 7 tests): the posterior against the generative joint; report
probabilities; factorisation; the value function against an expectimax over raw histories that uses no belief
state (3 × 3 grid, H ≤ 6, every start and horizon, to 10⁻¹²); and against exhaustive enumeration of every
deterministic history-dependent policy (3 × 3 grid, H ≤ 4, to 10⁻¹²).

## Regret

Episode-level regret is V*(s₀) − E[return]. It equals the expected sum of the local decision regrets
V*(s_t) − Q*(s_t, a_t) along the trajectory, which is how it is measured (no reward noise). Local regret is
also reported per decision on the fixed histories. Split of the episode regret by the decision that lost it:
**query** (a query that was not optimal, or a move where only a query was), **commit** (a commit that was not
optimal, or another action where only a commit was), **route** (any other suboptimal move).
