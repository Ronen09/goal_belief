# Pair terms: what H does when the belief is split between two cells

Written before `run.py` was run on any trained model. **Disclosure:** smoke-tested on the untrained checkpoint of
seed 0 only. Models: the maze10 experiment's six transformers; decisions, H and G as in the H-simplex experiment.
Brief: the H-simplex experiment's closing question.

## The question

The H-simplex experiment found H close to quadratic in the posterior (0.86 with pairwise products against 0.45
affine; MLP 0.94): the profile for a belief split between two cells is not the mixture of the two cells' profiles.
Beliefs in the aliased maze are often split between two candidate cells with the same recent symbols. On those
decisions, four readings of the pair term are distinguishable without a solver:

| rule | the goal-averaged profile it predicts, for a belief w on cell s and 1 − w on s′ |
|---|---|
| **mixture** (what an affine H would do) | w · h_s + (1 − w) · h_s′, with h_s the model's own mean H on decisions where b(s) ≥ 0.9 (`mixture`), or popt(s, ·) (`mixture_popt`) |
| **robust** | the share of goals for which the move is a shortest-path move from *both* cells |
| **commit** | popt at the likelier cell alone (a winner-take-all reading) |
| **disambiguate** | the expected information gain of each move under the two-cell belief (the maze10 experiment's measure) |
| **lookahead** | the goal-averaged one-step-lookahead value (QMDP + lookahead, the maze10 reference that values the next symbol) |

The mixture and the lookahead are the two readings in which the pair term is value-like; commit and disambiguate are
the two in which the network resolves the ambiguity rather than averaging over it.

## Data

Every live decision of the fit and test episodes of the H-simplex experiment (4 096 each per model). **Two-cell
decisions**: the two most likely cells hold ≥ 0.85 of the mass and the less likely of the two holds ≥ 0.15. For each
such decision: the pair (s, s′), w, H, the model's move and goal, the step, and the per-goal shortest-path moves from
s and from s′. **Conflict** of a two-cell decision: the share of goals for which no move is a shortest-path move from
both cells.

## Measures

**A. Where the mixture fails.** Relative departure ‖H − mixture‖² / ‖H‖² of two-cell decisions by conflict (0; (0, 0.5];
(0.5, 1]).

**B. Prediction of H on two-cell decisions** (held-out R², fit on fit episodes, test on test; each rule through a free
4 × 4 map plus step-bin intercepts, as in what-is-H): mixture, mixture_popt, robust, commit, disambiguate, lookahead,
and each of the last four added to the mixture. Also the MLP on b restricted to two-cell decisions (ceiling), and the
affine fit of the H-simplex experiment scored on them.

**C. The top action.** On two-cell decisions where a rule's top action differs from the mixture's (and the rule's top
action is unique): the share where H's top action is the rule's, and the share where it is the mixture's.

**D. Behaviour, per goal.** The model's own move on two-cell decisions under its goal, against per-goal versions of
the rules: `qmdp` (the mixture of values), `both` (a shortest-path move from both cells; defined when one exists),
`likelier` (a shortest-path move from the likelier cell), `eig` (the most informative move), `qmdp2` (one-step
lookahead). On decisions where `both` exists and differs from `likelier`: which the model takes. On decisions where no
common move exists: `likelier` against `eig` against `qmdp2`.

## Decision rule

Six models; medians; exact one-sided Wilcoxon signed-rank tests across models, p < 0.05.

| | criterion |
|---|---|
| **CONF** | departure (A) at conflict > 0.5 exceeds departure at conflict 0 in every model |
| **GAIN** | the best single rule's R² on two-cell decisions exceeds the mixture's by ≥ 0.1 |
| **WHICH** | the best rule beats the runner-up by ≥ 0.03 in median R² (otherwise: no single reading) |
| **TOP** | on the best rule's disagreement decisions (C), H's top action is the rule's more often than the mixture's (p < 0.05) |
| **BEH** | on two-cell decisions where `both` exists and differs from `likelier`, the model takes `both` in ≥ 0.6 |

| result | reading |
|---|---|
| GAIN with commit or disambiguate, TOP | **the pair term resolves the ambiguity**: the network acts as if in one cell (commit) or acts to find out (disambiguate) |
| GAIN with robust, BEH | the pair term hedges: it prefers moves that are right in either cell |
| GAIN with lookahead | the pair term is a value of the next observation |
| not GAIN | the pair term is none of these: it is not about the two candidate cells' routes |

## Expectations

| | expectation |
|---|---|
| E1 | CONF holds |
| E2 | GAIN holds |
| E3 | the best rule is **commit** (the H-simplex experiment: log b is the better coordinate, the top-two-cell code adds 0.1 to log b, and the maze10 agents deviate from QMDP for information only mildly) |
| E4 | disambiguate is the weakest of the four (information gain added 0.08 to the affine fit of all decisions) |
| E5 | TOP holds for commit |
| E6 | BEH fails: where a common move exists but differs from the likelier cell's, the model follows the likelier cell more often than not |
| E7 | mixture + commit reaches ≥ 0.9 × the MLP on two-cell decisions |

## Limits known in advance

Two-cell decisions are a selected part of the simplex (a fifth to a third of decisions, by the H-simplex quantiles);
the rules are goal-averaged shortest-path quantities, not values of the belief-MDP. No causal edit.
