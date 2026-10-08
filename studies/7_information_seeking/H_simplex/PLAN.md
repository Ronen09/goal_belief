# H on the belief simplex: is the history profile a belief-weighted per-cell table?

Written before `run.py` was run on any trained model. **Disclosure:** the code was smoke-tested on the untrained
initial checkpoint of seed 0 (`--untrained`); nothing was looked at on a trained model. Models: the maze10
experiment's six reward-trained transformers (55-cell aliased maze, four goals, no solver). Code: `run.py`, `tables.py`;
the maze10 experiment's `measure.py` supplies the episodes, the logits under every goal and the goal bias G.

## The question

The additive code reads the policy's logits as H(h) + G(g, step): H the goal-averaged, action-centred logit profile of
the history, G a fixed bias per goal and step. In the small maze (what-is-H, H-mixture) H was 0.82 explained by the
best *affine* function of the 14-cell posterior, equal to a two-term solver mixture (the share of goals for which an
action is optimal from each cell, and the goals' mean reachability), and 0.91 by a per-posterior table; editing with
the affine mixture moved decisions as far as editing with the table. The remainder was a record of the prefix tokens,
outside the belief code.

Here the posterior lives on a 55-cell simplex and evolves over 40 moves of the agent's own choosing. The hypothesis
under test is the **QMDP form**: H(b) = Σ_s b(s) h_s, a fixed four-number profile per cell, mixed by the belief. If it
holds, the policy is, to first order, "a per-cell table of goal-free action preferences, averaged under the belief,
plus a goal bias" — and the questions become what the per-cell profiles h_s are, how much of H is nonlinear in b, and
how much is not a function of b at all.

## Data

For each model: its own greedy episodes on the maze10 experiment's fixed evaluation spawns (4 096 episodes, `test`)
and on separate spawns (4 096, `fit`). At every live decision: the exact posterior b (55), the step, the entropy of b,
the prefix tokens so far, and the centred logits under each of the four goals (the goal token replaced, the history
the model's own). H = their mean over goals; G(g, step) = the mean goal deviation per step on the fit episodes (as in
maze10, 25 step bins). Everything is fitted on `fit` decisions and reported on `test` decisions (pooled R² over the
four centred numbers, weights chosen on a tenth of `fit`).

## Measures

**A. Prediction of H** (held-out R²):

| model of H | form | what it tests |
|---|---|---|
| `step` | intercept per step bin | the baseline |
| **`affine`** | step + Σ_s b(s) h_s (ridge on b; 55 × 4 free) | **the QMDP form** |
| `popt` | step + M · Σ_s b(s) popt(s, ·), M a 4 × 4 matrix | belief-weighted share of goals for which the move is a shortest-path move from s |
| `reach` | step + M · Σ_s b(s) reach(s, ·) | belief-weighted mean over goals of γ^d(next(s, a), g) |
| `mix` | step + M₁ · popt + M₂ · reach | the H-mixture code, per cell, without a solver |
| `mls` | step + h at the most likely cell (table 55 × 4, free) | a winner-take-all reading of the belief |
| `affine+sharp` | affine + Σ_s b̃(s) h'_s, b̃ ∝ b² | sharpening toward the likely cells |
| `affine+ent` | affine + entropy(b) · u + entropy(b) · Σ_s b(s) h''_s | an entropy-scaled profile |
| `mlp` | step + a two-layer MLP (64 units) on b | the nonlinear-in-b ceiling |
| `history` | mlp features + prefix features (counts of each symbol and each move so far, the last move, the last symbol) | beyond the posterior |

Gains: **nonlinear-in-b** = R²(mlp) − R²(affine); **beyond-b** = R²(history) − R²(mlp).

**B. The per-cell profiles.** h_s from the affine fit: (i) the share of cells whose top action in h_s is a shortest-path
move for at least one goal; (ii) the correlation, over cells × actions, of h_s with popt(s, ·), reach(s, ·) and the
best affine combination of the two; (iii) a figure of h_s as four arrows per cell on the maze.

**C. Decisions.** With G from the fit episodes, each fitted Ĥ replaces H in the additive code:
* offline: on `test` decisions where the model's move depends on the goal, the share where argmax(Ĥ + G) is the model's
  move; references: argmax(H + G) itself (maze10: 0.85) and the goal-blind argmax H (0.55);
* online: the policy argmax(Ĥ(b_t) + G(g, t)) run in the environment on the evaluation spawns, with b_t the exact
  posterior — a policy that is a function of (b, g, t) only; return and success against the natural policy, the
  additive code run online (maze10: 0.77 of the goal-directed return), goal-blind and history-blind; recovery
  = (return − goal-blind) / (natural − goal-blind), as in maze10.
  Policies run: `affine`, `mix`, `mlp`, and `mix` fitted with a single slope per term (α · popt + β · reach: a
  three-number model of H, no network beyond G).

## Decision rule

Six models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05 where a comparison is between models.

| | criterion |
|---|---|
| **QMDP** | R²(affine) ≥ 0.75 |
| **MIX** | R²(mix) ≥ 0.9 × R²(affine) |
| **NONLIN** | nonlinear-in-b gain ≤ 0.05 |
| **BEYOND** | beyond-b gain > nonlinear-in-b gain |
| **PROF** | (i) ≥ 0.85 and the correlation of h_s with the best affine combination of popt and reach ≥ 0.7 |
| **DEC** | offline agreement of affine ≥ 0.9 × that of H + G, and online recovery of affine ≥ 0.9 × the additive code's |
| **NONET** | online recovery of the three-number mix ≥ 0.8 × the additive code's |

| result | reading |
|---|---|
| QMDP, MIX, NONLIN, DEC | **H is a belief-weighted per-cell table, and the table is goal-averaged usefulness and reachability** |
| QMDP, DEC, not MIX | the table is real but its entries are not these two quantities |
| not QMDP, NONLIN fails | H is a nonlinear function of the belief: the small-maze result does not scale |
| BEYOND fails | in the larger maze the history matters through more than the belief and its nonlinearities |

## Expectations

| | expectation |
|---|---|
| E1 | QMDP holds (0.75–0.85) |
| E2 | MIX holds |
| E3 | NONLIN holds: the nonlinear-in-b gain is ≤ 0.05, and `mls` is well below `affine` |
| E4 | BEYOND holds: the prefix record adds 0.05–0.10, more than any nonlinearity in b |
| E5 | PROF holds |
| E6 | DEC holds |
| E7 | NONET holds: three numbers and G keep most of the additive code's return |

## Limits known in advance

Decisions are the model's own, so the fitted region of the simplex is the one its policy visits; no causal edit is
made here (the maze10 experiment's removals stand as the causal evidence for the history part). G is the maze10
experiment's step-binned goal bias, refitted on this run's fit episodes.
