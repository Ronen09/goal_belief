# What is H: candidates, tests, decision rule and expectations

Written before `run.py` was run on any trained model. The candidates were checked against the solver alone, and the code
was smoke-tested on the untrained initial checkpoint of seed 0. Brief: `BRIEF.md`.

## H and the candidates

H(h) is the mean over goals of the centred logits at the goal token (additive code). The candidates are computed per
posterior b at the reveal (goal-free), one value per action:

| candidate | definition |
|---|---|
| **maxQ** | max_g Q*(b, g, a), the brief's hypothesis |
| meanQ | mean_g Q*(b, g, a) (equal to mean_g of the advantage once centred over actions) |
| popt | the share of goals under which a is optimal |
| maxA | max_g (Q* − V*)(b, g, a): 0 for an action optimal under some goal |
| maxMDP, meanMDP | max / mean over goals of Q_MDP(b, g, a) = Σ_s b(s) γ^d(next(s, a), goal g). Fully observed value under the belief: reachability that ignores information gathering |
| nearest | Σ_s b(s) γ^(min_g d(next(s, a), g)): closeness to the nearest goal |
| ceilings | Htab, H's own mean per posterior (fit side), and blin, affine in b |

Solver-only facts: maxQ and maxMDP share their top action in 0.90 of histories; maxQ and popt in 0.54.

## Tests

* **Prediction** (held-out histories, length ≥ 1):
  * H (centred) ≈ α[a, L] + C·M, with M a 4 × 4 matrix (flexible), and ≈ α + β·C (a single slope);
  * held-out R²;
  * top-action agreement (H's argmax is among the candidate's top actions).
* **Decision**: the fitted H plus the additive-code experiment's goal bias G; optimal rate on goal-dependent cells.
* **Causal edits** at two goal-token sites, entering block 2 (primary) and the final residual stream:
  * the goal-free state (mean over goals) is regressed on C on the fit side: x ≈ α_L + C·W;
  * edit x(A, g) + (C(b_B) − C(b_A))·W, the same vector under every goal;
  * kinds: none, whole, every candidate, blin, Htab, and maxQ's edit rotated (5 draws);
  * the belief-encoding-edit experiment's main pairs and the direct-belief-edit experiment's measures: donor-optimal on change cells, goal-dependent pairs, preservation.
* **Reference**: the ten untrained checkpoints (prediction).

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05. "Best" is the candidate (not ceiling) with
the highest median flexible R².

| | criterion |
|---|---|
| **MQ** | maxQ is best, and above the runner-up (p < 0.05) |
| **EXPL** | best R² ≥ 0.8 × Htab's |
| **CE** | at block 2: best's edit gain (donor-optimal − none, change cells) ≥ 0.5 × whole's, and above rotated (p < 0.05) |
| **AG** | prediction and causation agree: best's edit gain ≥ every other candidate's − 0.02 |
| **D** | decision with best ≥ 0.9 × decision with H itself |

| result | reading |
|---|---|
| MQ, CE | **the brief's algorithm: score actions by their best value across goals, add a goal preference** |
| not MQ; CE, AG | H is the best candidate, causally |
| not EXPL | none of these candidates describes H well |

## Expectations

| | expectation |
|---|---|
| E1 | MQ fails (the additive-code experiment: H ranks the union of the goals' optimal actions on top, which favours popt or maxA) |
| E2 | best is popt or maxA |
| E3 | EXPL holds |
| E4 | CE holds |
| E5 | D holds |
| E6 | the best Q*-based candidate beats the best fully observed (MDP) one |
| E7 | blin's edit ≥ best's (more dimensions) |

## Limits known in advance

Candidates are functions of the posterior at the reveal only. The edits use linear maps from 4-number codes (rank 3
after centring). First decision; reward models only.
