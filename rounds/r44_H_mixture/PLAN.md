# Round 44: mixture, goal weights, decision rule and expectations

Written before `run.py` was run on any trained model. The goal-weight fit was checked on a synthetic H built from known
weights (0.5, 0.2, 0.3), which it recovered to 3 decimals. Brief: `BRIEF.md`.

Training sampled the three goals uniformly (round 23's `Env` default), so "weights follow training frequency" means
equal weights.

## Predictions of H (round 43's held-out setting)

* **single**: P(optimal | random goal), mean reachability, mean Q* (flexible, as round 43).
* **mix**: the three together (flexible).
* **goal weights**: for each per-goal quantity q (opt_g = 1[a optimal under g], Q*_g, reachability M_g),
  H ≈ α[a, L] + Σ_g β_g q_g, with one slope per goal. The weights are w_g = β_g / Σβ. The same model with equal weights
  is the uniform model.
* **weighted**: all nine (3 goals × 3 quantities) with one slope each. Its composite Σ β q_g is a 4-number code; the
  uniform version forces equal goal weights within each quantity.

## Causal edits (round 43's editor, goal token entering block 2)

Kinds: popt, meanMDP, mix, weighted (the composite), weighted_uniform, Htab (ceiling), and mix rotated. The measure is
the gain in donor-optimal on change cells over none, also as a share of whole's.

## Decision rule

Ten models; medians; exact Wilcoxon signed-rank tests, one-sided, p < 0.05.

| | criterion |
|---|---|
| **MIX** | mix R² ≥ the best single + 0.05 (p < 0.05) |
| **CEm** | mix edit gain ≥ popt edit gain + 0.05 (p < 0.05) |
| **W** | goal weights follow training frequency: weighted R² − weighted_uniform R² ≤ 0.02, and for the best single-quantity model max_g \|w_g − 1/3\| ≤ 0.1 |
| **WC** | the weighted composite's edit gain ≥ 0.9 × mix's |

| result | reading |
|---|---|
| MIX, CEm | **usefulness and reachability are two parts of one code, causally** |
| W | H averages the goals in proportion to their training frequency |
| not W | H weights the goals unequally; which goal is overweighted is reported |
| WC | nine numbers (three quantities × three goal weights) describe H as well as the flexible mixture, causally |

## Expectations

| | expectation |
|---|---|
| E1 | MIX holds |
| E2 | CEm holds |
| E3 | W fails |
| E4 | WC holds |
| E5 | the weighted model reaches ≥ 0.9 × the flexible mixture's R² |

## Limits known in advance

Linear combinations of per-goal quantities; one edit site; first decision; reward models only.
