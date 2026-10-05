# H as a mixture, and its goal weights (h mixture)

Run date: 2026-10-02. Brief: `BRIEF.md` (the what-is-H experiment's proposed next step). Models, edits and decision rule: `PLAN.md`,
committed (`a9bbc55`) **before `run.py` was run on any trained model**. The goal-weight fit was checked on a synthetic H
with known weights. Numbers from `tables.md`; data in `results.json`. Reproduce: `reproduce.py H_mixture` (1 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen; H is the goal-averaged action profile at the goal token (additive code),
predicted on held-out histories and edited causally as in the what-is-H experiment. Training sampled the three goals uniformly.

## 1. H is two codes together: which actions serve how many goals, and how close they bring the agent

| model of H | R² |
|---|---|
| P(optimal \| random goal) | 0.66 |
| mean reachability under the belief | 0.67 |
| mean Q* | 0.62 |
| **P(optimal) + mean reachability** | **0.82** (0.77–0.84) |
| all three | 0.83 |
| ceiling: affine in the posterior (what is H) | 0.82 |
| ceiling: H's own mean per posterior | 0.91 |

* **MIX holds**: together, the two leading candidates explain 0.82 of H, against 0.66–0.67 each (p < 0.001). Mean Q*
  adds 0.01. The pair matches the best affine function of the 14-cell posterior, using 8 numbers per action derived
  from the solver.
* So H combines a **discrete** part, how many goals each action is optimal for, with a **graded** part, how much closer
  it brings the agent to the goals on average under the belief.

## 2. The mixture is as causal as H itself

Edits of the goal token's state entering block 2 (the same vector under every goal):

| edit | share of whole replacement's gain | goal-dependent pairs | preserve: harm |
|---|---|---|---|
| P(optimal) | 0.59 | 0.098 | 0.015 |
| mean reachability | 0.53 | 0.078 | 0.030 |
| **mix** | **0.68** | 0.104 | 0.017 |
| **weighted composite (4 numbers)** | **0.69** | 0.108 | 0.018 |
| ceiling: H's own mean per posterior | 0.68 | 0.112 | 0.013 |
| mix, rotated | 0.00 | 0.038 | 0.005 |

* **CEm holds**: the mixture's edit gains 0.07 over P(optimal) alone (p 0.005). It reaches the same 0.68 of whole
  replacement as an edit built from H's own posterior table. No posterior-based code tested moves the decision further.
* **WC holds**: a single composite of four numbers per posterior (three per-goal quantities, each with three goal
  slopes) edits as well as the full mixture (1.01 ×).
* The remaining third of whole replacement's effect is not a function of the posterior, as the interaction-removal experiment found: per-history
  structure and the goal × history interaction. On goal-dependent pairs every goal-averaged edit stays near 0.1.

## 3. Goal weights: close to uniform, and causally uniform

| per-goal quantity | w_G1 | w_G2 | w_G3 | R² free / equal weights |
|---|---|---|---|---|
| 1[a optimal under g] | 0.38 | **0.22** | 0.39 | 0.66 / 0.63 |
| Q*(b, g, a) | 0.30 | 0.41 | 0.28 | 0.60 / 0.60 |
| reachability Q_MDP(b, g, a) | 0.22 | **0.54** | 0.25 | 0.63 / 0.61 |

* **W fails, narrowly**: free goal weights add 0.029 R² over equal weights in the nine-slope model (registered ≤ 0.02).
  For the best single quantity the largest deviation from 1/3 is 0.12 (registered ≤ 0.1).
* **But the deviations cancel across quantities.** G2 (the goal below the right arm) is underweighted in the
  optimal-action count (0.22) and overweighted in reachability (0.54). In the joint nine-slope model, some goal shares
  turn negative, a sign of collinearity between the per-goal quantities, not a stable preference.
* **Causally, equal weights suffice**: the composite with each quantity's goal weights forced equal edits as well as the
  free one (difference 0.006, p 0.053).
* Reading: H averages the goals essentially in proportion to their training frequency (uniform). G2 enters through its
  graded distance more than through its optimal set.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **MIX** | mix R² ≥ best single + 0.05 | 0.83 against 0.68; p < 0.001 | **yes** |
| **CEm** | mix edit gain ≥ P(optimal)'s + 0.05 | 0.472 against 0.403; p 0.005 | **yes** |
| **W** | free − equal weights ≤ 0.02 R²; max \|w − 1/3\| ≤ 0.1 | 0.029; 0.12 | no |
| **WC** | composite edit ≥ 0.9 × mix's | 1.01 × | **yes** |

**Registered readings:**
* MIX and CEm: **usefulness and reachability are two parts of one code, causally**.
* WC: **four numbers per posterior describe H as well as the flexible mixture, causally**.
* Not W: H weights the goals unequally, G2 in particular. §3 qualifies this: the inequality is small, reverses between
  quantities, and does not matter causally.

## Expectations

| | expectation | held |
|---|---|---|
| E1 | MIX holds | yes |
| E2 | CEm holds | yes |
| E3 | W fails | yes (narrowly) |
| E4 | WC holds | yes |
| E5 | weighted ≥ 0.9 × the mixture's R² | yes (0.93 ×) |

5 of 5.

## Conclusions

1. **H has a two-part, goal-averaged code.** For each action it combines how many of the possible goals it is optimal
   for and how much closer, on average over goals, it brings the agent under the current belief. Together they explain
   0.82 of H, as much as any linear function of the posterior.
2. **That code is what the network uses.** Editing the goal token's state by the donor's two-part code moves decisions
   0.68 of the way to the donor's, the same as editing by H's own per-posterior profile. Four numbers per posterior
   suffice.
3. **The goals are weighted about as they were seen in training: equally.** Small departures appear when one quantity
   is fitted alone, but they reverse between quantities and do not change the edits.
4. **The algorithm, the query-swap to H-mixture experiments:**
   * infer the belief;
   * for each action, count the goals it serves and measure its average progress toward the goals;
   * add the current goal's fixed preference;
   * choose.

   It ignores the value of information and the goal × belief interaction, which explains both its strength and the
   non-additive cases it fails (additive code).

Limits: linear combinations of per-goal solver quantities; one edit site; first decision; reward models only; the
nine-slope goal shares are collinear and should not be read individually.
