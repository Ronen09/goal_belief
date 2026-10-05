# Does the goal change which evidence route the policy relies on? (goal route selection)

Run date: 2026-10-01. Brief: `BRIEF.md`. Routes, measures and decision rule: `PLAN.md`, committed (`1d2e2a3`) **before
any route intervention was applied to a trained model**. Numbers from `tables.md` and `posthoc.json`; data in
`results.json`. Reproduce: `reproduce.py goal_route_selection` (2 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward-only models, frozen; the belief-encoding-edit experiment's pairs, each under all three goals, so the evidence
difference is fixed and only the goal changes. The decision is exactly a function of two routes, both entering
block 1:

* the **interface**: the prefix tokens' states (goal-free);
* the **direct route**: the goal token's own state, computed by block 0 from the raw tokens and the goal.

Each is replaced by the donor's (from the hybrid run, the donor's prefix with the same goal). Replacing both
reproduces the hybrid (largest |a − 1| 0.0014). *Reliance* a on a route is the projection of its intervention's
logit change onto the hybrid's (1 = the whole behavioural difference). *Direct-route adequacy* is defined without
any intervention: a per-goal affine probe from the goal token's state to the exact action values, fitted on separate
histories, picks an optimal action for both histories of the cell. Cells: TV(none, hybrid) ≥ 0.1.

**Registered and post hoc.** Sections 1–3 are registered. Section 4 (between- against within-pair variance) is post
hoc.

## 1. The two routes add up, and the direct route carries most of the decision

Main pairs; median over ten models:

| goal | a_interface | a_encoding | a_direct | interaction 1 − a_interface − a_direct | direct-adequate cells |
|---|---|---|---|---|---|
| G1 | 0.31 | 0.26 | 0.69 | −0.002 | 0.69 |
| G2 | 0.29 | 0.26 | 0.71 | 0.002 | 0.72 |
| G3 | 0.29 | 0.24 | 0.72 | −0.005 | 0.72 |
| one-step agreeing pairs, G1 / G2 / G3 | 0.47 / 0.50 / 0.58 | 0.35 / 0.39 / 0.50 | 0.54 / 0.51 / 0.43 | ≈ 0 | 0.64 / 0.56 / 0.46 |

* **Interface and direct route contribute additively** (interaction within ±0.01 in every goal and model), so
  reliance on one is one minus reliance on the other.
* The goal-level means are nearly the same for the three goals. On the same pairs under all three goals: 0.21, 0.23,
  0.27.

## 2. The registered questions

Within-pair regressions (pair fixed effects; controls: unedited behavioural difference TV(none, hybrid), its logit
norm, and the oracle action gap). Median (range) over ten models:

| | coefficient on a_interface | median (range) | p | criterion | held |
|---|---|---|---|---|---|
| Q1 | G2 − G1 | −0.018 (−0.059 to 0.111) | 0.32 (two-sided) | a goal coefficient, \|median\| ≥ 0.05 and p < 0.05 | **no** |
| | G3 − G1 | −0.053 (−0.081 to 0.062) | 0.11 (two-sided) | | |
| **Q2** | direct-adequate | **−0.014** (−0.038 to 0.047) | 0.053 (one-sided) | median ≤ −0.05, p < 0.05 | **no** |
| Q2b | direct-adequate, with goal dummies | −0.010 (−0.031 to 0.044) | 0.042 | sign, p < 0.05 | yes, at a hundredth |

* **Route dependence does not change appropriately with the goal.** For a fixed pair, taking the goal under which
  the direct route handles the distinction poorly raises interface reliance by about 0.01. That is consistent in
  sign (Q2b) but a fifth of the registered size.
* **Goal effects exist in individual models but point in different directions.** In 8 of 10 models some pair of
  goals differs by ≥ 0.05 at a fixed pair (largest contrast 0.08, range 0.02–0.11), but which goal draws more on the
  interface varies by model. G3 at −0.05 is the only coefficient near the threshold, and it is not consistent
  (p 0.11).
* The matched contrast agrees: two goals of the same pair, direct-inadequate against adequate, with TV within 0.1 and
  gap within 0.02 (2 929 contrasts per model) give Δ a_interface 0.013 (−0.001 to 0.027).
* The encoding edit behaves like the whole interface (Q2: −0.011, p 0.032; goal coefficients not significant).
  a_direct mirrors a_interface.
* One-step agreeing pairs: the same (Q2: −0.011, p 0.14; Q1 p ≥ 0.16).

By the plan's table: **neither**. Route dependence is similar across goals. The next question is how a shared
evidence estimate is turned into different action preferences.

## 3. Probes

Direct-route probe accuracy (held-out): 0.90 / 0.87 / 0.95 for G1 / G2 / G3; interface probe 0.91 / 0.87 / 0.96.
The two routes' linear decodability of the action values is nearly the same, and highest for G3.

## 4. Post hoc: the pair sets the route, not the goal

On pairs with at least two eligible goals, the variance of a_interface splits into between pairs (pair means) and
within pairs (across goals):

| | share of the variance between pairs | a_interface: pairs direct-inadequate under every goal | pairs direct-adequate under every goal | pair-level correlation (adequacy, reliance) |
|---|---|---|---|---|
| main pairs | **0.82** (0.72–0.89) | **0.46** (0.29–0.57) | 0.25 (0.15–0.36) | −0.20 (−0.26 to −0.15) |
| one-step agreeing | 0.84 (0.66–0.92) | 0.57 (0.37–0.74) | 0.39 (0.25–0.65) | −0.19 (−0.27 to −0.07) |

* **Interface reliance is mostly a property of the history pair.** Four fifths of its variance is between pairs.
  Pairs that the direct route cannot resolve under any goal draw on the interface almost twice as much (0.46
  against 0.25, higher in every model).
* So the brief's hypothesis holds at the level of the evidence, not the goal. The policy relies more on the
  interface where shallow reading of the raw tokens is insufficient. That is decided by the histories, and it hardly
  changes when the goal changes.
* This also explains the policy-belief-edit experiment's clue: one-step agreeing pairs are more often direct-inadequate, and their
  interface reliance is higher under every goal. The pair type, not the goal, raises it.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | a_both = 1 within 10⁻³ | no, by numerical precision | 1.1·10⁻³ (0.6–1.4·10⁻³; float32) |
| E2 | Q1 holds | **no** | p 0.11, 0.32 |
| E3 | Q2 holds | **no** | −0.014, p 0.053 |
| E4 | Q2b holds | yes | −0.010, p 0.042 |
| E5 | interaction ≤ 0.2 | yes | ≤ 0.01 |
| E6 | one-step agreeing pairs: a_interface higher, every goal | yes | 0.47–0.58 against 0.29–0.31 |
| E7 | direct-route probe least accurate under G1 and G2 | yes | 0.90, 0.87 against 0.95 |

4 of 7 held; the two that mattered for the question (E2, E3) failed.

## Conclusions

1. **The goal does not choose the evidence route.** For a fixed pair of histories, the policy takes nearly the same
   share of its evidence from the pre-goal interface under every goal: within ±0.05 at the median after controls.
   Under goals where the direct route is inadequate it takes about 0.01 more. Individual models have goal effects of
   up to 0.1, but not in a shared direction.
2. **The evidence chooses the route.** Interface reliance varies four times more between pairs than across goals.
   Where the raw-token route cannot resolve the histories, the policy relies on the interface twice as much (post
   hoc). Which evidence contributes is settled before the goal is read.
3. **The routes add.** The decision's difference between two histories is the sum of an interface part and a
   direct part, with no interaction, under every goal.
4. **So the goal acts after the evidence is combined.** The brief's second branch applies: a largely shared
   evidence estimate, assembled the same way for every goal, is turned into goal-specific action preferences. Where
   and how that transformation happens, in the goal token's later blocks, is the next question.

Limits: adequacy is a linear probe of the route states, not the policy's own read-out; one interface depth; the first
decision after the reveal; reward models only; the post hoc decomposition was chosen after the registered results.
