# A location belief in a fixed maze, and the goal-dependent decisions it supports (round 18)

Run date: 2026-09-29. Brief: `BRIEF.md`. Frozen task and the four pre-freeze checks: `SPEC.md`, `checks.md`. Design,
pilot and 12 predictions: `THEORY.md`, committed (`b95969b`) after the exploratory pilot and **before the main runs**.
Numbers from `tables.md`; per-seed data in `runs/<condition>/seed*/{log,measures}.json`. Reproduce: `reproduce.py r18`.

## Setup in brief

A fixed maze of 14 cells: a corridor whose left arm mostly emits symbol a and right arm b (each symbol is the
sibling 40 % of the time), a noiseless landmark, three goal cells. The agent never sees its cell. A passive prefix
of 0–4 imposed moves is followed by the goal's reveal; the agent then has 12 moves, and entering the goal with
move t pays 0.9^(t−1). At the reveal the exact posterior over cells is the same whatever the goal. Exact
posterior, action values and optimal actions are known for all 4.1 × 10⁶ reachable posteriors.

What the task demands (exact, `checks.md`): acting as if the most likely cell were certain loses 0.10 of an
optimal 0.65; averaging fully observed values loses 0.012; planning without future observations loses 0.003. The
task requires acting on the whole belief. It does not require planning to observe.

Model and training as in round 17: 4 layers, width 128, 4 heads, full causal attention; PPO on reward only,
5 seeds, 1 500 updates (3.8–4.1 × 10⁷ interactions). Measurements on 40 000 fixed histories from an ε-greedy
solver. Patches are made at the reveal, on pairs from histories held out of every fit.

## 1. Behaviour

| condition | greedy regret | G1 (centre) | G2 (side branch) | G3 (corridor end) |
|---|---|---|---|---|
| reward, seeds 1, 3, 4 | 0.0044–0.0050 | 0.003 | 0.008–0.010 | 0.002 |
| reward, seeds 0, 2 | 0.044–0.046 | 0.005 | **0.125–0.132** | 0.002 |
| solver-supervised | 0.0010 | 0.0004 | 0.0024 | 0.0003 |
| frozen backbone | 0.257–0.262 | 0.34–0.36 | 0.38–0.39 | 0.045 |

* Three seeds end below the averaging baseline (0.012); two had not learned G2 after 1 500 updates. **M1 failed**
  (it required 4 of 5).
* **Goals are learned in the order of how much they need the belief.** On the fixed histories, regret at the reveal
  falls below 0.01 at update 46–57 for G3, 406–505 for G1, and 505 or never for G2. G3 is the goal whose direction
  does not depend on the belief.
* The frozen backbone learns G3 partly (0.045) and nothing else: it reaches the goal in 56 % of episodes.

## 2. The posterior is decodable before training, a little more after, and most at tokens that never act

Held-out affine decoding of the exact 14-cell posterior (share of variance, best site, mean ± sd over seeds):

| where | initialisation | 1 500 updates |
|---|---|---|
| goal token at the reveal | 0.640 ± 0.007 | 0.770 ± 0.008 |
| prefix tokens (goal not yet shown; no decision is made there) | 0.836 ± 0.008 | 0.926 ± 0.005 |
| last prefix token | 0.64 | 0.86 |
| every decision token, including those after the reveal | 0.30 | 0.37 |

* Raw tokens give 0.00 at the goal token; the first attention layer gives 0.47 untrained and 0.52 trained; the rest
  arrives by block 2.
* Decodability at the reveal rises by 0.13 while regret at the reveal falls from 0.055 to 0.009 and use of the
  evidence rises from 0.00 to 0.64 (Spearman of use with regret −0.93 to −0.98).
* The decoded vector is not a probability vector: every decoded posterior has a component outside [0, 1], and the
  mean L1 error is 0.62. What decodes is a linear image of the posterior, not the posterior.
* After the reveal decodability is low (0.37). Not having terminated is evidence there, and the histories are
  off-policy.
* P(left arm) decodes at 0.67 and the exact action values at 0.62 at decision tokens.

## 3. Changing the goal with the history fixed

At the final checkpoint the action distribution changes as much under a goal swap (total variation 0.49 ± 0.04) as
under an evidence swap (0.48 ± 0.03). The decoded posterior moves by 0.41–0.46 (L1) under the goal swap and by
1.59–1.67 under the evidence swap: a ratio of 0.26–0.28 at every residual site. The read-out of the posterior at
the goal token is mostly, not entirely, independent of the goal.

A decoder fitted without one goal fails on that goal (R² −0.6 at the last site), and recovers 0.29 ± 0.11 once each
goal's mean activation is removed, against 0.83 for a decoder fitted on the goal itself. The posterior's code at
the goal token is shared across goals only in part.

## 4. Does the evidence reach the decision by a route that does not depend on the goal?

Donor: other evidence and another goal. "To hybrid" is the behaviour for the donor's evidence with the recipient's
goal, which a goal-free carrier of the belief should produce. 493–1 074 cross-goal pairs per seed.

| patch at the reveal | same goal: to hybrid | cross goal: to hybrid | to the donor's behaviour |
|---|---|---|---|
| L0 attention, whole | 0.62 ± 0.07 | −0.05 ± 0.02 | 0.91 ± 0.04 |
| L0 goal head (one per seed) | 0.10 ± 0.09 | −0.04 ± 0.04 | 0.67 ± 0.10 |
| L0 other three heads together | 0.45 ± 0.08 | **0.22 ± 0.15** | 0.31 ± 0.29 |
| residual stream at every prefix token, entering block 1 | 0.27 ± 0.08 | **0.37 ± 0.13** | 0.07 ± 0.03 |
| same, last prefix token only | 0.13 ± 0.04 | 0.19 ± 0.08 | 0.05 ± 0.04 |
| same, entering block 2 | 0.04 ± 0.03 | 0.09 ± 0.06 | 0.00 |
| embedding of the last prefix token | 0.38 ± 0.04 | 0.32 ± 0.03 | 0.08 ± 0.04 |

* **Every seed has exactly one goal head in the first attention layer**, chosen on the fitting histories and
  confirmed on the held-out ones (0.50–0.85 of the way to the behaviour for the donor's goal; other heads ≤ 0.12).
* **The other heads carry evidence across goals, but only part of it, and not in every seed.** Per seed: to hybrid
  0.23, 0.40, 0.17, −0.04, 0.32, with 0.13–0.20 to the donor, except seed 3, where the three heads together import
  the donor's goal (0.89) although none does alone. The pilot's 0.49 and 0.69 rested on 46 and 48 pairs and did
  not replicate. **M7 failed.**
* **Prefix tokens are a goal-free route by construction, and the decision uses it for about a third.** Replacing
  their state after the first block moves the decision 0.27 (same goal) and 0.37 (cross goal) of the way, with
  0.07 to the donor. After the second block nothing is read from them. The rest of the evidence is read from the
  raw tokens by the decision token's own first attention layer. **M6 failed** in one seed (0.37 against a
  threshold of 0.35); I had expected less reading of prefix states, from the pilot's 0.11 and 0.23.
* Whole sublayers at the goal token carry the goal with the evidence, as whole components carried the position in
  round 17.

## 5. Controls

The supervised model decodes the posterior as well (0.82 at the reveal) and uses it more (0.95), and has **no**
goal head by the same criterion: its three "other" heads together move a cross-goal recipient 0.86 of the way to
the donor. The separation into a goal head and evidence heads is a property of the reward-trained solutions, not
of the task. The frozen backbone has the initial decodability (0.65, 0.84) and use 0.00.

## Registered predictions

| | prediction | held | numbers |
|---|---|---|---|
| M1 | greedy regret ≤ 0.012 in ≥ 4 of 5 seeds | no | 3 of 5; two seeds 0.044, 0.046 (G2 not learned) |
| M2 | supervised ≤ 0.003 | **yes** | 0.0010 |
| M3 | frozen backbone ≥ 0.05 | **yes** | 0.257, 0.262 |
| M4 | decodable at the reveal before training, more after | **yes** | 0.640 → 0.770 |
| M5 | prefix tokens hold the posterior (≥ 0.85) | **yes** | 0.926 ± 0.005 |
| M6 | prefix states after block 0 move the decision ≤ 0.35 | no | 0.16–0.37, one seed over |
| M7 | evidence heads carry evidence across goals in ≥ 4 of 5 seeds | no | 1 of 5 met both thresholds |
| M8 | whole L0 attention is entangled | **yes** | −0.05 / 0.91 |
| M9 | goal swap moves the decoded posterior ≤ 0.35 × the evidence swap | **yes** | 0.26–0.29; action TV 0.49 |
| M10 | posterior code shared across goals in part | **yes** | 0.29 vs 0.83 |
| M11 | use tracks regret | **yes** | ρ −0.93 to −0.98 |
| M12 | G3 learned no later than G1 and G2 | **yes** | 5 of 5 |

9 of 12 held. The three that failed are the ones that mattered most for the central claim (M6, M7) and for training
(M1). Thresholds for M6 and M7 came from two pilot models with under 50 cross-goal pairs each.

## Conclusions

1. **An inferred location posterior is linearly available, and training teaches its use more than its formation.**
   Decodability at the reveal goes from 0.64 to 0.77; use from 0.00 to 0.64. This repeats round 17 on a posterior of
   effective dimension 6.7 in place of two binary marginals.
2. **The belief supports goal-dependent decisions causally, by routes that are only partly goal-free.** Evidence
   read from prefix tokens after the first block is goal-free and accounts for about a third of the decision's
   dependence on evidence. Three of the four first-layer heads carry evidence and not the goal in four of five
   seeds, and account for about a quarter. Most of the dependence passes through components that carry the goal
   as well.
3. **The transformer mostly recomputes.** Prefix tokens hold the posterior at R² 0.93 and the decision reads a
   third of its evidence from them; the rest comes from the raw tokens. Consistent with rounds 15 and 17, with a
   larger share for the carried state than in either.
4. **What needs the belief is learned later.** G3 before G1 before G2, in every seed.
5. **Occupancy was not measured.** Whether an occupancy quantity sits between the posterior and the decision is
   open, as agreed before the round.

Limits: one maze; patches at the reveal only; decoders are affine; two of five seeds had not finished learning,
and their patches are included in the means above.

## Figures

`fig1_learning.png` regret by condition and by goal · `fig2_layers.png` decodability by site and checkpoint ·
`fig3_timeline.png` regret, decodability, and the effect of evidence and goal swaps along training ·
`fig4_patches.png` cross-goal patches.
