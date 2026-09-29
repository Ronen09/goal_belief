# Round 18: design, pilot and registered predictions

Written after the exploratory pilot and before the main runs. The pilot: two reward-trained seeds (1 500 updates)
and one solver-supervised model (1 000 updates, learning rate 3 × 10⁻⁴), kept in the scratch directory and not part
of the results; the measurement code was run on all three at initialisation and at the end, and on one along
training. What the pilot showed is marked (pilot). Thresholds apply to the five new runs.

## Setup

Task: `SPEC.md`. Exact posterior, action values and optimal actions for every history: `goalgeo/mazegraph.py`
(4.1 × 10⁶ reachable posteriors, checked against the solver). Model: round 17's transformer (4 layers, width 128,
4 heads, full causal attention), tokens [OBS] [EVT]×prefix [GOAL] [EVT]…; an event token holds a move and the
symbol seen after it; the goal appears once, in its own token; decisions are read at the goal token and at every
event token after it.

Training (`train.py`): PPO, 4 096 episodes per update, 3 epochs × 4 minibatches, Adam 10⁻⁴, clip 0.2, GAE λ 0.95,
γ 0.9, entropy bonus 0.03 → 0.003, 1 500 updates, 24 log-spaced checkpoints and the initialisation.

| condition | seeds | updates |
|---|---|---|
| ppo (primary) | 0–4 | 1 500 |
| frozen backbone, trained heads | 0, 1 | 1 000 |
| solver-supervised (learning rate 3 × 10⁻⁴) | 0 | 1 000 |

Bank (`goalgeo/mazemeasure.py`, seed 180018): 40 000 histories from an ε-greedy solver (ε = 0.1, 0.3, 0.6, 1).
Decoders are fitted on histories whose prefix evidence hashes to the fit side (70 %) and tested on the rest.
Pairs at the reveal, prefix ≥ 2, posteriors ≥ 0.5 apart in L1: recipient (evidence A, goal g), hybrid (B, g),
donor (B, g′), swapped (A, g′). Head groups are chosen on pairs from the fit histories and tested on pairs from the
held-out ones.

## Measures

* **regret**: exact, on-policy (16 384 fixed episodes) and on the bank.
* **use**: on pairs with disjoint exact optimal sets, P(own optimal set) − P(the hybrid's).
* **decoding**: held-out affine decoding of the 14-cell posterior (share of total variance), at decision tokens, at
  the reveal, and at prefix tokens.
* **patches**: moved-to-X = 1 − TV(patched, X) / TV(base, X), X ∈ {hybrid, donor, swapped}, on pairs where the
  model's own distributions differ by more than 0.3.

## Predictions

| | prediction | threshold |
|---|---|---|
| M1 | Reward training succeeds, and does better than acting on averaged fully observed values | greedy regret ≤ 0.012 in ≥ 4 of 5 seeds (the averaging baseline's exact loss on the same episodes is reported beside it) |
| M2 | Supervision succeeds | greedy regret ≤ 0.003 |
| M3 | A frozen backbone does not | greedy regret ≥ 0.05, both seeds |
| M4 | The posterior is decodable at the reveal before training, and more so after (pilot: 0.65 → 0.78) | best site: initial R² ≥ 0.5; final − initial in [0.05, 0.25], mean over seeds |
| M5 | Prefix tokens, which never act and never see the goal, hold the posterior (pilot: 0.84 → 0.93) | best site, final: R² ≥ 0.85 in every seed |
| M6 | The decision is computed from the tokens, not from the posterior held at prefix tokens (pilot: 0.11, 0.23) | replacing the residual stream entering block 1 at every prefix token: same-goal moved-to-hybrid ≤ 0.35 in every seed |
| M7 | The first attention layer splits into goal heads and evidence heads, and the evidence heads carry the evidence across goals (pilot: 0.69 / 0.14 and 0.49 / 0.14) | in ≥ 4 of 5 seeds: 1 or 2 layer-0 goal heads; the other layer-0 heads patched together, held-out cross-goal pairs: to hybrid ≥ 0.4 and to donor ≤ 0.25 |
| M8 | Whole sublayers are entangled | L0 attention, cross-goal: to hybrid ≤ 0.2 and to donor ≥ 0.7, every seed |
| M9 | Changing the goal leaves the decoded posterior almost where it was, and changes the action | every residual site: displacement under goal swap ≤ 0.35 × displacement under evidence swap; TV of actions under goal swap ≥ 0.15 |
| M10 | The posterior's code is shared across goals only in part | decoder fitted without one goal, goal means removed, last site: R² ≥ 0.25 on the held-out goal (mean over goals and seeds) and ≥ 0.15 below the own-goal decoder |
| M11 | Use develops with the policy | per seed, over checkpoints: Spearman(use, bank regret at the reveal) ≤ −0.8 |
| M12 | The goal whose direction does not depend on the belief is learned first | the first checkpoint with bank regret at the reveal < 0.01 comes no later for G3 than for G1 and G2, in ≥ 4 of 5 seeds |

## What is and is not claimed

Claimed if M4–M9 hold: an inferred location posterior is available in the network, reading it is what training
adds, and a goal-free part of the first attention layer carries it to decisions for whichever goal is set.
Not claimed: that an occupancy quantity mediates the computation (not measured); that the network plans to
observe (the task does not require it: planning without future observations loses ≤ 0.008); anything about the
posterior after the reveal, where not having terminated is goal-dependent evidence (decoding there is reported,
patches are made at the reveal only).
