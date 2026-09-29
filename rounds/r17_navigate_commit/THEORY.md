# Round 17: design, pilot and registered predictions

Written after the exploratory pilot and before the main runs. The pilot: three reward-trained models on the
fixed-q task (two seeds, 2 000 to 2 600 updates, in the scratch directory, not part of the results), one
solver-supervised model, and the measurement code run on one of them at initialisation and at 2 000 updates.
Everything below that the pilot already showed is marked (pilot); the main runs use five new runs and the
thresholds apply to them.

## Setup

Environment: `SPEC.md`. Model: `goalgeo/navmodel.py`, 4 layers, width 128, 4 heads, pre-LN, learned absolute
positions, full causal attention, no carry, gate or bottleneck. Tokens: two configuration tokens (station cell
and numeric reliability), then for each step a decision token (own cell, remaining actions, used flags) and an
event token (action taken, report returned). A report appears once, in its event token.

Training (`train.py`): PPO, 4 096 complete episodes per update, 3 epochs × 4 minibatches, Adam 10⁻⁴ (50 warm-up
updates), clip 0.2, GAE λ 0.95, γ 1, value coefficient 0.5, entropy bonus 0.05 → 0.005 linearly, gradient norm
0.5. What the pilot decided:
* entropy 0.01 collapses onto "walk to the nearest corner and commit" (regret 0.085; never querying costs
  0.101) and stays there; 0.05 leaves it after about 1 000 updates;
* budget **3 000 updates** (about 1.1 × 10⁸ environment interactions) for the fixed-q task: the pilot seeds
  reached greedy regret 0.014 at 2 000 and 0.007 at 2 600;
* the transition lay between 900 and 1 500 updates in both seeds: checkpoints every 100 updates from 600 to
  1 800, on top of 20 log-spaced ones and the initialisation.

| condition | reliabilities | seeds | updates |
|---|---|---|---|
| ppo (primary) | fixed 0.8 | 0–4 | 3 000 |
| ppo | grid, 32 trained pairs | 0–2 | 3 500 (not piloted) |
| frozen backbone, trained heads | fixed | 0, 1 | 2 000 |
| solver-supervised | fixed; grid | 0 | 1 500 |

Evaluation bank (`goalgeo/navbank.py`, seed 170017, built without any model): 24 000 broad histories over all
layouts and 48 000 dense ones (40 layouts × 3 starts × 400), from an ε-greedy solver, random wanderers and
investigators that visit the stations in either order. Probes are fitted on 70 % of the layouts and tested
on the rest; the dense layouts are split into 20 for choosing components and 20 for testing them. Matched
pairs (`goalgeo/navcausal.py`): twins (same history, one seen report flipped), same-belief donors (same public
state and belief, different token sequence), cross-position donors (the twin's evidence, another cell with the
same legal actions).

## Measures

* **regret**: on-policy greedy and sampled (16 384 episodes); on the bank, the expected local regret of the
  model's action distribution, by number of clues.
* **use**: on twin pairs with disjoint exact optimal sets, P(own optimal set) − P(the twin's). 0 = evidence
  ignored, 1 = perfect.
* **decoding**: held-out R² of affine (ridge) decoders for P(right), P(bottom), their product and the four goal
  probabilities at 9 sites; invalid decoded probabilities; Adam-trained affine and 64-unit decoders with one
  budget.
* **causal**: a component's output at the decision token replaced by the donor's. cf_mass: gain in probability
  of the counterfactual optimal set as a share of the model's own gain on the twin. Cross-position: to_twin and
  to_donor, how far the recipient's distribution moves to the model's behaviour at (recipient's cell, donor's
  belief) and at the donor's cell.

## Predictions

| | prediction | threshold |
|---|---|---|
| N1 | Reward training succeeds | greedy regret ≤ 0.02 at 3 000 updates in ≥ 4 of 5 seeds |
| N2 | Supervision succeeds, more precisely than reward | supervised greedy regret ≤ 0.005 and below every PPO seed |
| N3 | The frozen backbone does not get past the no-query plateau | greedy regret ≥ 0.05 and use ≤ 0.2, both seeds |
| N4 | The marginals are decodable before any training (pilot) | initialisation, best site: mean R² of the two marginals ≥ 0.6; raw-input baseline ≤ 0.05 |
| N5 | Training does not make the marginals more affinely decodable | final minus initial best-site R², mean over seeds, in [−0.15, +0.15]; final best-site R² < 0.9 in every seed |
| N6 | The joint posterior is not affinely represented | product term R² < 0.4 at every site in every seed at 3 000 updates |
| N7 | Use develops with the policy, not with decodability | per seed over checkpoints: Spearman(use, bank regret with ≥ 1 clue) ≤ −0.8; Spearman(best-site marginal R², bank regret) > −0.5 |
| N8 | Evidence enters at the first attention layer (pilot) | final, held-out layouts, twin patch: cf_mass ≥ 0.7 for L0.attn, ≤ 0.2 for each of L1–L3.attn, every seed |
| N9 | Controls are flat | final: \|cf_mass\| ≤ 0.05 for every component with same-belief donors and with norm-matched random moves |
| N10 | A report moves its own marginal only | final, twin patch of L0.attn: decoded displacement of the other marginal ≤ 0.2 × that of the reported one |
| N11 | No component carries the belief apart from the position (pilot: L0.attn to_twin 0.11, to_donor 0.90) | final, cross-position: L0.attn to_donor ≥ 0.6; no head or sublayer has to_twin ≥ 0.6 with to_donor ≤ 0.3 |
| N12 | The upstream computation changes too, not only the read-out | the twin difference in L0.attn's output, relative to the output's norm, grows ≥ 3× from initialisation to 3 000 updates (pilot: 0.05 → 0.84) |
| N13 | Reliability is used when it varies | grid models: greedy regret ≤ 0.03 on trained pairs in ≥ 2 of 3 seeds; on held-out values and combinations within 0.01 of trained pairs |
| N14 | Decoding generalises over reliability | grid models: best-site marginal R² on held-out pairs ≥ that on trained pairs − 0.1 |
| N15 | The fixed-q models ignore the reliability tokens | fixed-q models evaluated on the grid: greedy regret at (0.6, 0.6) ≥ 0.02 above the grid models' |

## Which row of the brief's table is predicted

Row 2 with a qualification from row 3. The belief's two marginals are in the residual stream from the
start, because untrained attention already averages report embeddings into the decision token (N4); what training
adds is use (N7). But the upstream computation changes as well (N12), and the trained representation is
organised around the action at the current position (N11) and leaves out the product of the marginals (N6).
Outcomes that would contradict this: decodability that rises with falling regret (row 1: N5 and N7 fail), or
a component that transfers the belief across positions (N11 fails).

## What is not claimed

Decoding is not evidence of use (N4 holds at initialisation, where use is 0). The fixed-q task has nine beliefs
and does not show a general Bayesian computation; N13–N15 address only reliability. Non-factorising likelihoods
are out of scope.
