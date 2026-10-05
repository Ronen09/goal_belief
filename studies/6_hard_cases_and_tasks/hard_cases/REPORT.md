# The additively impossible cases: what they are, training on them, and harder tasks (hard cases)

Run date: 2026-10-05. Brief: `BRIEF.md`. **Exploratory: no plan was committed before any of these runs**, and every
reading below is post hoc. The observation-prediction experiment's reward models (ten seeds) and the additive-code experiment's additive code and solvability test
throughout. Numbers are medians over seeds unless stated. Data: the JSON files in this directory; checkpoints in
`runs/*/seed*/ckpt` (not committed, as in earlier rounds); the case file `cache/later_cases.pt` is rebuilt by
`later_cases.py`.

## 1. Hallmarks of the additively unsolvable posteriors (`hallmarks.py`, `hallmark_beliefs.py`)

At the first decision (the additive-code experiment's cells), against solvable posteriors:

* **Always a two-goal bias inversion.** The binding cycle of the additive-code experiment's difference constraints has length 2 in > 99 %
  of cases: goal X must take a, goal Y must take b, and Y's fixed bias favours a over b at least as much as X's.
  Goals moving against their own bias: 1.2 per posterior against 0.2.
* **Uncertain, multimodal beliefs** (entropy 1.20 against 1.00; most likely cell 0.40 against 0.47; prefix 2.0
  against 2.6 tokens). The dominant pattern (G1 Up, G2 and G3 Right; 2 945 histories, 12 posteriors, unsolvable in 9
  of 10 models) splits its mass over the left end, the left stub and the right arm: the right move localises.
* **Near-ties**: smallest Q* margin over goals 0.006 against 0.037.
* **Failures are mostly G2's** (≈ 64 %; optimal 0.20 against 0.76 elsewhere), and the policy takes its goal bias's
  favourite candidate in ≈ 75 % of failing cells.

## 2. More cases: decisions after the reveal (`later_cases.py`)

40 000 episodes of the solver with 20 % random moves; every decision replayed under all three goals through the
belief graph (replay checked against the true-goal nodes). G per prefix length and step, fitted on one half of the
episodes.

* Later decisions: **≈ 20 % of goal-dependent cells are unsolvable** (step 0: 5–9 %). Cases unsolvable in ≥ half the
  models: 41 831 decisions, **9 204 distinct posterior triples** (step 0 alone: about a dozen per pattern).
* What carries over: the bias inversion (1.1 goals against own bias, against 0.12), near-ties (0.016 against 0.050),
  and the network failing like the additive code (natural 0.48–0.51 against additive 0.46–0.47; solvable 0.78–0.84).
* What does not: uncertainty (later unsolvable beliefs are slightly *more* certain, most likely cell 0.64 against
  0.59), and the concentration on G2 (failures ≈ 1/3 per goal, apart from seeds 3 and 6).

## 3. Fine-tuning on the cases (`finetune.py`, `finetune_diag.py`)

3 000 supervised steps (cross-entropy towards the optimal set), test posteriors disjoint from training. Targeted: half
of each batch from the unsolvable cases; control: ordinary decisions only.

| | before | targeted | control |
|---|---|---|---|
| unsolvable cells: natural optimal | 0.49 | **0.91** | 0.88 |
| unsolvable cells: additive optimal | 0.46 | 0.55 | 0.53 |
| other goal-dependent cells | 0.82 | 0.93 | 0.94 |
| additive share of goal dependence | 0.47 | 0.32 | 0.32 |
| **on-policy greedy regret** | **0.006** | **0.034** | **0.034** |

The interaction is learned and generalises to new posteriors, and the targeted half adds only 0.026. But the endgame
breaks: seed 0's local regret at steps 9–10 rises from ≈ 0.003 to 0.14–0.18, success from 0.993 to 0.945, while the
first decision improves (0.87 → 0.90). Late states are rare in the case histories (not verified as the cause).

## 4. Training from scratch with the cases (`train_cases.py`, `eval_scratch.py`, `policy_diff.py`)

The observation-prediction experiment's reward-only PPO with the same seeds, initialisations (u = 0 regret identical) and settings, plus two
supervised steps per update on the train-side cases (coefficient 1, batch 1 024 per model). Ten seeds per arm.

| | the observation-prediction experiment | + hard-case supervision | + ordinary supervision |
|---|---|---|---|
| unsolvable cells: natural optimal | 0.49 | **0.91** | 0.86 |
| unsolvable cells: additive optimal | 0.46 | 0.53 | 0.49 |
| other goal-dependent cells | 0.82 | 0.94 | 0.95 |
| goal-irrelevant cells | 0.93 | 0.96 | 0.97 |
| additive share of goal dependence | 0.47 | 0.32 | 0.34 |
| **on-policy greedy regret** | **0.0060** | **0.0015** | **0.0015** |

* Regret falls 4× in every seed (0.0010–0.0023; one control seed 0.0044), including seeds 3 and 6 that never learned
  G2 under reward alone. No endgame damage: reward training keeps it.
* The policies are no longer additive: natural minus additive on unsolvable cells is 0.37 (the observation-prediction experiment: 0.03).
* **Targeted and control are the same policy up to seed noise**: same move in 0.906 of cells against 0.907–0.908
  between two seeds of one arm (total variation 0.071 against 0.070–0.072). Targeted is 4–6 points better on the
  unsolvable cells under every goal and at every stage of the episode; control is slightly better elsewhere. On-policy
  regret, regret per goal, length (5.59) and success (99.7–99.8 %) do not differ. Against the observation-prediction experiment the change is
  large (same move on unsolvable cells 0.51–0.53 against 0.86 between observation-prediction seeds; G2 regret 0.0113 → 0.003).
* The supervised targets come from the exact solver, so these are not reward-only agents.

## 5. Visual simulation (`sim_data.py`, `build_sim.py`, `aliased_corridor.html`)

Matched episodes (same prefix, goal and noise until the first different move) of a policy and the Bayes-optimal
solver, with beliefs, Q*, the policy's H + G decomposition and the goal-swap view; the observation-prediction experiment and both new arms (seed 0).
Over 8 192 matched episodes: return 0.644 / 0.649 / 0.647 against 0.651, success 99.29 / 99.68 / 99.71 % against
99.94 %; failed hard cases 112 / 27 / 19.

## 6. A harder task: model-free screen (`hardness.py`, `screen.py`, `screen_confirm.py`)

Hardness without a network: fit the additive code H + G to the solver's own Q*. In the maze-belief experiment's task only 16 % of Q*'s
goal dependence is additive, yet the additive move is optimal in 75 % of goal-dependent cells and loses 0.082 where
it errs.

| task (full size) | nodes | additive move fails | loss per decision | unsolvable | loss where it fails |
|---|---|---|---|---|---|
| the maze-belief experiment | 4.1 M | 0.256 | 0.0145 | 0.18 | 0.081 |
| goals (0,6) (1,4) (2,2) | 4.5 M | 0.358 | 0.0218 | 0.31 | 0.078 |
| goal = any of 10 non-start cells | 14.1 M | 0.302 | 0.0240 | 0.60 | 0.086 |
| mirror layout, best goals | 28.7 M | 0.335 | 0.0199 | 0.29 | 0.072 |
| twin-stubs layout, best goals | 11.1 M | 0.336 | 0.0188 | 0.27 | 0.064 |

Across all 360 three-goal placements on three layouts (`screen_search.json`, reduced size) the best costs the
additive policy ≈ 1.4× the maze-belief experiment. Placement matters more than layout; no variant raises the cost of an error.

## 7. Multiple goals with random values (`goalgeo/multigoal.py`, `screen_multi.py`)

Four fixed goal cells, each worth 0–3 (drawn per episode, shown at the reveal; 256 settings), collected for
discounted value within 12 moves. A pickup is observed whatever its value, so the belief depends on the collected set
but not on the values, and one belief graph (moves left × collected set × belief) serves every setting. Exact against
brute-force recursion (1e-12) and identical to the maze-belief experiment for a single goal (1e-16, equal node counts;
`tests/test_multigoal.py`).

All 210 four-goal placements, solver with 20 % random moves. Best balanced placement, the four stubs
(0,2) (0,6) (2,2) (2,6): 181 k nodes; 33 % of decisions with the location uncertain (a pickup reveals it); additive
share of the value dependence 0.10; the additive move wrong in 11 % of decisions; its loss 0.18 of return (the maze-belief experiment
≈ 0.14 on the same scale; best placement 0.205); QMDP 0.048 and most-likely-cell 0.105 of return. The values interact
with the state more than the goals did, but the additive decision still mostly works, and the belief matters only
until the first pickup.

## Limits

No registered predictions; seed-0 diagnosis of the fine-tune; hard cases defined with the observation-prediction experiment's goal biases; case
histories from the solver with random moves, not the models' own; the screen ranks tasks by the solver's additive
fit, not by what a network learns.

Next, if wanted: anonymous pickups (the agent learns it collected something and its value, not which goal), so the
belief stays uncertain through the episode; re-screen, then train with the observation-prediction recipe and a registered plan.
