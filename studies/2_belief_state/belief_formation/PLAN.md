# Belief formation: measures, decision rule and expectations

Written before any measurement of this experiment was run. The reward-bandit experiment's six trained transformers
(2 blocks, width 128, 4 heads) and their untrained checkpoints; no training. Brief: `BRIEF.md`. Code: `measure.py`
(results, tables).

## The question

The counts of the tokens are a sufficient statistic, the log-odds y are affine in them, and the probabilities b are
a softmax of y. A transformer can gather counts by attention (a linear operation over the history) but can only make
b from y with a nonlinearity, which its MLPs have. Where along the two blocks does b appear, which component makes
it, and is that component's nonlinearity necessary for the policy?

## Sites

At decision positions (the last cue token and every event token), on the evaluation episodes of the reward-bandit
experiment: res0 (the embedding), attn0, mid0 = res0 + attn0, mlp0, res1, attn1, mid1, mlp1, res2 (the state the
heads read). Component outputs are taken alone as well as in the residual.

## Measures (`measure.py`)

**A. Where the belief appears.** At every site, affine probes (`goalgeo/beliefprobe.py`): the counts (13 numbers:
5 cue, 8 outcome), IID; y and b, IID and EXT (the reward-bandit split: fit on decisions after the first with every
|y| < 2, test on those with one ≥ 3). The same for the untrained checkpoints.

**B. What each component computes.** For each component output (attn0, mlp0, attn1, mlp1) at decision positions,
the IID R² of a least-squares fit from: y (affine); b (affine); y with its squares and product (quadratic); and the
one-hot belief node (the table, the ceiling of any function of the belief). The gap between the y-affine and the
b-affine fits is the component's softmax-like nonlinearity.

**C. Attention.** From the decision position, the attention mass on cue tokens, event tokens and BOS, per head and
block; and the evenness of the mass over the history tokens (entropy over them, relative to uniform).

**D. Causal patches at the decision token, online.** The policy is re-run in the environment with one component's
output at the decision position (the last position of each prefix) replaced:
* *mean*: by its mean over decisions at that step (ablation);
* *affine in y*: by its best affine fit from the exact log-odds of the current state (fitted on natural runs);
* *affine in b*: by its best affine fit from the exact probabilities;
for mlp0, mlp1 and both MLPs; mean ablation also for attn0 and attn1. For each patched policy: return, exact regret,
and the EXT R² of b and of y at res2, measured on the patched run step by step (the same patch applied at the
decision position of each prefix). The affine-in-y patch keeps what an affine function of the counts can give and
removes the rest; the affine-in-b patch keeps the probability code. If an MLP's job is the y → b nonlinearity, the
affine-in-y patch of it should cost the policy and the b code, and the affine-in-b patch should not.

## Decision rule

Six models; medians.

| | criterion |
|---|---|
| **GATHER** | attn0's output holds the counts: IID R² ≥ 0.9; res0 does not: ≤ 0.3 |
| **FORM** | the first site in order whose EXT R² of b is ≥ 0.6 is an MLP output or the residual after one, not an attention output or mid0 / mid1 |
| **NONLIN** | replacing both MLPs by their affine-in-y fits raises the regret to ≥ 0.15 and lowers EXT R² of b at res2 to ≤ 0.5, while replacing them by their affine-in-b fits keeps the regret ≤ 0.10 |

| result | reading |
|---|---|
| GATHER, FORM and NONLIN | **the belief in probabilities is made by an MLP's nonlinearity from counts that attention gathers**, and the policy needs that nonlinearity |
| FORM without NONLIN | the probability code appears at an MLP but the policy can do without it (the heads read a log-odds-linear part) |
| not FORM | the probability code is assembled across attention and MLPs with no single maker |

## Expectations

| | expectation |
|---|---|
| E1 | GATHER holds: res0 holds no counts (the embedding of one token), attn0 holds them (≥ 0.9) |
| E2 | FORM holds at **mlp0 / res1**: EXT R² of b ≥ 0.6 at res1 and < 0.3 at mid0 |
| E3 | mlp0's output is fitted better by b than by y (R² difference ≥ 0.1), and the quadratic fit in y recovers most of that difference |
| E4 | NONLIN holds: affine-in-y on both MLPs gives regret 0.2–0.3 (the GRU's level) and EXT b ≤ 0.5; affine-in-b keeps regret ≤ 0.10 |
| E5 | mean-ablating mlp0 costs more regret than mean-ablating mlp1 |
| E6 | block 0's attention from the decision position is a counting pattern: at least one head puts ≥ 0.8 of its mass on history tokens with evenness ≥ 0.8 of uniform |
| E7 | the untrained checkpoints hold the counts at attn0 as well (≥ 0.8) and no b code anywhere (EXT b < 0.3) |

## Limits known in advance

Patches act at the decision position only, so what earlier positions compute and pass through block 1's attention
is left in place. Affine fits in y and b use the exact belief, not a decoded one. One task; six models.
