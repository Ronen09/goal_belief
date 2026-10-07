# How the probability-space belief is formed (belief formation)

Run date: 2026-10-07. Brief: `BRIEF.md` (given in conversation). Measures and decision rule: `PLAN.md`, committed
(`45440ac`) **before any measurement was run**. The reward-bandit experiment's six trained transformers and their
untrained checkpoints; no training. Numbers from `tables.md`; data in `results.json`. Reproduce:
`reproduce.py belief_formation` (10 min on one GPU). The three combined patches at the end of §4 were added after a
one-seed smoke run showed why the registered patch is weak; they are marked post hoc.

**Setting.** Two blocks, width 128, four heads. At decision positions: res0 (the embedding of the decision token),
attn0, mid0 = res0 + attn0, mlp0, res1, attn1, mid1, mlp1, res2 (what the heads read). y: the log-odds, affine in the
token counts; b: the probabilities, a softmax of y. EXT: the probe fitted on decisions with every |y| < 2, tested
where one is ≥ 3.

## 1. Where the belief appears: block 0's MLP makes the probability code, block 1 completes it

| site | counts R² | EXT R² y | **EXT R² b** | untrained: counts | untrained: EXT b |
|---|---|---|---|---|---|
| res0 | 0.53 | −0.03 | −0.04 | 0.44 | −0.12 |
| attn0 | 0.88 | 0.28 | 0.35 | 0.98 | −0.37 |
| mid0 | 0.89 | 0.33 | 0.33 | 0.98 | −1.13 |
| **mlp0** | 0.87 | 0.48 | **0.75** (0.64–0.80) | 0.99 | 0.10 |
| res1 | 0.91 | 0.49 | 0.64 (0.30–0.81) | 0.99 | 0.02 |
| attn1 | 0.95 | 0.50 | 0.74 | 0.99 | 0.10 |
| mid1 | 0.93 | 0.50 | 0.80 | 0.99 | −0.06 |
| mlp1 | 0.86 | 0.49 | 0.81 | 0.99 | 0.09 |
| **res2** | 0.92 | 0.50 | **0.85** (0.72–0.90) | 0.99 | 0.09 |

* **FORM holds at mlp0.** Across block 0's MLP the extrapolating probability code goes from 0.33 (its input, mid0)
  to 0.75 (its output). Nothing before it has one, and the untrained MLP does not make one (0.10). Block 1's attention
  then imports it from the earlier positions (attn1 0.74, built from res1 at every position) and block 1's MLP adds
  a little (mid1 0.80 → res2 0.85).
* **GATHER fails by the letter and holds in substance.** The decision token's own embedding already gives the counts
  at R² 0.53 (the step and the last event are in it), and block 0's attention raises that to 0.88, just under the
  0.9 asked. The untrained attention gathers the counts *better* (0.98): a near-uniform attention over the history is
  a counter, and training makes the pattern less even (evenness 0.62–0.84 against uniform) rather than more. Counts
  are not what training builds; the nonlinearity after them is.
* The log-odds never extrapolate beyond 0.50 from any trained site, although they are affine in the counts the
  network holds at 0.9: the trained network's code is not the count-affine one.

## 2. What each component computes

IID R² of each component's output at decision positions from functions of the belief:

| component | affine in y | **affine in b** | quadratic in y | the step alone | the belief-node table (any function of the belief) |
|---|---|---|---|---|---|
| attn0 | 0.16 | 0.21 | 0.44 | 0.51 | 0.994 |
| **mlp0** | 0.41 | **0.56** | 0.53 | 0.18 | 0.995 |
| attn1 | 0.62 | 0.72 | 0.67 | 0.11 | 0.994 |
| mlp1 | 0.71 | **0.82** | 0.79 | 0.09 | 0.998 |

* Every component's output is a function of the belief state (node table 0.994–0.998): there is nothing else in
  them. But none is affine in either parametrisation: mlp0's output is 0.41 affine in y, 0.56 in b, 0.53 quadratic
  in y (E3 holds: b beats y by 0.15 and the quadratic recovers most of it). The code is "probability-like", a
  nonlinear function of the log-odds that extrapolates in b rather than y, not the probabilities themselves.
* attn0's output is half the step (0.51): block 0's attention mixes the position in with the history, and the
  counting is only part of what it does.

## 3. Attention from the decision position

| block 0 | mass on BOS | cue tokens | event tokens | evenness over history tokens |
|---|---|---|---|---|
| head 2 | 0.08 | 0.46 | 0.46 | 0.80 |
| head 3 | 0.12 | 0.44 | 0.43 | 0.84 |
| heads 0, 1 | 0.07–0.08 | 0.33–0.36 | 0.58–0.60 | 0.62–0.66 |

* E6 holds: two heads of block 0 put 0.87–0.92 of their mass on the history with evenness 0.80–0.84 of uniform, a
  counting pattern; the other two weight the event tokens, where the rewards are.

## 4. Patches at the decision token, online

The policy re-run with a component's output at the decision position replaced by its mean at that step, or by its
best affine fit from the **exact** log-odds (keeps what an affine function of the counts can give) or from the exact
probabilities.

| patch | exact regret | EXT R² b at res2 |
|---|---|---|
| none | 0.046 | 0.85 |
| mlp0 mean | **0.201** | 0.72 |
| mlp1 mean | 0.052 | 0.79 |
| attn0 mean | 0.736 | 0.32 |
| attn1 mean | 0.089 | 0.73 |
| mlp0 affine in y | 0.085 | −0.09 |
| **both MLPs affine in y** | 0.080 | −0.22 |
| **both MLPs affine in b** | **0.027** | 1.00 |
| post hoc: attn1 mean + mlp0 affine in y | 0.188 | −1.04 |
| **post hoc: attn1 mean + both MLPs affine in y** | **0.158** | −0.20 |
| post hoc: attn1 mean + both MLPs affine in b | 0.062 | 1.00 |

* Mean-ablating block 0's MLP costs 0.15 of regret, block 1's 0.006 (E5 holds); block 0's attention is everything
  (0.74: without it the decision token has no history).
* **NONLIN fails as registered**: with both MLPs replaced by affine functions of the exact log-odds at the decision
  token, the probability code at res2 is gone (−0.22) but the regret is only 0.080, because block 1's attention still
  imports the probability code that block 0's MLP made at the earlier positions (the plan's known limit). With that
  route cut as well (post hoc), a decision that is affine in the log-odds costs **0.158** of regret, against
  **0.062** for the same network affine in the probabilities. The nonlinearity is worth about 0.1 of regret, the
  distance between the model and the myopic policy and more.
* The affine-in-b patches *improve* the policy (0.027 against 0.046): an affine function of the exact probabilities
  is all the heads need, and it is better than the network's own estimate. So of what the MLPs compute (a function
  of the belief node at 0.995, only 0.56–0.82 affine in b), the affine-in-b part is the part the decision uses.

## 4b. Post hoc: the belief simplex (`simplex.py`, `simplex.md`, `simplex_*.png`)

Not registered. The exact posterior of every decision state is a point of the triangle with vertices G1, G2, G3;
`simplex_exact.png` shows them (a lattice, since the counts are discrete), coloured by decision, optimal action,
entropy and the belief itself as RGB. `simplex_decoded_*.png` shows the beliefs an affine probe decodes from held-out
activations: in distribution every site, trained or not, fills a recognisable triangle (R² 0.93–0.99), which is the
in-distribution non-result again. The discriminating picture is `simplex_ext_*.png`: the probe fitted on the
uncertain states only (every |log-odds| < 2, grey) and applied to the confident ones.

* **Untrained**: the confident states fly out of the triangle along straight lines (R² −2.5 to −3.4 at every site).
  A code affine in the log-odds extrapolates linearly, and the probabilities it predicts leave the simplex.
* **Trained, mid0** (block 0's MLP input): still outside (0.21). **mlp0**: the confident states fold back toward the
  vertices (0.64). **res2**: all three land on the corners (0.86). The MLP's nonlinearity is the saturation that
  bends the log-odds line into the simplex's corners; this is what "a probability code" looks like.

How curved the embedding is (fits within each step, pooled; `simplex.md`):

| trained | affine in y | affine in b | cubic in b | MLP from (b, step) | node table | PCA within a step: top 2 / 3 components |
|---|---|---|---|---|---|---|
| mlp0 | 0.82 | 0.86 | 0.95 | 0.94 | 0.995 | 0.88 / 0.96 |
| **res2** | 0.86 | **0.91** | 0.96 | 0.95 | 0.997 | **0.96** / 0.98 |
| untrained res2 | 0.67 | 0.68 | 0.74 | 0.70 | 0.958 | 0.58 / 0.72 |

* Within a step the trained final state is nearly planar (two principal components hold 0.96 of its variance;
  untrained 0.58) and mostly an affine image of the probabilities (0.91; cubic terms add 0.05). The rest, up to the
  node table's 0.997 (cross-validated 0.996: 3 247 distinct belief states over 98 304 decisions, 30 visits each),
  is not a smooth function of b: `simplex_pca_trained.png` shows the plane striped by the count lattice.
* So the manifold is a curved but nearly flat embedding of the simplex, folded at the corners, with the discrete
  count structure printed on it; the untrained network's is a plane of log-odds.

## 5. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **GATHER** | attn0 holds the counts (≥ 0.9), res0 does not (≤ 0.3) | 0.88 / 0.53 | no |
| **FORM** | the first site with EXT R² b ≥ 0.6 is an MLP output or the residual after one | mlp0 | **yes** |
| **NONLIN** | both MLPs affine in y: regret ≥ 0.15 and EXT b ≤ 0.5; affine in b: regret ≤ 0.10 | 0.080, −0.22; 0.027 | no |

**Registered reading.** FORM without NONLIN: *the probability code appears at an MLP but the policy can do without
it.* Too strong, for the reason the plan listed as a limit: the registered patch left block 1's attention free to
import the code from other positions. With that cut, the policy cannot do without it (0.158 against 0.062).

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | GATHER | no | attn0 0.88; res0 0.53 |
| E2 | FORM at mlp0 / res1: res1 ≥ 0.6, mid0 < 0.3 | in part | 0.64; 0.33 |
| E3 | mlp0 fitted better by b than y by ≥ 0.1; the quadratic in y recovers most of it | yes | 0.56 − 0.41; 0.53 |
| E4 | NONLIN: affine-in-y regret 0.2–0.3 (the GRU's level) | no | 0.080 (0.158 with attn1 cut) |
| E5 | mean-ablating mlp0 costs more than mlp1 | yes | 0.201 against 0.052 |
| E6 | a block-0 head with ≥ 0.8 of its mass on the history, evenness ≥ 0.8 | yes | heads 2 and 3 |
| E7 | untrained: counts at attn0 ≥ 0.8, no b code (EXT b < 0.3) | yes | 0.98; ≤ 0.10 |

4 of 7, one in part.

## Conclusions

1. **The probability code is made by block 0's MLP** from an input that is affine in the counts (extrapolation of
   b across it: 0.33 → 0.75; untrained 0.10), carried to the decision by block 1's attention from every earlier
   position (0.74) and finished by block 1's MLP (0.85). Mean-ablating that MLP at the decision token costs 0.15 of
   regret; block 1's MLP almost nothing.
2. **Counting is free; the softmax is learned.** An untrained attention over the history already gives the counts
   (0.98), and training makes the pattern less even, not more. The log-odds, affine in the counts, never extrapolate
   from the trained network (0.50); what it builds is a nonlinear function of them that extrapolates in the
   probabilities.
3. **The nonlinearity is what the policy pays for.** A decision affine in the exact log-odds costs 0.158 of regret
   against 0.062 affine in the exact probabilities, with the same network otherwise. The affine-in-b part of the
   MLPs' output is all the heads use; it does better than the network's own estimate (0.027).
4. **Seen on the simplex** (post hoc): a probe fitted on the uncertain states sends the untrained network's confident
   states out of the triangle along straight lines; block 0's MLP folds them back toward the vertices and the final
   state puts them on the corners. Within a step the final state is a nearly flat image of the simplex (two
   components, 0.96 of the variance; 0.91 affine in b), striped by the count lattice.
5. **The GRU's shortfall (0.29) is larger than this parametrisation cost (0.16).** The reward-bandit report's
   reading, that a log-odds code with a linear readout caps the GRU, explains part of it; what else limits the GRU
   is open.

Limits: patches at the decision position only (the post-hoc combinations cut block 1's attention instead of
patching the earlier positions); affine fits from the exact belief, which hands the patched network cleaner
information than it had; one task; six models.
