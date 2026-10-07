# Reward-trained hidden goal: does reward alone produce a Bayesian belief? (reward bandit)

Run date: 2026-10-07. Brief: `BRIEF.md`. Task, measures and decision rule: `PLAN.md`, committed (`b34f3f4`) **before any
model was trained**; the pilot it discloses (hyperparameters, and the clipping of the reward probabilities) came first.
Numbers from `tables.md`; data in `results.json`; training logs in `runs/`. Reproduce: `reproduce.py reward_bandit`
(25 min on one GPU).

**Setting.** A hidden goal among three, four noisy cue tokens, then six decisions among four actions whose Bernoulli
reward probabilities depend on the goal (the brief's vectors, clipped to [0.1, 0.9]); the reward is seen, so it is
evidence too. The posterior depends on the history only through its counts, so the belief MDP has 210 210 states and
Q* is exact. A 2-layer transformer and a 1-layer GRU, six seeds each, trained by PPO on reward only.

## 1. The transformer plays near the optimum and seeks information; the GRU stops short of the myopic policy

| policy | return | exact regret | optimal actions | where information pays: the optimal action | there: the myopic action |
|---|---|---|---|---|---|
| **transformer** | 4.489 (4.449–4.525) | **0.046** (0.010–0.098) | 0.92 | **0.51** (0.30–0.84) | 0.46 |
| **GRU** | 4.256 (4.249–4.260) | **0.290** (0.288–0.295) | 0.77 | 0.07 | 0.88 |
| optimal | 4.535 | 0 | 1.00 | 1.00 | 0 |
| myopic (ignores the value of information) | 4.472 | 0.068 | 0.88 | 0 | 1.00 |
| evidence-blind (the best fixed action) | 3.394 | 1.143 | 0.42 | | |

* **LEARN holds for the transformer** (regret 0.046 against the threshold 0.286) and **fails for the GRU by 0.004**
  (0.290): every GRU seed ends at the same regret, as every pilot setting did (learning rate 1e-3, 1 000 updates).
* Five of six transformers have a lower regret than the myopic policy, and where information pays they take the
  optimal, non-myopic action in half of the decisions (0.30–0.84 by seed). The median return is above the myopic
  policy's (4.489 against 4.472), but four of six seeds are above it and the registered test does not pass
  (p 0.078): **SEEK fails by its test**, with the behaviour of information seeking present in every seed.
* The GRU takes the myopic action in 0.88 of the decisions where it does not pay and is optimal in 0.77 of all
  decisions: it uses the evidence (the evidence-blind policy is at 0.42) but not as Bayes would.

## 2. What the states hold: the transformer's interface is in probabilities, the GRU's in log-odds

Affine probes; IID five-fold by episode; EXT fit on decisions after the first with every |log-odds| < 2, tested on
those with one ≥ 3. y: the log-odds (affine in the token counts); b: the probabilities (not).

| final site | IID R² y | IID R² b | **EXT R² y** | **EXT R² b** | within-action R² y |
|---|---|---|---|---|---|
| transformer trained (res2) | 0.982 | 0.990 | 0.50 (0.41–0.66) | **0.85** (0.72–0.90) | 0.993 |
| transformer untrained | 0.983 | 0.925 | 0.85 (0.66–0.99) | 0.09 (−2.5–0.64) | 0.984 |
| GRU trained (h) | 0.999 | 0.989 | **0.94** (0.89–0.96) | −0.06 (−0.5–0.36) | 0.999 |
| GRU untrained | 0.998 | 0.950 | 0.98 (0.74–1.00) | −0.30 | 0.998 |

* In distribution everything decodes from everything, untrained included (R² ≥ 0.98 for y): with i.i.d. evidence the
  log-odds are affine in the counts, which any network that sums token embeddings holds. As the plan said, IID is not
  diagnostic here, and neither is EXT for y: the untrained transformer extrapolates the log-odds at 0.85.
* **Training turns the transformer's interface into probabilities.** EXT R² of b goes from 0.09 untrained to 0.85
  (every seed 0.72–0.90) while EXT R² of y falls from 0.85 to 0.50. Q* is linear in b, and the final residual comes
  to hold b beyond the fit region. **BELIEF fails by its threshold** (0.85 against 0.9) and by that much only.
* **The GRU keeps the counts.** Its state extrapolates the log-odds at 0.94 and the probabilities not at all (−0.06).
  The heads read the state linearly, so its policy is a linear function of the log-odds, which cannot be the Bayes
  policy (the optimal regions are linear in b, not in y): this is where its regret of 0.29 comes from. It matches the
  belief-state study's finding that a GRU carries log-odds whatever it is trained on.
* The ladder (E7): y stays decodable within each optimal-action class at the final site (0.993 transformer, 0.999
  GRU): the belief is not quotiented by the action, which the value head needs anyway.

## 3. Both policies act on the belief alone

| | transformer | GRU |
|---|---|---|
| equal-belief pairs (cue and event tokens permuted: the same counts, the same posterior): JS divergence of the action distribution | 0.037 | 0.014 |
| random pairs at the same step | 0.421 | 0.358 |
| **ratio** | **0.087** (0.065–0.138) | **0.038** (0.032–0.047) |
| equal-belief pairs: the greedy action changes | 0.064 | 0.028 |
| random pairs: the greedy action differs | 0.616 | 0.526 |
| **belief-node table**: share of the logits' variance explained by the belief state | **0.998** | **0.999** |

* **STATE and TABLE hold in both.** Permuting the history leaves the action distribution nearly unchanged (the
  greedy action changes in 6 % of transformer pairs and 3 % of GRU pairs, against 62 % and 53 % for random pairs), and
  a table over belief states explains 0.998–0.999 of the logits' variance. The policies depend on the history only
  through the sufficient statistic. The GRU is the more order-invariant of the two (E4), the transformer having
  positions to learn away.

## 4. Moving the decoded belief moves two thirds of the decision

At the state the heads read, A's state is moved to B's log-odds along the probe's encoder (probe_e), along the
decoder's pseudo-inverse (probe_d), or by a random direction of the same length; the heads' output is compared with
the model's own on B.

| | transformer | GRU |
|---|---|---|
| **probe_e: gap closed** | **0.62** (0.55–0.65) | **0.68** (0.68–0.73) |
| probe_d: gap closed | 0.00 | 0.00 |
| random direction: gap closed | 0.01 | 0.01 |
| probe_e: the action is optimal at B (model's own on B: 0.92 / 0.77; no edit: 0.35 / 0.36) | 0.71 | 0.68 |
| untrained, probe_e | 0.23 | 0.02 |

* **STEER fails by its threshold** (0.62 and 0.68 against 0.8), as expected (E6). The encoder's directions carry about
  two thirds of the decision: after the edit the transformer's action is optimal at B in 0.71 of pairs (0.35 without
  the edit, 0.92 for the model's own decision on B). The decoder's pseudo-inverse moves nothing and a random direction
  of the same length moves nothing: the effect is specific to the belief directions, and the remaining third lies in
  directions the probe does not pick out (the probe-e / probe-d gap is the belief-state study's, again).

## 4b. Post hoc: where the missing third of the edit is (`edit_followup.py`, `edit_followup.md`)

Not registered. The same pairs, other edits of A's state at the state the heads read:

| edit | transformer | GRU |
|---|---|---|
| probe_e from the log-odds y (the registered edit) | 0.62 | 0.68 |
| **probe_e from the probabilities b** | **0.85** (0.77–0.89) | 0.76 |
| probe_e from Q* | 0.86 | 0.70 |
| **the mean state of B's belief node** (the ceiling of any belief-only edit) | **1.00** | **1.00** |
| A + (B's node mean − A's node mean) | 1.00 | 1.00 |

| the y edit by stratum | transformer | GRU |
|---|---|---|
| B certain (max b ≥ 0.8; 0.47 of pairs) | 0.89 | 0.84 |
| B uncertain (max b < 0.6; 0.32 of pairs) | 0.19 | 0.53 |
| first decision (0.17 of pairs) | 0.07 | 0.38 |
| decisions 4–6 (0.50 of pairs) | 0.87 | 0.71 |

* **The heads read nothing beyond the belief.** Replacing A's state by the mean state of B's belief node closes the
  gap completely (1.00 in every seed, both architectures), and so does adding the difference of node means to A's
  own state. The missing third is the probe's, not the network's.
* **It is the parametrisation.** The registered encoder is affine in the log-odds, which are affine in the counts;
  the transformer's state holds the belief as probabilities (§2), a nonlinear function of the log-odds, so the
  encoder's two directions reach the right log-odds reading (decoded R² 1.00) while leaving the probability code
  behind. An encoder fitted from the probabilities closes 0.85, from Q* 0.86. The move along the log-odds encoder is
  also too short: 0.74 of the true distance between the two states.
* The y edit fails exactly where the two parametrisations differ most: with B uncertain (0.19 closed) and at the
  first decision (0.07), and works where B is near a vertex (0.89), where log-odds and probabilities are both near
  saturation. For the GRU, whose state is in log-odds, the b encoder gains less (0.68 → 0.76) and the uncertain
  stratum is better served by y (0.53).
* An encoder from y and b together collapses (−0.30): the joint target has two degrees of freedom in four
  coordinates, so the reverse regression is ill-conditioned (a method artefact, reported for completeness).

## 5. Decision rule

| | criterion | transformer | GRU |
|---|---|---|---|
| **LEARN** | regret ≤ a quarter of the evidence-blind policy's (0.286) | 0.046: **yes** | 0.290: no |
| **SEEK** | return > the myopic policy's (4.472) | 4.489, p 0.078: no | 4.256: no |
| **BELIEF** | final site: EXT R² of b ≥ 0.9 | 0.85: no | −0.06: no |
| **STATE** | equal-belief JS ratio ≤ 0.1 | 0.087: **yes** | 0.038: **yes** |
| **TABLE** | belief-node table R² ≥ 0.95 | 0.998: **yes** | 0.999: **yes** |
| **STEER** | probe_e gap closed ≥ 0.8, random ≤ 0.2 | 0.62: no | 0.68: no |

**Registered reading (transformer).** STATE and TABLE without STEER: *the policy is a function of the belief, but
the decoded directions are not the ones the heads use.* That is too strong on the second half: the directions carry
0.62 of the decision and a random direction 0.01. With BELIEF missed by 0.05, the summary is that reward alone made
the transformer act on the belief and nothing else, hold the posterior's probabilities beyond the fit region
(0.85, from 0.09 untrained), and use the decoded directions for two thirds of its decision. The GRU is not
interpreted under the rule (LEARN fails by 0.004); its numbers say it acts on the belief as a function of the counts
and never forms the probabilities.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | LEARN: transformer yes, GRU narrowly no | yes | 0.046; 0.290 |
| E2 | SEEK holds for the transformer; optimal where information pays ≥ 0.5; fails for the GRU | in part: the test fails (p 0.078), the behaviour is there (0.51) | |
| E3 | IID ≥ 0.98 trained and ≥ 0.95 untrained; EXT y high untrained; EXT b trained ≥ 0.9, untrained < 0.5 | in part: all but the 0.9 (0.85) | |
| E4 | STATE in both, the GRU's ratio lower | yes | 0.087, 0.038 |
| E5 | TABLE in both | yes | 0.998, 0.999 |
| E6 | STEER fails, probe_e 0.4–0.75, probe_d ≈ 0, random ≤ 0 | yes | 0.62, 0.68; 0.00; 0.01 |
| E7 | within-action R² of y ≥ 0.95 at every site | yes (res0, the embedding, 0.77) | 0.993, 0.999 |

5 of 7, two in part.

## Conclusions

1. **Reward alone is enough for a transformer to infer the hidden goal and act on the belief and nothing else.**
   Permuting the evidence changes 6 % of its greedy actions (random pairs 62 %), a table over belief states explains
   0.998 of its logits, and it plays within 0.05 of the exact optimum, taking the information-seeking action in half
   of the decisions where it pays.
2. **The belief it holds is in the form the decision needs.** Training moves the final state's extrapolating code
   from the log-odds (which the counts give for free, 0.85 untrained) to the probabilities (0.09 → 0.85), in which
   the Bayes value is linear. Moving the state along the log-odds probe's directions carries two thirds of the
   decision, along a probability probe's 0.85, and to the mean state of the target belief all of it (post hoc); a
   random move carries nothing.
3. **The GRU stays with the counts.** Its state is the log-odds (0.94 beyond the fit region) and never the
   probabilities; it ends at a regret of 0.29, between the evidence-blind and the myopic policy, in every seed and
   every setting tried. The belief-formation experiment (later) finds that a decision affine in the exact log-odds
   costs 0.16 of regret, so the linear readout of a log-odds code explains part of the GRU's shortfall, not all.
4. The usual caution holds: in distribution everything is decodable from everything, trained or not. The
   discriminating measures were extrapolation of the non-affine quantity, the permutation test, and the edit.

Limits: one task and reward matrix; i.i.d. cues, so counts are sufficient and the cue phase cannot distinguish a
filter from a tally (the hidden-goal experiment's channel environment would); the transplant moves one position's
state and does not re-run the future; the GRU is one layer with a linear readout, which is what makes its result
sharp and also what limits it; six seeds per architecture; SEEK was judged on a six-seed sign test.
