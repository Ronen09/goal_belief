# goalgeo — summary of results

Condensed from the per-round reports; numbers are seed means as reported there. See
`README.md` for the round-to-file map.

## Rounds 1–4: what shapes the learned geometry

**Round 1 — occupancy geometry (`rounds/r01_occupancy/REPORT.md`).** A 3-layer MLP trained by behavioural
cloning toward a supplied goal learns a hidden geometry only weakly predicted by the occupancy
vectors (RSA 0.29–0.35 at the last two layers) and strongly predicted by the policy table
(0.73–0.78 in the base grid, 0.77–0.85 in the twins maze, where every other hypothesis is near
zero). States with identical futures merge (portal pockets at 0.35 of the median distance);
mirror states requiring opposite actions do not. The representation is largely additive (≈44 %
state, 46 % goal, 11 % interaction at the middle layer). Goal-difference steering flips 88–100 %
of eligible states versus 41–49 % for random directions. With stochastic transitions the
geometry follows the stochastic-optimal policy table (partial RSA 0.61–0.69 vs 0.12).

**Round 2 — policy quotient vs occupancy (`rounds/r02_policy_quotient/REPORT.md`).** Under argmax targets the
policy table leads a six-way RDM regression in all nine environments (β 0.49–0.86). When no
optimal action changes between two λ values, the trained networks are bit-for-bit identical.
Under Boltzmann targets softmax(Q/0.02), hidden change tracks occupancy change smoothly
(Spearman 0.27–0.37 among non-flipping pairs) and advantage geometry leads (β 0.51–0.61). The
interaction component is not causally necessary (removing it costs ≤ 7 pp accuracy). Steering
thresholds are predicted by the linearised action-boundary crossing (Spearman 0.92–1.00);
occupancy predictors add nothing.

**Round 3 — the supervision bottleneck (`rounds/r03_supervision/REPORT.md`).** Last-layer policy RSA falls
monotonically with target temperature, from 0.78 (hard) to 0.05 (τ=5), while PR rises from 3.1
to 16. Hidden dimensionality follows target informativeness, not target rank. Under hard
targets the within/between class ratio falls 0.98 → 0.46; held-out rows classify at 92–94 %.
The hard-target quotient replicates across MLP, 6-layer MLP, residual MLP and a 2-layer
transformer (RSA 0.71–0.78); the soft-target geometry replicates in the MLPs but not in the
normalised architectures (RSA ≤ 0.17). Six of eight success criteria met, one in part.

**Round 4 — HMM objectives (`rounds/r04_hmm_objectives/REPORT.md`).** For two one-step-equivalent
beliefs, a window MLP collapses them under one-step training (0.24 of the control distance)
and separates them under 2-step training (1.05). A GRU does not collapse them (0.85) but stops
amplifying them; the belief stays decodable at R² ≈ 1. Sequential next-token training behaves
like one-step training in geometry, with exact or sampled targets.

## Rounds 5–7: what makes a distinction prominent (GRU, TASK4 HMM family)

**Round 5 — prominence (`rounds/r05_prominence/REPORT.md`).** In 27 of 35 conditions the delayed cue is
99 % decodable one step before use while its prominence ranges 0.25–1.0 of the control.
Frequency, loss weight, expected forgetting cost and gradient magnitude do not set prominence;
the required log-odds gap at the time of use does — the frequency, strength, weight and grid
sweeps fall on one curve against it.

**Round 6 — readout scale (`rounds/r06_readout_scale/REPORT.md`).** The required gap is a constraint on the
product of readout gain, hidden separation and alignment. Halving a fixed readout gain doubles
the hidden separation; an eightfold gain shrinks it to a quarter, with identical outputs and
loss. With a free readout, the gain barely changes with δ and the representation absorbs it.

**Round 7 — allocation (`rounds/r07_allocation/REPORT.md`).** All 30 runs reach the Bayes floor with an
achieved gap of 4.305. The split among gain, separation and alignment is set by (1) relative
speed during fitting — changing the readout learning rate changes the split in proportion,
(2) initial scale, asymmetrically — a large initial gain is kept, a small one is grown, and
(3) a weak post-floor drift toward an intermediate allocation.

## Rounds 8–11: invariants and implementation freedom

**Round 8 — representation invariants (`rounds/r08_invariants/REPORT.md`).** Across 64 functionally
equivalent GRUs:

| measure | coordinate-invariant | stable across equivalent models | sensitive to the function |
|---|---|---|---|
| raw distances, norm, PR, P_metric, gain, cos θ | no | no (CV 0.15–0.9) | weak |
| Euclidean / cosine RSA | no | yes (CV 0.02) | no |
| OLS decodability, rank | yes | yes | no |
| readout contrast (w_x−w_y)·Δh | yes | yes (CV 0.004) | yes |
| ‖JΔh‖, Fisher metric | yes | no (CV 0.27 / 0.22) | weak |
| divergence after finite propagation | yes | yes (CV 0.000) | yes |

Criteria 1–4 and 7–9 hold; 5 and 6 fail (local linearised measures are no more stable than raw
distance).

**Round 9 — intervention equivalence (`rounds/r09_interventions/REPORT.md`).** Steering vectors in equivalent
models, matched by effect, give the same output distribution at every strength; matched by
geometry they do not. The geometries are linear images of each other on the reachable set
(R² 0.997; mapped steering vectors cosine 0.997). Local functional metrics disagree across the
same models by CV 0.4–1.2 at arbitrarily small scale, in every direction. Finite-propagation
divergence is invariant at every horizon.

**Round 10 — transformer reproduction (`rounds/r10_transformer/REPORT.md`).** 2-layer pre-LN transformer,
with and without final LayerNorm, 60 models (53 at the Bayes floor). Decodability itself tracks
use (carrying a cue forward requires an attention pattern built only when the cue matters). The
required-gap identity holds at the readout interface in every model; with final LayerNorm the
freedom goes into alignment, without it separation regains about half its GRU role. Readout
contrast and interface decodability are stable to CV 0.004; raw geometry, gain, separation,
alignment and local metrics are not. Single-position patching effects vary by seed; patching
all positions between cue and use restores the invariant.

**Round 11 — implementation freedom (`rounds/r11_implementation_freedom/REPORT.md`, theory and pre-registered
predictions in `rounds/r11_implementation_freedom/THEORY.md`).**

- *Claim A, cut identifiability.* Over 20 models, 5 routing regimes and 3 complete cuts, the
  patched prediction equals the counterfactual prediction bitwise (deviation 0.0). The
  single-node effect ranges 0 → 0.368 (CV 1.13) across equivalent models; the sum of the two
  route effects ranges 0.367 → 0.798. Claim A holds as stated.
- *Claim B, factorisation freedom.* C = g·D·cos θ is fixed by the function (CV 0.006 across 77
  equivalent models whose factors span 37×; the three log-slopes sum to zero to three
  decimals). The frozen-LayerNorm bound C ≤ 4c√d held in all 49 frozen models but is not
  tight: the attained ceiling is C = 14.4c at δ=0.4 (non-antipodal rows ×0.65, non-antipodal
  states ×0.70), predicting a boundary of 0.306 against an observed 0.4 (0.220 vs 0.3 at
  δ=0.2). A 3× learning rate or 4× steps moves capped models by < 1 %. The separation D
  saturates at 11.3 (δ=0.4) and 5.8 (δ=0.2). Low-gain δ=0.2 frozen models (c ≤ 0.07) are stuck
  (cos θ ≈ 0.3), not capped. The identity half of Claim B holds; the architecture-predicts-
  factorisation half holds only ordinally; the quantitative boundary is falsified.
- 8 of 12 pre-registered predictions held (P1, P2, P3, P3b, P4, P7, P9a, P9c).

## Round 12: a hidden goal

**Round 12 — hidden goal (`rounds/r12_hidden_goal/REPORT.md`, theory and pre-registered predictions in
`rounds/r12_hidden_goal/THEORY.md`).** A latent goal (K = 4; also 3, 5) is inferred from 24 noisy tokens, and
the exact Bayesian posterior is computed. Two evidence processes are used: i.i.d. (log-odds
exactly affine in token counts) and a latent sticky reliability channel (best count-affine R²
0.869). Four objectives (posterior, soft and hard Bayes-optimal actions, next observation) are
trained on a 2-layer transformer and a GRU, and an affine probe to the K−1 log-odds is scored on
held-out sequences, confident posteriors (EXT), a held-out posterior region (CONF, also withheld
from network training) and later time steps.

- Posterior-trained models: interface R²_y ≥ 0.975 on every held-out split, both architectures,
  K = 3–5. In the channel env the probe removes 94–97 % of the best tally's error. Networks never
  trained on the conflict region decode it at 0.994–1.000 with unchanged output KL.
- In-distribution R² does not discriminate: every trained iid transformer reaches ≥ 0.969 at its
  best site and an untrained one 0.82. Extrapolation does: the `goal` interface is affine in
  log-odds (EXT R²_y 0.98, R²_b −0.97) and the `act_soft` interface in probabilities (EXT R²_y
  0.55, R²_b 0.999).
- Gain over counts at the channel-env interface: goal 0.94 / 0.97, next_obs 0.51 / 0.79,
  act_soft 0.39 / 0.73, untrained −0.78 / −1.28 (transformer / GRU). In the transformer the
  nonlinear filter appears in block 2.
- Hard-target transformers carry the posterior upstream and compress it to the decision at the
  last block. Trained without the conflict region, they fail there (output KL 2.7 nats vs 0.01),
  while posterior-trained models do not. In the channel env hard-target models are capped at
  ≈ 98.5 % optimal-action agreement, and 4× training does not raise it (not pre-registered).
- 4 of 11 pre-registered predictions held (P1, P2, P5, P10). The failures are three threshold
  near-misses, one mis-specified criterion (P8) and three substantive: the GRU carries log-odds
  whatever the target, the first transformer block is not a running mean, and the channel
  hard-target models are capped.

**Round 12, part 2 — is the decoded posterior the causal state? (`rounds/r12_hidden_goal/causal/REPORT.md`,
predictions in `rounds/r12_hidden_goal/causal/THEORY.md`).** Interventions on the round-12 models (no
retraining). The GRU state is a complete cut; no single transformer position is.

- Posterior transplant, GRU `goal`: setting the decoded log-odds to another history's value along
  the encoder directions makes the whole future indistinguishable from having seen that history
  (gap closed 1.000 over 18 steps, iid). A half move gives the Bayes future of half the evidence
  (0.999). The minimum-norm move with the same decoded value closes only 0.78 (iid) and 0.46
  (channel), and a random direction ≈ 0. In the channel env the probe edit follows the Bayes
  future of "b′ with the original channel belief" (KL 0.021 vs 0.030 to genuine-B).
- Single transformer positions: the edit switches the immediate output (res2 SWAP 1.000) and
  leaves the future almost untouched (≤ 0.16).
- Equal-belief histories (8 of 12 tokens different): downstream effect < 10⁻⁵ of random pairs
  in iid, although the states differ (0.20 of random distance, 0.8 % of it in the belief span).
  In the channel env, equal-b histories with different channel beliefs diverge by exactly the
  Bayes amount (ratio 1.01, Spearman 1.00); equal joint states do not.
- Same action, different belief: every network, hard-action ones included, acts differently
  once later evidence separates the pair (flip rate 0.991–0.999, iid). In the hard-action
  transformer the last block makes the action exactly linear (0.95 → 1.000), discards history
  detail (0.80 → 0.43) and compresses the belief within an action only slightly (0.973 → 0.932),
  in the sites the future never reads.
- 10 of 14 pre-registered predictions held.

## Round 13: the full filter state

**Round 13 — does a recurrent network learn the minimal predictive state? (`rounds/r13_filter_state/REPORT.md`,
predictions in `rounds/r13_filter_state/THEORY.md`).** In the channel environment the minimal predictive
state is the 8-state joint filter over (goal, channel), with 7 coordinates (observability rank 8).
The marginals (b, P(on)) are not sufficient, and the round-12 objectives expose only 3 (`goal`,
`act_soft`) or 5 (`next_obs`) of the 7 coordinates.

- Every trained GRU decodes all 7 coordinates (goal block 0.96–0.996, channel block 0.99), but an
  untrained GRU also decodes the channel block (0.98), a short-memory quantity. Only the goal
  block separates trained from untrained (0.996 vs 0.70).
- Matched edits of the decoded state: moving both blocks to another history's values closes
  0.99 of the gap to that history's future (round 12's goal-only edit: 0.95–0.97). The goal block
  alone follows its own counterfactual (0.96). The channel block alone is imprecise (0.54–0.63 of a
  gap 1/40 as large).
- Equivalence hierarchy (`goal` GRU): histories matched on b or on the marginals diverge by
  exactly the Bayes amount (ratio 1.01, Spearman 1.00). Histories matched on the full state
  agree to ~10⁻⁷ JS, below the model's own error against Bayes (~10⁻⁵). Histories of length 8
  and 16 with the same full state diverge at 1.13× Bayes and 2 × 10⁻⁴ of random. The same holds
  for `act_soft` and `next_obs`.
- Hidden-size bottleneck: in iid the GRU is exact at n = 3, the minimal dimension (KL 0.028 →
  10⁻⁴ from n = 2 to 3). In the channel env the largest single-step KL drop comes at n = 7, and
  Bayes-level equivalence across lengths needs n ≥ 12. Under a bottleneck the supervised goal
  block is kept before the channel block at every n ≤ 6.
- 4 of 10 pre-registered predictions held. Two of the failures (F5, F7) come from a criterion
  that divides by a Bayes divergence of ~10⁻⁸.

## Round 14: windowed transformers

**Round 14 — does restricting attention force a steerable belief state? (`rounds/r14_window_carry/REPORT.md`,
predictions in `rounds/r14_window_carry/THEORY.md`).** A 2-layer transformer attends only to the last l
positions (l = 1, 2, 4, 8, full), with or without a recurrent carry of the previous position's
readout interface. It is trained to output the exact posterior in both round-12 environments.

- Without the carry, every windowed model fails beyond its receptive field of 2l − 1 tokens (KL
  0.04–0.45); only full attention converges. The window removes access to evidence and gives the
  network nowhere to keep it.
- With the carry, every window converges (KL ≤ 0.0003) and the carried vector holds the full
  filter state (goal R² ≥ 0.985, channel 0.99).
- With window 1 the carried vector is a complete cut and as steerable as the GRU: swapping it
  transfers the whole future (1.000), and the affine probe edit closes 0.96–1.00. The share of the
  future one edit controls falls with the window (l = 2: 0.81–0.84; l = 4: 0.42–0.46; l = 8:
  0.27–0.29). At full attention the network ignores the carry (iid: −0.005).
- Cross-length equal filter states: 1.63× Bayes for carry, l = 1 (GRU 1.13×).
- 7 of 7 pre-registered predictions held.

## Round 15: prior or recomputation in a next-token transformer

**Round 15 — does position t+1 use the belief exported by position t as a prior? (`rounds/r15_prior_vs_recompute/REPORT.md`,
predictions in `rounds/r15_prior_vs_recompute/THEORY.md`).** Standard full-attention transformers (2 or 4 layers,
context 24 or 64) are trained on sampled next tokens in the channel environment. Position t+1 is
recomputed from edited K/V sources of positions ≤ t: position t's belief-carrying exports swapped
for another history's (or probe-edited toward its decoded state) with all tokens kept, or the
older evidence replaced or hidden.

- The transplanted belief gets 0.4–8 % of the weight at position t+1 (output predictive, calibrated
  so baseline = 0 and the consistent counterfactual = 1).
- With conflicting sources, position t+1 follows the tokens. Its output equals the exact Bayes
  update of the token sequence it can attend to, within −0.016 to +0.029 over all 24 (model,
  position) cells.
- Hiding the older evidence breaks the inference even with position t's state intact.
- The prior's weight rises slightly with context (≤ 0.02 → 0.03–0.06; the brief's hypothesised
  direction) and does not track attention to position t (ρ = −0.61).
- 2 of 9 pre-registered predictions held. Three failures come from reading raw probe-decoded λ
  without calibrating against the baseline and positive control, and one (the sum rule) ignored
  that the corrupted-evidence cell keeps the raw token at t.

## Round 16: inducing recurrence by incentive

**Round 16 — random historical K/V dropout during training (`rounds/r16_kv_dropout/REPORT.md`, predictions in
`rounds/r16_kv_dropout/THEORY.md`).** Next-token models on the channel process are trained with each
historical K/V entry removed with probability p. Plain 4-layer transformers keep self and the previous
position; carry transformers (round 14) always keep the carried state. The prior's weight is round 15's
calibrated λ when everything position t exports comes from another history.

- Carry: the prior's weight rises smoothly with p, 0.36 (p = 0) → 0.65 (0.05) → 0.80 (0.2) → 0.90
  (0.5) → 0.975 (0.9) → 1.000 (1). It is logit-linear in p (slope 0.69, R² 0.97), and the weight on
  older evidence falls in step. The task is solved at every p (KL ≤ 0.006, best at high p), and the
  learned weighting persists with full attention at test.
- Plain: the weight rises from 0.03 to ~0.25 by p = 0.1–0.25 and saturates. Without same-layer
  recurrence, extra dropout only costs accuracy.
- A carry is used even at p = 0 (0.36), unlike round 14's goal-trained model; at low p its weight falls
  with position.
- 5 of 7 pre-registered predictions held. The carry transition is front-loaded, below the registered
  grid's intermediate points; a follow-up grid (p = 0.05–0.3) resolves it.

## Round 17: navigate, investigate, commit

**Round 17 — how does a reward-trained full-attention transformer come to use a belief? (`rounds/r17_navigate_commit/REPORT.md`,
predictions in `rounds/r17_navigate_commit/THEORY.md`).** A 5 × 5 grid with a hidden reward corner and two noisy clue
stations; the agent moves, queries and commits. Exact belief-state solver; PPO on reward only, 5 seeds; all
representation measures on fixed histories replayed at every checkpoint.

- All seeds reach greedy regret 0.009–0.013 (never querying: 0.101). Supervised: 0.002. Frozen backbone: 0.117.
- The belief's marginals are affinely decodable from the first attention layer before training (R² 0.76) and
  no more so after (0.76, with a peak of 0.91 in between). Use of the evidence goes from 0.00 to 0.93 and tracks
  regret (ρ = −0.97). The product of the marginals, which the joint posterior needs, stays at R² ≤ 0.30.
- Two transitions 800 updates apart: acting on evidence that is present (updates 80–270), then walking to the
  stations (1 000–1 800). The second gives most of the on-policy improvement and is a change in visitation.
- Evidence reaches the decision through the first attention layer in every seed (patch: 1.00; controls ≤ 0.02).
  Two of five seeds also use layer-1 attention. No component carries the evidence to another position without
  importing that position's behaviour (0.88).
- Trained over a reliability grid, models do as well on held-out reliabilities (0.0125 vs 0.0110); trained at
  0.8 only, they ignore the stated reliability (regret 0.061 at 0.6).
- 14 of 15 pre-registered predictions held.

## Round 18: a location belief and goal-dependent decisions

**Round 18 — does an inferred location belief support goal-dependent decisions? (`rounds/r18_maze_belief/REPORT.md`,
predictions in `rounds/r18_maze_belief/THEORY.md`).** A fixed 14-cell maze with aliased noisy symbols; the agent
never sees its cell. A passive prefix, then one of three goals is revealed, so the exact posterior at the reveal is
the same for every goal. PPO on reward only, 5 seeds.

- Three seeds reach greedy regret 0.004–0.005 (acting on the most likely cell loses 0.10); two had not learned
  the goal that needs the belief most. Goals are learned in the order of how much they need the belief.
- The posterior over cells decodes at the reveal at R² 0.64 before training and 0.77 after; at prefix tokens,
  which never act, at 0.93. Use of the evidence goes from 0.00 to 0.64.
- A goal swap changes the actions as much as an evidence swap and moves the decoded posterior a quarter as much.
- Every seed has one goal head in the first attention layer. The other heads carry evidence across goals in
  part (0.22 of the way, 4 of 5 seeds); states of prefix tokens after the first block, goal-free by construction,
  carry 0.37. Whole sublayers carry the goal with the evidence (0.91 to the donor's behaviour).
- 9 of 12 pre-registered predictions held; the cross-goal transfer seen in the pilot (0.5–0.7) did not replicate.

## Round 19: occupancy

**Round 19 — is goal-conditioned occupancy represented beyond the posterior and the action values?
(`rounds/r19_occupancy/REPORT.md`, criteria in `rounds/r19_occupancy/PLAN.md`).** Round 18's models, no training.
Occupancy is the expected discounted future visitation from the posterior at the reveal, exact under the solver's
policy and estimated by rollouts under each model's own.

- A linear function of the posterior and the action values, per goal, gives 79 % of the solver's occupancy.
- At the goal token the action values decode best (0.91–0.94), then the posterior (0.79–0.82), then occupancy
  (0.74–0.82). What is specific to occupancy decodes at 0.07–0.26 in reward-trained models (supervised: 0.31).
- Futures that share a first action and differ in visitation are told apart (slope 0.70–0.84), and an untrained
  network does it too (0.69).
- The two seeds that had not learned the hardest goal differ from the solver's occupancy by 4.5 (L1) on that goal
  and by 0.4–0.6 on the others.
- Two of three criteria met; the causal test, conditional on all three, was not run.

## Round 20: editing the decoded belief

**Round 20 — does the policy use the decoded belief? (`rounds/r20_belief_edit/REPORT.md`, expectations in
`rounds/r20_belief_edit/PLAN.md`).** Round 18's models, no training. The states of the prefix tokens, which cannot
depend on the goal, are edited towards a donor with other evidence.

- Replacing the donor's component in the posterior decoder's row space makes the decoder read the donor's
  posterior (R² 0.9) and changes nothing else: decisions move 0.00, in all six models, with or without the
  decision token's direct access to the raw tokens. That row space holds 2–3 % of the difference between states.
- The whole prefix state moves decisions 0.12–0.45 of the way, and all the way once the direct route is removed.
  The effect persists over later decisions.
- Post hoc: the 14 directions along which the state co-varies with the posterior hold 90 % of the difference and
  carry nearly the whole effect; one such edit gives different, appropriate actions under different goals in
  47–91 % of cases. The edit is not selective for the posterior.
- Registered: whole-state patches work, belief-directed edits do not.

## Round 21: specificity of the ΣW edit

**Round 21 — is the posterior-associated edit more specific than replacing the dominant history representation?
(`rounds/r21_pattern_specificity/REPORT.md`, expectations in `rounds/r21_pattern_specificity/PLAN.md`).** Round 18's
models, no training. Round 20's post hoc edit, frozen, on new histories, against controls.

- It replicates: the pattern edit (rank 13) moves decisions 0.79–0.93 of the way with the direct route removed;
  the decoder's row space 0.00–0.04.
- It is not specific. The top 13 principal components move 0.89–0.96, and a random subspace capturing the same
  share of the donor difference (rank 115) 0.85–0.93. At equal captured share the pattern edit is within
  −0.07 to 0.05 of PCA. Thirteen random directions move 0.03–0.09.
- Between histories with the same posterior (L1 < 0.05) the pattern edit carries 0.88–0.96 of the whole patch's
  effect.
- Across goals the pattern edit is appropriate by the solver as often as the model on the donor's evidence is, and
  so are the controls.
- Uncertainty pairs respond more than mode pairs at equal solver stake (the difference is unchanged by matching).
- 4 of 9 expectations held, 3 in part; the one that would have shown specificity did not.

## Round 22: three types of pair

**Round 22 — does history matter beyond the belief, and is evidence kept beyond one goal's action?
(`rounds/r22_pair_types/REPORT.md`, expectations in `rounds/r22_pair_types/PLAN.md`).** Round 18's models, no
training. Pairs with the same posterior and different histories; with the same optimal action under one goal and a
different one under another; with different optimal actions under every goal. Whole and rank-13 PCA patches, each
pair under all three goals, natural access and direct route removed kept apart.

- Same posterior, different history: the action distribution differs by 0.04–0.09, a tenth or less of the effect
  of a different posterior; it grows with prefix length and is a difference in which cases the model gets wrong.
- The same patched goal-free states give 0.93–0.98 of the model's own change under the goal where the optimal
  action differs (direct route removed; 0.20–0.36 under natural access).
- Where the solver predicts no change, the supervised model changes 0.07 and the reward-trained models 0.23–0.34:
  their errors depend on the evidence, and the patch reproduces the model on the donor's evidence, errors included.
- With the direct route removed the prefix states carry the model's whole response, and the ablated model is on
  the optimal set with probability 0.56–0.80.
- The PCA patch equals the whole patch within 0.02 under natural access.
- 3 of 7 expectations held, 4 in part.

## Round 23: reward plus observation prediction

**Round 23 — does an observation-prediction objective reduce history dependence, incorrect action changes and
regret? (`rounds/r23_obs_prediction/REPORT.md`, decision rule in `rounds/r23_obs_prediction/PLAN.md`).** Round 18's
task and network; reward only against reward plus next-symbol prediction (coefficient 1 and 0.1), ten seeds each;
round 22's pairs.

- Regret falls from 0.0063 to 0.0042 at coefficient 1 (p < 0.001); no seed fails G2 (reward only: 2 of 10);
  learning is about twice as fast.
- Greedy action changes where the optimal action does not change fall from 28 % to 19 %. At equal regret the
  difference is 0.016 [−0.039, 0.007]: most of it is a better policy.
- Identical-posterior histories are treated as differently as before (0.072 against 0.066), and with the direct
  route removed more differently (0.14 against 0.08).
- Patch transfer from the prefix states does not change (0.15 against 0.13 of the way).
- At coefficient 0.1 the head learns as well and the policy gains little.
- 4 of 7 expectations held, 1 in part.

## Round 24: the prediction head on identical-posterior histories

**Round 24 — does the prediction head agree on identical-posterior histories?
(`rounds/r24_head_consistency/REPORT.md`, criterion in `rounds/r24_head_consistency/PLAN.md`).** Round 23's models,
no training.

- The head's predictions for two histories with the same posterior differ by 0.026 (total variation) at the goal
  token; relative to its response to a different posterior that is 0.82 of the policy's inconsistency.
- It is 0.54 of the policy's on the move the policy takes and 0.33 at prefix tokens, and never absent.
- The difference is below the head's error against the exact predictive distribution (0.043) in every seed.
- It is larger in the cells where the policy's greedy action differs between the two histories (0.044 against
  0.026).
- By the registered criterion the auxiliary task is itself solved with history-dependent approximations.

## Round 25: balanced candidate-action supervision

**Round 25 — does balanced candidate-action supervision make predictions, and then the policy, more
belief-consistent? (`rounds/r25_balanced_prediction/REPORT.md`, rule in `rounds/r25_balanced_prediction/PLAN.md`).**
Round 23's setting; the head is given a simulator-sampled symbol for every candidate move (or one random candidate)
and not only for the move taken. Ten seeds per arm.

- The head's error against the exact predictive distribution falls from 0.043 to 0.017 and its inconsistency
  between identical-posterior histories from 0.026 to 0.013, all of it on moves the policy does not take; on the
  taken move nothing changes (0.014).
- One random candidate per token does as well as four.
- Policy inconsistency falls from 0.072 to 0.063 (p 0.018), to the reward-only level. At equal regret the
  difference is 0.001 [−0.008, 0.011] (post hoc).
- Across seeds the two inconsistencies are correlated (ρ 0.85 within the balanced arm).
- 4 of 5 expectations held; the registered verdict is that both improve, and the policy's share is small.

## Cross-round findings

1. Hidden geometry mirrors the distinctions the training target contains (policy quotient for
   argmax targets, advantage for soft targets), across MLP, residual and transformer
   architectures for hard targets.
2. The scale of a distinction is set by the output gap it must produce, divided between readout
   gain, separation and alignment by optimisation dynamics, not by frequency or gradient
   pressure.
3. Raw geometry and local (Jacobian, Fisher) metrics are implementation-dependent; decodability,
   readout contrast and finite-propagation effects on a complete causal cut are invariant
   across functionally equivalent models, in both GRU and transformer.

## Round 26: prediction-trained, reward-trained and random backbones under a frozen head

**Round 26 — does learning to predict the maze produce a representation from which different goals can be solved
efficiently? (`rounds/r26_predictive_transfer/REPORT.md`, rule in `rounds/r26_predictive_transfer/PLAN.md`).**
Backbones trained only to predict the next symbol after each supplied move on goal-free random walks (and, secondary,
the next two after each move pair), round 23's reward-only models, and the same initialisations untrained; ten
seeds each. A 16-unit head reads the last prefix token plus the goal and is trained on optimal actions, the same
examples for every backbone, N = 30 to 100 000.

- One-step prediction leaves the optimal action undetermined in 43–85 % of decisions (5 of 13 belief dimensions);
  checked before training.
- The predictor learns its objective (error 0.008) and more: the posterior decodes at 0.93 (its objective needs
  0.45), and identical posteriors give states 1.5 % as far apart as different ones (reward 12 %, random 29 %).
- Area under the learning curve: prediction 0.0047, reward 0.0061, random 0.0079 (each pair p < 0.001); half the gap
  from goal-only closed with 65, 121 and 162 examples (exact posterior 57, raw tokens 451).
- The predictor plateaus at 0.0020 (exact posterior 0.0002), mostly on decisions one-step prediction does not
  determine; two-step prediction lowers it to 0.0014. Post hoc, reward is ahead from 10 000 examples (0.0012).
- 6 of 8 expectations held; the two that failed were no difference from reward and random being worse than raw
  tokens.

## Round 27: belief-encoding edits read by the frozen head

**Round 27 — can the transferable representation support a selective causal belief edit?
(`rounds/r27_belief_encoding_edit/REPORT.md`, rule in `rounds/r27_belief_encoding_edit/PLAN.md`).** Round 26's
backbones and heads, frozen (heads rebuilt, regret reproduced exactly). An encoding h ≈ c + E b fitted on separate
histories; held-out recipients edited to h_A + E (b_B − b_A) and read by the head under each goal.

- Prediction-trained state: the donor-belief action in 0.77 of cells where it must change (whole replacement 0.90);
  the same vectors rotated into generic directions 0.15, a random rank-13 patch 0.10 (p < 0.001).
- Works where one-step predictions are identical and actions differ (0.72; a one-step edit is exactly zero there),
  on two histories sharing a belief (0.71, agreement unchanged), and with a third belief (its action in 0.73).
- Not selective by the rule: it changes 9 % of decisions that should stay (margin 0.05); whole replacement changes
  none. Round 20's decoder direction fails again (0.15).
- Every backbone shows the same pattern, the random one included (0.68 / 0.79): the edit shows causal use by the
  head, not that the information was learned. Learning enlarges it (moved 0.70 against 0.48).
- 8 of 11 expectations held.

## Round 28: the same edit, read by the original policy and the new head

**Round 28 — does the same belief-associated change control the newly trained head and the original policy?
(`rounds/r28_policy_belief_edit/REPORT.md`, rule in `rounds/r28_policy_belief_edit/PLAN.md`).** Round 23's reward
models, frozen. A belief-encoding edit of every prefix state entering block 1 (the pre-goal interface), read in one
forward pass by the policy at the goal token and by round 26's head at the last prefix token; all three goals.

- Natural access: the edit moves the head 0.81 of the way to the donor's behaviour and the policy 0.19 (ratio 0.22;
  registered dissociation, at the median).
- Replacing the whole interface moves the policy only 0.21: it recomputes the evidence from raw tokens at the goal
  token. Of what it takes from the interface, the encoding edit carries 0.89 (head 0.81); generic directions carry
  0.02–0.11. The dissociation is in access, not form.
- Direct route removed: the edit moves the policy 0.89 of the way, but the ablated policy is much less competent
  (0.48 donor-optimal on the donor's own evidence, against 0.81).
- On one-step-agreeing pairs the policy leans more on the interface (whole 0.48) and the edit transfers 0.40.
- Preservation: the policy's correct decisions are untouched (−0.005); the head loses 0.085.
- 7 of 7 expectations held.

## Round 29: does the goal choose the evidence route?

**Round 29 — does the policy become more sensitive to the pre-goal interface when the goal requires distinctions the
direct route handles poorly? (`rounds/r29_goal_route_selection/REPORT.md`, rule in
`rounds/r29_goal_route_selection/PLAN.md`).** Round 23's reward models, frozen; fixed history pairs under all three
goals. The decision is exactly split between the interface (prefix states entering block 1) and the direct route
(the goal token's own state entering block 1); each is replaced by the donor's.

- The routes add up (interaction within ±0.01); the direct route carries 0.69–0.72 of the behavioural difference.
- At a fixed pair, with behavioural difference and oracle gap controlled, interface reliance is the same across
  goals in direction (Q1 p ≥ 0.11), and only 0.014 higher where the direct route is inadequate (registered −0.05;
  p 0.053).
- Post hoc: 82 % of interface reliance varies between pairs; pairs the direct route cannot resolve rely on the
  interface 0.46 against 0.25. The evidence chooses the route; the goal acts on a shared evidence estimate.
- 4 of 7 expectations held; the two central ones failed.

## Round 30: goal swap with the history fixed

**Round 30 — which components after the two-route interface carry the switch from one goal's action to another's?
(`rounds/r30_goal_swap_components/REPORT.md`, rule in `rounds/r30_goal_swap_components/PLAN.md`).** Round 23's
reward models, frozen. One history under two goals with disjoint optimal actions; single head and MLP outputs at the
goal token patched from the donor goal's run; components found on fit-side histories, tested on held-out ones.

- Selected (patch ≥ 0.3) in every model: block-1 and block-2 MLPs (8 and 7 of 10), never a head. On held-out
  histories they transfer 0.71 (discovery 0.73), under every goal pair.
- Not localised: the other MLPs also carry 0.62 (registered D3 failed). All three MLPs of blocks 1–3: 0.92; all
  twelve heads: none (post hoc ratio-of-means: 0.98 against 0.12).
- Direct logit attribution grows along the MLP stack (0.15, 0.29, 0.39); the goal embedding writes nothing directly.
- With round 29: attention reads a goal-independent evidence estimate, and the MLPs make the goal-specific
  decision from it. 7 of 7 expectations held (necessity was not separately measurable: cell symmetry).

## Round 31: what the goal token's MLPs carry

**Round 31 — a goal instruction, an evidence–goal combination, or an action preference?
(`rounds/r31_cross_history_mlp/REPORT.md`, rule in `rounds/r31_cross_history_mlp/PLAN.md`).** Round 23's reward
models, frozen. MLP outputs at the goal token from history A under goal g′ (correct action a_A′) patched into history
B under g (correct a_B; would choose a_B′ under g′); three different actions.

- All three MLPs of blocks 1–3: the donor's action in 0.97 of cells, B's own action under the donor goal in 0.01,
  retained 0.01. Same-goal control: the donor's action 0.97. An action preference, not a goal instruction.
- A single MLP from another history is outvoted by the other two (retained 0.75–0.93). Block 0's MLP alone mixes:
  donor 0.40, instruction-like 0.20, retained 0.35.
- Representation: block 0's MLP output is goal × posterior (R² 0.78 against 0.59 for goal × action); blocks 2–3 are
  goal × action (0.92–0.96 of what both explain). The goal is folded into a decision early, and carried.
- 5 of 6 expectations held, one in part.

## Round 32: a belief edit at the goal token's state after block 0

**Round 32 — does a shared belief edit at the direct route transfer donor behaviour across goals?
(`rounds/r32_direct_belief_edit/REPORT.md`, rule in `rounds/r32_direct_belief_edit/PLAN.md`).** Round 23's reward
models, frozen. The goal token's state entering block 1, fitted as c_{g,L} + E b without action labels. Each edit
replaces only that state with z(A,g) + E (b_B − b_A); the prefix states are kept.

- Fit: shared R² 0.81 (0.56 within goal and length); goal-specific maps 0.86 (0.67).
- Main change cells: donor-optimal 0.11 → 0.42 (shared), 0.56 (goal-specific), 0.60 whole route, 0.81 hybrid.
  Shared edit: 0.66 of whole; goal-specific: 0.94. Rotated (same norms) 0.00; PCA-13 of the donor's state 0.98.
- Preservation harm ≤ 0.004; one-step-agreeing pairs 0.85 of whole; across four equivalent recipients the effect
  follows the belief change (between-change variance 0.80). Agreement among them falls by 0.09.
- Goal-dependent pairs (different actions under two goals): shared 0.36 of whole, goal-specific 0.85.
- Registered reading: belief is encoded at the site, but goal-specifically; the goal is mixed in after block 0.
  8 of 10 expectations held; the shared-map ones (C6, C7) failed.

## Round 33: the belief edit before block 0's MLP

**Round 33 — is the belief map shared across goals one step earlier?
(`rounds/r33_attention_belief_edit/REPORT.md`, rule in `rounds/r33_attention_belief_edit/PLAN.md`).** Round 32's
design at the goal token's state after block 0's attention, before its MLP (embedding + attention output).

- Shared map fits as well as goal-specific ones (R² within 0.65 against 0.67; round 32's site 0.56 against 0.67).
  Goal × history interaction 0.045 (round 32's site 0.17), smaller in every model.
- Main change cells: the shared edit reaches donor-optimal 0.54, 0.90 of whole (round 32: 0.66). Within 0.02 of the
  goal-specific edit and 0.05 of PCA-13; rotated 0.02. Harm 0.006; one-step agreeing 1.08 of whole; equivalent
  recipients 0.93 of whole.
- Goal-dependent pairs: 0.61 of whole (round 32: 0.36; registered 0.75 not met). The goal-specific linear edit reaches
  0.79 (post hoc; first misreported as short of the line, corrected after round 34).
- Through block 0's MLP the shared edit becomes the goal-specific one (cosine 0.92). The goal is combined with a
  goal-free belief code by block 0's MLP. 8 of 10 expectations held; the misses were both "more shared than expected".

## Round 34: a nonlinear encoding of the posterior

**Round 34 — does a nonlinear encoding close the gap on goal-dependent pairs?
(`rounds/r34_nonlinear_belief_edit/REPORT.md`, rule in `rounds/r34_nonlinear_belief_edit/PLAN.md`).** Round 33's site.
A shared MLP f(b) and a per-posterior table μ(b) (the most any function of b can give), fitted without action labels.

- Fit improves a lot (R² within 0.65 linear → 0.77 MLP = table). Edits improve little: on goal-dependent pairs 0.61
  (linear) → 0.68 (MLP), 0.65 (table) of whole. P and the ceiling T fail.
- The donor's difference without its posterior part does nothing alone (0.15 of whole; no edit 0.13).
- Goal-conditioned encodings of the posterior clear the line: f(b, g) 0.86, μ_g(b) 0.81. Post hoc: another goal's map
  gives 0.56, below the shared map, in all ten models.
- Main pairs: all posterior encodings 0.91–0.98 of whole; harm ≤ 0.008; equivalent recipients 0.90–0.97.
- What was missing was not nonlinearity in the posterior, nor information beyond it, but a small goal × belief
  interaction at block 0's attention output that the policy reads on exactly the decisions where the goal must turn
  one belief change into different actions. 8 of 10 expectations held.

## Round 35: block 0, step by step

**Round 35 — is the goal × belief part already in block 0's attention output, and does anything change it before the
MLP? (`rounds/r35_block0_steps/REPORT.md`, rule in `rounds/r35_block0_steps/PLAN.md`).** Exact steps at the goal
token: attention output (self and prefix terms, heads) → + embedding → layer norm → MLP input.

- Attention output → residual MLP input: only the embedding is added (1·10⁻⁷); the same edit vectors give the same
  outputs (6·10⁻⁶). The registered edit comparison (2·10⁻³) failed through the ridge fits' offsets, not the network.
- The attention output already holds the goal × belief part: interaction 0.045, goal × posterior gain 0.025;
  right-goal table edits 0.78 of whole against another goal's 0.50 on goal-dependent pairs (all ten models).
- Mostly from the prefix term (the history's goal-free values read with a goal-dependent query): 3 × the self term's
  interaction size, in every model; spread over heads, differently per model.
- The layer norm raises interaction to 0.070, but with σ fixed across goals it is still 0.062. Behaviour is not
  significantly changed (shared 0.64 → 0.64; right − wrong +0.07, p 0.065). 0.78 of the route's goal-dependent effect
  passes through block 0's MLP. 3 of 6 expectations held, one in part.

## Round 36: block 0's query, swapped

**Round 36 — does the goal × belief part move with block 0's query? (`rounds/r36_query_swap/REPORT.md`, rule in
`rounds/r36_query_swap/PLAN.md`).** Block 0's attention at the goal token recomputed with the goal token's query, or its
own key and value, from another goal (both: exactly the other goal's output).

- The query carries 0.64 of the goal × history part of the attention output, 0.61 of its posterior part (untrained
  reference 0.79). The two shares sum to 1 by construction over ordered goal pairs (found after the run).
- Swapping the query costs only 0.037 of correct decisions where the goal matters (registered 0.05 not met).
- Not registered: with the goal token's own key and value from g′, the model takes g′'s action in 0.90 of goal-matters
  cells, with the embedding still g; block 0's attention output from g′ gives 0.99. The goal's identity reaches the
  decision through the goal token's self-attention in block 0; the belief's goal-dependence through the query.
- 2 of 6 expectations held with content (one empty by construction).

## Round 37: the goal's identity is the goal token's own value

**Round 37 — value or key, which head? (`rounds/r37_self_value/REPORT.md`, rule in `rounds/r37_self_value/PLAN.md`).**
Block 0's attention at the goal token recomputed with the goal token's own value or own key from another goal g′, all
heads or one; complement: the goal embedding in the residual stream from g′.

- Own value from g′: the policy takes g′'s action in 0.83 of goal-matters cells. Own key: 0.20, as natural (0.19).
  Residual embedding: 0.19. The goal is read only through the self value.
- Usually one head (best single head 0.59 of the effect; ≥ 0.5 in 6 of 10 models), always the most self-attending one
  (10 of 10); which head differs by model. Single-head effects sum to 0.65: they combine superadditively.
- 5 of 6 expectations held.

## Round 38: the self value and the goal × belief interaction

**Round 38 — does swapping the goal's self value move I(b,g) = f(b,g) − f(b) at the MLP input?
(`rounds/r38_self_value_interaction/REPORT.md`, rule in `rounds/r38_self_value_interaction/PLAN.md`).** I split into the
goal's main effect M and the goal × belief part J; swap-induced changes regressed on their changes, relative to the
natural goal change.

- At the MLP input: the self-value swap moves M 0.77 of the way, J only 0.37 (J fails). J is small there (6 % of M) and
  follows the query about as much (0.42).
- After the MLP: J follows the self value (0.67) and hardly the query (0.19). Block 0's MLP makes the interaction from
  the goal identity and the shared belief.
- Shared belief: a linear decoder shifts (registered F fails), but the shared code's subspace moves less than under a
  natural goal change, and belief-only decisions are intact (round 37). Most likely an off-manifold decoder effect.
  3 of 6 expectations held.

## Round 39: how block 0's MLP computes J(b, g)

**Round 39 — a low-rank bilinear interaction? (`rounds/r39_mlp_bilinear/REPORT.md`, rule in
`rounds/r39_mlp_bilinear/PLAN.md`).** J_y, the goal × belief part of block 0's MLP output (posterior-level, noise ceiling
1.00), described from the MLP's belief input B and the goal G; untrained reference.

- J_y ≈ W_g B(b): R² 0.95. Low rank: Σ_{k≤4} (u_k·B)(v_k·G) w_k gives 0.82; rank 3 suffices (0.8 × full) in 7 of 10
  models. The bilinear form is largely architectural (untrained 0.92); training makes it 7 × larger and low-rank.
- The MLP applied to its rebuilt input (ū + B + G) makes 0.68 of J_y; that interaction is 0.88 second-order: one
  GELU-curvature × (belief projection) × (goal projection) product per hidden unit, spread over many units (61 for
  half the energy) along a few directions.
- Removing J_y from the MLP output costs only 0.007 of correct decisions where the goal matters: later blocks
  compensate. 4 of 8 expectations held.

## Round 40: the goal × belief part across depth

**Round 40 — bilinear at every MLP, and necessary but redundant? (`rounds/r40_mlp_depth/REPORT.md`, rule in
`rounds/r40_mlp_depth/PLAN.md`).** Round 39's analysis for each of the goal token's four MLPs; J removed from one, three
or all four MLP outputs.

- Every MLP's J is low-rank bilinear in its input's belief code and the goal (rank 4: 0.82–0.86; full 0.88–0.95), lower
  rank with depth. Block 0 makes its J (0.68); blocks 2–3 mostly pass on what arrives (0.30, 0.18).
- Removing J from any one MLP costs ≤ 0.007; from all four 0.136 where the goal matters (6 × the sum of the singles);
  a wrong posterior's J costs 0.018. Necessary and redundant across depth.
- Goal-neutral cells also drop (0.099), so the registered specificity margin failed: J carries each goal's own belief →
  action map, against the three-goal average (the contrast was defined over two goals). 3 of 7 expectations held.
