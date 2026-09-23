# Claim 2: is the decoded posterior the causal state? (TASK11, round 12, part 2)

Brief: `rounds/r12_hidden_goal/BRIEF.md` (claim 2 and the three tests that followed it). Part 1, recoverability, is
`rounds/r12_hidden_goal/THEORY.md` and `rounds/r12_hidden_goal/REPORT.md`. This part uses the **already-trained round-12
models** (K = 4, main grid, converged models only for the predictions). No model is retrained.
The predictions in §6 were committed **before any intervention or measurement of this part was
run**. Environment-level facts (§3) come from the filter alone.

---

## 1. Cuts: where a belief can be the state

A site S is a *complete cut* for the future if every path from the history tokens o_≤t to the
outputs at positions u > t passes through S (round 11, Definition A.1). Patching a complete cut
from history H₂ into H₁ reproduces the run on H₂ exactly (Lemma A.2), so its downstream effect
*is* the behavioural difference between H₁⊕c and H₂⊕c.

* **GRU**: h_t is a complete cut. The future is F(h_t, o_{t+1..}), so the GRU state is the only
  thing that can carry a belief forward. If a posterior is the state, it is here.
* **Transformer**: no single-position site is a cut. Positions u > t attend to the raw tokens
  o_≤t in block 1 and to res1(s ≤ t) in block 2. Nothing downstream reads res2(t) or u(t)
  except the output at t itself. The transformer does not carry a belief forward: it recomputes
  it at every position. A patch at one position is a single-node effect (round 11,
  Proposition A.4), and its future effect is a route share, not a test of sufficiency.

So the sufficiency tests (A and B) are made at the GRU's complete cut, and the transformer
supplies the single-position contrast. Test C adds the depth dimension, which only the
transformer has.

## 2. The posterior is not always the sufficient state

With i.i.d. evidence, b_t is a sufficient statistic for every future quantity: future
posteriors, optimal actions and predictive distributions depend on the history only through
b_t. With the latent channel, the sufficient statistic is the joint filter P(G, c_t | o_≤t), and
two histories with the same b_t but different channel beliefs have different futures. Two
readings of "behaves as though it had observed evidence producing b′" then separate:

* **genuine-B**: the run on the spliced sequence B_≤t ⊕ c_A. It moves the whole joint state.
* **transplant**: the Bayes future from the joint state b′(g)·P(c | g, A_≤t). The goal belief
  is moved and the channel belief given the goal is kept.

In iid these coincide. In the channel env, an intervention that moves only the decoded goal
coordinates should follow the transplant, and a full state swap should follow genuine-B.

## 3. Tests

Histories at t ∈ {6, 12} from fresh sequences; continuations run to position 24. Every model
output is compared at each future offset k = u − t, where k = 0 is the immediate output at t. The
divergence is JS between output distributions. "Gap closed toward R" at offset k is
F_R(k) = 1 − E[JS(p_int, R)] / E[JS(p_A, R)], a ratio of means over pairs.

**Probe and encoder.** At a site h, the decoder z(h) = Ah + c is fit by least squares on held-out
fit states (all positions), as in part 1. The encoder h ≈ B y + c_B is the reverse regression. An
intervention that sets the decoded belief to y* is h ← h + Bγ with γ = (AB)⁻¹(y* − z(h)), a move
along the directions in which the network's own states vary with belief (**PROBE-E**), or
h ← h + A⁺(y* − z(h)), the minimum-norm move (**PROBE-D**). Both set z exactly to y*.

**A. Posterior transplant.** Pairs (A, B) with ‖y_A − y_B‖ ≥ 2 nats. Interventions on A's state
at t: SWAP (h_B itself), PROBE-E and PROBE-D to y_B, PROBE-E halfway (y_A + ½(y_B − y_A)), and
RAND (a random direction with the norm of the PROBE-E shift). References: genuine-B model
output; genuine-B Bayes target; transplant Bayes target (at the intervention's own dose). The
continuation is A's own.

**B. Equal-belief equivalence.** Pairs (H₁, H₂) at t = 12:
* iid: exactly equal posteriors (|Δy| < 1e-7) with count vectors differing by ≥ 8 of the 12
  tokens (different evidence, different order);
* channel, equal-b: |Δy|∞ < 0.05 with channel beliefs P(on) differing by ≥ 0.15;
* channel, equal-joint: |Δy|∞ < 0.05 and |ΔP(on | g)| < 0.02 for every g.

Each pair gets 8 continuations sampled from H₁'s posterior predictive. Effect: JS between
H₁⊕c and H₂⊕c (at the GRU's complete cut this is exactly the patch effect), normalised by the
same quantity for random pairs, and compared with the Bayes divergence of the same pair.

**C. Same action, different belief.** Pairs (x₁, x₂) at t = 12 with the same, untied optimal
action, ‖Δy‖ ≥ 1.5, and a continuation (sampled from x₁'s predictive) under which the Bayes-
optimal action differs between x₁⊕c and x₂⊕c at some later step. Measured:
1. *causal, GRU (complete cut)*: on the future steps where the Bayes actions differ, how often
   the model's argmax differs between the two runs (flip rate). A network that merged the
   pair at the cut would have flip rate 0.
2. *the depth ladder, transformer (and GRU h)*: per site, for states grouped by optimal action,
   the within-action affine R² of y (fit within each action class, held out by sequence). This
   is the part of the belief that the action does not already determine. Next to it: the
   overall R²_y, the action's linear decodability, and the decodability of a pure-history
   quantity with zero evidential value (the number of neutral tokens so far). This gives
   history → belief → action quotient → output as four numbers per site.

## 4. What the theory forces

* SWAP at the GRU cut equals genuine-B exactly (an identity; checked, not predicted).
* For a `goal` model at the interface, PROBE-E and PROBE-D at k = 0 set the output to b′ up to
  the probe's error: the immediate switch is guaranteed. Every claim is about k ≥ 1.
* A transformer patch at res2(t) or u(t) has exactly zero effect at k ≥ 1 (nothing reads them).
  A patch at res1(t) reaches k ≥ 1 only through block-2 attention to position t.

## 5. Grid

GRU and transformer, iid and channel, objectives `goal`, `act_soft`, `act_hard` (iid only for
the predictions; channel models did not converge), `next_obs`; 4 seeds; 1500 pairs per (test,
t, env). Transformer interventions are made at res1(t) and res2(t).

## 6. Pre-registered predictions

GRU unless stated, seed means, future means over k ≥ 1.

* **A1.** iid `goal`: PROBE-E closes ≥ 0.9 of the gap to genuine-B at k ≥ 1 and ≥ 0.95 at
  k = 0. PROBE-D closes ≥ 0.8 at k ≥ 1.
* **A2.** Channel `goal`: after PROBE-E, the output is closer to the transplant Bayes target
  than to the genuine-B Bayes target (mean KL at k ≥ 1). The gap closed toward the transplant
  target is ≥ 0.8.
* **A3.** Halfway PROBE-E (iid `goal`) closes ≥ 0.8 of the gap to its own transplant target:
  the future follows the log-odds line, not just its end point.
* **A4.** RAND closes ≤ 0.2 of the gap to genuine-B, in every GRU cell.
* **A5.** iid `act_soft`, `act_hard`, `next_obs`: PROBE-E closes ≥ 0.7 of the gap to genuine-B
  at k ≥ 1.
* **A6.** Transformer `goal`, both envs: a single-position PROBE-E or SWAP at res1(t) closes
  ≤ 0.3 of the gap to genuine-B at k ≥ 1; SWAP at res2(t) closes ≥ 0.95 at k = 0.
* **B1.** iid `goal`: the equal-belief effect is ≤ 0.05 of the random-pair effect at every k.
* **B2.** Channel `goal`, equal-b pairs: the model's downstream divergence is between 0.5× and
  2× the Bayes divergence of the same pairs (ratio of means over pairs and k ≥ 1), and it tracks
  the Bayes divergence across pairs (Spearman ≥ 0.5). The goal posterior is not sufficient
  here, and the network's state reflects that.
* **B3.** Channel `goal`, equal-joint pairs: effect ≤ 0.1 of the random-pair effect.
* **B4.** iid `act_hard`, `act_soft`, `next_obs`: equal-belief effect ≤ 0.1 of random.
* **C1.** iid `act_hard` GRU: flip rate ≥ 0.9. The recurrent state cannot merge beliefs
  that later evidence will separate, because it is the only carrier of the future.
* **C2.** iid transformer: within-action R²_y at u is at least 0.2 lower for `act_hard` than
  for `goal`, while at res1 it is ≥ 0.8 for both. The quotient forms in the last block.
* **C3.** iid transformer `goal`: within-action R²_y ≥ 0.95 at res2 and u. Nothing is
  quotiented when the target is the posterior.
* **C4.** iid transformer, every objective: neutral-token-count R² is lower at u than at res1.
  History detail with no evidential value is discarded with depth.
