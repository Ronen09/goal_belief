# Implementation freedom: two claims, and how to kill them

Companion to `rounds/r08_invariants/THEORY.md` (which defines ∼_coord, ∼_func and descent)
and to `rounds/r11_implementation_freedom/REPORT.md`. This note states two claims formally, derives
what each one forbids, and pre-registers the two experiments meant to falsify
them. Everything is stated for, and tested on, **one architecture**: the
2-layer pre-LN causal transformer of `goalgeo/tfm.py` on the TASK4 HMM. The
GRU and MLP results of rounds 4–9 are background, not evidence, here.

Predictions in §3.4 and §4.5 were written and committed **before any model in
`rounds/r11_implementation_freedom/` was trained**.

---

## 1. Setup

A model M is a finite DAG of nodes. Input nodes carry tokens x_1..x_T; every
other node v carries a value a_v(x) computed from its parents. For the
transformer of `goalgeo/tfm.py` the nodes are the residual-stream sites
res_ℓ(u), ℓ = 0..L, u = 1..T (res_0 = embedding + position, res_ℓ = output of
block ℓ), together with the internal attention/MLP nodes of each block, and the
output nodes out(u) = softmax(W·LN_f(res_L(u)) + b). The edges follow the
causal mask: res_ℓ(u) has parents res_{ℓ−1}(u′) for all u′ ≤ u.

**Interchange intervention.** For a set S of non-input nodes and a second input
x′, write M[x ; S←x′] for the run on x in which every v ∈ S has its value
overwritten by a_v(x′) and every other node is recomputed from its parents.
The *interchange effect* at a read node o is

    Φ_M(S ; x, x′, o) = D( out_o(M[x ; S←x′]) , out_o(M[x]) ),

D a divergence on output distributions (Jensen–Shannon below). S = {v} is a
*single-node effect*; this is what activation patching reports.

**The counterfactual pair.** Fix a position t and let Δ be the set of input
positions at which x and x′ differ. Throughout, x′ is the *minimal* flip of x:
Δ = {t}, and in the HMM the flip swaps the cue token p ↔ q at position t. Both
members of the pair have positive probability under the chain, so both are
on-distribution.

---

## 2. Claim A — cut identifiability

> Functional equivalence need not identify effects associated with individual
> internal nodes, because equivalent implementations may redistribute
> computation across parallel paths. Effects defined on a complete causal cut
> do descend to functional equivalence.

**Definition A.1 (complete cut).** S is a *complete cut for (Δ, o)* if every
directed path from an input node in Δ to o contains a node of S.

**Lemma A.2 (cut substitution).** Let x, x′ agree outside Δ and let S be a
complete cut for (Δ, o). Then for every node w that is o or a descendant of S,
and more generally for every w all of whose Δ-paths meet S,

    a_w(M[x ; S←x′]) = a_w(x′).

*Proof.* Induction in topological order. If w ∈ S the value is a_w(x′) by
construction. Otherwise take a parent u of w. Either u is a descendant of Δ, in
which case every Δ-path to o through w extends a Δ-path to u, so u's Δ-paths
also meet S (else a path would bypass S), and by the induction hypothesis
a_u = a_u(x′); or u is not a descendant of Δ, and then a_u(x) = a_u(x′) because
x and x′ agree outside Δ and u's ancestors among the inputs lie outside Δ. In
both cases every parent carries its x′ value, hence so does w. ∎

**Theorem A.3 (complete-cut effects descend to ∼_func).** With the hypotheses of
Lemma A.2,

    Φ_M(S ; x, x′, o) = D( p_M(· | x′)_o , p_M(· | x)_o ),

the *behavioural* divergence between the model's outputs on the two inputs. In
particular (i) the value is the same for every complete cut, at any layer and
of any size; (ii) it is a functional of the input–output map, so M₁ ∼_func M₂
implies Φ_{M₁} = Φ_{M₂}; and (iii) it can be computed without any intervention.

*Proof.* Apply Lemma A.2 at w = o. ∎

Theorem A.3 is the precise version of the second sentence of the claim, and it
is also a deflation of it: what a complete cut measures is behaviour. An
interchange effect carries information beyond behaviour exactly to the extent
that the patched set is *not* a complete cut — which is the first sentence.

**Proposition A.4 (single-node effects are not identified).** There is a family
of models, all computing the same function, whose single-node effects take every
value in an interval. *Construction.* Let two nodes v₁, v₂ receive the same
upstream signal s and let the read node compute f(α·s_{v₁} + (1−α)·s_{v₂}).
Every α ∈ [0,1] gives the same input–output map; patching v₁ with the
counterfactual value s′ gives f(α s′ + (1−α)s), which varies with α and is
constant only if f is. Hence Φ({v₁}) is not a function of the input–output map.
∎ The claim's "may redistribute" is exactly the existence of the α direction,
and it is a property of the architecture: a strictly serial model (a GRU state
sequence) has no such α, which is why single-node = complete cut there.

**Corollary A.5 (the transformer's cuts).** For the read node o = out(t+1) and
Δ = {t} in a 2-block causal transformer:

| set | complete? | why |
|---|---|---|
| {res_0(t)} | yes | the token enters only through its own embedding |
| {res_1(t)} | **no** | block-1 attention at t+1 copies the cue into res_1(t+1) |
| {res_1(t+1)} | **no** | block-2 attention at t+1 reads res_1(t) |
| {res_1(t), res_1(t+1)} | yes | block 2 at t+1 sees no other position ≤ t+1 that depends on t |
| {res_1(t), res_1(t+1), res_1(t+2)} | yes | superset of a complete cut |
| {res_1(t), res_1(t+2)} | **no** | leaves the res_1(t+1) route open |

The two incomplete singletons are the two parallel routes of Proposition A.4,
with α the share of the cue that block 1 has already moved to t+1.

### 2.1 What the claim forbids

1. No complete cut may disagree with behaviour: Φ(complete) = Φ_behav exactly,
   in every model, under every training manipulation. This is a theorem, so it
   fails only if the cut enumeration in Corollary A.5 is wrong.
2. Complete-cut effects may not vary across functionally equivalent models
   beyond what their behavioural differences allow.
3. Single-node effects *may* vary — and the claim is only interesting if they
   actually do. If routing cannot be moved without changing the function, the
   first sentence of Claim A is vacuous for this architecture.

---

## 3. Experiment 1 — force the routing

### 3.1 Design

Same task (HMM4, r=1, δ=0.4, k=1), same model, same optimiser, five training
conditions that differ only in how attention may route the cue, 4 seeds each:

| condition | manipulation | intended route |
|---|---|---|
| `free` | none | whatever the optimiser picks |
| `l1_diag` | block-1 attention hard-masked to the diagonal | through res_1(t): block 2 must fetch it |
| `l2_diag` | block-2 attention hard-masked to the diagonal | through res_1(t+1): block 1 must copy it |
| `pen_l2` | + μ·(mean block-2 attention to position q−1) in the loss | pushed toward res_1(t+1) |
| `pen_l1` | + μ·(mean block-1 off-diagonal attention) in the loss | pushed toward res_1(t) |

The hard masks are architectural route restrictions; the penalties are the soft
version, and check that the effect is not an artefact of masking. Both
architecturally-restricted models can still compute the task: the cue must
travel one position, and either block can move it.

### 3.2 Measurement

For 200 anchors (i, t) with a cue at t, build the minimal flip x′ and measure at
o = out(t+1): Φ_behav (two clean forward passes), and Φ(S) for the six sets of
Corollary A.5. Also record the route weights attn₁(t+1→t), attn₂(t+1→t) and the
KL to the Bayes floor. Functional equivalence is enforced by admitting only
models within 0.003 nats of the floor.

### 3.3 Statistics

Across the admitted models: CV(Φ(S)) = sd/mean for each set, and
max_M |Φ(complete) − Φ_behav|.

### 3.4 Pre-registered predictions

P1. max over all models and all three complete cuts of |Φ − Φ_behav| < 10⁻⁵
    (numerical, not statistical, agreement).
P2. CV(Φ) over the admitted models ≤ 0.02 for each complete cut — and the
    residual variation is the spread of Φ_behav itself, not of the patching.
P3. CV(Φ({res_1(t)})) ≥ 0.5 over the admitted models, with condition means
    ordered l1_diag > free ≳ pen_l1 > pen_l2 > l2_diag, and
    Φ({res_1(t)}) < 0.02 in `l2_diag`.
P4. CV(Φ({res_1(t+1)})) ≥ 0.3, with the reverse ordering.
P5. Route weights track the single-node effects: across models, the correlation
    between attn₂(t+1→t) and Φ({res_1(t)}) is > 0.8.
P6. Φ({res_1(t)}) + Φ({res_1(t+1)}) is *not* a constant across conditions
    (CV ≥ 0.1): the routes do not add up to the cut, JS is not additive.

### 3.5 Falsifiers

* P1 fails → a path was missed; Corollary A.5, hence Theorem A.3's application,
  is wrong for this architecture.
* P2 fails while P1 holds → functional equivalence was not achieved (the models
  differ behaviourally); the claim survives but the experiment is void.
* P3 and P4 both fail (single-node effects stable at CV < 0.1 despite hard
  route restriction) → routing is *not* free in this architecture: single-node
  effects would be identified after all, and the first sentence of Claim A
  would be false as stated for transformers of this size.

---

## 4. Claim B — factorization freedom

> The achieved contrast C = g·D·cos θ is fixed by the function, while
> architecture and optimisation determine how C is factorised. LayerNorm
> changes the available degrees of freedom without changing the functional
> requirement.

### 4.1 The interface identity

At the decision position (the last filler, where the next token is emitted by C
or D) let h̃_A, h̃_B be the mean post-final-norm vectors of the two branches and
w_d = w_x − w_y the readout contrast. Define

    C = w_d · (h̃_A − h̃_B),   g = ‖w_d‖,   D = ‖h̃_A − h̃_B‖,   cos θ = C/(gD).

**Proposition B.1 (C is functional).** z_x − z_y = log p(x)/p(y) at any
position, so C = ⟨log-odds⟩_A − ⟨log-odds⟩_B: a functional of the input–output
map, exactly (the readout bias and the LN bias cancel in the difference, and the
map h̃ ↦ z is affine, so means commute with it). At the Bayes floor of HMM4,

    C = C*(δ) = 2 log((0.5+δ)/(0.5−δ)).

**Proposition B.2 (no final norm: the factorisation is pure gauge).** Without a
final norm the interface admits the compensated change h̃ → Ah̃, W → WA⁻¹ for any
A ∈ GL(d) (TASK7's ∼_coord). C is invariant; the orbit of (g, D, cos θ) is the
whole level set {g D cos θ = C, g, D > 0, 0 < cos θ ≤ 1}. *Construction:* for a
target (g′, D′, c′) pick an A acting as a scaling by D′/D along (h̃_A − h̃_B), a
scaling on the complementary direction chosen to give w_d the norm g′, and a
rotation in the plane they span to set the angle; the three parameters are
independent and the constraint g′D′c′ = C fixes exactly one of them.

**Proposition B.3 (a frozen final LayerNorm imposes a hard ceiling).** With
γ ≡ 1, β ≡ 0, LN_f(u) is zero-mean with unit per-coordinate variance, so
‖h̃‖ = √d exactly for every input, hence D ≤ 2√d. With a fixed-gain readout
(‖w_v‖ = c for every row) g = ‖w_x − w_y‖ ≤ 2c. Therefore

    C ≤ 4c√d.

A model with frozen final LN and row-gain c *cannot* reach the Bayes floor when
C*(δ) > 4c√d, i.e. when

    c < c_crit(δ) = C*(δ) / (4√d).

For d = 64: c_crit(0.4) = 0.1373, c_crit(0.2) = 0.0530, c_crit(0.1) = 0.0253.
Note also that LN is invariant to the scale of its input, so the residual
stream's scale is not a degree of freedom at this interface at all: it is
quotiented out, not merely discouraged.

**Proposition B.4 (a learnable γ restores the scale).** h̃ = γ ⊙ u with
‖u‖ = √d, so √d·min|γ_i| ≤ ‖h̃‖ ≤ √d·max|γ_i|: the ceiling of B.3 is multiplied
by the LN gain, and the scale degree of freedom re-enters through γ rather than
through the residual stream.

### 4.2 What the claim forbids

1. C must equal C*(δ) in every converged model, whatever the norm and the gain.
2. Which term absorbs a change of budget is *not* free-floating: it is
   predictable from the architecture ahead of training (the content of "the
   architecture determines the degrees of freedom").
3. A hard ceiling that the architecture imposes must show up as a failure to
   reach the floor at a location computable in advance — c_crit(δ) — and must
   *not* appear in the architectures that do not impose it.

### 4.3 Design

3 norm variants × fixed-gain readout c × 3 seeds, δ = 0.4, everything else as in
TASK9 (`--steps 10000`, lr 1e-3, GPU):

* `none` — no final norm;
* `frozen` — final LayerNorm with γ ≡ 1, β ≡ 0, not trained;
* `learn` — ordinary learnable final LayerNorm;

with c ∈ {0.1, 0.125, 0.15, 0.2, 0.3, 0.6, 1.2, 2.4}, straddling
c_crit(0.4) = 0.1373. A second block moves the boundary: δ = 0.2 with
c ∈ {0.03, 0.04, 0.05, 0.07, 0.1}, for `frozen` and `none`, straddling
c_crit(0.2) = 0.0530.

### 4.4 Measurement

C, g, D, cos θ at the post-norm interface, ‖γ‖, the pre-norm separation, KL and
KL on relevant positions, and the excess over the Bayes floor.

### 4.5 Pre-registered predictions

P7. (identity) In every model with KL within 0.003 of the floor, C is within 3 %
    of C*(δ). CV(C) ≤ 0.03 across all converged models of a δ block, while
    CV(D) ≥ 0.5 across the same set.
P8. (`none` absorbs in D) Regressing log D on log c over c ≥ c_crit gives slope
    −1.00 ± 0.15; cos θ varies little (CV ≤ 0.2) and shows no trend.
P9. (`frozen` cannot absorb in D) D is capped: measured D ≤ 16 in every model,
    slope of log D on log c ≥ −0.5, and cos θ increases monotonically as c
    falls, reaching ≥ 0.9 at the smallest converged c.
P10. (the boundary) For `frozen`, models with c < 0.1373 fail to reach the floor
    (excess KL on relevant positions > 0.01) and their achieved C tracks the
    ceiling, C ≈ 4c√d within 25 %; models with c ≥ 0.2 converge. For `none` at
    the same c, and for `learn` at the same c, every model converges.
P11. (the boundary moves with δ) At δ = 0.2 the same pattern appears at
     c_crit = 0.0530: `frozen` fails at c ≤ 0.04 and converges at c ≥ 0.07.
P12. (`learn` absorbs in γ) For `learn`, the slope of log‖γ‖ on log c is
     −1.0 ± 0.3 for c ≤ 0.3, and D/√d stays within a factor 2 of its value at
     c = 1.2 once divided by ‖γ‖/√d.

### 4.6 Falsifiers

* P7 fails → C is not the functional invariant; the identity is not the right
  bookkeeping.
* P8/P9 fail (e.g. `none` absorbs in cos θ, or `frozen` finds D > 2√d) → either
  the measurement is at the wrong interface or the "which term absorbs"
  prediction is not derivable from the architecture, which is the substantive
  half of Claim B.
* P10/P11 fail — `frozen` reaching the floor below c_crit → Proposition B.3 is
  violated, which is impossible unless the interface model is incomplete (e.g.
  another path reaches the logits), so this doubles as a check that the
  factorisation is stated over the right cut.
* P10 fails in the other direction — `none` or `learn` also failing below
  c_crit — → the ceiling is not the norm's doing, and LayerNorm is not what
  changes the degrees of freedom.

---

## 5. Relation to rounds 7–9

TASK7 and TASK9 measured coefficients of variation across an equivalence set
that the optimiser happened to produce. The two experiments here *construct* the
equivalence set by intervening on the architecture: Experiment 1 moves the
routing on purpose and asks which measure notices, Experiment 2 removes a degree
of freedom on purpose and predicts in advance where the variation must go and
where the model must break. The claims are worth the name only if those
predictions can be, and are not, falsified.

---

## 6. Post-hoc: what the experiments changed

*Written after the round; §§1–5 are the pre-registration and are unmodified.*

### 6.1 Claim A stands; the interesting content moved

Nothing in §2 needed repair. Theorem A.3 held bitwise, three distinct complete cuts agreed, and
Proposition A.4's α — the share of the computation routed through a node — was moved from 0 to
1 by masks, by penalties and by seed alone. What the experiment added is that the *sum* of the
two route effects is not a substitute for the cut: with exclusive routing it equals the cut
value, with duplicated routing it exceeds it by 2.2×. Interchange effects do not decompose.

### 6.2 Claim B: the ceiling is a capacity, not the norm's algebra

Proposition B.3 bounds the contrast a frozen-LayerNorm interface can express by C ≤ 4c√d. The
bound held in every model and was never tight, because it assumes two things training does not
deliver:

    g = 2c        antipodal readout rows        attained  g/c = 1.294   → factor 0.65
    D = 2√d       antipodal branch states       attained  D   = 11.27   → factor 0.70

and the two losses multiply: the attained ceiling is κ·c with κ = 14.4 at δ=0.4 (flat to 16 %
across the capped gains), i.e. 0.45 of the formal 32c — the product of the two factors. Both
are needed: quoting only the readout-row factor would put the attained ceiling at 20c, which
the data exclude.

The attained law predicts the boundary as c = C*/κ: 0.306 at δ=0.4 against an observed 0.4, and
0.220 at δ=0.2 against an observed 0.3, where the formal c_crit is off by 3× and 6×. So the
non-antipodality accounts for most of the distance but **not** for the boundary: a factor ≈ 1.3
survives at both δ, and at δ=0.2 κ is not even well defined (it varies 11× over the
sub-boundary gains, because the small-gain models there are degenerate rather than capped).
What the data do pin down is that **D has a task-dependent plateau**: 11.3 at δ=0.4 and ≈ 6–7
at δ=0.2, with the same architecture, the same d and the same frozen norm. What limits the separation of one pair of states is not the
norm's sphere but the other predictions the same interface must make.

The plateau is not an optimisation artefact: at 3× the learning rate and 4× the steps the
capped models move by under 1 % in C and stay exactly at cos θ = 1.000.

**Revised proposition (to be tested, not assumed).** For a distinction (A, B) read at an
interface shared with a task T, define the *attainable separation* D\*(T, A, B) as the largest
‖h̃_A − h̃_B‖ compatible with Bayes-optimal behaviour on T at that interface. Then the
architecture's contribution to the ceiling is

    C ≤ g_max · D*(T, A, B),

with g_max set by the readout parameterisation and D\* ≤ 2√d for a frozen LayerNorm, ≤ √d·max|γ|
for a learned one, unbounded without a norm. B.3 is the special case D\* = 2√d. The predictive
content of Claim B therefore rests on D\*, which is measurable — it is the plateau — and which
the present round measured only as a by-product.

### 6.3 What "architecture determines the factorisation" is worth

The pre-registered form — each architecture has one term that absorbs a change of budget — is
false: all three conditions split it between separation and alignment
(d log D / d log c between −0.55 and −0.83, d log cos θ between −0.21 and −0.62), with the
three slopes summing to zero to three decimals in each case. What survives is ordinal and
mechanistic: a frozen norm removes the scale degree of freedom, so alignment saturates first
and the model runs into D\*; a learned γ puts the scale back, so separation carries more and
alignment saturates later; no norm at all leaves both free. That is a mechanism, not yet a
quantitative prediction, and the honest summary of this round is one claim promoted to a
theorem with a passed falsification test, and one claim reduced to an identity plus a mechanism
whose predictive form is still missing.
