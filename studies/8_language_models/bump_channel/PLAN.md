# Bump channel: does the belief reach the decision through the goal's four directions?

Written before any of the measurements below were run. **Disclosure:** the models (`../bump_goals/runs/llm_40000`,
`tfm_120000`, `gru_120000`), the goal-swap pool and the fine-tuned model's goal subspaces (`../bump_coords`:
`cache/states.pt`, `rank.json`, `cache/rank_subspaces.pt`) and the goal-swap study's history-swap curve
(`../bump_goals/runs/llm_40000/factorial.json`) exist; nothing of this experiment has been run. Brief: given in
conversation as a three-step proposal (patch the belief upstream of the goal and measure how much of the induced
decision change passes through the four-dimensional causal subspace; intervene on that subspace directly; compare
known and uncertain states), with the hypothesis in the author's words: "the belief set (history) combines with the
goal to output a four-dimensional subspace, so intervening on this subspace should change actions"; the thresholds
(0.65 to hold, 0.40 to drop) and the per-architecture expectations (GRU 0.8, transformers 0.65–0.7) were set there.

## The question

The bump coordinates study found that in the fine-tuned four-goal model the goal's effect on the decision is read
from four directions of the 896-dimensional residual at layer 23: the full goal map's change restricted to the top
four singular directions of the per-history goal difference switches 0.73 of the 0.79 the full map switches, one
pooled subspace serves every goal pair at 0.935 of the full effect, and k random directions give nothing until k is
half the space. The goal-swap study found that the history's belief arrives in the decision token at layers 13–19
(the same-goal history swap 0.58 at 15, 0.78 at 19) and the goal only at 20–23. The hypothesis under test is that
those four directions are not the goal's channel alone but the channel into which belief and goal are combined: the
decision reads the belief's consequence for the action from the same directions it reads the goal's. If so, a
change of belief installed upstream of the goal's arrival should change the decision only through those four
directions at the top, the natural difference between two histories at the top should switch the decision when
restricted to them, and both should fail when restricted to their complement. The same question is asked of the
from-scratch four-goal transformer and GRU, whose goal subspaces are found first.

## Models, sites and the decision pool

**The fine-tuned model** (`llm_40000`, exactly optimal under four goals): the decision token's residual entering
layers **15, 19** (upstream sites: the belief is in the token, the goal is not) and **21, 23** (downstream sites: the
goal has arrived; the goal subspace's causal rank is 6 and 4). Each site's **goal subspace** is the top-k singular
directions of the pooled, per-step-centred goal differences of all twelve ordered pairs over the fit histories, as
`rank.py` built it, with **k the median pooled causal rank at that site: 4 at 23, 8 at 21** (64 at 19 and 128 at 15,
where the subspace is reported but the goal is not yet read). The sweep over k ∈ {1, 2, 4, 8, 16, 32, 64, 128, 896}
is reported alongside at every site.

**The from-scratch models** (`tfm_120000`: 4 blocks, width 128, success 0.79; `gru_120000`: width 128, success
0.87; the 40 000-episode models at 0.40–0.47 success are too weak to carry a causal measure and are not used). The
goal is the start token. Transformer sites: the decision position's residual entering blocks **1, 2, 3** and the
**final** residual entering the head's layer norm (block 0's entrance is the step token's embedding, which holds no
goal). GRU sites: the hidden state at the decision **h_t** (entering the linear head), and the hidden states **h_{t−1}**
and **h_{t−2}** (before the last one and two tokens' updates). Their goal subspaces come from part 0.

**The decision pool** is the coordinates study's: decisions after the first step along the optimum under every goal,
deduplicated by history and split by history, 1 979 distinct fit histories and 660 test histories, each run under
all four goals; the fit histories fit every subspace and map, every number below is on the test histories. **History
pairs**: 512 same-step pairs of test histories with different goal-free sets, drawn as the goal-swap study draws its
pairs (seed 1), each under all four goals: 2 048 (pair, goal) cells. A cell is **history-dependent** when the two
histories' natural decisions under that goal differ; every switch below is counted on history-dependent cells, and
the share of cells that are history-dependent is reported per goal. Bootstraps cluster by the receiving history h1.

**Causal convention** (the swap studies'): the prompt (or token sequence) of the receiving history h1 is run under
the goal g; the state at a site is replaced by the candidate; **switch** is the share of history-dependent cells whose
patched decision is h2's natural decision under g; the natural state of h2 installed at the site is the ceiling (the
natural history swap), a random vector of the natural change's norm the floor.

**Projectors.** For a site with goal subspace basis U (k columns): P = U Uᵀ; the complement I − P; **bottom-k**, the
last k singular directions; **random-k**, k random orthonormal directions (three draws); the **readout subspace**
P_J, the top-k right singular directions of the stacked Jacobians of the three centred logits with respect to the
site's state at the decision position over the fit decisions (for the GRU's h_t and the transformer's final residual
this is the head's row space, two directions after centring: there any change acts on the decision through those
rows alone, which is why those two sites are reported as ceilings and excluded from the decision rule); the
**history subspace** P_H, the top-k singular directions of the per-step-centred differences x(h2, g) − x(h1, g) over
512 same-step fit pairs under each goal (the belief effect's own principal directions, fitted on fit histories only).
The **overlap** of two rank-k projectors is trace(P₁ P₂) / k (the mean squared cosine of their principal angles;
random pairs of rank-4 subspaces in 896 dimensions give 0.004, in 128 dimensions 0.03).

## Part 0. The goal subspace of the from-scratch models

The coordinates study's part B on each from-scratch model at each of its sites: for every ordered goal pair, the
ridge map from the state under A plus the one-hot step to the state under B (λ tuned on a held-out quarter of the
fit histories), the per-history goal difference with its per-step mean removed and its singular spectrum (k50, k90,
k99; the pooled spectrum), and the causal sweep: the full map's change restricted to the top-k directions (k ∈
{1, 2, 4, 8, 16, 32, 64, 128}) installed under A on the test histories' goal-dependent decisions, switch against
the full map's; the effective causal rank k_c per pair (the smallest k at 0.9 × the full map) and the pooled
subspace's k_c; controls bottom-k and random-k (three draws). The natural goal swap per site is the ceiling. For
the GRU the swap at h_{t−1} installs the state and runs the last token's update; at h_{t−2}, the last two. **Each
from-scratch model's registered k is the median, over the twelve ordered pairs, of the pooled subspace's k_c at its
registered site** (below), and its goal subspace the pooled top-k directions there; k ≤ 8 is expected but not
required. Conventions (the builder's calls, fixed here so that the fine-tuned model's 4 at 23 and 8 at 21 follow the
same rule: its pooled k_c values there are 4 × 10, 8 × 2 and 4 × 6, 8, 16 × 4, 32): a median that falls between
two grid values is rounded **up** to the next grid value (a larger subspace is the conservative choice for a test
that asks whether the subspace is enough); a pair whose k_c is undefined (no k on the grid reaches 0.9 × the full
map, or the full map's own switch on that pair is below 0.10) is left out of the median; if more than six pairs are
undefined at a site, that site has no registered k. **The part 0 reference has the same floor as the through
ratios**: a site whose natural goal swap is below 0.10 has no goal subspace worth the name, and a from-scratch model
with no such site among its non-linear ones (the transformer's entrances 1–3, the GRU's h_{t−1} and h_{t−2}) has no
registered site; its THRU, COMP, CTRL and KNOWN are then reported as undefined, and parts 1–3 are run on its linear
site only, as the ceiling they are.

## Part 1. The upstream belief swap, measured through the subspace downstream

For an (upstream site s_in, downstream site s_out) pair and a (pair, goal) cell: h1 is run under g with its state at
s_in replaced by h2's natural state at s_in under g; the patched logits give the **upstream swap** (switch_in, the
reference for this part), and the induced state x̃ at s_out is recorded. With δ = x̃ − x1 (x1 the natural state of
h1 at s_out), the following are installed at s_out in the natural run of h1 under g and the switch counted:
x1 + δ (reproduces the upstream swap exactly: the patch only changes the decision position, so this is the check),
**x1 + P δ** (through the goal subspace), **x1 + (I − P) δ** (through its complement), x1 + P_bottom δ, x1 + P_rand δ
(three draws), x1 + P_J δ, x1 + P_H δ, and the sweep x1 + P_k δ over k. The **through ratio** of a projector is its
switch over switch_in. Also reported: the share of δ's squared norm in the goal subspace (expected tiny: the
decision reads few directions of a large change, as the coordinates study found for the goal), the same for the
complement and the controls.

Fine-tuned model: s_in ∈ {15, 19}, s_out ∈ {21, 23}. Transformer: s_in ∈ {1, 2}, s_out ∈ {3, final}. GRU:
s_in = h_{t−2}, s_out = h_{t−1} (and h_t, as the linear ceiling). **Part 1 is decision-bearing for the fine-tuned
model only** (the builder's call): there the belief sits in the token at 15 and 19 before the goal is read, so a
swap there is a change of belief that the goal layers must re-read. In the from-scratch models the goal is the
start token and every state at the decision position already carries it, so no site is upstream of the goal; their
part 1 is the belief swap one or two positions earlier measured through the subspace, reported as an extra (the
induced change differs from the natural difference because the context stays h1's), and their verdict rests on
THRU-D, COMP, CTRL and KNOWN.

## Part 2. The natural history difference at the downstream site

At each site, with Δ = x2 − x1 the natural difference between the two histories' states under g: x2 (the natural
history swap, the ceiling for this part), **x1 + P Δ**, **x1 + (I − P) Δ**, the controls and the sweep as in part 1,
the random vector of |Δ| as the floor. The through ratio is the projector's switch over the natural swap's. The
squared-norm share of Δ in each subspace is reported, and the **overlaps**: goal subspace with the readout subspace,
goal subspace with the history subspace, history subspace with the readout subspace, each against random subspaces
of the same rank. Fine-tuned model: every site (15, 19, 21, 23). Transformer: every site. GRU: every site.

## Part 3. Known and uncertain states

Every through ratio of parts 1 and 2 split by the goal-free set size of the receiving history h1: **known** (size 1)
against **uncertain** (size > 1), with the bootstrap interval of their difference; the same split by h2's size is
reported. The share of history-dependent cells in each bin is reported with it, since at known states the optimal
action is more often goal-dependent and less often history-dependent.

## Decision rule

Per model, at its **registered site**: the fine-tuned model at layer 23 (the largest natural goal swap, k = 4), with
21 reported; the transformer at the entrance (1, 2 or 3) with the largest natural goal swap in part 0, the final
residual reported as the linear ceiling; the GRU at h_{t−1}, h_t reported as the linear ceiling. Part 1's registered
pair is the registered s_out with the upstream site at which the natural history swap is largest among those where
the natural goal swap is below half its maximum (the fine-tuned model: 19). Bootstrap 95 % intervals over cells,
clustered by h1; a criterion holds if its interval excludes the threshold.

| | criterion |
|---|---|
| **THRU-D** | part 2: the natural history difference restricted to the goal subspace switches ≥ 0.65 × the natural history swap |
| **THRU-U** | part 1, the fine-tuned model only: the upstream swap's induced change restricted to the goal subspace switches ≥ 0.65 × the upstream swap |
| **COMP** | the complement's through ratio is ≤ 0.35 in both parts (the effect is in the subspace, not merely reproducible from a large enough piece of the change) |
| **CTRL** | bottom-k and every random-k draw give through ratios ≤ 0.20 in both parts |
| **DISTINCT** | the goal subspace's overlap with the readout subspace is ≤ 0.50 (the subspace is not the logit readout under another name) |
| **KNOWN** | the part 2 through ratio at known states exceeds the one at uncertain states (the interval of the difference excludes 0) |

**Reading.** THRU-D holding with COMP and CTRL (and THRU-U, for the fine-tuned model) is the hypothesis: belief
and goal are combined into the same few directions. A through ratio **below 0.40** on THRU-D rejects it for that model (the belief's effect on the
decision uses directions the goal does not); between 0.40 and 0.65 is borderline and is reported as such. DISTINCT
failing while THRU holds means the subspace is the readout: belief and goal both pass through it because everything
does, which is true of any model at its last linear step and says nothing about how they are combined; then the
earlier site's numbers carry the reading.

## Expectations

| | expectation |
|---|---|
| E1 | THRU-D holds for every model: the GRU at about 0.8, the fine-tuned model and the transformer at 0.65–0.7 (the author's numbers); the sweep rises steeply and saturates by k = 8 (fine-tuned) or 16 (small models), as the goal's own sweep did |
| E2 | THRU-U holds at the same level as THRU-D: a belief installed at 19 is re-read by the goal layers and its consequence lands in the same directions |
| E3 | COMP and CTRL hold at every registered site: the complement carries most of the change's norm and little of its effect |
| E4 | DISTINCT holds at 23 and 21 for the fine-tuned model (overlap 0.2–0.5: one block, its attention over the context and the final norm lie between the site and the logits) and at the transformer's registered entrance; at the two linear sites it fails by construction (overlap ≥ 0.8), which is why they are ceilings |
| E5 | the history subspace and the goal subspace overlap at ≥ 0.5 at the registered sites (random 0.004–0.03): the belief effect's own principal directions are the goal's |
| E6 | KNOWN holds: at known states the decision is a goal × position readout and the belief's consequence is fully in the goal's directions; at uncertain states part of the history's effect (the information-seeking move) passes outside them, so the through ratio is lower |
| E7 | part 0: the from-scratch models' goal subspaces are low-rank too (median k_c ≤ 8 at the registered site, k90 of the differences far above), as the fine-tuned model's were |

The alternative E1 would reject: THRU-D below 0.40 with the complement switching as the full difference does. Then the
goal's four directions are a readout of the goal alone, the belief acts on the decision elsewhere, and "combined into
one subspace" is the wrong picture: the two are added at the logits, not before.

## Limits known in advance

Linear subspaces of one site's state, fitted on the optimum's trajectories (the coordinates study found the goal
maps do not transfer to shortest histories; the pairs here are from the same pool as the subspaces' fit, held out by
history, so the result is about the model on its own distribution); one seed per model; the from-scratch models'
goal subspaces are found in the same experiment that tests them (part 0 is registered here as a procedure, not as a
result); the readout subspace is a local linearisation, pooled over decisions; the known / uncertain split uses h1's
goal-free set size, and at known states fewer cells are history-dependent, so KNOWN has wider intervals; the through
ratio is a ratio of switch rates and is undefined where the reference switch is below 0.10 (then reported as such).

## Compute

About 2–3 h on one GPU. Fine-tuned model: the test histories' natural states at the four sites under four goals are
in the coordinates study's cache (recomputed by `collect.py` if absent, 7 168 prompts); part 1: 2 × 2 (s_in, s_out)
× 2 048 cells × 11 variants ≈ 90 000 patched prompts; part 2: 4 sites × 2 048 cells × 17 variants ≈ 140 000;
Jacobians: 1 979 × 4 backward passes per site. From-scratch models: part 0 ≈ 12 pairs × 8 ranks × 5 variants × 4
sites × 660 histories, parts 1–2 as above, all in batches of 2 048 on a 128-wide model: minutes.
