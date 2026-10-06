# Ablating the additive code's parts, live (additive ablation)

Run date: 2026-10-06. Brief: `BRIEF.md` (given in conversation). Kinds and decision rule: `PLAN.md`, written **before
`run.py` was run on any trained model** (smoke-tested on the untrained checkpoint of seed 0 only). Not committed before
the run; its SHA-256 at that time is in `plan.sha256`. Numbers from `tables.md`; data in `results.json` (registered)
and `explore.json` (post hoc). Reproduce: `reproduce.py additive_ablation` (1 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen; the query-swap experiment's cells. At
the goal token each of the eight component outputs splits, as in the interaction-removal experiment, into a history
part xbar_k(h) (mean over goals), the goal's main effect M_k(g, L) and the interaction I_k(h, g). The brief: remove the
history part and see whether the policy collapses, which would establish the additive code as the policy.

Setting the additive state with a history-blind history part (`F_hist_mean`) collapses the policy by construction,
because nothing at the goal token is then computed live. The test that can fail runs live: remove the history part and
let later components recompute, keeping the interaction, which is also history-dependent.

## 1. Registered: the live edits overshoot

| edit at the goal token | optimal, all cells | history dependence |
|---|---|---|
| nothing | 0.854 | 0.317 |
| history-blind ceiling B (best action per goal and length) | 0.625 | 0 |
| fixed: Xbar(L) + M(g) (history-blind by construction) | 0.625 | 0.000 |
| **live: history part removed at all 8** | **0.362** | **0.293** |
| live: history part removed at the four attention outputs | 0.613 | 0.068 |
| live: interaction removed at all 8 | 0.596 | 0.262 |
| live: rotated history part (control) | 0.805 | 0.309 |
| live: history part swapped to h′, all 8 | 0.331 | 0.507 |

* Removing the history part at all eight components takes the policy far **below** the history-blind ceiling (0.36
  against 0.63), and it stays as history-dependent as the natural policy (0.29 against 0.32). That is not a collapse.
  It is a wrong history code driving the decision.
* The cause is double subtraction. The edits are fixed vectors measured in natural runs. Once the attention outputs'
  history part is removed, the MLPs downstream no longer produce theirs. Subtracting their natural history part anyway
  inserts its negative. The same edit at the four attention outputs alone, where the history enters, collapses cleanly
  (0.613, history dependence 0.068). The interaction removal (0.596 against the fixed version's 0.821) and the goal
  removal (goal dependence unchanged, optimal 0.26) overshoot the same way.
* Swaps are not affected in direction. Moving the history part to a donor's takes the donor's natural action in 0.929
  of the pairs where it differs (stay 0.002). Moving the goal part to g′'s takes g′'s action in 0.847.

| | criterion | value | held |
|---|---|---|---|
| **HC** | live_noH collapse fraction ≥ 0.8, above rotated | 2.30 against 0.23; p < 0.001 | by the letter; **not a collapse** (history dependence 0.29) |
| **ID** | natural − live_noI ≤ 0.2 × (natural − live_noH) | ratio 0.47 | no |
| **SW** | live_swapH follow ≥ 0.75, above stay | 0.929 against 0.002; p < 0.001 | **yes** |
| **GC** | live_noG goal dependence ≤ 0.25 × natural's, and live_swapG follow ≥ 0.75 | 0.94 ×; 0.847 | no |

The collapse fraction was registered as the collapse measure, but it cannot tell a collapse from an overshoot. HC
passes on a number above 1, which the history dependence shows is not a collapse. **Registered reading: SW only.** The
history part steers the decision to a donor's, but the registered removals cannot show that it is necessary, because
they overshoot.

## 2. Post hoc: recomputed removals show the collapse (`explore.py`)

Every history is run under all three goals in one batch. At each component, in order, the parts are measured from the
edited outputs themselves, so later components lose exactly the part they still make. Not registered; designed after
seeing section 1.

| edit at the goal token, recomputed | optimal, all cells | goal-matters | history dependence | goal dependence |
|---|---|---|---|---|
| nothing | 0.854 | 0.834 | 0.317 | 0.807 |
| **history part removed, interaction kept** | **0.620** | 0.704 | **0.044** | |
| history part and interaction removed (history-blind by construction) | 0.590 | 0.673 | 0.000 | |
| **interaction removed** | **0.788** | 0.746 | 0.237 | |
| **goal part removed, interaction kept** | 0.635 | 0.466 | | **0.119** |

* **Without the history part, the policy collapses to the history-blind ceiling** (0.620 against 0.625; collapse
  fraction 1.02). History dependence falls from 0.32 to 0.04 although the goal × history interaction is left in
  place. Of what the history contributes (natural − history-blind 0.26), the interaction alone recovers 0.019 (0.06).
* **Without the interaction, the policy keeps most of its decisions** (0.788; drop 0.063), as in the
  interaction-removal experiment's fixed version (0.821). Against the history part's drop (0.227) that is a ratio of
  0.27, just outside the 0.2 that ID required.
* **Without the goal part, goal dependence falls from 0.81 to 0.12**, with the interaction left in place.
* Edits at the entry points alone, with natural vectors, agree. Swapping the history part at the four attention outputs
  takes the donor's action in 0.927 of pairs (stay 0.026). Removing the goal part at block 0's attention leaves goal
  dependence at 0.19; swapping it there takes g′'s action in 0.953.

## Conclusions

1. **The additive code is the route the history and the goal take to the decision at the goal token.** Removed
   consistently, the history part leaves a history-blind policy (at the ceiling, history dependence 0.04), and the goal
   part a goal-blind one (0.12). In both cases the interaction is still present and carries almost nothing. Swapped,
   either part moves the decision to the donor's (0.93, 0.95). Removing the interaction costs 0.06.
2. That rests on the post hoc recomputed removals. The registered fixed-vector removals overshot, so by the registered
   rule only SW held, and the necessity claim is post hoc.
3. **Method:** removing a part from several components in series with vectors measured in natural runs subtracts it
   twice. Collapse should be measured by history dependence, not by accuracy alone. The interaction-removal experiment
   was not affected because it set every component (fixed) rather than subtracting live.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | F_additive reproduces all_hist within 0.005 | yes | goal-matters 0.788 |
| E2 | F_hist_mean history dependence 0, optimal ≤ B | yes | 0.000; 0.625 |
| E3 | HC holds; live_noH within 0.05 of B | **no** | 0.362 (overshoot) |
| E4 | ID holds | **no** | 0.47 |
| E5 | SW holds, follow 0.8–0.9 | yes (above) | 0.929 |
| E6 | live_noH_attn collapses less than live_noH | **no**: it collapses more cleanly | 0.613, dependence 0.068 |
| E7 | GC holds | **no** | 0.94 × |

3 of 7.

Limits: the necessity result is post hoc; recomputed parts use in-batch means per prefix length (4 096 cells per batch);
goal token and first decision only; reward models only; the history part is the whole goal-averaged state, not a
low-dimensional code. Which low-dimensional variable carries it (H in R^4, the posterior, the H-mixture code) is not
tested here.
