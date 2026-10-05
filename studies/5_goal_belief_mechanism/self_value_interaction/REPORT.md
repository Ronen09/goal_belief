# Does the self-value swap move the goal × belief interaction at the MLP input? (self-value interaction)

Run date: 2026-10-02. Brief: `BRIEF.md`. Decomposition, swaps, measures and decision rule: `PLAN.md`, committed
(`bf32595`) **before any decomposition was fitted on a trained model or any swap of this experiment applied to one**. Numbers
from `tables.md`; data in `results.json`. Reproduce: `reproduce.py self_value_interaction` (20 min on one GPU).

**Setting.** The observation-prediction experiment's ten reward models, frozen; held-out histories under the six ordered goal pairs g₁ → g₂. At the
goal token, from natural fit-side states:
* f(b, g) = c[g, L] + S_g(b), with S_g a table per posterior and goal;
* f(b) = c̄[L] + S(b), with S shared;
* **I(b, g) = f(b, g) − f(b) = M(g, L) + J(b, g)**: the goal's **main effect** M and the **goal × belief part** J.

Block 0's attention output at the goal token is recomputed with the goal token's **own value** from g₂ (the self-value experiment's goal
identity) or its **query** from g₂ (query swap), and the network runs on. The state change is regressed on the changes of
M and J. Each coefficient is reported relative to the full change (the natural run under g₂), as ρ_M and ρ_J; 1 means
the component moved fully to the other goal's. Three sites:
* **u**, the MLP input (primary);
* m, before its layer norm;
* z, after the MLP.

## 1. At the MLP input, the self value moves the goal's main effect, not its interaction with belief

| at u (the MLP input) | ρ_M: main effect | ρ_J: goal × belief | share of the change explained |
|---|---|---|---|
| **own value from g₂** | **0.77** (0.55–0.90) | **0.37** (0.15–0.54) | 0.72 |
| query from g₂ | 0.11 | 0.42 (0.32–0.61) | 0.17 |
| natural under g₂ | 1 | 1 | 0.97 |

* **J fails**: the goal × belief part moves 0.37 of the way to the other goal's, below 0.5.
* **M holds**: the main effect moves 0.77 of the way. The rest is the goal embedding, which the swap leaves at g₁ and
  which the self-value experiment showed the policy ignores.
* At the MLP input, the goal × belief part is small: |dJ|² is 0.06 of |dM|². It follows the self value and the query
  about equally (0.37 against 0.42, p 0.22). Before the layer norm (m) the query leads (0.56 against 0.26), consistent
  with the query-swap experiment. The layer norm's scaling mixes some of the main effect into it.

## 2. After the MLP, the interaction follows the self value

| at z (after block 0's MLP) | ρ_M | ρ_J | explained |
|---|---|---|---|
| **own value from g₂** | 0.76 | **0.67** (0.40–0.86) | 0.80 |
| query from g₂ | 0.11 | 0.19 | 0.17 |

* **Z holds**: after the MLP, swapping the self value moves the goal × belief part 0.67 of the way. Swapping the query
  moves it 0.19. The part is twice as large there relative to the main effect (|dJ|²/|dM|² 0.10 against 0.06 at u,
  1.96 ×).
* **So the goal × belief interaction that matters is made by block 0's MLP from the goal identity.** The MLP receives
  the goal's main effect (from the self value) and a mostly shared belief code, and its output combines them in a
  belief-specific way. That output's interaction follows the goal identity it was given. It does not follow the small
  goal × belief part already present at its input.

## 3. Is the shared belief left in place? The two measures disagree

| at u | decoded posterior change (L1) | change inside the shared code's subspace, / natural goal change's |
|---|---|---|
| own value from g₂ | **1.03** | **0.66** |
| query from g₂ | 0.77 | 0.21 |
| natural under g₂ (same history: belief unchanged) | 0.26 | 1 |
| scale: decoded difference between real belief pairs | 1.34 | |

* **By the registered measure, F fails.** A goal-free linear decoder reads a posterior shift of 1.03 under the swap,
  0.55 of a real belief change beyond what a natural goal change shows.
* **The decoder-free measure says otherwise.** Inside the subspace where the shared belief code varies, the swap moves
  the state less than a natural goal change does (0.66 of it). The decoder shift therefore comes from directions
  outside the shared code. The swapped state combines the g₁ embedding with a g₂ self value, a state that never occurs
  naturally, and a linear decoder fitted on natural states need not be stable there. The query swap, which changes the
  state much less, also shifts the decoder by 0.77.
* **Behaviour sides with the subspace measure.** In the self-value experiment, the same swap left the optimal rate on goal-neutral cells
  (where only the belief decides) at 0.876 against 0.887. The belief the policy uses is intact.

I report F as registered (failed). The plan's reading for a failed F is "the swap also moves the shared belief
component". The subspace measure and the self-value experiment's behaviour argue against that reading; the decoder shift is better
explained by the off-manifold state.

## 4. Decision rule

| | criterion | value | held |
|---|---|---|---|
| **J** | at u: ρ_J(selfv) ≥ 0.5 | 0.37 | no |
| **M** | at u: ρ_M(selfv) ≥ 0.75 | 0.77 | **yes** |
| **F** | at u: decoded change beyond the natural goal change ≤ 0.1 × pair difference | 0.55 × | no |
| **Z** | at z: ρ_J(selfv) ≥ 0.5 | 0.67 | **yes** |
| Q | at u: ρ_J(query), above selfv | 0.42; p 0.22 | (comparison) |

No row of the plan's reading table matches exactly. The closest, "not J; Z; F", fails only on F, and §3 explains why
F's failure is probably the decoder's.

## Expectations

| | expectation | held | numbers |
|---|---|---|---|
| E1 | M holds | yes | 0.77 |
| E2 | J fails, ρ_J between 0.2 and 0.5 | yes | 0.37 |
| E3 | Q: ρ_J(query) ≥ 0.5, above selfv, at u | **no** | 0.42, p 0.22 (at m: 0.56) |
| E4 | Z holds | yes | 0.67 |
| E5 | F holds | **no** | 0.55 × by the decoder |
| E6 | interaction / main effect at z ≥ 2 × at u | **no** (just) | 1.96 × |

3 of 6.

## Conclusions

1. **The answer to the brief: not at the MLP input.** Swapping only the goal's self value moves the goal's main effect
   at the MLP input to the other goal's (0.77), but its goal × belief part only 0.37 of the way. That part is small at
   the input (6 % of the main effect's size) and follows the query about as much as the self value.
2. **After the MLP, yes.** There, the goal × belief part follows the self value (0.67) and hardly the query (0.19). Block
   0's MLP builds the interaction from the goal identity the self value gives it and the belief code it reads.
3. **The shared belief.** A linear decoder shifts under the swap; the shared code's own subspace does not move beyond a
   natural goal change; and the policy's belief-only decisions are unaffected (self value). Most likely the belief is
   left in place and the decoder is misled by an unnatural state. The registered criterion still failed.

The mechanism is now:
* the self value sets the goal (main effect);
* the query adds a small goal-dependent weighting of the evidence;
* block 0's MLP turns goal identity × shared belief into the goal-specific belief code (the I(b, g) that matters);
* later MLPs turn that into the action.

Next, if wanted: the brief's question one site later, as an intervention. Swap the self value and test whether
I(b, g₁) at z moves toward I(b, g₂) for each posterior separately: does the MLP's output follow the posterior-specific
pattern of the new goal, and not just its average?

Limits: M and J are posterior-level tables; the shared belief is read by one linear decoder and one subspace; the
swapped states are off the natural manifold; reward models only; the first decision.
