# maze10: task, measures, decision rule and expectations

Written before the models of this experiment were trained. **Disclosure:** a design probe of reference policies
(`design.md`, no network) and a learnability pilot came first: 3 seeds, 200 updates, at noise 0.2 and 0.4; only returns
were looked at (0.51–0.52 at noise 0.4, 0.53–0.54 at 0.2). The pilot's models were discarded. Brief: `BRIEF.md`. Code:
`goalgeo/bigmaze.py`, `train.py`, `measure.py`.

## Task (choices of mine where the brief left them open)

* **Maze**: 55 open cells in a 9 × 9 box (a perfect maze on a 5 × 5 lattice of rooms plus 6 loops; `design.md` shows
  it). The brief said 10 × 10; rooms at even coordinates give an odd box. Four times the cells of the earlier maze;
  paths to a goal are 9.2 moves on average and at most 19.
* **Observations**: one noisy symbol pair (each cell shows its own symbol with probability 0.6, the other with 0.4),
  assigned to cells at random, and two noiseless landmarks. The noise and the pair are the maze-belief experiment's.
* **No passive prefix**: the goal (one of four fixed cells, spread out) is shown with the first symbol and every move
  is the agent's. **Spawn**: uniform over the 51 cells that are not goals.
* 40 moves, discount 0.97 (0.9 would leave almost no reward on paths of 15 moves).
* **No solver.** Every environment carries the exact posterior over its location (a closed-form filter). Action values
  and optimal moves are not available. References use the filter and shortest paths only:

| reference | what it is | return (success) on the evaluation episodes |
|---|---|---|
| oracle | knows the cell: an upper bound no agent can reach | 0.786 (1.00) |
| QMDP + lookahead | values the information of the next symbol | 0.564 (0.95) |
| QMDP | exact belief, ignores the value of information | 0.551 (0.93) |
| most likely cell | | 0.489 (0.84) |
| random moves | | 0.07 (0.11) |

Training: the maze-belief transformer (4 layers, width 128, 4 heads) and the observation-prediction experiment's PPO
settings, reward only; **6 seeds, 1 000 updates** of 4 096 episodes.

## Measures (`measure.py`), on each model's own greedy episodes

**A. Behaviour.** Return, success and length on 8 192 fixed episodes (the same spawns and goals for every model and
reference).

**B. Information seeking, from behaviour.** At every decision the exact filter gives, for each move, its QMDP value
and its expected information gain (entropy of the posterior now, minus its expected entropy after the move and the
next symbol).
* the share of decisions where the model's move is not a QMDP-best move (**deviations**);
* at deviations: the information gain of the model's move minus that of QMDP's move, and the QMDP value given up;
* null: QMDP with uniformly random moves inserted at the model's deviation rate (random deviations also change the
  information gain);
* reference: the same for QMDP + lookahead, which seeks information by construction;
* posterior entropy by step, and the share of episodes that step on a landmark, against QMDP on the same episodes.

**C. The additive code, tested by behaviour.** Logits at every decision under each of the four goals (the goal token
replaced, the history the model's own). H(h) = their mean over goals, centred over actions; G(g, step) = the mean goal
deviation per step on separate fit episodes.
* offline: the share of decisions where argmax H + G is the model's move, on decisions where the model's move depends
  on the goal; the additive share of the logits' goal dependence;
* **online**: policies run in the environment on the evaluation episodes: natural; **additive** (argmax H + G, the
  history being the additive policy's own); goal-blind (argmax H); history-blind (argmax G).
  **Recovery** = (additive − goal-blind) / (natural − goal-blind), in return.

**D. The additive state, tested by intervention.** As the additive-ablation experiment's recomputed removals, run
online: at every decision the network runs under all four goals, and at the decision token every attention and MLP
output is replaced before the next component reads it.
* interaction removed: output → history part (mean over goals) + goal part (mean over the batch's histories);
* history part removed, interaction kept; goal part removed, interaction kept.
Returns on the evaluation episodes; recovery as in C.
All tokens here come after the goal token, so earlier tokens can already mix goal and history. The removal makes the
*decision token's* state additive; it does not remove what earlier tokens computed.

**E. What the state holds.** Ridge decoders from the final state at decision tokens, held-out episodes:
* the exact posterior (R²), and the true cell (accuracy, against the posterior's own most likely cell);
* **occupancy**: the discounted future visitation of each cell along the model's own trajectory. R² from the state;
  from the exact posterior × goal × step (what a belief-and-goal code can give); and from both.

## Decision rule

Six models; medians; exact one-sided Wilcoxon signed-rank tests, p < 0.05.

| | criterion |
|---|---|
| **LEARN** | greedy return ≥ the most-likely-cell policy's |
| **SEEK-R** | greedy return > QMDP's (the model collects more than a policy that ignores information can) |
| **SEEK-B** | at deviations from QMDP the model's move has a higher expected information gain than QMDP's move, and by more than the random-deviation null |
| **AD-B** | additive policy: recovery ≥ 0.9 |
| **AD-S** | interaction removed at the decision token: recovery ≥ 0.9 |
| **OCC** | occupancy R² from the state exceeds that from posterior × goal × step by ≥ 0.05 |

| result | reading |
|---|---|
| AD-B and AD-S | the additive policy holds in a maze where the move toward a goal depends on the position |
| neither | **the additive policy was a property of the small maze**: here the policy needs goal × history interaction |
| one of them | additive at one level only; reported as such |
| SEEK-R or SEEK-B | the agent seeks information; not SEEK-B: its deviations from QMDP are errors, not exploration |
| not LEARN | the models did not learn the task; nothing else is interpreted |

## Expectations

| | expectation |
|---|---|
| E1 | LEARN holds |
| E2 | SEEK-R fails: the models end between the most-likely-cell policy and QMDP |
| E3 | SEEK-B fails: deviations from QMDP are not more informative than random ones |
| E4 | AD-B fails, recovery < 0.5: in a two-dimensional maze the move toward a goal depends on where the agent is, so a fixed bias per goal cannot pick it |
| E5 | AD-S fails likewise |
| E6 | the posterior decodes at R² ≥ 0.5 |
| E7 | OCC holds: the state says where the agent is going, beyond belief and goal |

## Limits known in advance

No optimal policy: competence is relative to heuristics, and "information seeking" is relative to QMDP. One maze,
one goal set. G indexed by step only. Occupancy targets are single trajectories. The removals in D act at the decision
token only.
