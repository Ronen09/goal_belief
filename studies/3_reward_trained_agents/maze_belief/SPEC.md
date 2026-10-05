# Maze belief: frozen task specification

Frozen after the solver checks (`checks.md`) and before any model was trained. Code: `goalgeo/mazebelief.py`
(`cross_maze`, filter, solver, baselines). Validation: `tests/test_mazebelief.py`.

```
  ·    ·   0L    ·    ·    ·   1b    ·    ·    ·    ·
  ·   2a   3a   4a   5a   6a*  7b   8b   9b  10b  11b*
  ·    ·  12a    ·    ·    ·  13b*   ·    ·    ·    ·
```

| | |
|---|---|
| world | 14 cells, fixed for all of training; number = cell, letter = symbol, L = landmark, * = goal cell |
| hidden state | the agent's cell |
| start | uniform over cells 2, 3, 9, 10 (two per arm); no goal cell is a start |
| symbols | a cell emits its own symbol with probability 0.6 and the sibling (a ↔ b) with 0.4; the landmark's symbol is unique and noiseless |
| actions | N, S, W, E; deterministic; a move into a wall leaves the agent in place |
| prefix | 0–4 moves (uniform), imposed uniformly at random, shown to the model with the symbols that follow; no goal is shown, entering a goal cell does nothing |
| reveal | the goal (G1 = cell 6, G2 = cell 13, G3 = cell 11, uniform) is shown; the agent acts from here |
| reward | entering the goal with the t-th move after the reveal pays 0.9^(t−1) and ends the episode; after 12 moves the episode ends with nothing |
| model input | symbols, actions, and the goal from the reveal on |
| hidden from the model | the cell, the posterior, the transition and emission tables, solver values |

## What the checks established

1. **Belief geometry.** At the start the posterior has two values (one noisy symbol). After a 4-move prefix there are
   452 distinct posteriors on 65 supports, with participation ratio 6.7 and 9 components for 95 % of the variance.
   "Probability of being in the left arm" explains 12 % of it. The number of start cells changes the count of
   posteriors (115 / 452 / 885 for 2 / 4 / 6) and hardly the dimension (6.5 / 6.7 / 6.5).
2. **Matched evidence.** At the reveal the posterior is the goal-free filter's and is the same for the three goals by
   construction. After the reveal, not having terminated is evidence and depends on the goal.
3. **Action and predictive relevance.** G1 and G2: the first action depends on the belief. G3: east is optimal for
   80–100 % of beliefs (the exceptions stand in a side branch), while the value varies more across beliefs than for
   the other goals (sd 0.15 vs 0.07) and the expected arrival time by 2.2 moves. G3 controls for action relevance
   only.
4. **Baselines** (value lost, prefix 0 → 4): most likely cell taken as certain 0.15 → 0.08 (G1), 0.18 → 0.09 (G2),
   0.00 → 0.02 (G3); belief-weighted fully observed values ≤ 0.010 (G1), 0.020–0.034 (G2); planning without future
   observations ≤ 0.002 (G1), ≤ 0.008 (G2). **The task requires acting on the whole belief, not on its mode. It
   hardly requires planning to observe**: evidence arrives while moving.
