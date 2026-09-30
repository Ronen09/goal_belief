# Round 25, post hoc: at equal regret

Checkpoints from update 200, binned by greedy regret; mean over checkpoints (number). Difference from the selected arm within bins, weighted by the number of checkpoints, with a bootstrap interval over seeds.


## Q1: type A, TV between the two histories' action distributions

| arm | < 0.004 | 0.004–0.006 | 0.006–0.01 | 0.01–0.02 | 0.02–0.05 | > 0.05 | final median: difference from selected | within bins [95 %] |
|---|---|---|---|---|---|---|---|---|
| selected | 0.072 (7) | 0.079 (25) | 0.096 (33) | 0.114 (12) | 0.129 (15) | 0.128 (8) |  |  |
| balanced, all | 0.072 (20) | 0.074 (25) | 0.096 (24) | 0.132 (13) | 0.133 (10) | 0.122 (8) | -0.008 | 0.001 [-0.008, 0.011] |
| balanced, one | 0.068 (18) | 0.072 (19) | 0.091 (29) | 0.135 (9) | 0.095 (13) | 0.121 (12) | -0.011 | -0.007 [-0.019, 0.006] |
| reward only | — (0) | 0.056 (5) | 0.085 (30) | 0.122 (13) | 0.109 (27) | 0.117 (25) | -0.005 | -0.012 [-0.024, -0.001] |

## O2: greedy changed where the optimal set is the same

| arm | < 0.004 | 0.004–0.006 | 0.006–0.01 | 0.01–0.02 | 0.02–0.05 | > 0.05 | final median: difference from selected | within bins [95 %] |
|---|---|---|---|---|---|---|---|---|
| selected | 0.199 (7) | 0.217 (25) | 0.276 (33) | 0.318 (12) | 0.388 (15) | 0.405 (8) |  |  |
| balanced, all | 0.194 (20) | 0.238 (25) | 0.288 (24) | 0.365 (13) | 0.388 (10) | 0.402 (8) | 0.010 | 0.014 [-0.005, 0.031] |
| balanced, one | 0.185 (18) | 0.242 (19) | 0.287 (29) | 0.380 (9) | 0.337 (13) | 0.385 (12) | 0.016 | 0.005 [-0.019, 0.030] |
| reward only | — (0) | 0.243 (5) | 0.307 (30) | 0.380 (13) | 0.362 (27) | 0.402 (25) | 0.087 | 0.016 [-0.007, 0.039] |
