# TASK16 tables (round 17: pricing recomputation with a learned read gate)

Read rate = fraction of (block, position ≥ 2) with an open deterministic gate on 3000 held-out sequences. Prior weight = calibrated λ of R2 (position t's exports from B, A's tokens) at the output of t+1; 'open' = every gate forced open at test, 'own' = the learned gates. Entropy / movement split = mean exact-filter entropy / KL(J_u ‖ J_{u−1}) at open minus closed (block, position ≥ 8) pairs. 3 seeds, seed means. Positions 0–1 carry no gate, so 'u=2–4' is the earliest gated window.

## carry

| c | read rate | KL own | KL open | prior t=4 (open) | prior t=8 (open) | prior t=16 (open) | prior t=23 (open) | prior t=4 (own) | prior t=8 (own) | prior t=16 (own) | prior t=23 (own) | R3 t=16 (open) | entropy split | movement split | rate u=2–4 | rate u≥16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.0058 | 0.0058 | 0.531 | 0.437 | 0.401 | 0.379 | 0.531 | 0.437 | 0.401 | 0.379 | 0.524 | — | — | 1.000 | 1.000 |
| 0.003 | 0.500 | 0.0038 | 0.0038 | 0.595 | 0.513 | 0.474 | 0.450 | 0.595 | 0.512 | 0.473 | 0.449 | 0.438 | 0.000 | 0.000 | 0.500 | 0.500 |
| 0.01 | 0.095 | 0.0025 | 0.0040 | 0.945 | 0.935 | 0.931 | 0.929 | 0.964 | 0.955 | 0.950 | 0.949 | 0.052 | -0.074 | 0.013 | 0.095 | 0.096 |
| 0.03 | 0.000 | 0.0020 | 0.0021 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | -0.000 | — | — | 0.000 | 0.000 |
| 0.1 | 0.000 | 0.0019 | 0.0020 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | — | — | 0.000 | 0.000 |
| 0.3 | 0.000 | 0.0019 | 0.0019 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | — | — | 0.000 | 0.000 |

crossover c* (carry) = 0.003

## plain

| c | read rate | KL own | KL open | prior t=4 (open) | prior t=8 (open) | prior t=16 (open) | prior t=23 (open) | prior t=4 (own) | prior t=8 (own) | prior t=16 (own) | prior t=23 (own) | R3 t=16 (open) | entropy split | movement split | rate u=2–4 | rate u≥16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.0080 | 0.0080 | 0.001 | 0.006 | 0.021 | 0.034 | 0.001 | 0.006 | 0.021 | 0.034 | 0.878 | — | — | 1.000 | 1.000 |
| 0.003 | 0.380 | 0.0052 | 0.0114 | 0.044 | 0.040 | 0.049 | 0.043 | 0.191 | 0.205 | 0.231 | 0.248 | 0.876 | 0.174 | 0.001 | 0.251 | 0.447 |
| 0.01 | 0.250 | 0.0049 | 0.0157 | 0.058 | 0.045 | 0.026 | 0.021 | 0.255 | 0.284 | 0.310 | 0.319 | 0.880 | 0.000 | -0.000 | 0.250 | 0.250 |
| 0.03 | 0.110 | 0.0052 | 0.0286 | 0.293 | 0.235 | 0.190 | 0.173 | 0.678 | 0.680 | 0.710 | 0.707 | 0.787 | -0.055 | 0.011 | 0.110 | 0.110 |
| 0.1 | 0.000 | 0.0215 | 0.0452 | 0.285 | 0.235 | 0.176 | 0.136 | 0.809 | 0.806 | 0.803 | 0.811 | 0.834 | — | — | 0.000 | 0.000 |
| 0.3 | 0.000 | 0.0214 | 0.0527 | 0.355 | 0.245 | 0.145 | 0.129 | 0.803 | 0.807 | 0.818 | 0.805 | 0.865 | — | — | 0.000 | 0.000 |

crossover c* (plain) = 0.003

Qualifying intermediate carry price for R4/R5: c = 0.003


## Pre-registered predictions (rounds/r17_read_cost/THEORY.md §4)

| prediction | held | numbers |
|---|---|---|
| R1 carry: read rate non-increasing in c, ≥ 0.5 at c=0, ≤ 0.05 at c=0.3 | yes | c=0.0: 1.000 → c=0.003: 0.500 → c=0.01: 0.095 → c=0.03: 0.000 → c=0.1: 0.000 → c=0.3: 0.000 |
| R2 carry: KL under own gates < 0.01 at every c | yes | c=0.0 0.0058; c=0.003 0.0038; c=0.01 0.0025; c=0.03 0.0020; c=0.1 0.0019; c=0.3 0.0019 |
| R3 carry, read rate ≤ 0.05: prior weight own ≥ 0.95 and open ≥ 0.8 | yes | c=0.03: own 1.000, open 1.000; c=0.1: own 1.000, open 1.000; c=0.3: own 1.000, open 1.000 |
| R4 carry, intermediate c: entropy split ≥ +0.05, movement split ≥ +0.02 | **no** | c=0.003: entropy 0.000, movement 0.000 |
| R5 carry, intermediate c: read rate at u=2–4 exceeds u≥16 by ≥ 0.2 | **no** | c=0.003: 0.500 vs 0.500 |
| R6 plain: prior weight under forced-open ≤ 0.4 at every c | yes | c=0.0 0.021; c=0.003 0.049; c=0.01 0.026; c=0.03 0.190; c=0.1 0.176; c=0.3 0.145 |
| R7 plain, read rate ≤ 0.1: KL under own gates ≥ 0.03 | **no** | c=0.1: 0.0215; c=0.3: 0.0214 |
| R8 c*_carry < c*_plain; plain read rate ≥ 0.5 at c=0.01 | **no** | c* carry 0.003, plain 0.003; plain rate at 0.01: 0.250 |

4 of 8 held.
