# TASK15 tables (round 16: inducing recurrence by K/V dropout)

Prior weight = calibrated λ of R2 (position t's exports from B, A's tokens) at the output of t+1; R3 = older positions from B, t from A; 'recompute' = round 15's exact pure-recomputation prediction for R3. Full = full attention at test; drop = the model's own training dropout at test (query t+1 always sees t). 3 seeds, seed means.

## carry

| p | KL full | KL drop | prior t=4 (full) | prior t=8 (full) | prior t=16 (full) | prior t=23 (full) | prior t=4 (drop) | prior t=8 (drop) | prior t=16 (drop) | prior t=23 (drop) | R3 t=16 (full) | recompute t=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0060 | 0.0060 | 0.469 | 0.392 | 0.361 | 0.339 | 0.469 | 0.392 | 0.361 | 0.339 | 0.574 | 0.861 |
| 0.5 | 0.0028 | 0.0027 | 0.965 | 0.924 | 0.900 | 0.885 | 0.975 | 0.938 | 0.909 | 0.893 | 0.072 | 0.861 |
| 0.75 | 0.0022 | 0.0022 | 0.981 | 0.964 | 0.951 | 0.943 | 0.993 | 0.978 | 0.960 | 0.952 | 0.037 | 0.861 |
| 0.9 | 0.0021 | 0.0020 | 0.988 | 0.979 | 0.975 | 0.969 | 0.998 | 0.994 | 0.987 | 0.980 | 0.020 | 0.861 |
| 0.97 | 0.0021 | 0.0021 | 0.996 | 0.995 | 0.995 | 0.993 | 1.000 | 0.999 | 0.998 | 0.998 | 0.005 | 0.861 |
| 1.0 | 0.0033 | 0.0023 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | -0.000 | 0.861 |

## plain

| p | KL full | KL drop | prior t=4 (full) | prior t=8 (full) | prior t=16 (full) | prior t=23 (full) | prior t=4 (drop) | prior t=8 (drop) | prior t=16 (drop) | prior t=23 (drop) | R3 t=16 (full) | recompute t=16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0079 | 0.0079 | 0.010 | 0.012 | 0.027 | 0.042 | 0.010 | 0.012 | 0.027 | 0.042 | 0.886 | 0.861 |
| 0.5 | 0.0035 | 0.0037 | 0.351 | 0.277 | 0.245 | 0.209 | 0.457 | 0.391 | 0.335 | 0.312 | 0.500 | 0.861 |
| 0.75 | 0.0049 | 0.0046 | 0.351 | 0.293 | 0.250 | 0.214 | 0.565 | 0.509 | 0.461 | 0.449 | 0.526 | 0.861 |
| 0.9 | 0.0098 | 0.0079 | 0.337 | 0.288 | 0.243 | 0.206 | 0.625 | 0.620 | 0.574 | 0.573 | 0.565 | 0.861 |
| 0.97 | 0.0245 | 0.0143 | 0.333 | 0.254 | 0.236 | 0.193 | 0.665 | 0.665 | 0.663 | 0.676 | 0.651 | 0.861 |
| 1.0 | 0.0515 | 0.0228 | 0.362 | 0.254 | 0.140 | 0.108 | 0.623 | 0.667 | 0.621 | 0.608 | 0.892 | 0.861 |

## Follow-up (not pre-registered): the finer dropout grid, merged

Carry p ∈ {0.05, 0.1, 0.2, 0.3} and plain p ∈ {0.1, 0.25} were added after the main grid showed the carry transition lies between p = 0 and 0.5 (`rounds/r16_kv_dropout/fine/`). Prior weight at t = 16.

| family | p | KL full | KL drop | prior (full) | prior (drop) | R3 (full) |
|---|---|---|---|---|---|---|
| carry | 0.0 | 0.0060 | 0.0060 | 0.361 | 0.361 | 0.574 |
| carry | 0.05 | 0.0051 | 0.0052 | 0.651 | 0.653 | 0.275 |
| carry | 0.1 | 0.0043 | 0.0044 | 0.729 | 0.732 | 0.210 |
| carry | 0.2 | 0.0039 | 0.0039 | 0.795 | 0.800 | 0.156 |
| carry | 0.3 | 0.0034 | 0.0034 | 0.840 | 0.846 | 0.118 |
| carry | 0.5 | 0.0028 | 0.0027 | 0.900 | 0.909 | 0.072 |
| carry | 0.75 | 0.0022 | 0.0022 | 0.951 | 0.960 | 0.037 |
| carry | 0.9 | 0.0021 | 0.0020 | 0.975 | 0.987 | 0.020 |
| carry | 0.97 | 0.0021 | 0.0021 | 0.995 | 0.998 | 0.005 |
| carry | 1.0 | 0.0033 | 0.0023 | 1.000 | 1.000 | -0.000 |
| plain | 0.0 | 0.0079 | 0.0079 | 0.027 | 0.027 | 0.886 |
| plain | 0.1 | 0.0033 | 0.0039 | 0.180 | 0.189 | 0.683 |
| plain | 0.25 | 0.0033 | 0.0038 | 0.228 | 0.260 | 0.536 |
| plain | 0.5 | 0.0035 | 0.0037 | 0.245 | 0.335 | 0.500 |
| plain | 0.75 | 0.0049 | 0.0046 | 0.250 | 0.461 | 0.526 |
| plain | 0.9 | 0.0098 | 0.0079 | 0.243 | 0.574 | 0.565 |
| plain | 0.97 | 0.0245 | 0.0143 | 0.236 | 0.663 | 0.651 |
| plain | 1.0 | 0.0515 | 0.0228 | 0.140 | 0.621 | 0.892 |

## Pre-registered predictions (rounds/r16_kv_dropout/THEORY.md §4)

| prediction | held | numbers |
|---|---|---|
| I1 carry continuum: ≤ 0.1 at p=0, ≥ 0.95 at p=1, non-decreasing, ≥ 2 intermediate | **no** | p=0.0: 0.361 → p=0.5: 0.900 → p=0.75: 0.951 → p=0.9: 0.975 → p=0.97: 0.995 → p=1.0: 1.000 |
| I2 carry: KL < 0.01 under own dropout at every p | yes | p=0.0 0.0060; p=0.5 0.0027; p=0.75 0.0022; p=0.9 0.0020; p=0.97 0.0021; p=1.0 0.0023 |
| I3 carry p=0.9: prior weight under full access ≥ 0.3 | yes | 0.975 |
| I4 dropout at test shifts weight to the prior | yes | carry p=0.5: drop 0.909 vs full 0.900; carry p=0.75: drop 0.960 vs full 0.951; carry p=0.9: drop 0.987 vs full 0.975; carry p=0.97: drop 0.998 vs full 0.995; plain p=0.5: drop 0.335 vs full 0.245; plain p=0.75: drop 0.461 vs full 0.250; plain p=0.9: drop 0.574 vs full 0.243; plain p=0.97: drop 0.663 vs full 0.236 |
| I5 plain: rises from ≤ 0.05 at p=0, > 0.2 at p=0.9 | yes | p=0.0: 0.027 → p=0.5: 0.245 → p=0.75: 0.250 → p=0.9: 0.243 → p=0.97: 0.236 → p=1.0: 0.140 |
| I6 plain p=1: KL under own dropout > 0.05 | **no** | 0.0228 |
| I7 carry p ≥ 0.75: R3 ≤ recompute − 0.5 × prior weight | yes | p=0.75: R3 0.037 vs bound 0.386; p=0.9: R3 0.020 vs bound 0.374; p=0.97: R3 0.005 vs bound 0.364; p=1.0: R3 -0.000 vs bound 0.361 |

5 of 7 held.
