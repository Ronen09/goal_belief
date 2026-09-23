# TASK13 tables (round 14: windowed transformers)

Window l = attention to the last l positions (itself included) in both layers; carry = the previous position's readout interface u added to the input. Objective `goal`, K = 4, 3 seeds, seed means. F = gap to the genuine-B future closed by a single-site edit at position t, pooled over future offsets k ≥ 1; 'no gap' where the unedited future already equals B's.

## 1. Accuracy and recoverability

| env | l | carry | converged (KL < 0.01) | KL | KL at positions 1–7 | KL at positions 17–24 | goal R² from u | channel R² from u |
|---|---|---|---|---|---|---|---|---|
| iid | 1 | no | 0/3 | 0.4304 | 0.1385 | 0.6746 | 0.109 | — |
| iid | 1 | yes | 3/3 | 0.0001 | 0.0000 | 0.0003 | 0.999 | — |
| iid | 2 | no | 0/3 | 0.3400 | 0.0628 | 0.5793 | 0.297 | — |
| iid | 2 | yes | 3/3 | 0.0001 | 0.0000 | 0.0002 | 0.999 | — |
| iid | 4 | no | 0/3 | 0.2038 | 0.0018 | 0.4117 | 0.576 | — |
| iid | 4 | yes | 3/3 | 0.0001 | 0.0000 | 0.0001 | 0.999 | — |
| iid | 8 | no | 0/3 | 0.0487 | 0.0003 | 0.1404 | 0.893 | — |
| iid | 8 | yes | 3/3 | 0.0001 | 0.0000 | 0.0001 | 0.999 | — |
| iid | 25 | no | 3/3 | 0.0001 | 0.0000 | 0.0001 | 1.000 | — |
| iid | 25 | yes | 3/3 | 0.0001 | 0.0000 | 0.0001 | 1.000 | — |
| channel | 1 | no | 0/3 | 0.4514 | 0.1853 | 0.6575 | 0.135 | 0.423 |
| channel | 1 | yes | 3/3 | 0.0002 | 0.0001 | 0.0004 | 0.987 | 0.992 |
| channel | 2 | no | 0/3 | 0.3166 | 0.0729 | 0.5136 | 0.367 | 0.805 |
| channel | 2 | yes | 3/3 | 0.0003 | 0.0001 | 0.0005 | 0.985 | 0.992 |
| channel | 4 | no | 0/3 | 0.1645 | 0.0021 | 0.3263 | 0.652 | 0.948 |
| channel | 4 | yes | 3/3 | 0.0002 | 0.0001 | 0.0003 | 0.986 | 0.991 |
| channel | 8 | no | 0/3 | 0.0388 | 0.0005 | 0.1095 | 0.912 | 0.955 |
| channel | 8 | yes | 3/3 | 0.0001 | 0.0000 | 0.0002 | 0.988 | 0.991 |
| channel | 25 | no | 3/3 | 0.0002 | 0.0001 | 0.0003 | 0.992 | 0.982 |
| channel | 25 | yes | 3/3 | 0.0001 | 0.0000 | 0.0002 | 0.990 | 0.991 |

## 2. Steerability: how much of the future one edit at position t controls

| env | l | carry | SWAP u_t, t=6 | t=12 | PROBE-E u_t, t=6 | t=12 | SWAP res1(t), t=6 | t=12 | PROBE-E u_t, immediate (k=0), t=6 |
|---|---|---|---|---|---|---|---|---|---|
| iid | 1 | no | no gap | no gap | no gap | no gap | no gap | no gap | -0.270 |
| iid | 1 | yes | 1.000 | 1.000 | 0.996 | 0.996 | 1.000 | 1.000 | 0.999 |
| iid | 2 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.311 | 0.291 | 0.739 |
| iid | 2 | yes | 0.806 | 0.832 | 0.731 | 0.745 | 0.834 | 0.850 | 0.999 |
| iid | 4 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.315 | 0.298 | 0.983 |
| iid | 4 | yes | 0.464 | 0.426 | 0.414 | 0.381 | 0.527 | 0.481 | 1.000 |
| iid | 8 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.159 | 0.147 | 0.997 |
| iid | 8 | yes | 0.288 | 0.245 | 0.256 | 0.227 | 0.363 | 0.293 | 1.000 |
| iid | 25 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.086 | 0.050 | 1.000 |
| iid | 25 | yes | -0.005 | 0.002 | -0.009 | -0.001 | 0.094 | 0.058 | 1.000 |
| channel | 1 | no | no gap | no gap | no gap | no gap | no gap | no gap | -0.254 |
| channel | 1 | yes | 1.000 | 1.000 | 0.961 | 0.924 | 1.000 | 1.000 | 0.997 |
| channel | 2 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.360 | 0.332 | 0.852 |
| channel | 2 | yes | 0.837 | 0.844 | 0.699 | 0.614 | 0.844 | 0.855 | 0.996 |
| channel | 4 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.268 | 0.266 | 0.977 |
| channel | 4 | yes | 0.423 | 0.350 | 0.360 | 0.263 | 0.490 | 0.403 | 0.996 |
| channel | 8 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.135 | 0.104 | 0.995 |
| channel | 8 | yes | 0.265 | 0.183 | 0.221 | 0.147 | 0.340 | 0.230 | 0.996 |
| channel | 25 | no | 0.000 | 0.000 | 0.000 | 0.000 | 0.173 | 0.076 | 0.998 |
| channel | 25 | yes | 0.174 | 0.089 | 0.135 | 0.067 | 0.340 | 0.195 | 0.998 |

Reference: round-12 GRU (complete cut) SWAP 1.000; round-12 full-attention transformer, SWAP res1(t) ≤ 0.16.

## 3. Equal full state at lengths 8 vs 16 (channel)

| l | carry | model divergence | Bayes divergence | model ÷ Bayes | model ÷ cross-length random |
|---|---|---|---|---|---|
| 1 | no | 0.000000 | 0.000045 | 0.00 | — |
| 1 | yes | 0.000073 | 0.000045 | 1.63 | 0.00032 |
| 2 | no | 0.001108 | 0.000045 | 24.57 | 0.09683 |
| 2 | yes | 0.000072 | 0.000045 | 1.60 | 0.00032 |
| 4 | no | 0.004723 | 0.000045 | 104.75 | 0.07390 |
| 4 | yes | 0.000067 | 0.000045 | 1.49 | 0.00030 |
| 8 | no | 0.004671 | 0.000045 | 103.60 | 0.02502 |
| 8 | yes | 0.000063 | 0.000045 | 1.41 | 0.00028 |
| 25 | no | 0.000097 | 0.000045 | 2.15 | 0.00043 |
| 25 | yes | 0.000055 | 0.000045 | 1.23 | 0.00024 |

## 4. Pre-registered predictions (docs/task13_window_theory.md §4)

| prediction | held | numbers |
|---|---|---|
| W1 no carry: l ≤ 8 fail (KL ≥ 0.01), full window converges | yes | iid l=1 0.4304; iid l=2 0.3400; iid l=4 0.2038; iid l=8 0.0487; iid l=25 0.0001; channel l=1 0.4514; channel l=2 0.3166; channel l=4 0.1645; channel l=8 0.0388; channel l=25 0.0002 |
| W2 carry: every window converges | yes | iid l=1 0.0001; iid l=2 0.0001; iid l=4 0.0001; iid l=8 0.0001; iid l=25 0.0001; channel l=1 0.0002; channel l=2 0.0003; channel l=4 0.0002; channel l=8 0.0001; channel l=25 0.0001 |
| W3 carry l = 1: SWAP u_t ≥ 0.95, PROBE-E ≥ 0.9 | yes | iid t=6: SWAP 1.000, PROBE-E 0.996; iid t=12: SWAP 1.000, PROBE-E 0.996; channel t=6: SWAP 1.000, PROBE-E 0.961; channel t=12: SWAP 1.000, PROBE-E 0.924 |
| W4 carry: SWAP u_t decreases with l, ≤ 0.5 at l = 25 | yes | iid t=6: 1.000 > 0.806 > 0.464 > 0.288 > -0.005; iid t=12: 1.000 > 0.832 > 0.426 > 0.245 > 0.002; channel t=6: 1.000 > 0.837 > 0.423 > 0.265 > 0.174; channel t=12: 1.000 > 0.844 > 0.350 > 0.183 > 0.089 |
| W5 no carry: u_t edit moves nothing; SWAP res1 ≤ 0.3 (converged models) | yes | largest change from the u_t edit 0.0e+00; SWAP res1: iid l=25 t=6 0.086; iid l=25 t=12 0.050; channel l=25 t=6 0.173; channel l=25 t=12 0.076 |
| W6 carry l = 1 channel: cross-length equal state ≤ 3× Bayes | yes | 1.63× |
| W7 carry l = 1 channel: goal R² ≥ 0.98, channel ≥ 0.9 from u | yes | goal 0.987, channel 0.992 |

7 of 7 held.
