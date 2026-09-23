# TASK12 tables (round 13: the full filter state)

Channel environment, K = 4, GRU unless stated, seed means over converged models (`act_hard` models did not converge in round 12 and are shown in brackets). Full state = goal block y (3 log-odds) + channel block l (logit P(on | g), 4).

## 1. Recoverability of the full state

| model | goal IID | channel IID | channel TIME | channel EXT | goal TIME | goal EXT |
|---|---|---|---|---|---|---|
| GRU goal | 0.996 | 0.997 | 0.988 | 0.984 | 0.992 | 0.994 |
| GRU act_soft | 0.964 | 0.988 | 0.975 | 0.948 | 0.928 | 0.940 |
| GRU next_obs | 0.973 | 0.992 | 0.982 | 0.959 | 0.935 | 0.948 |
| GRU act_hard | [0.893] | [0.965] | [0.935] | [0.871] | [0.827] | [0.829] |
| GRU init | 0.701 | 0.978 | 0.970 | 0.963 | 0.611 | 0.673 |

Baselines: channel block from the true goal block 0.504; from the true marginals (y, logit P(on)) 0.559; full state from the token counts: goal 0.869, channel 0.623.

| transformer | site | goal IID | channel IID | channel TIME | channel EXT |
|---|---|---|---|---|---|
| goal | res1 | 0.901 | 0.986 | 0.933 | 0.964 |
| goal | res2 | 0.992 | 0.984 | 0.945 | 0.952 |
| goal | u | 0.992 | 0.979 | 0.892 | 0.951 |
| act_soft | res1 | 0.928 | 0.982 | 0.884 | 0.944 |
| act_soft | res2 | 0.965 | 0.975 | 0.912 | 0.928 |
| act_soft | u | 0.920 | 0.955 | 0.877 | 0.832 |
| next_obs | res1 | 0.930 | 0.988 | 0.935 | 0.969 |
| next_obs | res2 | 0.960 | 0.986 | 0.941 | 0.949 |
| next_obs | u | 0.935 | 0.977 | 0.917 | 0.901 |

## 2. Matched interventions in both coordinates (PROBE-E on the 7 full-state coordinates)

F = fraction of the gap closed toward the intervention's own Bayes future (k ≥ 1). 'vs genuine-B' = toward the model's own run on B's history. Round 12's goal-only probe closed 0.950 (t = 6) and 0.968 (t = 12) toward genuine-B.

| objective | t | G-only | R-only | both | swap | random | both vs genuine-B | swap vs genuine-B | gap G-only | gap R-only | gap both |
|---|---|---|---|---|---|---|---|---|---|---|---|
| goal | 6 | 0.962 | 0.544 | 0.994 | 1.000 | -0.122 | 0.991 | 1.000 | 1.0079 | 0.0361 | 0.9022 |
| goal | 12 | 0.965 | 0.629 | 0.993 | 1.000 | -0.053 | 0.988 | 1.000 | 1.8597 | 0.0253 | 1.8451 |
| act_soft | 6 | 0.674 | -0.835 | 0.921 | 0.999 | -0.009 | 0.861 | 1.000 | 2.4760 | 0.0916 | 2.1784 |
| act_soft | 12 | 0.772 | -1.088 | 0.925 | 0.999 | 0.009 | 0.854 | 1.000 | 3.5809 | 0.0587 | 3.4772 |
| next_obs | 6 | 0.558 | -0.333 | 0.925 | 0.998 | -0.045 | 0.925 | 1.000 | 0.0345 | 0.0078 | 0.0543 |
| next_obs | 12 | 0.620 | 0.067 | 0.921 | 0.998 | -0.021 | 0.923 | 1.000 | 0.0541 | 0.0108 | 0.0866 |
| [act_hard] | 6 | 0.363 | -1.503 | 0.786 | 0.991 | 0.034 | 0.613 | 1.000 | 14.0304 | 1.2086 | 12.9655 |
| [act_hard] | 12 | 0.513 | -1.820 | 0.825 | 0.992 | 0.040 | 0.660 | 1.000 | 17.0308 | 0.7642 | 16.9463 |

## 3. The equivalence hierarchy

Model divergence = JS between the model's outputs after the two histories (k ≥ 1, 8 continuations). Random = same-length random pairs (cross-length random pairs for Ex).

| objective | pair set | model | Bayes | model ÷ Bayes | model ÷ random | Spearman(model, Bayes) |
|---|---|---|---|---|---|---|
| goal | equal b | 0.004416 | 0.004365 | 1.01 | 0.02334 | 1.00 |
| goal | equal marginals (b, P(on)) | 0.000860 | 0.000851 | 1.01 | 0.00454 | 1.00 |
| goal | equal full state | 0.000000 | 0.000000 | 8.14 | 0.00000 | 0.17 |
| goal | equal full state, ≥ 3 positions differ beyond neutral relabel | 0.000001 | 0.000000 | 1.92 | 0.00000 | 0.74 |
| goal | equal full state, lengths 8 vs 16 | 0.000051 | 0.000045 | 1.13 | 0.00022 | 0.88 |
| act_soft | equal b | 0.014333 | 0.014416 | 0.99 | 0.04746 | 0.99 |
| act_soft | equal marginals (b, P(on)) | 0.002234 | 0.002213 | 1.01 | 0.00740 | 0.99 |
| act_soft | equal full state | 0.000000 | 0.000000 | 13.00 | 0.00000 | 0.41 |
| act_soft | equal full state, ≥ 3 positions differ beyond neutral relabel | 0.000002 | 0.000001 | 3.36 | 0.00001 | 0.27 |
| act_soft | equal full state, lengths 8 vs 16 | 0.000137 | 0.000099 | 1.38 | 0.00039 | 0.75 |
| next_obs | equal b | 0.001022 | 0.001026 | 1.00 | 0.06329 | 1.00 |
| next_obs | equal marginals (b, P(on)) | 0.000110 | 0.000110 | 1.00 | 0.00681 | 1.00 |
| next_obs | equal full state | 0.000000 | 0.000000 | 9.94 | 0.00000 | 0.42 |
| next_obs | equal full state, ≥ 3 positions differ beyond neutral relabel | 0.000000 | 0.000000 | 3.45 | 0.00000 | 0.59 |
| next_obs | equal full state, lengths 8 vs 16 | 0.000002 | 0.000001 | 1.35 | 0.00009 | 0.75 |
| act_hard | equal b | 0.059538 | 0.069430 | 0.86 | 0.15201 | 0.95 |
| act_hard | equal marginals (b, P(on)) | 0.012447 | 0.016404 | 0.76 | 0.03179 | 0.79 |
| act_hard | equal full state | 0.000018 | 0.000000 | 17983998.43 | 0.00005 | — |
| act_hard | equal full state, ≥ 3 positions differ beyond neutral relabel | 0.000072 | 0.000043 | 1.65 | 0.00018 | 0.13 |
| act_hard | equal full state, lengths 8 vs 16 | 0.002879 | 0.003990 | 0.72 | 0.00645 | 0.31 |

## 4. Bottleneck (goal objective)

| hidden | KL iid | KL channel | goal R² iid | goal R² channel | channel R² channel | Ej model ÷ Bayes | Ex model ÷ Bayes | Ex model ÷ random |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.0275 | 0.0913 | 0.936 | 0.636 | 0.439 | 3060.6 | 175.8 | 0.03890 |
| 3 | 0.0001 | 0.0314 | 0.999 | 0.913 | 0.563 | 5381.1 | 104.9 | 0.02111 |
| 4 | 0.0001 | 0.0128 | 0.999 | 0.938 | 0.671 | 448.3 | 31.6 | 0.00635 |
| 5 | 0.0000 | 0.0082 | 0.999 | 0.941 | 0.730 | 593.9 | 24.6 | 0.00495 |
| 6 | 0.0000 | 0.0052 | 0.999 | 0.959 | 0.847 | 301.4 | 21.4 | 0.00426 |
| 7 | 0.0000 | 0.0016 | 0.999 | 0.982 | 0.944 | 71.4 | 10.1 | 0.00200 |
| 8 | 0.0000 | 0.0012 | 0.999 | 0.983 | 0.956 | 27.2 | 7.8 | 0.00156 |
| 12 | 0.0000 | 0.0003 | 1.000 | 0.989 | 0.965 | 10.4 | 2.4 | 0.00048 |
| 16 | 0.0000 | 0.0002 | 1.000 | 0.991 | 0.981 | 9.3 | 2.0 | 0.00041 |
| 32 | 0.0000 | 0.0001 | 1.000 | 0.994 | 0.994 | 5.0 | 1.2 | 0.00023 |
| 64 | — | 0.0001 | — | 0.996 | 0.997 | 8.1 | 1.1 | 0.00022 |

## 5. Pre-registered predictions (rounds/r13_filter_state/THEORY.md §5)

| prediction | held | numbers |
|---|---|---|
| F1 channel IID ≥ 0.85, goal ≥ 0.98 (goal, next_obs); > marginals baseline; untrained channel ≤ 0.5 | **no** | goal: channel 0.997, goal 0.996; next_obs: channel 0.992, goal 0.973; marginals baseline 0.559; untrained channel 0.978 |
| F2 goal: channel TIME ≥ 0.8, EXT ≥ 0.6 | yes | TIME 0.988, EXT 0.984 |
| F3 goal: G-only, R-only, both ≥ 0.9 of own Bayes gap; random ≤ 0.1 | **no** | g_only t=6 0.962; g_only t=12 0.965; r_only t=6 0.544; r_only t=12 0.629; both t=6 0.994; both t=12 0.993; rand t=6 -0.122; rand t=12 -0.053 |
| F4 goal: both ≥ 0.97 toward genuine-B and above round 12's goal-only probe | yes | t=6: 0.991 vs goal-only 0.950; t=12: 0.988 vs goal-only 0.968 |
| F5 goal: Eb, Em model ÷ Bayes in [0.5, 2] (Spearman ≥ 0.5); Ej ≤ 0.002 of random and ≤ 3× Bayes | **no** | Eb 1.01 (ρ 1.00); Em 1.01 (ρ 1.00); Ej ÷ random 0.00000, ÷ Bayes 8.14 |
| F6 goal cross-length equal state: ≤ 0.01 of random, ≤ 3× Bayes (tolerance 0.05, see report) | yes | ÷ random 0.00022, ÷ Bayes 1.13 |
| F7 Ej and Ex also for next_obs, act_soft | **no** | next_obs Ej: ÷ random 0.00000, ÷ Bayes 9.94; next_obs Ex: ÷ random 0.00009, ÷ Bayes 1.35; act_soft Ej: ÷ random 0.00000, ÷ Bayes 13.00; act_soft Ex: ÷ random 0.00039, ÷ Bayes 1.38 |
| F8 KL < 0.01 at n ≤ 4 (iid), needs n ≥ 6 (channel) | **no** | smallest n with KL < 0.01: iid 3, channel 5 |
| F9 channel n ∈ {3, 4, 5}: goal R² ≥ 0.9, channel R² ≤ 0.6 | **no** | n=3: goal 0.913, channel 0.563; n=4: goal 0.938, channel 0.671; n=5: goal 0.941, channel 0.730 |
| F10 transformer goal: channel R² at res2 ≥ at u | yes | res2 0.984, u 0.979 |

4 of 10 held.
