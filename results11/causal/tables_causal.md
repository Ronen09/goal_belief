# TASK11 part 2 tables: causal tests of the decoded posterior

Seed means over converged models (channel `act_hard` models, which did not converge, are shown in brackets from all seeds). 'fut' = mean over future offsets k ≥ 1; 'k0' = the output at the intervention position. F = fraction of the gap closed (1 − E[div(intervened, ref)] / E[div(unintervened, ref)]).

## A. Posterior transplant

Reference 'model B' = the model's own run on B's history followed by A's continuation. Bayes S = genuine-B Bayes target; Bayes T = transplant target (goal belief moved, channel belief given the goal kept).

### t = 6

| arch | site | env | objective | intervention | F model B, k0 | F model B, fut | F Bayes T, fut | KL to Bayes S, fut | KL to Bayes T, fut | F Bayes T½, fut |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | h | iid | goal | swap | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 |  |
| gru | h | iid | goal | probe_e | 1.000 | 1.000 | 1.000 | 0.0002 | 0.0002 |  |
| gru | h | iid | goal | probe_d | 0.839 | 0.778 | 0.814 | 0.1165 | 0.1165 |  |
| gru | h | iid | goal | probe_e_half | 0.719 | 0.683 | 0.735 | 0.1673 | 0.1673 | 0.999 |
| gru | h | iid | goal | rand | -0.017 | -0.023 | -0.045 | 0.6636 | 0.6636 |  |
| gru | h | iid | act_soft | swap | 1.000 | 1.000 | 1.000 | 0.0002 | 0.0002 |  |
| gru | h | iid | act_soft | probe_e | 0.776 | 0.890 | 0.934 | 0.1214 | 0.1214 |  |
| gru | h | iid | act_soft | probe_d | -0.034 | 0.138 | 0.191 | 1.4765 | 1.4765 |  |
| gru | h | iid | act_soft | probe_e_half | 0.374 | 0.442 | 0.566 | 0.7812 | 0.7812 | 0.921 |
| gru | h | iid | act_soft | rand | -0.003 | -0.005 | -0.021 | 1.8416 | 1.8416 |  |
| gru | h | iid | act_hard | swap | 1.000 | 1.000 | 0.999 | 0.0117 | 0.0117 |  |
| gru | h | iid | act_hard | probe_e | 0.366 | 0.613 | 0.775 | 3.0982 | 3.0982 |  |
| gru | h | iid | act_hard | probe_d | 0.002 | 0.147 | 0.203 | 11.0889 | 11.0889 |  |
| gru | h | iid | act_hard | probe_e_half | 0.069 | 0.241 | 0.338 | 9.1476 | 9.1476 | 0.778 |
| gru | h | iid | act_hard | rand | 0.012 | 0.005 | 0.011 | 13.5906 | 13.5906 |  |
| gru | h | iid | next_obs | swap | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 |  |
| gru | h | iid | next_obs | probe_e | 0.872 | 0.952 | 0.952 | 0.0011 | 0.0011 |  |
| gru | h | iid | next_obs | probe_d | -0.126 | 0.093 | 0.093 | 0.0196 | 0.0196 |  |
| gru | h | iid | next_obs | probe_e_half | 0.555 | 0.589 | 0.591 | 0.0087 | 0.0087 | 0.943 |
| gru | h | iid | next_obs | rand | -0.118 | -0.033 | -0.034 | 0.0220 | 0.0220 |  |
| gru | h | channel | goal | swap | 1.000 | 1.000 | 0.948 | 0.0002 | 0.0507 |  |
| gru | h | channel | goal | probe_e | 0.998 | 0.950 | 0.977 | 0.0299 | 0.0213 |  |
| gru | h | channel | goal | probe_d | 0.537 | 0.460 | 0.534 | 0.4245 | 0.4656 |  |
| gru | h | channel | goal | probe_e_half | 0.677 | 0.585 | 0.652 | 0.3064 | 0.3446 | 0.956 |
| gru | h | channel | goal | rand | -0.000 | -0.017 | -0.037 | 0.9545 | 1.0444 |  |
| gru | h | channel | act_soft | swap | 1.000 | 1.000 | 0.935 | 0.0024 | 0.1494 |  |
| gru | h | channel | act_soft | probe_e | 0.752 | 0.727 | 0.788 | 0.3943 | 0.4993 |  |
| gru | h | channel | act_soft | probe_d | -0.008 | 0.040 | 0.055 | 2.0423 | 2.3048 |  |
| gru | h | channel | act_soft | probe_e_half | 0.315 | 0.305 | 0.413 | 1.2130 | 1.4071 | 0.701 |
| gru | h | channel | act_soft | rand | 0.011 | 0.007 | 0.006 | 2.1488 | 2.4168 |  |
| gru | h | channel | act_hard | swap | [1.000] | [1.000] | [0.873] | [0.1030] | [1.6486] |  |
| gru | h | channel | act_hard | probe_e | [0.374] | [0.421] | [0.552] | [4.8514] | [6.0580] |  |
| gru | h | channel | act_hard | probe_d | [0.002] | [0.044] | [0.066] | [11.8994] | [13.0414] |  |
| gru | h | channel | act_hard | probe_e_half | [0.055] | [0.139] | [0.220] | [9.5730] | [10.7543] | 0.428 |
| gru | h | channel | act_hard | rand | [0.003] | [0.006] | [0.020] | [12.5679] | [13.6553] |  |
| gru | h | channel | next_obs | swap | 1.000 | 1.000 | 0.806 | 0.0000 | 0.0090 |  |
| gru | h | channel | next_obs | probe_e | 0.719 | 0.733 | 0.761 | 0.0146 | 0.0075 |  |
| gru | h | channel | next_obs | probe_d | -0.033 | 0.041 | 0.045 | 0.0525 | 0.0323 |  |
| gru | h | channel | next_obs | probe_e_half | 0.415 | 0.361 | 0.399 | 0.0340 | 0.0193 | 0.764 |
| gru | h | channel | next_obs | rand | -0.023 | -0.008 | -0.010 | 0.0543 | 0.0340 |  |

| tfm | res1 | iid | goal | swap | 0.989 | 0.031 | 0.039 | 0.6105 | 0.6105 |  |
| tfm | res1 | iid | goal | probe_e | 0.719 | 0.014 | 0.018 | 0.6240 | 0.6240 |  |
| tfm | res1 | iid | goal | probe_d | 0.004 | -0.000 | -0.000 | 0.6358 | 0.6358 |  |
| tfm | res1 | iid | goal | probe_e_half | 0.379 | 0.007 | 0.009 | 0.6301 | 0.6301 | 0.017 |
| tfm | res1 | iid | goal | rand | 0.003 | 0.001 | 0.001 | 0.6354 | 0.6354 |  |
| tfm | res1 | iid | act_soft | swap | 0.995 | 0.010 | 0.015 | 1.7731 | 1.7731 |  |
| tfm | res1 | iid | act_soft | probe_e | 0.598 | 0.004 | 0.008 | 1.7868 | 1.7868 |  |
| tfm | res1 | iid | act_soft | probe_d | 0.001 | 0.000 | 0.000 | 1.8016 | 1.8016 |  |
| tfm | res1 | iid | act_soft | probe_e_half | 0.274 | 0.002 | 0.004 | 1.7936 | 1.7936 | 0.008 |
| tfm | res1 | iid | act_soft | rand | 0.021 | 0.000 | 0.000 | 1.8012 | 1.8012 |  |
| tfm | res1 | iid | act_hard | swap | 0.928 | 0.000 | 0.003 | 10.7964 | 10.7964 |  |
| tfm | res1 | iid | act_hard | probe_e | 0.257 | 0.000 | 0.002 | 10.8052 | 10.8052 |  |
| tfm | res1 | iid | act_hard | probe_d | 0.000 | -0.000 | -0.000 | 10.8311 | 10.8311 |  |
| tfm | res1 | iid | act_hard | probe_e_half | 0.080 | 0.000 | 0.001 | 10.8171 | 10.8171 | 0.002 |
| tfm | res1 | iid | act_hard | rand | 0.014 | 0.000 | 0.001 | 10.8182 | 10.8182 |  |
| tfm | res1 | iid | next_obs | swap | 0.756 | 0.075 | 0.075 | 0.0196 | 0.0196 |  |
| tfm | res1 | iid | next_obs | probe_e | 0.568 | 0.023 | 0.024 | 0.0208 | 0.0208 |  |
| tfm | res1 | iid | next_obs | probe_d | 0.025 | -0.000 | -0.000 | 0.0213 | 0.0213 |  |
| tfm | res1 | iid | next_obs | probe_e_half | 0.292 | 0.011 | 0.011 | 0.0211 | 0.0211 | 0.021 |
| tfm | res1 | iid | next_obs | rand | -0.009 | 0.004 | 0.004 | 0.0212 | 0.0212 |  |
| tfm | res1 | channel | goal | swap | 0.791 | 0.156 | 0.197 | 0.7340 | 0.8057 |  |
| tfm | res1 | channel | goal | probe_e | 0.451 | 0.054 | 0.081 | 0.8380 | 0.9228 |  |
| tfm | res1 | channel | goal | probe_d | 0.006 | 0.000 | 0.000 | 0.9139 | 1.0041 |  |
| tfm | res1 | channel | goal | probe_e_half | 0.219 | 0.026 | 0.038 | 0.8776 | 0.9656 | 0.071 |
| tfm | res1 | channel | goal | rand | 0.002 | 0.003 | 0.006 | 0.9089 | 0.9990 |  |
| tfm | res1 | channel | act_soft | swap | 0.819 | 0.108 | 0.128 | 1.8566 | 2.0972 |  |
| tfm | res1 | channel | act_soft | probe_e | 0.310 | 0.031 | 0.043 | 2.0352 | 2.2966 |  |
| tfm | res1 | channel | act_soft | probe_d | 0.001 | 0.000 | 0.001 | 2.1309 | 2.3999 |  |
| tfm | res1 | channel | act_soft | probe_e_half | 0.107 | 0.014 | 0.021 | 2.0856 | 2.3513 | 0.047 |
| tfm | res1 | channel | act_soft | rand | 0.009 | 0.001 | 0.002 | 2.1282 | 2.3973 |  |
| tfm | res1 | channel | act_hard | swap | [0.755] | [0.093] | [0.172] | [9.0360] | [9.8557] |  |
| tfm | res1 | channel | act_hard | probe_e | [0.212] | [0.014] | [0.041] | [10.4391] | [11.3536] |  |
| tfm | res1 | channel | act_hard | probe_d | [0.001] | [0.001] | [0.001] | [10.8898] | [11.8134] |  |
| tfm | res1 | channel | act_hard | probe_e_half | [0.040] | [0.006] | [0.016] | [10.7207] | [11.6429] | 0.032 |
| tfm | res1 | channel | act_hard | rand | [0.002] | [0.001] | [0.003] | [10.8701] | [11.7966] |  |
| tfm | res1 | channel | next_obs | swap | 0.936 | 0.118 | 0.128 | 0.0493 | 0.0298 |  |
| tfm | res1 | channel | next_obs | probe_e | 0.498 | 0.029 | 0.031 | 0.0527 | 0.0326 |  |
| tfm | res1 | channel | next_obs | probe_d | 0.011 | -0.000 | -0.001 | 0.0539 | 0.0336 |  |
| tfm | res1 | channel | next_obs | probe_e_half | 0.223 | 0.014 | 0.016 | 0.0533 | 0.0331 | 0.035 |
| tfm | res1 | channel | next_obs | rand | 0.051 | 0.004 | 0.005 | 0.0537 | 0.0334 |  |
| tfm | res2 | iid | goal | swap | 1.000 | 0.000 | 0.000 | 0.6357 | 0.6357 |  |
| tfm | res2 | iid | goal | probe_e | 0.807 | 0.000 | 0.000 | 0.6357 | 0.6357 |  |
| tfm | res2 | iid | goal | probe_d | 0.015 | 0.000 | 0.000 | 0.6357 | 0.6357 |  |
| tfm | res2 | iid | goal | probe_e_half | 0.443 | 0.000 | 0.000 | 0.6357 | 0.6357 | 0.000 |
| tfm | res2 | iid | goal | rand | 0.014 | 0.000 | 0.000 | 0.6357 | 0.6357 |  |
| tfm | res2 | iid | act_soft | swap | 1.000 | 0.000 | 0.000 | 1.8016 | 1.8016 |  |
| tfm | res2 | iid | act_soft | probe_e | 0.731 | 0.000 | 0.000 | 1.8016 | 1.8016 |  |
| tfm | res2 | iid | act_soft | probe_d | 0.002 | 0.000 | 0.000 | 1.8016 | 1.8016 |  |
| tfm | res2 | iid | act_soft | probe_e_half | 0.359 | 0.000 | 0.000 | 1.8016 | 1.8016 | 0.000 |
| tfm | res2 | iid | act_soft | rand | 0.034 | 0.000 | 0.000 | 1.8016 | 1.8016 |  |
| tfm | res2 | iid | act_hard | swap | 1.000 | 0.000 | 0.000 | 10.8310 | 10.8310 |  |
| tfm | res2 | iid | act_hard | probe_e | 0.276 | 0.000 | 0.000 | 10.8310 | 10.8310 |  |
| tfm | res2 | iid | act_hard | probe_d | 0.000 | 0.000 | 0.000 | 10.8310 | 10.8310 |  |
| tfm | res2 | iid | act_hard | probe_e_half | 0.065 | 0.000 | 0.000 | 10.8310 | 10.8310 | 0.000 |
| tfm | res2 | iid | act_hard | rand | 0.007 | 0.000 | 0.000 | 10.8310 | 10.8310 |  |
| tfm | res2 | iid | next_obs | swap | 1.000 | 0.000 | 0.000 | 0.0213 | 0.0213 |  |
| tfm | res2 | iid | next_obs | probe_e | 0.864 | 0.000 | 0.000 | 0.0213 | 0.0213 |  |
| tfm | res2 | iid | next_obs | probe_d | -0.047 | 0.000 | 0.000 | 0.0213 | 0.0213 |  |
| tfm | res2 | iid | next_obs | probe_e_half | 0.525 | 0.000 | 0.000 | 0.0213 | 0.0213 | 0.000 |
| tfm | res2 | iid | next_obs | rand | -0.141 | 0.000 | 0.000 | 0.0213 | 0.0213 |  |
| tfm | res2 | channel | goal | swap | 1.000 | 0.000 | 0.000 | 0.9143 | 1.0045 |  |
| tfm | res2 | channel | goal | probe_e | 0.860 | 0.000 | 0.000 | 0.9143 | 1.0045 |  |
| tfm | res2 | channel | goal | probe_d | 0.088 | 0.000 | 0.000 | 0.9143 | 1.0045 |  |
| tfm | res2 | channel | goal | probe_e_half | 0.443 | 0.000 | 0.000 | 0.9143 | 1.0045 | 0.000 |
| tfm | res2 | channel | goal | rand | 0.019 | 0.000 | 0.000 | 0.9143 | 1.0045 |  |
| tfm | res2 | channel | act_soft | swap | 1.000 | 0.000 | 0.000 | 2.1322 | 2.4013 |  |
| tfm | res2 | channel | act_soft | probe_e | 0.856 | 0.000 | 0.000 | 2.1322 | 2.4013 |  |
| tfm | res2 | channel | act_soft | probe_d | 0.005 | 0.000 | 0.000 | 2.1322 | 2.4013 |  |
| tfm | res2 | channel | act_soft | probe_e_half | 0.401 | 0.000 | 0.000 | 2.1322 | 2.4013 | 0.000 |
| tfm | res2 | channel | act_soft | rand | 0.047 | 0.000 | 0.000 | 2.1322 | 2.4013 |  |
| tfm | res2 | channel | act_hard | swap | [1.000] | [0.000] | [0.000] | [10.9044] | [11.8280] |  |
| tfm | res2 | channel | act_hard | probe_e | [0.493] | [0.000] | [0.000] | [10.9044] | [11.8280] |  |
| tfm | res2 | channel | act_hard | probe_d | [0.000] | [0.000] | [0.000] | [10.9044] | [11.8280] |  |
| tfm | res2 | channel | act_hard | probe_e_half | [0.128] | [0.000] | [0.000] | [10.9044] | [11.8280] | 0.000 |
| tfm | res2 | channel | act_hard | rand | [0.004] | [0.000] | [0.000] | [10.9044] | [11.8280] |  |
| tfm | res2 | channel | next_obs | swap | 1.000 | 0.000 | 0.000 | 0.0539 | 0.0336 |  |
| tfm | res2 | channel | next_obs | probe_e | 0.728 | 0.000 | 0.000 | 0.0539 | 0.0336 |  |
| tfm | res2 | channel | next_obs | probe_d | -0.004 | 0.000 | 0.000 | 0.0539 | 0.0336 |  |
| tfm | res2 | channel | next_obs | probe_e_half | 0.403 | 0.000 | 0.000 | 0.0539 | 0.0336 | 0.000 |
| tfm | res2 | channel | next_obs | rand | 0.105 | 0.000 | 0.000 | 0.0539 | 0.0336 |  |

### t = 12

| arch | site | env | objective | intervention | F model B, k0 | F model B, fut | F Bayes T, fut | KL to Bayes S, fut | KL to Bayes T, fut | F Bayes T½, fut |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | h | iid | goal | swap | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 |  |
| gru | h | iid | goal | probe_e | 1.000 | 0.999 | 1.000 | 0.0006 | 0.0006 |  |
| gru | h | iid | goal | probe_d | 0.830 | 0.765 | 0.826 | 0.2223 | 0.2223 |  |
| gru | h | iid | goal | probe_e_half | 0.697 | 0.656 | 0.746 | 0.3250 | 0.3250 | 0.999 |
| gru | h | iid | goal | rand | -0.001 | -0.005 | -0.041 | 1.3360 | 1.3360 |  |
| gru | h | iid | act_soft | swap | 1.000 | 1.000 | 1.000 | 0.0003 | 0.0003 |  |
| gru | h | iid | act_soft | probe_e | 0.795 | 0.895 | 0.947 | 0.1664 | 0.1664 |  |
| gru | h | iid | act_soft | probe_d | -0.026 | 0.094 | 0.148 | 2.6585 | 2.6585 |  |
| gru | h | iid | act_soft | probe_e_half | 0.375 | 0.412 | 0.591 | 1.2664 | 1.2664 | 0.901 |
| gru | h | iid | act_soft | rand | 0.008 | 0.011 | 0.004 | 3.0946 | 3.0946 |  |
| gru | h | iid | act_hard | swap | 1.000 | 1.000 | 0.999 | 0.0147 | 0.0147 |  |
| gru | h | iid | act_hard | probe_e | 0.589 | 0.706 | 0.846 | 2.7077 | 2.7077 |  |
| gru | h | iid | act_hard | probe_d | 0.003 | 0.112 | 0.168 | 14.7306 | 14.7306 |  |
| gru | h | iid | act_hard | probe_e_half | 0.096 | 0.236 | 0.352 | 11.3846 | 11.3846 | 0.758 |
| gru | h | iid | act_hard | rand | 0.004 | 0.001 | 0.016 | 17.3379 | 17.3379 |  |
| gru | h | iid | next_obs | swap | 1.000 | 1.000 | 1.000 | 0.0000 | 0.0000 |  |
| gru | h | iid | next_obs | probe_e | 0.889 | 0.959 | 0.959 | 0.0016 | 0.0016 |  |
| gru | h | iid | next_obs | probe_d | -0.149 | 0.046 | 0.046 | 0.0370 | 0.0370 |  |
| gru | h | iid | next_obs | probe_e_half | 0.620 | 0.609 | 0.611 | 0.0149 | 0.0149 | 0.923 |
| gru | h | iid | next_obs | rand | -0.150 | -0.033 | -0.033 | 0.0399 | 0.0399 |  |
| gru | h | channel | goal | swap | 1.000 | 1.000 | 0.981 | 0.0003 | 0.0357 |  |
| gru | h | channel | goal | probe_e | 0.998 | 0.968 | 0.986 | 0.0337 | 0.0259 |  |
| gru | h | channel | goal | probe_d | 0.497 | 0.428 | 0.550 | 0.8577 | 0.8467 |  |
| gru | h | channel | goal | probe_e_half | 0.635 | 0.546 | 0.670 | 0.6267 | 0.6188 | 0.965 |
| gru | h | channel | goal | rand | 0.005 | -0.003 | -0.029 | 1.9305 | 1.9433 |  |
| gru | h | channel | act_soft | swap | 1.000 | 1.000 | 0.970 | 0.0051 | 0.1069 |  |
| gru | h | channel | act_soft | probe_e | 0.834 | 0.806 | 0.878 | 0.3782 | 0.4359 |  |
| gru | h | channel | act_soft | probe_d | -0.008 | 0.026 | 0.044 | 3.4003 | 3.4884 |  |
| gru | h | channel | act_soft | probe_e_half | 0.430 | 0.342 | 0.527 | 1.6466 | 1.7039 | 0.750 |
| gru | h | channel | act_soft | rand | 0.021 | 0.016 | 0.011 | 3.5128 | 3.6042 |  |
| gru | h | channel | act_hard | swap | [1.000] | [1.000] | [0.930] | [0.1329] | [1.1902] |  |
| gru | h | channel | act_hard | probe_e | [0.568] | [0.597] | [0.746] | [3.8256] | [4.3229] |  |
| gru | h | channel | act_hard | probe_d | [0.004] | [0.030] | [0.076] | [15.9596] | [16.0921] |  |
| gru | h | channel | act_hard | probe_e_half | [0.175] | [0.177] | [0.345] | [11.0098] | [11.2705] | 0.527 |
| gru | h | channel | act_hard | rand | [0.003] | [0.006] | [0.032] | [16.7521] | [16.8312] |  |
| gru | h | channel | next_obs | swap | 1.000 | 1.000 | 0.804 | 0.0001 | 0.0128 |  |
| gru | h | channel | next_obs | probe_e | 0.837 | 0.839 | 0.772 | 0.0133 | 0.0129 |  |
| gru | h | channel | next_obs | probe_d | -0.049 | 0.034 | 0.038 | 0.0847 | 0.0524 |  |
| gru | h | channel | next_obs | probe_e_half | 0.574 | 0.454 | 0.509 | 0.0450 | 0.0255 | 0.656 |
| gru | h | channel | next_obs | rand | -0.055 | -0.003 | -0.016 | 0.0877 | 0.0553 |  |

| tfm | res1 | iid | goal | swap | 0.984 | 0.016 | 0.023 | 1.2551 | 1.2551 |  |
| tfm | res1 | iid | goal | probe_e | 0.934 | 0.012 | 0.019 | 1.2609 | 1.2609 |  |
| tfm | res1 | iid | goal | probe_d | 0.005 | -0.000 | -0.000 | 1.2852 | 1.2852 |  |
| tfm | res1 | iid | goal | probe_e_half | 0.531 | 0.006 | 0.009 | 1.2738 | 1.2738 | 0.018 |
| tfm | res1 | iid | goal | rand | 0.025 | 0.001 | 0.003 | 1.2817 | 1.2817 |  |
| tfm | res1 | iid | act_soft | swap | 0.992 | 0.005 | 0.008 | 3.0624 | 3.0624 |  |
| tfm | res1 | iid | act_soft | probe_e | 0.867 | 0.003 | 0.006 | 3.0678 | 3.0678 |  |
| tfm | res1 | iid | act_soft | probe_d | 0.001 | 0.000 | 0.000 | 3.0878 | 3.0878 |  |
| tfm | res1 | iid | act_soft | probe_e_half | 0.387 | 0.002 | 0.004 | 3.0764 | 3.0764 | 0.007 |
| tfm | res1 | iid | act_soft | rand | 0.044 | 0.000 | 0.000 | 3.0865 | 3.0865 |  |
| tfm | res1 | iid | act_hard | swap | 0.953 | 0.000 | 0.001 | 13.4226 | 13.4226 |  |
| tfm | res1 | iid | act_hard | probe_e | 0.579 | 0.000 | 0.001 | 13.4299 | 13.4299 |  |
| tfm | res1 | iid | act_hard | probe_d | 0.000 | -0.000 | -0.000 | 13.4398 | 13.4398 |  |
| tfm | res1 | iid | act_hard | probe_e_half | 0.151 | -0.000 | 0.001 | 13.4288 | 13.4288 | 0.001 |
| tfm | res1 | iid | act_hard | rand | 0.006 | -0.000 | 0.000 | 13.4346 | 13.4346 |  |
| tfm | res1 | iid | next_obs | swap | 0.819 | 0.027 | 0.027 | 0.0375 | 0.0375 |  |
| tfm | res1 | iid | next_obs | probe_e | 0.756 | 0.017 | 0.018 | 0.0379 | 0.0379 |  |
| tfm | res1 | iid | next_obs | probe_d | 0.037 | -0.000 | -0.000 | 0.0386 | 0.0386 |  |
| tfm | res1 | iid | next_obs | probe_e_half | 0.392 | 0.008 | 0.009 | 0.0382 | 0.0382 | 0.016 |
| tfm | res1 | iid | next_obs | rand | 0.019 | 0.004 | 0.004 | 0.0384 | 0.0384 |  |
| tfm | res1 | channel | goal | swap | 0.555 | 0.069 | 0.101 | 1.6766 | 1.6826 |  |
| tfm | res1 | channel | goal | probe_e | 0.456 | 0.041 | 0.072 | 1.7280 | 1.7383 |  |
| tfm | res1 | channel | goal | probe_d | 0.006 | 0.000 | 0.000 | 1.8586 | 1.8719 |  |
| tfm | res1 | channel | goal | probe_e_half | 0.189 | 0.018 | 0.031 | 1.8015 | 1.8138 | 0.061 |
| tfm | res1 | channel | goal | rand | 0.007 | 0.001 | 0.002 | 1.8564 | 1.8696 |  |
| tfm | res1 | channel | act_soft | swap | 0.603 | 0.058 | 0.069 | 3.2402 | 3.3225 |  |
| tfm | res1 | channel | act_soft | probe_e | 0.442 | 0.026 | 0.039 | 3.3392 | 3.4277 |  |
| tfm | res1 | channel | act_soft | probe_d | 0.001 | 0.000 | 0.000 | 3.4756 | 3.5673 |  |
| tfm | res1 | channel | act_soft | probe_e_half | 0.150 | 0.011 | 0.018 | 3.4150 | 3.5053 | 0.041 |
| tfm | res1 | channel | act_soft | rand | 0.012 | 0.000 | 0.001 | 3.4753 | 3.5671 |  |
| tfm | res1 | channel | act_hard | swap | [0.552] | [0.077] | [0.145] | [12.4649] | [12.5084] |  |
| tfm | res1 | channel | act_hard | probe_e | [0.398] | [0.013] | [0.060] | [13.6839] | [13.7588] |  |
| tfm | res1 | channel | act_hard | probe_d | [0.004] | [0.000] | [0.001] | [14.5188] | [14.6058] |  |
| tfm | res1 | channel | act_hard | probe_e_half | [0.105] | [0.004] | [0.022] | [14.2242] | [14.3089] | 0.041 |
| tfm | res1 | channel | act_hard | rand | [-0.001] | [0.001] | [0.002] | [14.5127] | [14.6002] |  |
| tfm | res1 | channel | next_obs | swap | 0.908 | 0.047 | 0.046 | 0.0834 | 0.0519 |  |
| tfm | res1 | channel | next_obs | probe_e | 0.693 | 0.010 | 0.008 | 0.0862 | 0.0538 |  |
| tfm | res1 | channel | next_obs | probe_d | 0.013 | -0.001 | -0.001 | 0.0868 | 0.0542 |  |
| tfm | res1 | channel | next_obs | probe_e_half | 0.298 | 0.009 | 0.010 | 0.0862 | 0.0537 | 0.020 |
| tfm | res1 | channel | next_obs | rand | 0.087 | 0.004 | 0.005 | 0.0865 | 0.0539 |  |
| tfm | res2 | iid | goal | swap | 1.000 | 0.000 | 0.000 | 1.2851 | 1.2851 |  |
| tfm | res2 | iid | goal | probe_e | 0.978 | 0.000 | 0.000 | 1.2851 | 1.2851 |  |
| tfm | res2 | iid | goal | probe_d | 0.018 | 0.000 | 0.000 | 1.2851 | 1.2851 |  |
| tfm | res2 | iid | goal | probe_e_half | 0.594 | 0.000 | 0.000 | 1.2851 | 1.2851 | 0.000 |
| tfm | res2 | iid | goal | rand | 0.048 | 0.000 | 0.000 | 1.2851 | 1.2851 |  |
| tfm | res2 | iid | act_soft | swap | 1.000 | 0.000 | 0.000 | 3.0879 | 3.0879 |  |
| tfm | res2 | iid | act_soft | probe_e | 0.807 | 0.000 | 0.000 | 3.0879 | 3.0879 |  |
| tfm | res2 | iid | act_soft | probe_d | 0.002 | 0.000 | 0.000 | 3.0879 | 3.0879 |  |
| tfm | res2 | iid | act_soft | probe_e_half | 0.434 | 0.000 | 0.000 | 3.0879 | 3.0879 | 0.000 |
| tfm | res2 | iid | act_soft | rand | 0.063 | 0.000 | 0.000 | 3.0879 | 3.0879 |  |
| tfm | res2 | iid | act_hard | swap | 1.000 | 0.000 | 0.000 | 13.4397 | 13.4397 |  |
| tfm | res2 | iid | act_hard | probe_e | 0.456 | 0.000 | 0.000 | 13.4397 | 13.4397 |  |
| tfm | res2 | iid | act_hard | probe_d | 0.000 | 0.000 | 0.000 | 13.4397 | 13.4397 |  |
| tfm | res2 | iid | act_hard | probe_e_half | 0.138 | 0.000 | 0.000 | 13.4397 | 13.4397 | 0.000 |
| tfm | res2 | iid | act_hard | rand | 0.012 | 0.000 | 0.000 | 13.4397 | 13.4397 |  |
| tfm | res2 | iid | next_obs | swap | 1.000 | 0.000 | 0.000 | 0.0386 | 0.0386 |  |
| tfm | res2 | iid | next_obs | probe_e | 0.886 | 0.000 | 0.000 | 0.0386 | 0.0386 |  |
| tfm | res2 | iid | next_obs | probe_d | -0.061 | 0.000 | 0.000 | 0.0386 | 0.0386 |  |
| tfm | res2 | iid | next_obs | probe_e_half | 0.602 | 0.000 | 0.000 | 0.0386 | 0.0386 | 0.000 |
| tfm | res2 | iid | next_obs | rand | -0.115 | 0.000 | 0.000 | 0.0386 | 0.0386 |  |
| tfm | res2 | channel | goal | swap | 1.000 | 0.000 | 0.000 | 1.8592 | 1.8726 |  |
| tfm | res2 | channel | goal | probe_e | 0.960 | 0.000 | 0.000 | 1.8592 | 1.8726 |  |
| tfm | res2 | channel | goal | probe_d | 0.094 | 0.000 | 0.000 | 1.8592 | 1.8726 |  |
| tfm | res2 | channel | goal | probe_e_half | 0.513 | 0.000 | 0.000 | 1.8592 | 1.8726 | 0.000 |
| tfm | res2 | channel | goal | rand | 0.028 | 0.000 | 0.000 | 1.8592 | 1.8726 |  |
| tfm | res2 | channel | act_soft | swap | 1.000 | 0.000 | 0.000 | 3.4773 | 3.5691 |  |
| tfm | res2 | channel | act_soft | probe_e | 0.917 | 0.000 | 0.000 | 3.4773 | 3.5691 |  |
| tfm | res2 | channel | act_soft | probe_d | 0.006 | 0.000 | 0.000 | 3.4773 | 3.5691 |  |
| tfm | res2 | channel | act_soft | probe_e_half | 0.491 | 0.000 | 0.000 | 3.4773 | 3.5691 | 0.000 |
| tfm | res2 | channel | act_soft | rand | 0.065 | 0.000 | 0.000 | 3.4773 | 3.5691 |  |
| tfm | res2 | channel | act_hard | swap | [1.000] | [0.000] | [0.000] | [14.5367] | [14.6237] |  |
| tfm | res2 | channel | act_hard | probe_e | [0.616] | [0.000] | [0.000] | [14.5367] | [14.6237] |  |
| tfm | res2 | channel | act_hard | probe_d | [0.001] | [0.000] | [0.000] | [14.5367] | [14.6237] |  |
| tfm | res2 | channel | act_hard | probe_e_half | [0.256] | [0.000] | [0.000] | [14.5367] | [14.6237] | 0.000 |
| tfm | res2 | channel | act_hard | rand | [0.010] | [0.000] | [0.000] | [14.5367] | [14.6237] |  |
| tfm | res2 | channel | next_obs | swap | 1.000 | 0.000 | 0.000 | 0.0868 | 0.0541 |  |
| tfm | res2 | channel | next_obs | probe_e | 0.819 | 0.000 | 0.000 | 0.0868 | 0.0541 |  |
| tfm | res2 | channel | next_obs | probe_d | -0.006 | 0.000 | 0.000 | 0.0868 | 0.0541 |  |
| tfm | res2 | channel | next_obs | probe_e_half | 0.516 | 0.000 | 0.000 | 0.0868 | 0.0541 | 0.000 |
| tfm | res2 | channel | next_obs | rand | 0.157 | 0.000 | 0.000 | 0.0868 | 0.0541 |  |

## B. Equal-belief equivalence (t = 12, 8 continuations per pair)

Effect = JS between the model's outputs after H₁⊕c and H₂⊕c; at the GRU's complete cut this is the patch effect. '/random' = the same effect for random pairs.

| arch | env | objective | pairs | effect k0 | effect fut | /random (mean) | /random (max over k) | Bayes divergence fut | model / Bayes | Spearman(model, Bayes) over pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| gru | iid | goal | iid_exact | 0.00000 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | — (Bayes = 0) | — |
| gru | iid | act_soft | iid_exact | 0.00002 | 0.00002 | 0.0001 | 0.0001 | 0.00000 | — (Bayes = 0) | — |
| gru | iid | act_hard | iid_exact | 0.00048 | 0.00114 | 0.0024 | 0.0047 | 0.00000 | — (Bayes = 0) | — |
| gru | iid | next_obs | iid_exact | 0.00000 | 0.00000 | 0.0000 | 0.0000 | -0.00000 | — (Bayes = 0) | — |
| gru | channel | goal | chan_equal_b | 0.00005 | 0.00442 | 0.0211 | 0.0286 | 0.00437 | 1.01 | 1.00 |
| gru | channel | goal | chan_equal_joint | 0.00001 | 0.00001 | 0.0001 | 0.0001 | 0.00001 | 1.12 | 0.91 |
| gru | channel | act_soft | chan_equal_b | 0.00018 | 0.01433 | 0.0428 | 0.0585 | 0.01442 | 0.99 | 0.99 |
| gru | channel | act_soft | chan_equal_joint | 0.00003 | 0.00003 | 0.0001 | 0.0001 | 0.00002 | 1.31 | 0.65 |
| gru | channel | act_hard | chan_equal_b | 0.00841 | 0.05954 | 0.1388 | 0.1908 | 0.06943 | 0.86 | 0.95 |
| gru | channel | act_hard | chan_equal_joint | 0.00076 | 0.00043 | 0.0011 | 0.0015 | 0.00074 | 0.59 | 0.41 |
| gru | channel | next_obs | chan_equal_b | 0.00517 | 0.00102 | 0.0742 | 0.1252 | 0.00103 | 1.00 | 1.00 |
| gru | channel | next_obs | chan_equal_joint | 0.00000 | 0.00000 | 0.0001 | 0.0001 | 0.00000 | 1.26 | 0.94 |
| tfm | iid | goal | iid_exact | 0.00000 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | — (Bayes = 0) | — |
| tfm | iid | act_soft | iid_exact | 0.00000 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | — (Bayes = 0) | — |
| tfm | iid | act_hard | iid_exact | 0.00000 | 0.00004 | 0.0001 | 0.0003 | 0.00000 | — (Bayes = 0) | — |
| tfm | iid | next_obs | iid_exact | 0.00000 | 0.00000 | 0.0000 | 0.0000 | -0.00000 | — (Bayes = 0) | — |
| tfm | channel | goal | chan_equal_b | 0.00007 | 0.00446 | 0.0214 | 0.0291 | 0.00437 | 1.02 | 0.99 |
| tfm | channel | goal | chan_equal_joint | 0.00001 | 0.00001 | 0.0001 | 0.0001 | 0.00001 | 1.59 | 0.94 |
| tfm | channel | act_soft | chan_equal_b | 0.00025 | 0.01446 | 0.0431 | 0.0584 | 0.01442 | 1.00 | 0.99 |
| tfm | channel | act_soft | chan_equal_joint | 0.00003 | 0.00003 | 0.0001 | 0.0001 | 0.00002 | 1.62 | 0.76 |
| tfm | channel | act_hard | chan_equal_b | 0.00823 | 0.05851 | 0.1357 | 0.1806 | 0.06943 | 0.84 | 0.95 |
| tfm | channel | act_hard | chan_equal_joint | 0.00050 | 0.00044 | 0.0011 | 0.0019 | 0.00074 | 0.60 | 0.41 |
| tfm | channel | next_obs | chan_equal_b | 0.00517 | 0.00103 | 0.0747 | 0.1252 | 0.00103 | 1.00 | 0.99 |
| tfm | channel | next_obs | chan_equal_joint | 0.00000 | 0.00000 | 0.0001 | 0.0002 | 0.00000 | 1.89 | 0.87 |

## C. Same action, different belief

Flip rate: on future steps where the Bayes-optimal actions of the two runs differ, how often the model's actions differ (`goal` models act on their output posterior). Ladder on the evaluation set: R²_y overall and within the optimal-action classes, action decodability, and decodability of the neutral-token count (centred on its expectation given t).

| arch | env | objective | flip rate |
|---|---|---|---|
| gru | iid | goal | 0.999 |
| gru | iid | act_soft | 0.995 |
| gru | iid | act_hard | 0.991 |
| gru | channel | goal | 0.974 |
| gru | channel | act_soft | 0.964 |
| gru | channel | act_hard | [0.912] |
| tfm | iid | goal | 0.999 |
| tfm | iid | act_soft | 0.999 |
| tfm | iid | act_hard | 0.999 |
| tfm | channel | goal | 0.963 |
| tfm | channel | act_soft | 0.958 |
| tfm | channel | act_hard | [0.930] |

| arch | env | objective | site | R²_y | R²_y within action | action acc | neutral-count R² |
|---|---|---|---|---|---|---|---|
| tfm | iid | goal | res0 | 0.110 | 0.143 | 0.439 | 0.081 |
| tfm | iid | goal | res1 | 0.992 | 0.992 | 0.897 | 0.911 |
| tfm | iid | goal | res2 | 0.994 | 0.995 | 0.878 | 0.899 |
| tfm | iid | goal | u | 1.000 | 1.000 | 0.861 | 0.975 |
| tfm | iid | act_soft | res0 | 0.110 | 0.143 | 0.439 | 0.081 |
| tfm | iid | act_soft | res1 | 0.973 | 0.982 | 0.927 | 0.838 |
| tfm | iid | act_soft | res2 | 0.982 | 0.989 | 0.937 | 0.779 |
| tfm | iid | act_soft | u | 0.977 | 0.988 | 0.936 | 0.810 |
| tfm | iid | act_hard | res0 | 0.110 | 0.143 | 0.439 | 0.081 |
| tfm | iid | act_hard | res1 | 0.969 | 0.973 | 0.949 | 0.798 |
| tfm | iid | act_hard | res2 | 0.960 | 0.971 | 0.997 | 0.557 |
| tfm | iid | act_hard | u | 0.878 | 0.932 | 1.000 | 0.433 |
| tfm | iid | next_obs | res0 | 0.110 | 0.143 | 0.439 | 0.081 |
| tfm | iid | next_obs | res1 | 0.999 | 0.999 | 0.871 | 0.990 |
| tfm | iid | next_obs | res2 | 0.998 | 0.999 | 0.929 | 0.968 |
| tfm | iid | next_obs | u | 0.997 | 0.998 | 0.931 | 0.963 |
| tfm | channel | goal | res0 | 0.138 | 0.182 | 0.479 | 0.092 |
| tfm | channel | goal | res1 | 0.901 | 0.907 | 0.865 | 0.880 |
| tfm | channel | goal | res2 | 0.992 | 0.993 | 0.904 | 0.941 |
| tfm | channel | goal | u | 0.992 | 0.994 | 0.886 | 0.962 |
| tfm | channel | act_soft | res0 | 0.138 | 0.182 | 0.479 | 0.092 |
| tfm | channel | act_soft | res1 | 0.928 | 0.921 | 0.877 | 0.927 |
| tfm | channel | act_soft | res2 | 0.965 | 0.975 | 0.947 | 0.930 |
| tfm | channel | act_soft | u | 0.920 | 0.962 | 0.953 | 0.911 |
| tfm | channel | act_hard | res0 | [0.138] | [0.182] | [0.479] | [0.092] |
| tfm | channel | act_hard | res1 | [0.927] | [0.902] | [0.888] | [0.889] |
| tfm | channel | act_hard | res2 | [0.930] | [0.945] | [0.958] | [0.891] |
| tfm | channel | act_hard | u | [0.794] | [0.869] | [0.982] | [0.798] |
| tfm | channel | next_obs | res0 | 0.138 | 0.182 | 0.479 | 0.092 |
| tfm | channel | next_obs | res1 | 0.930 | 0.929 | 0.875 | 0.900 |
| tfm | channel | next_obs | res2 | 0.960 | 0.970 | 0.915 | 0.885 |
| tfm | channel | next_obs | u | 0.935 | 0.957 | 0.926 | 0.892 |
| gru | iid | goal | h | 1.000 | 1.000 | 0.872 | 0.921 |
| gru | iid | act_soft | h | 0.996 | 0.997 | 0.938 | 0.706 |
| gru | iid | act_hard | h | 0.982 | 0.981 | 0.985 | 0.225 |
| gru | iid | next_obs | h | 0.998 | 0.998 | 0.936 | 0.764 |
| gru | channel | goal | h | 0.996 | 0.996 | 0.900 | 0.872 |
| gru | channel | act_soft | h | 0.964 | 0.971 | 0.945 | 0.806 |
| gru | channel | act_hard | h | [0.893] | [0.895] | [0.975] | [0.693] |
| gru | channel | next_obs | h | 0.973 | 0.977 | 0.929 | 0.739 |

## Pre-registered predictions (docs/task11_causal_theory.md §6)

| prediction | held | numbers |
|---|---|---|
| A1 GRU iid goal: PROBE-E ≥ 0.9 fut, ≥ 0.95 k0; PROBE-D ≥ 0.8 fut | **no** | t=6: E fut 1.000, k0 1.000, D fut 0.778; t=12: E fut 0.999, k0 1.000, D fut 0.765 |
| A2 GRU channel goal: PROBE-E closer to transplant than genuine-B; F_T ≥ 0.8 | yes | t=6: KL to T 0.0213 vs to S 0.0299, F_T 0.977; t=12: KL to T 0.0259 vs to S 0.0337, F_T 0.986 |
| A3 GRU iid goal: halfway PROBE-E ≥ 0.8 of its own transplant gap | yes | t=6: 0.999; t=12: 0.999 |
| A4 RAND ≤ 0.2 in every GRU cell | yes | max 0.016 |
| A5 GRU iid act_soft / act_hard / next_obs: PROBE-E ≥ 0.7 fut | **no** | act_soft t=6: 0.890; act_soft t=12: 0.895; act_hard t=6: 0.613; act_hard t=12: 0.706; next_obs t=6: 0.952; next_obs t=12: 0.959 |
| A6 tfm goal: res1 single-position PROBE-E / SWAP ≤ 0.3 fut; res2 SWAP ≥ 0.95 at k0 | yes | res1 fut max 0.156; res2 swap k0 min 1.000 |
| B1 GRU iid goal: equal-belief effect ≤ 0.05 of random at every k | yes | max over k 0.0000, mean 0.0000 |
| B2 GRU channel goal equal-b: model/Bayes in [0.5, 2], Spearman ≥ 0.5 | yes | model/Bayes 1.01, Spearman 1.00 |
| B3 GRU channel goal equal-joint: ≤ 0.1 of random | yes | 0.0001 |
| B4 GRU iid other objectives: equal-belief ≤ 0.1 of random | yes | act_hard 0.0024; act_soft 0.0001; next_obs 0.0000 |
| C1 GRU iid act_hard flip rate ≥ 0.9 | yes | 0.991 |
| C2 tfm iid: within-action R²_y at u, goal − act_hard ≥ 0.2; res1 ≥ 0.8 both | **no** | u: goal 1.000 − act_hard 0.932 = 0.067; res1 goal 0.992, act_hard 0.973 |
| C3 tfm iid goal: within-action R²_y ≥ 0.95 at res2 and u | yes | res2 0.995, u 1.000 |
| C4 tfm iid: neutral-count R² lower at u than res1, every objective | **no** | goal res1 0.911 → u 0.975; act_soft res1 0.838 → u 0.810; act_hard res1 0.798 → u 0.433; next_obs res1 0.990 → u 0.963 |

10 of 14 held.
