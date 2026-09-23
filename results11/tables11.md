# TASK11 tables (round 12: hidden goal)

Seed means over converged models; K = 4 unless stated. Probe: unregularised least squares on [h, 1].
R²_y: log-odds; R²_b: posterior probabilities; G_count = 1 − SSE(probe)/SSE(best affine-in-counts predictor).

## 1. Convergence

| arch | env | K | objective | pool | converged | KL to target (nats) | argmax agrees with optimal action |
|---|---|---|---|---|---|---|---|
| gru | channel | 3 | act_hard | full | 1/3 | 0.0271 | 0.9899 |
| gru | channel | 3 | goal | full | 3/3 | 0.0000 | — |
| gru | channel | 4 | act_hard | full | 0/4 | 0.0359 | 0.9866 |
| gru | channel | 4 | act_hard | no-conflict | 0/3 | 0.1027 | 0.9770 |
| gru | channel | 4 | act_soft | full | 4/4 | 0.0005 | 0.9942 |
| gru | channel | 4 | act_soft | no-conflict | 3/3 | 0.0006 | 0.9936 |
| gru | channel | 4 | goal | full | 4/4 | 0.0001 | — |
| gru | channel | 4 | goal | no-conflict | 3/3 | 0.0001 | — |
| gru | channel | 4 | next_obs | full | 4/4 | 0.0000 | — |
| gru | channel | 4 | next_obs | no-conflict | 3/3 | 0.0000 | — |
| gru | channel | 5 | act_hard | full | 0/3 | 0.0450 | 0.9839 |
| gru | channel | 5 | goal | full | 3/3 | 0.0001 | — |
| gru | iid | 3 | act_hard | full | 3/3 | 0.0009 | 0.9998 |
| gru | iid | 3 | goal | full | 3/3 | 0.0000 | — |
| gru | iid | 4 | act_hard | full | 4/4 | 0.0054 | 0.9985 |
| gru | iid | 4 | act_hard | no-conflict | 3/3 | 0.0208 | 0.9945 |
| gru | iid | 4 | act_soft | full | 4/4 | 0.0001 | 0.9985 |
| gru | iid | 4 | act_soft | no-conflict | 3/3 | 0.0003 | 0.9978 |
| gru | iid | 4 | goal | full | 4/4 | 0.0000 | — |
| gru | iid | 4 | goal | no-conflict | 3/3 | 0.0000 | — |
| gru | iid | 4 | next_obs | full | 4/4 | 0.0000 | — |
| gru | iid | 4 | next_obs | no-conflict | 3/3 | 0.0000 | — |
| gru | iid | 5 | act_hard | full | 3/3 | 0.0224 | 0.9927 |
| gru | iid | 5 | goal | full | 3/3 | 0.0000 | — |
| tfm | channel | 3 | act_hard | full | 1/3 | 0.0282 | 0.9889 |
| tfm | channel | 3 | goal | full | 3/3 | 0.0001 | — |
| tfm | channel | 4 | act_hard | full | 0/4 | 0.0312 | 0.9879 |
| tfm | channel | 4 | act_hard | no-conflict | 0/3 | 0.0789 | 0.9806 |
| tfm | channel | 4 | act_soft | full | 4/4 | 0.0007 | 0.9919 |
| tfm | channel | 4 | act_soft | no-conflict | 3/3 | 0.0006 | 0.9930 |
| tfm | channel | 4 | goal | full | 4/4 | 0.0002 | — |
| tfm | channel | 4 | goal | no-conflict | 3/3 | 0.0001 | — |
| tfm | channel | 4 | next_obs | full | 4/4 | 0.0000 | — |
| tfm | channel | 4 | next_obs | no-conflict | 3/3 | 0.0000 | — |
| tfm | channel | 5 | act_hard | full | 0/3 | 0.0406 | 0.9849 |
| tfm | channel | 5 | goal | full | 3/3 | 0.0003 | — |
| tfm | iid | 3 | act_hard | full | 3/3 | 0.0004 | 1.0000 |
| tfm | iid | 3 | goal | full | 3/3 | 0.0000 | — |
| tfm | iid | 4 | act_hard | full | 4/4 | 0.0004 | 0.9999 |
| tfm | iid | 4 | act_hard | no-conflict | 3/3 | 0.0266 | 0.9964 |
| tfm | iid | 4 | act_soft | full | 4/4 | 0.0000 | 0.9996 |
| tfm | iid | 4 | act_soft | no-conflict | 3/3 | 0.0001 | 0.9990 |
| tfm | iid | 4 | goal | full | 4/4 | 0.0000 | — |
| tfm | iid | 4 | goal | no-conflict | 3/3 | 0.0000 | — |
| tfm | iid | 4 | next_obs | full | 4/4 | 0.0000 | — |
| tfm | iid | 4 | next_obs | no-conflict | 3/3 | 0.0000 | — |
| tfm | iid | 5 | act_hard | full | 3/3 | 0.0193 | 0.9944 |
| tfm | iid | 5 | goal | full | 3/3 | 0.0000 | — |

## 2. The readout interface (tfm: post-final-LN u; GRU: h)

| arch | env | objective | IID R²_y | IID R²_b | EXT R²_y | EXT R²_b | CONF R²_y | TIME R²_y | IID RMSE (nats) | probe KL (nats) | G_count | action acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfm | iid | goal | 1.000 | 0.947 | 0.999 | 0.293 | 1.000 | 0.999 | 0.070 | 0.0000 | — | 0.861 |
| tfm | iid | act_soft | 0.977 | 1.000 | 0.687 | 1.000 | 0.987 | 0.948 | 0.626 | 0.0089 | — | 0.936 |
| tfm | iid | act_hard | 0.878 | 0.958 | 0.515 | 0.807 | 0.624 | 0.798 | 1.430 | 0.0456 | — | 1.000 |
| tfm | iid | next_obs | 0.997 | 0.999 | 0.951 | 0.995 | 0.998 | 0.983 | 0.219 | 0.0012 | — | 0.931 |
| tfm | iid | init | 0.819 | 0.761 | 0.579 | 0.647 | 0.887 | 0.656 | 1.748 | 0.0786 | — | 0.809 |
| tfm | channel | goal | 0.992 | 0.954 | 0.982 | -0.972 | 0.993 | 0.983 | 0.450 | 0.0010 | 0.940 | 0.886 |
| tfm | channel | act_soft | 0.920 | 0.999 | 0.553 | 0.999 | 0.722 | 0.849 | 1.433 | 0.0342 | 0.390 | 0.953 |
| tfm | channel | act_hard | — | — | — | — | — | — | — | — | — | — |
| tfm | channel | next_obs | 0.935 | 0.991 | 0.667 | 0.965 | 0.797 | 0.854 | 1.287 | 0.0301 | 0.506 | 0.926 |
| tfm | channel | init | 0.766 | 0.807 | 0.437 | 0.710 | 0.531 | 0.617 | 2.448 | 0.1217 | -0.782 | 0.847 |
| gru | iid | goal | 1.000 | 0.971 | 1.000 | 0.333 | 1.000 | 1.000 | 0.031 | 0.0000 | — | 0.872 |
| gru | iid | act_soft | 0.996 | 1.000 | 0.977 | 0.998 | 0.996 | 0.990 | 0.262 | 0.0010 | — | 0.938 |
| gru | iid | act_hard | 0.982 | 0.989 | 0.899 | 0.925 | 0.983 | 0.964 | 0.551 | 0.0053 | — | 0.985 |
| gru | iid | next_obs | 0.998 | 1.000 | 0.982 | 0.989 | 0.998 | 0.994 | 0.194 | 0.0006 | — | 0.936 |
| gru | iid | init | 0.682 | 0.633 | 0.489 | 0.547 | 0.670 | 0.589 | 2.313 | 0.1608 | — | 0.691 |
| gru | channel | goal | 0.996 | 0.970 | 0.991 | -1.644 | 0.996 | 0.992 | 0.303 | 0.0004 | 0.973 | 0.900 |
| gru | channel | act_soft | 0.964 | 0.998 | 0.757 | 0.993 | 0.889 | 0.928 | 0.955 | 0.0098 | 0.728 | 0.945 |
| gru | channel | act_hard | — | — | — | — | — | — | — | — | — | — |
| gru | channel | next_obs | 0.973 | 0.993 | 0.859 | 0.845 | 0.940 | 0.935 | 0.833 | 0.0070 | 0.794 | 0.929 |
| gru | channel | init | 0.701 | 0.713 | 0.413 | 0.662 | 0.272 | 0.611 | 2.765 | 0.1674 | -1.275 | 0.772 |

Baselines (features instead of a network):

| env | features | IID R²_y | IID R²_b | EXT R²_y | CONF R²_y | TIME R²_y | action acc |
|---|---|---|---|---|---|---|---|
| iid | counts | 1.000 | 0.802 | 1.000 | 1.000 | 1.000 | 0.755 |
| iid | mean_t | 0.560 | 0.547 | 0.282 | 0.675 | 0.402 | 0.748 |
| channel | counts | 0.869 | 0.782 | 0.609 | 0.492 | 0.867 | 0.776 |
| channel | mean_t | 0.548 | 0.655 | 0.264 | 0.711 | 0.399 | 0.782 |

## 3. Transformer sites

IID R²_y (TIME R²_y) [G_count in the channel env]. res0 TIME is ill-posed: positions t > 12 lie outside the span of the fitted positional embeddings, and the least-squares extrapolation diverges (R² ~ −1e10).

| env | objective | res0 | res1 | res2 | u |
|---|---|---|---|---|---|
| iid | goal | 0.110 (n/a) | 0.992 (0.929) | 0.994 (0.953) | 1.000 (0.999) |
| iid | act_soft | 0.110 (n/a) | 0.973 (0.886) | 0.982 (0.947) | 0.977 (0.948) |
| iid | act_hard | 0.110 (n/a) | 0.969 (0.888) | 0.960 (0.903) | 0.878 (0.798) |
| iid | next_obs | 0.110 (n/a) | 0.999 (0.984) | 0.998 (0.986) | 0.997 (0.983) |
| iid | init | 0.110 (n/a) | 0.791 (0.406) | 0.819 (0.653) | 0.819 (0.656) |
| channel | goal | 0.138 (n/a) [-5.565] | 0.901 (0.725) [0.245] | 0.992 (0.970) [0.937] | 0.992 (0.983) [0.940] |
| channel | act_soft | 0.138 (n/a) [-5.566] | 0.928 (0.804) [0.453] | 0.965 (0.912) [0.731] | 0.920 (0.849) [0.390] |
| channel | act_hard | — (n/a) [—] | — (—) [—] | — (—) [—] | — (—) [—] |
| channel | next_obs | 0.138 (n/a) [-5.569] | 0.930 (0.811) [0.466] | 0.960 (0.900) [0.696] | 0.935 (0.854) [0.506] |
| channel | init | 0.139 (n/a) [-5.563] | 0.746 (0.380) [-0.934] | 0.770 (0.619) [-0.752] | 0.766 (0.617) [-0.782] |

## 4. Network-level held-out: models trained without any sequence that enters the conflict region

Probe fit outside the region, scored inside it. KL: the model's own output against its exact target.

| arch | env | objective | CONF R²_y, full pool | CONF R²_y, no-conflict pool | KL in / out, full | KL in / out, no-conflict |
|---|---|---|---|---|---|---|
| tfm | iid | goal | 1.000 | 1.000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| tfm | iid | act_soft | 0.987 | 0.987 | 0.0002 / 0.0000 | 0.0006 / 0.0001 |
| tfm | iid | act_hard | 0.624 | 0.750 | 0.0115 / 0.0003 | 2.7086 / 0.0007 |
| tfm | iid | next_obs | 0.998 | 0.998 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| tfm | channel | goal | 0.993 | 0.994 | 0.0006 / 0.0002 | 0.0004 / 0.0001 |
| tfm | channel | act_soft | 0.722 | 0.726 | 0.0071 / 0.0006 | 0.0070 / 0.0005 |
| tfm | channel | act_hard | — | — | — / — | — / — |
| tfm | channel | next_obs | 0.797 | 0.792 | 0.0001 / 0.0000 | 0.0002 / 0.0000 |
| gru | iid | goal | 1.000 | 1.000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| gru | iid | act_soft | 0.996 | 0.995 | 0.0010 / 0.0001 | 0.0020 / 0.0002 |
| gru | iid | act_hard | 0.983 | 0.979 | 0.0968 / 0.0045 | 1.6932 / 0.0046 |
| gru | iid | next_obs | 0.998 | 0.998 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| gru | channel | goal | 0.996 | 0.996 | 0.0004 / 0.0001 | 0.0005 / 0.0001 |
| gru | channel | act_soft | 0.889 | 0.849 | 0.0067 / 0.0004 | 0.0129 / 0.0004 |
| gru | channel | act_hard | — | — | — / — | — / — |
| gru | channel | next_obs | 0.940 | 0.941 | 0.0001 / 0.0000 | 0.0001 / 0.0000 |

## 5. Number of goals

| arch | env | K | objective | IID R²_y | EXT R²_y | CONF R²_y | TIME R²_y | G_count (best site) | counts-baseline R²_y |
|---|---|---|---|---|---|---|---|---|---|
| tfm | iid | 3 | goal | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 |
| tfm | iid | 3 | act_hard | 0.931 | 0.758 | 0.438 | 0.882 | — | 1.000 |
| tfm | iid | 3 | init | 0.835 | 0.736 | 0.875 | 0.655 | — | 1.000 |
| tfm | iid | 4 | goal | 1.000 | 0.999 | 1.000 | 0.999 | — | 1.000 |
| tfm | iid | 4 | act_hard | 0.878 | 0.515 | 0.624 | 0.798 | — | 1.000 |
| tfm | iid | 4 | init | 0.819 | 0.579 | 0.887 | 0.656 | — | 1.000 |
| tfm | iid | 5 | goal | 0.999 | 0.998 | 1.000 | 0.997 | — | 1.000 |
| tfm | iid | 5 | act_hard | 0.860 | 0.384 | 0.860 | 0.757 | — | 1.000 |
| tfm | iid | 5 | init | 0.798 | 0.458 | 0.877 | 0.639 | — | 1.000 |
| tfm | channel | 3 | goal | 0.996 | 0.991 | 0.994 | 0.991 | 0.965 | 0.890 |
| tfm | channel | 3 | act_hard | 0.819 | 0.372 | -1.995 | 0.687 | 0.527 | 0.890 |
| tfm | channel | 3 | init | 0.805 | 0.520 | -0.610 | 0.645 | -0.769 | 0.890 |
| tfm | channel | 4 | goal | 0.992 | 0.982 | 0.993 | 0.983 | 0.940 | 0.869 |
| tfm | channel | 4 | act_hard | — | — | — | — | — | 0.869 |
| tfm | channel | 4 | init | 0.766 | 0.437 | 0.531 | 0.617 | -0.752 | 0.869 |
| tfm | channel | 5 | goal | 0.988 | 0.975 | 0.992 | 0.977 | 0.921 | 0.854 |
| tfm | channel | 5 | act_hard | — | — | — | — | — | 0.854 |
| tfm | channel | 5 | init | 0.712 | 0.372 | 0.620 | 0.571 | -0.917 | 0.854 |
| gru | iid | 3 | goal | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 |
| gru | iid | 3 | act_hard | 0.984 | 0.904 | 0.983 | 0.969 | — | 1.000 |
| gru | iid | 3 | init | 0.687 | 0.625 | 0.676 | 0.592 | — | 1.000 |
| gru | iid | 4 | goal | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 |
| gru | iid | 4 | act_hard | 0.982 | 0.899 | 0.983 | 0.964 | — | 1.000 |
| gru | iid | 4 | init | 0.682 | 0.489 | 0.670 | 0.589 | — | 1.000 |
| gru | iid | 5 | goal | 1.000 | 1.000 | 1.000 | 1.000 | — | 1.000 |
| gru | iid | 5 | act_hard | 0.967 | 0.815 | 0.974 | 0.934 | — | 1.000 |
| gru | iid | 5 | init | 0.689 | 0.445 | 0.726 | 0.593 | — | 1.000 |
| gru | channel | 3 | goal | 0.998 | 0.995 | 0.997 | 0.996 | 0.981 | 0.890 |
| gru | channel | 3 | act_hard | 0.918 | 0.610 | 0.405 | 0.862 | 0.257 | 0.890 |
| gru | channel | 3 | init | 0.745 | 0.499 | -0.452 | 0.657 | -1.312 | 0.890 |
| gru | channel | 4 | goal | 0.996 | 0.991 | 0.996 | 0.992 | 0.973 | 0.869 |
| gru | channel | 4 | act_hard | — | — | — | — | — | 0.869 |
| gru | channel | 4 | init | 0.701 | 0.413 | 0.272 | 0.611 | -1.275 | 0.869 |
| gru | channel | 5 | goal | 0.995 | 0.985 | 0.995 | 0.989 | 0.964 | 0.854 |
| gru | channel | 5 | act_hard | — | — | — | — | — | 0.854 |
| gru | channel | 5 | init | 0.671 | 0.357 | 0.342 | 0.580 | -1.254 | 0.854 |

## 6. Seed stability of interface IID R²_y (CV across seeds)

| arch | env | goal | act_soft | act_hard | next_obs |
|---|---|---|---|---|---|
| tfm | iid | 0.0000 | 0.0024 | 0.0163 | 0.0003 |
| tfm | channel | 0.0007 | 0.0042 | — | 0.0085 |
| gru | iid | 0.0000 | 0.0005 | 0.0028 | 0.0003 |
| gru | channel | 0.0007 | 0.0029 | — | 0.0022 |

## 7. Pre-registered predictions (docs/task11_theory.md §5)

| prediction | held | numbers |
|---|---|---|
| P1 forced identity | yes | tfm/iid: IID 1.000, EXT 0.999, CONF 1.000, TIME 0.999; tfm/channel: IID 0.992, EXT 0.982, CONF 0.993, TIME 0.983; gru/iid: IID 1.000, EXT 1.000, CONF 1.000, TIME 1.000; gru/channel: IID 0.996, EXT 0.991, CONF 0.996, TIME 0.992 |
| P2 coordinates follow the target | yes | tfm/iid: goal y 1.000 vs b 0.947; act_soft y 0.977 vs b 1.000; tfm/channel: goal y 0.992 vs b 0.954; act_soft y 0.920 vs b 0.999; gru/iid: goal y 1.000 vs b 0.971; act_soft y 0.996 vs b 1.000; gru/channel: goal y 0.996 vs b 0.970; act_soft y 0.964 vs b 0.998 |
| P3 extrapolation exposes the coordinate | **no** | tfm/iid: goal 0.999 − act_soft 0.687 = 0.312; tfm/channel: goal 0.982 − act_soft 0.553 = 0.429; gru/iid: goal 1.000 − act_soft 0.977 = 0.023; gru/channel: goal 0.991 − act_soft 0.757 = 0.233 |
| P4 hard targets keep the region, not the posterior | **no** | tfm/iid: goal 1.000 − act_hard 0.878 = 0.122, action acc 1.000; tfm/channel: goal 0.992 − act_hard — = —, action acc —; gru/iid: goal 1.000 − act_hard 0.982 = 0.018, action acc 0.985; gru/channel: goal 0.996 − act_hard — = —, action acc — |
| P5 iid is not diagnostic (tfm) | yes | goal: u 1.000; act_soft: res2 0.982; act_hard: res1 0.969; next_obs: res1 0.999 |
| P6 channel is diagnostic | **no** | tfm goal: u 0.940; tfm next_obs: res2 0.696; tfm act_hard — < goal 0.940; gru goal: h 0.973; gru next_obs: h 0.794; gru act_hard — < goal 0.973 |
| P7 init control | **no** | tfm/iid: max R²_y 0.819; tfm/channel: max R²_y 0.770, max G -0.752; gru/iid: max R²_y 0.682; gru/channel: max R²_y 0.701, max G -1.275 |
| P8 network-level held-out | **no** | tfm/iid: CONF R²_y 1.000, KL in/out 1.45 (full-pool models, not part of the test: 1.48); tfm/channel: CONF R²_y 0.994, KL in/out 4.00 (full-pool models, not part of the test: 3.75); gru/iid: CONF R²_y 1.000, KL in/out 1.96 (full-pool models, not part of the test: 2.25); gru/channel: CONF R²_y 0.996, KL in/out 4.79 (full-pool models, not part of the test: 3.50) |
| P9 block 1 codes a running mean | **no** | res1 IID 0.992 − TIME 0.929 = 0.063 |
| P10 seed stability | yes | violations: none |
| P11 K = 3, 5 | **no** | failures: K=5 tfm/channel: IID 0.988, EXT 0.975, CONF 0.992, TIME 0.977 |

4 of 11 held.

## 8. Follow-up (not pre-registered): channel-env act_hard at 4× the steps

Transformers 80k steps, GRUs 60k (main grid: 20k / 15k). All models shown, converged or not; the goal row is the main grid's for comparison.

| arch | pool | steps | converged | argmax agrees | IID R²_y | IID R²_b | EXT R²_y | CONF R²_y | G_count | action acc |
|---|---|---|---|---|---|---|---|---|---|---|
| tfm | full | main | 0/4 | 0.9879 | 0.794 | 0.955 | 0.335 | 0.107 | -0.568 | 0.982 |
| tfm | full | 4× | 0/4 | 0.9852 | 0.789 | 0.953 | 0.354 | 0.334 | -0.605 | 0.979 |
| tfm | no-conflict | main | 0/3 | 0.9806 | 0.791 | 0.942 | 0.311 | 0.091 | -0.590 | 0.977 |
| tfm | no-conflict | 4× | 0/3 | 0.9784 | 0.763 | 0.935 | 0.320 | 0.166 | -0.802 | 0.976 |
| tfm | full | goal (main) | — | — | 0.992 | 0.954 | 0.982 | 0.993 | 0.940 | 0.886 |
| gru | full | main | 0/4 | 0.9866 | 0.893 | 0.984 | 0.546 | 0.629 | 0.187 | 0.975 |
| gru | full | 4× | 0/4 | 0.9847 | 0.885 | 0.982 | 0.555 | 0.632 | 0.123 | 0.971 |
| gru | no-conflict | main | 0/3 | 0.9770 | 0.853 | 0.972 | 0.508 | 0.528 | -0.117 | 0.969 |
| gru | no-conflict | 4× | 0/3 | 0.9756 | 0.841 | 0.968 | 0.503 | 0.496 | -0.213 | 0.966 |
| gru | full | goal (main) | — | — | 0.996 | 0.970 | 0.991 | 0.996 | 0.973 | 0.900 |
