# Belief state

A goal that must be inferred from noisy evidence. The exact posterior is decodable everywhere, but it is the state the network computes with only in recurrent networks; a transformer recomputes it from the tokens, unless a carried state and an incentive (unreliable or priced access to the history) make keeping it worthwhile.

| experiment | question | report | cost |
|---|---|---|---|
| [Hidden goal](hidden_goal/) | A hidden goal: is the posterior affinely recoverable, and is it the causal state? | [report](hidden_goal/REPORT.md), [causal](hidden_goal/causal/REPORT.md) | 40 min GPU + CPU, then 6 min |
| [Filter state](filter_state/) | Is the recurrent state the environment's minimal predictive state? | [report](filter_state/REPORT.md) | 7 min, 96 CPU workers |
| [Window carry](window_carry/) | Does a narrow attention window force a steerable belief state? | [report](window_carry/REPORT.md) | 31 min CPU workers |
| [Prior vs recompute](prior_vs_recompute/) | Does a next-token transformer use its previous belief as a prior? | [report](prior_vs_recompute/REPORT.md) | 11 min GPU |
| [K/V dropout](kv_dropout/) | Can K/V dropout induce recurrence continuously? | [report](kv_dropout/REPORT.md) | 1 h GPU + CPU workers |
| [Read cost](read_cost/) | Does a price on reading the history induce a selective, recurrent belief state? | [report](read_cost/REPORT.md) | 3 h (32 min GPU + carry family on CPU workers) |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py hidden_goal`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
