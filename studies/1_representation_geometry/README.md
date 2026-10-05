# Representation geometry

What a small network's hidden geometry reflects, what sets the size of a distinction, and which measures of a representation can be trusted across functionally equivalent models. The answer that the later studies lean on: raw geometry and local metrics depend on the implementation; decodability, readout contrast and interventions on a complete causal cut do not.

| experiment | question | report | cost |
|---|---|---|---|
| [Occupancy geometry](occupancy/) | Does a goal-conditioned policy learn goal-occupancy geometry? | [report](occupancy/REPORT.md) | 6 min CPU |
| [Policy quotient](policy_quotient/) | Policy quotient vs occupancy geometry; soft targets | [report](policy_quotient/REPORT.md) | 15 min CPU |
| [Supervision bottleneck](supervision/) | The supervision bottleneck: what the target keeps | [report](supervision/REPORT.md) | 18 min CPU |
| [HMM objectives](hmm_objectives/) | One-step vs k-step vs sequential objectives on an HMM | [report](hmm_objectives/REPORT.md) | 5 min |
| [Prominence](prominence/) | What makes information geometrically prominent? | [report](prominence/REPORT.md) | 20 min, 8 CPU workers |
| [Readout scale](readout_scale/) | Readout gain vs hidden separation | [report](readout_scale/REPORT.md) | 8 min, 8 CPU workers |
| [Allocation](allocation/) | What sets the gain/separation split | [report](allocation/REPORT.md) | 6 min, 8 CPU workers |
| [Invariants](invariants/) | Which representation measures are invariant | [report](invariants/REPORT.md) | 13 min, 8 CPU workers |
| [Intervention equivalence](interventions/) | Intervention equivalence and local metrics | [report](interventions/REPORT.md) | 5 min |
| [Transformer reproduction](transformer/) | Transformer reproduction of the prominence to intervention-equivalence experiments | [report](transformer/REPORT.md) | 10 min GPU |
| [Implementation freedom](implementation_freedom/) | Cut identifiability and factorisation freedom | [report](implementation_freedom/REPORT.md) | 1.5 h GPU |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py occupancy`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
