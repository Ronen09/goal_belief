# Goal × belief mechanism

How the maze policy combines the goal with the evidence, followed component by component. Attention reads a goal-free evidence estimate; the goal arrives through the goal token's own self-attention value and is combined in block 0's MLP as a low-rank bilinear term; and in the end the decision is, to a first approximation, additive: a goal-averaged usefulness profile of the history plus a fixed bias per goal.

| experiment | question | report | cost |
|---|---|---|---|
| [Goal route selection](goal_route_selection/) | Does the goal change which evidence route (pre-goal interface or direct route) the policy relies on? | [report](goal_route_selection/REPORT.md) | 2 min GPU |
| [Goal-swap components](goal_swap_components/) | Goal swap with the history fixed: which heads or MLPs after the interface carry the switch? | [report](goal_swap_components/REPORT.md) | 4 min GPU |
| [Cross-history MLP](cross_history_mlp/) | Cross-history patches of the goal token's MLP outputs: goal instruction, evidence-goal combination, or action preference? | [report](cross_history_mlp/REPORT.md) | 1 min GPU |
| [Direct belief edit](direct_belief_edit/) | A shared belief encoding at the goal token's state after block 0, edited with the donor's posterior under all goals | [report](direct_belief_edit/REPORT.md) | 2 min GPU |
| [Attention belief edit](attention_belief_edit/) | The direct-belief-edit experiment's belief edit one step earlier, before block 0's MLP: is the belief map shared across goals there? | [report](attention_belief_edit/REPORT.md) | 2 min GPU |
| [Nonlinear belief edit](nonlinear_belief_edit/) | Nonlinear (MLP, per-posterior table) encodings of the posterior before block 0's MLP; goal-dependent pairs | [report](nonlinear_belief_edit/REPORT.md) | 6 min GPU |
| [Block-0 steps](block0_steps/) | The goal x belief part at each step from block 0's attention output (self, prefix, heads) through the layer norm to the MLP input | [report](block0_steps/REPORT.md) | 2 min GPU |
| [Query swap](query_swap/) | Swap block 0's goal-token query, or its own key and value, to another goal's: representation and decisions | [report](query_swap/REPORT.md) | 1 min GPU |
| [Self value](self_value/) | The goal token's own value against its own key in block 0, per head; the residual embedding as the complement | [report](self_value/REPORT.md) | 1 min GPU |
| [Self-value interaction](self_value_interaction/) | Does the self-value swap move the goal x belief part I(b,g) = f(b,g) - f(b) at block 0's MLP input, and after it? | [report](self_value_interaction/REPORT.md) | 20 min GPU |
| [MLP bilinear](mlp_bilinear/) | How block 0's MLP computes J(b,g): low-rank bilinear description, second-order mechanism, hidden units, ablation | [report](mlp_bilinear/REPORT.md) | 30 min GPU |
| [MLP depth](mlp_depth/) | The goal x belief part at every MLP of the goal token: bilinear description, own share, removal from one or all four | [report](mlp_depth/REPORT.md) | 55 min GPU |
| [Interaction removal](interaction_removal/) | Per-history removal of the goal x history interaction from the goal token's MLP and attention outputs | [report](interaction_removal/REPORT.md) | 1 min GPU |
| [Additive code](additive_code/) | The additive code at the goal token: logits ~ H(h) + G(g, L), and where additivity is impossible | [report](additive_code/REPORT.md) | 20 s GPU |
| [What is H](what_is_H/) | What is the history profile H: max_g Q*, goal-averaged values, P(optimal), reachability? Prediction and causal edits | [report](what_is_H/REPORT.md) | 3 min GPU |
| [H mixture](H_mixture/) | H as a mixture of P(optimal) and reachability; goal weights inside H against uniform training frequency | [report](H_mixture/REPORT.md) | 1 min GPU |

Rerun one: `python reproduce.py <experiment>` (e.g. `python reproduce.py goal_route_selection`); results of the whole project: [FINDINGS.md](../../FINDINGS.md).
