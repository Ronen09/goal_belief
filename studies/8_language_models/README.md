# 8. Language models

Can a small pretrained language model do what the project's small transformers do: track a hidden state from a
history and act on it? One task, built to be as partially observed as possible while keeping the exact belief and the
exact optimum available.

| experiment | question | result | cost |
|---|---|---|---|
| [Bump maze](bump_maze/) | A MiniGrid maze with a known map and goal, an unknown start cell and heading, and one bit of observation per step (a forward move moved or bumped). Zero-shot: do Qwen2.5-Instruct models from 0.5B to 7B act well, and do they track where they could be? Trained: does a fine-tuned 0.5B model reach the exact optimum, is it more data-efficient than a transformer from scratch, and is the consistent set of states decodable from its layers? | [report](bump_maze/REPORT.md) | 4 h GPU |
| [Bump goals](bump_goals/) | The same maze with four corner goals, one per episode named in the prompt, where the exact additive ceiling is 0.88 (maze10's 0.96). The fine-tuned 0.5B is exactly optimal under four goals at 40 000 episodes (the models from scratch 0.40–0.47 success); its logits are 0.20 goal × history interaction, concentrated at known states, and its additive part run as a policy recovers 0.14 of the goal-directed return (maze10: 0.81). Post hoc, the exact ceiling rule itself recovers 0.14 online when fitted on the optimum's decisions and about 0.5 refitted on its own episodes. Goal swaps of the decision token's residual work only from layer 20 of 24 (0.98 at 23), history swaps from layer 13; at layer 23 a linear map installs 0.90 of the swap and the mean goal offset nothing. | [report](bump_goals/REPORT.md) | 8 h GPU |

Rerun: `python reproduce.py bump_maze`, `python reproduce.py bump_goals`; results of the whole project: [FINDINGS.md](../../FINDINGS.md).
