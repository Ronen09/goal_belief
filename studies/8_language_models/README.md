# 8. Language models

Can a small pretrained language model do what the project's small transformers do: track a hidden state from a
history and act on it? One task, built to be as partially observed as possible while keeping the exact belief and the
exact optimum available.

| experiment | question | result | cost |
|---|---|---|---|
| [Bump maze](bump_maze/) | A MiniGrid maze with a known map and goal, an unknown start cell and heading, and one bit of observation per step (a forward move moved or bumped). Zero-shot: do Qwen2.5-Instruct models from 0.5B to 7B act well, and do they track where they could be? Trained: does a fine-tuned 0.5B model reach the exact optimum, is it more data-efficient than a transformer from scratch, and is the consistent set of states decodable from its layers? | [report](bump_maze/REPORT.md) | 4 h GPU |

Rerun: `python reproduce.py bump_maze`; results of the whole project: [FINDINGS.md](../../FINDINGS.md).
