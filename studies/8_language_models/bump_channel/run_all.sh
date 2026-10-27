#!/bin/bash
# The bump channel study end to end on one GPU (PLAN.md "Compute": about 2–3 h): part 0 on the two from-scratch
# models, parts 1–3 on the three models, the tables. Logs under logs/.
set -e
cd /workspace/goal_belief
D=studies/8_language_models/bump_channel; mkdir -p $D/logs
export PYTHONPATH=/workspace/goal_belief:$D
PY=.venv/bin/python
echo "start $(date -u +%FT%TZ)"
for k in tfm gru; do $PY $D/subspace_small.py --kind $k > $D/logs/subspace_$k.log 2>&1; echo "subspace $k done $(date -u +%FT%TZ)"; done
for m in llm tfm gru; do $PY $D/channel.py --model $m > $D/logs/channel_$m.log 2>&1; echo "channel $m done $(date -u +%FT%TZ)"; done
$PY $D/tables.py > $D/logs/tables.log 2>&1
echo "tables done $(date -u +%FT%TZ)"
