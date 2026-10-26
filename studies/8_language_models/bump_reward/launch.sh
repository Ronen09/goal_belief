#!/bin/bash
# all seeds of both architectures in parallel on one GPU
cd /workspace/goal_belief
for k in tfm gru; do for s in 0 1 2 3 4; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/workspace/goal_belief nohup .venv/bin/python studies/8_language_models/bump_reward/train_reward.py --kind $k --seed $s > studies/8_language_models/bump_reward/train_${k}_seed${s}.log 2>&1 &
done; done
