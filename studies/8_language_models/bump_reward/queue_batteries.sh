#!/bin/bash
# wait for each seed's model, then run its battery (seeds of one architecture in parallel once all are saved)
cd /workspace/goal_belief
k=$1
for s in 0 1 2 3 4; do while [ ! -f studies/8_language_models/bump_reward/runs/$k/seed$s/model.pt ]; do sleep 30; done; done
for s in 0 1 2 3 4; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=.:studies/8_language_models/bump_policy nohup .venv/bin/python studies/8_language_models/bump_reward/battery.py --kind $k --model studies/8_language_models/bump_reward/runs/$k/seed$s/model.pt --name ppo_${k}_seed$s > studies/8_language_models/bump_reward/battery_ppo_${k}_seed$s.log 2>&1 &
done
wait
