#!/bin/bash
# the main runs, one after the other (they share one GPU)
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=studies/3_reward_trained_agents/navigate_commit/runs
.venv/bin/python studies/3_reward_trained_agents/navigate_commit/train.py --cond ppo --q fixed --seeds 0 1 2 3 4 --updates 3000 --dense 600 1800 100 --out $R/ppo_fixed > $R/ppo_fixed.log 2>&1
.venv/bin/python studies/3_reward_trained_agents/navigate_commit/train.py --cond frozen --q fixed --seeds 0 1 --updates 2000 --out $R/frozen_fixed > $R/frozen_fixed.log 2>&1
.venv/bin/python studies/3_reward_trained_agents/navigate_commit/train.py --cond ppo --q grid --seeds 0 1 2 --updates 3500 --out $R/ppo_grid > $R/ppo_grid.log 2>&1
