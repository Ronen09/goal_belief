#!/bin/bash
# main runs and measurements, one after the other (one GPU)
export PYTHONPATH=. OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=rounds/r18_maze_belief; R=$D/runs
.venv/bin/python $D/train.py --cond ppo --seeds 0 1 2 3 4 --updates 1500 --out $R/ppo > $R/ppo.log 2>&1
.venv/bin/python $D/train.py --cond frozen --seeds 0 1 --updates 1000 --out $R/frozen > $R/frozen.log 2>&1
.venv/bin/python $D/train.py --cond supervised --seeds 0 --updates 1000 --lr 3e-4 --out $R/supervised > $R/supervised.log 2>&1
.venv/bin/python $D/measure.py $R/ppo/seed0 $R/ppo/seed1 $R/ppo/seed2 $R/ppo/seed3 $R/ppo/seed4 > $R/measure_ppo.log 2>&1
.venv/bin/python $D/measure.py --only-final $R/frozen/seed0 $R/frozen/seed1 $R/supervised/seed0 > $R/measure_controls.log 2>&1
echo finished
