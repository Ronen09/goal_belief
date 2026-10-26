#!/bin/bash
# the whole study in sequence on one GPU: collect → compose → rank → preserve → posthoc_pool → tables; logs in logs/
cd /workspace/goal_belief; B=studies/8_language_models/bump_coords; export PYTHONPATH=/workspace/goal_belief:$B; mkdir -p $B/logs
for s in collect compose rank preserve posthoc_pool tables; do echo "=== $s $(date -u +%H:%M:%S)"; .venv/bin/python $B/$s.py > $B/logs/$s.log 2>&1 || { echo "$s FAILED"; exit 1; }; done; echo "=== all done $(date -u +%H:%M:%S)"
