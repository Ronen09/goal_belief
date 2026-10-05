"""The balanced-prediction experiment: the observation-prediction experiment's trainer with --sup all / one (counterfactual targets for the prediction head).

    .venv/bin/python studies/3_reward_trained_agents/balanced_prediction/train.py --aux 1.0 --sup all --seeds 0 1 2 3 4 5 6 7 8 9 --out studies/3_reward_trained_agents/balanced_prediction/runs/all
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[3] / "3_reward_trained_agents" / "obs_prediction" / "train.py"), run_name="__main__")
