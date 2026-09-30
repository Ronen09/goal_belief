"""Round 25: round 23's trainer with --sup all / one (counterfactual targets for the prediction head).

    .venv/bin/python rounds/r25_balanced_prediction/train.py --aux 1.0 --sup all --seeds 0 1 2 3 4 5 6 7 8 9 --out rounds/r25_balanced_prediction/runs/all
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent.parent / "r23_obs_prediction" / "train.py"), run_name="__main__")
