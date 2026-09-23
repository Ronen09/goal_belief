"""TASK11 follow-up (not pre-registered): the channel-env act_hard models missed the
pre-registered convergence criterion (argmax agreement >= 0.99) at 20k / 15k steps. Retrain
the K = 4 cells with 4x the steps to separate "capped" from "not yet trained", and measure
them with the same probes. Output: results11/long/results11.json, results11/long/models/.

    .venv/bin/python scripts/task11_followup.py
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_task11 as R11  # noqa: E402
from goalgeo import belief_train as BT  # noqa: E402


def main():
    out = Path("results11/long")
    jobs = []
    for arch in ("tfm", "gru"):
        jobs += [BT.Job(arch, "channel", 4, "act_hard", s) for s in range(4)]
        jobs += [BT.Job(arch, "channel", 4, "act_hard", s, netho=True) for s in range(3)]
    t0 = time.time()
    R11.train_all(jobs, out, tfm_steps=80000, gru_steps=60000, workers=16)
    R11._log(f"training {time.time() - t0:.0f}s")
    R11.measure_all(jobs, out, workers=16)


if __name__ == "__main__":
    main()
