"""Round 17: run every measurement on every saved run (measure.py, qgen.py, traces.py)."""

from __future__ import annotations

import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "runs"


def run(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], check=True)


def seeds(group):
    return sorted((R / group).glob("seed*"))


if __name__ == "__main__":
    run("measure.py", *seeds("ppo_fixed"), *seeds("sup_fixed"))
    run("measure.py", "--only-final", *seeds("frozen_fixed"))
    run("measure.py", "--q", "grid", "--only-final", *seeds("ppo_grid"), *seeds("sup_grid"))
    run("qgen.py", *seeds("ppo_fixed"), *seeds("ppo_grid"), *seeds("sup_fixed"), *seeds("sup_grid"))
    run("traces.py", R / "ppo_fixed" / "seed0")
