"""Round 18: run every measurement on every saved run."""

from __future__ import annotations

import subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "runs"

if __name__ == "__main__":
    seeds = lambda g: [str(p) for p in sorted((R / g).glob("seed*"))]
    subprocess.run([sys.executable, str(HERE / "measure.py"), *seeds("ppo")], check=True)
    subprocess.run([sys.executable, str(HERE / "measure.py"), "--only-final", *seeds("frozen"), *seeds("supervised")], check=True)
