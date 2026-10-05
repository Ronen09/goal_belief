# tests/test_r17_smoke.py
"""Round 17's run.py in quick mode produces a record with every key tables.py reads, and the R2 cell
records its own gate decision (it may differ from R1's)."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_r17_quick_run_record_structure(tmp_path):
    out = tmp_path / "r17"
    subprocess.run([sys.executable, str(ROOT / "rounds/r17_read_cost/run.py"), "--quick", "--out", str(out), "--workers", "4"],
                   cwd=ROOT, check=True, timeout=1500)
    runs = json.load(open(out / "results16.json"))["runs"]
    assert {(r["family"], r["c"]) for r in runs} == {("plain", 0.0), ("plain", 0.3), ("carry", 0.0), ("carry", 0.3)}
    r = runs[0]
    for k in ("kl_own", "kl_open", "read_rate", "entropy_split", "movement_split", "rate_pos", "rate_layer"):
        assert k in r
    assert len(r["rate_pos"]) == 25
    cell = r["t"]["16"]["own"]["R2"]
    assert set(cell) == {"lam", "gate"} and len(cell["gate"]) == (4 if r["family"] == "plain" else 2)
    assert all(r["t"][t][reg]["R1"]["gate"] is not None for t in ("4", "8", "16", "23") for reg in ("open", "own") for r in runs)
