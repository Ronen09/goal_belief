"""Prediction checks of round 17 on a synthetic results file that should pass every prediction."""
import importlib.util, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r17tables", ROOT / "rounds/r17_read_cost/tables.py")
TB = importlib.util.module_from_spec(spec); spec.loader.exec_module(TB)


def _rec(fam, c, seed, rate, kl, lam_open, lam_own, ent, mov, rate_pos):
    cells = lambda lam: {k: {"lam": v, "gate": [1.0, 1.0]} for k, v in (("R1", 0.0), ("R2", lam), ("R3", 0.5), ("R4", 1.0))}   # noqa: E731
    return {"family": fam, "c": c, "seed": seed, "kl_own": kl, "kl_open": kl, "read_rate": rate,
            "entropy_split": ent, "movement_split": mov, "rate_pos": rate_pos, "rate_layer": [rate, rate],
            "t": {str(t): {"open": cells(lam_open), "own": cells(lam_own)} for t in (4, 8, 16, 23)}}


def test_all_predictions_hold_on_a_passing_synthetic_file(tmp_path):
    runs = []
    carry = {0.0: (0.9, 0.3), 0.003: (0.7, 0.5), 0.01: (0.4, 0.8), 0.03: (0.2, 0.9), 0.1: (0.04, 0.95), 0.3: (0.01, 0.99)}
    plain = {0.0: (0.95, 0.03), 0.003: (0.9, 0.05), 0.01: (0.8, 0.1), 0.03: (0.4, 0.2), 0.1: (0.05, 0.25), 0.3: (0.01, 0.25)}
    for s in range(3):
        for c, (rate, lam) in carry.items():
            pos = [0.0] * 2 + [1.0] * 3 + [rate] * 20
            runs.append(_rec("carry", c, s, rate, 0.003, lam, max(lam, 1 - rate), 0.1, 0.05, pos))
        for c, (rate, lam) in plain.items():
            runs.append(_rec("plain", c, s, rate, 0.005 if rate > 0.1 else 0.05, lam, lam, 0.0, 0.0, [rate] * 25))
    p = tmp_path / "results16.json"; json.dump({"runs": runs}, open(p, "w"))
    d = TB.D(p)
    P = TB.predictions(d)
    assert len(P) == 8 and all(ok for _, ok, _ in P), [(n, x) for n, ok, x in P if not ok]
    assert TB.crossover(d, "carry") == 0.01 and TB.crossover(d, "plain") == 0.03
