"""Round 25: the fixed evaluation on every arm. Prediction measures are round 24's, policy measures round 23's.

    .venv/bin/python rounds/r25_balanced_prediction/measure.py            # writes results.json
"""

from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path

import torch

from goalgeo import mazeaux as AX, mazebelief as MB, mazegraph as MG

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent
sys.path.insert(0, str(ROUNDS / "r18_maze_belief"))
import train as T18                                                  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R24 = load("r24run", ROUNDS / "r24_head_consistency" / "run.py")
R23 = load("r23measure", ROUNDS / "r23_obs_prediction" / "measure.py")
ARMS = (("selected", ROUNDS / "r23_obs_prediction" / "runs" / "aux1"), ("all", HERE / "runs" / "all"), ("one", HERE / "runs" / "one"),
        ("reward", ROUNDS / "r23_obs_prediction" / "runs" / "reward"))


def main():
    dev = "cuda"
    t = T18.tables(dev)
    p_obs = torch.tensor(MG.cached(MB.cross_maze(), ROUNDS / "r18_maze_belief" / "cache" / "graph.npz").p_obs, dtype=torch.float32, device=dev)
    data = R24.R22.Data(t)
    res = dict(types=data.info, runs={})
    for arm, path in ARMS:
        res["runs"][arm] = {}
        for run in sorted(path.glob("seed*"), key=lambda p: int(p.name[4:])):
            log = json.load(open(run / "log.json"))["log"][-1]
            net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(dev).eval()
            net.load_state_dict(torch.load(sorted((run / "ckpt").glob("u*.pt"))[-1]))
            r = dict(regret=log["greedy_regret"], **{f"regret_G{g}": log[f"greedy_regret_G{g}"] for g in (1, 2, 3)}, excess_nll=log["obs_nll"] - log["obs_nll_exact"],
                     policy=R23.outcomes(net, data), head=R24.analyse(net, data, p_obs))
            res["runs"][arm][run.name] = r
            h = r["head"]
            print(arm, run.name, f"regret {r['regret']:.4f} | P1 {h['A']['error']:.4f} P2 {h['A']['d_head']:.4f} (greedy move {h['A']['greedy_move']['error']:.4f} {h['A']['greedy_move']['d_head']:.4f}) | "
                  f"Q1 {r['policy']['A']['tv']:.4f} O2 {r['policy']['B']['greedy_changed']:.3f}", flush=True)
            (HERE / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
