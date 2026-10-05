"""The balanced-prediction experiment, post hoc: policy inconsistency (Q1) at equal regret. Q1 at every checkpoint from update 200 for the two
balanced arms (the observation-prediction experiment's results.json has the same curves for the selected and reward-only arms), and the
difference from the selected arm within bins of greedy regret, as in the observation-prediction experiment.

    .venv/bin/python studies/3_reward_trained_agents/balanced_prediction/matched.py            # writes matched.json and matched.md
"""

from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazeaux as AX

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/
sys.path.insert(0, str(ROUNDS / "3_reward_trained_agents" / "maze_belief"))
import train as T18                                                  # noqa: E402

spec = importlib.util.spec_from_file_location("r23measure", ROUNDS / "3_reward_trained_agents" / "obs_prediction" / "measure.py")
R23 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R23)
EDGES = (0.004, 0.006, 0.01, 0.02, 0.05)
NAMES = ["< 0.004", "0.004–0.006", "0.006–0.01", "0.01–0.02", "0.02–0.05", "> 0.05"]
f = lambda x: "—" if x != x else f"{x:.3f}"


def curves():
    t = T18.tables("cuda"); data = R23.R22.Data(t)
    out = {}
    for arm in ("all", "one"):
        out[arm] = {}
        for run in sorted((HERE / "runs" / arm).glob("seed*"), key=lambda p: int(p.name[4:])):
            net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to("cuda").eval()
            rows = []
            for r in json.load(open(run / "log.json"))["log"]:
                if r["update"] >= R23.FROM:
                    net.load_state_dict(torch.load(run / "ckpt" / f"u{r['update']:06d}.pt"))
                    o = R23.outcomes(net, data)
                    rows.append(dict(update=r["update"], regret=r["greedy_regret"], A_tv=o["A"]["tv"], B_greedy_changed=o["B"]["greedy_changed"]))
            out[arm][run.name] = rows
            print(arm, run.name, flush=True)
    return out


def binned(C, key, rng=None):
    seeds = list(C)
    if rng is not None:
        seeds = [seeds[i] for i in rng.integers(len(seeds), size=len(seeds))]
    x = np.array([c["regret"] for s in seeds for c in C[s]]); y = np.array([c[key] for s in seeds for c in C[s]])
    b = np.digitize(x, EDGES)
    return np.array([y[b == i].mean() if (b == i).any() else np.nan for i in range(len(EDGES) + 1)]), np.bincount(b, minlength=len(EDGES) + 1)


def matched(Ca, Cr, key, rng=None):
    ya, na = binned(Ca, key, rng); yr, nr = binned(Cr, key, rng)
    ok = (na > 0) & (nr > 0)
    w = (na + nr) * ok
    return float(((ya - yr) * w)[ok].sum() / w.sum())


def main():
    p = HERE / "matched.json"
    if not p.exists():
        p.write_text(json.dumps(curves()))
    C = json.load(open(p))
    r23 = json.load(open(ROUNDS / "3_reward_trained_agents" / "obs_prediction" / "results.json"))["runs"]
    C["selected"] = {s: v["curve"] for s, v in r23["aux1"].items()}; C["reward"] = {s: v["curve"] for s, v in r23["reward"].items()}
    L = ["# The balanced-prediction experiment, post hoc: at equal regret", "", "Checkpoints from update 200, binned by greedy regret; mean over checkpoints (number). Difference from the selected arm within bins, "
         "weighted by the number of checkpoints, with a bootstrap interval over seeds.", ""]
    for key, title in (("A_tv", "Q1: type A, TV between the two histories' action distributions"), ("B_greedy_changed", "O2: greedy changed where the optimal set is the same")):
        L += ["", f"## {title}", "", "| arm | " + " | ".join(NAMES) + " | final median: difference from selected | within bins [95 %] |", "|" + "---|" * (len(NAMES) + 3)]
        for arm, name in (("selected", "selected"), ("all", "balanced, all"), ("one", "balanced, one"), ("reward", "reward only")):
            y, n = binned(C[arm], key)
            ext = ["", ""]
            if arm != "selected":
                rng = np.random.default_rng(0)
                bs = [matched(C[arm], C["selected"], key, rng) for _ in range(2000)]
                fin = lambda a: np.median([C[a][s][-1][key] for s in C[a]])
                ext = [f(fin(arm) - fin("selected")), f"{f(matched(C[arm], C['selected'], key))} [{f(np.nanquantile(bs, 0.025))}, {f(np.nanquantile(bs, 0.975))}]"]
            L.append("| " + " | ".join([name] + [f"{f(v)} ({k})" for v, k in zip(y, n)] + ext) + " |")
    (HERE / "matched.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[4:]))


if __name__ == "__main__":
    main()
