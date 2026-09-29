"""Round 18, step 1: search small mazes for one where the location belief changes the optimal action.

For each candidate maze and each goal, exactly:
    V*        the optimal value from the start prior
    V_map     the value of acting as if the most likely cell were certain
    V_qmdp    the value of belief-weighted fully observed action values (no value for information)
and, over the decision states the optimal policy reaches (weighted by probability):
    info      the optimal action is optimal for no cell in the belief's support, if the cell were known
    map_diff  the optimal action set excludes the certainty-equivalent action
    uncertain the most likely cell has probability < 0.9
"""

from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from goalgeo import mazebelief as MB

HERE = Path(__file__).resolve().parent


CAP = 600_000                       # mazes with more reachable beliefs per goal than this are skipped


def assess(args):
    try:
        return _assess(args)
    except MemoryError:
        return None


def _assess(args):
    seed, kw = args
    m = MB.random_maze(seed, **kw)
    rows = []
    for g in m.goals:
        S = MB.Solver(m, g, cap=CAP)
        v = S.start_value()
        vm = MB.Solver(m, g, MB.map_policy(m, g), cap=CAP).start_value()
        vq = MB.Solver(m, g, MB.qmdp_policy(m, g), cap=CAP).start_value()
        Q = m.mdp_values(g)
        mp = MB.map_policy(m, g)
        tot = info = diff = unc = 0.0
        for b, k, p in MB.reach(m, g, S):
            q = S.q(b, k)
            opt = q >= q.max() - MB.TIE
            sup = np.nonzero(b > 1e-9)[0]
            good = np.zeros(4, bool)
            for s in sup:
                good |= Q[k, s] >= Q[k, s].max() - MB.TIE
            tot += p; info += p * (not (opt & good).any()); diff += p * (not opt[mp(b, k)]); unc += p * (b.max() < 0.9)
        rows.append(dict(goal=int(g), v_opt=v, v_map=vm, v_qmdp=vq, info=info / tot, map_diff=diff / tot, uncertain=unc / tot,
                         beliefs=len(S.memo)))
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    return dict(seed=seed, goals=rows, v_opt=mean("v_opt"), map_regret=mean("v_opt") - mean("v_map"),
                qmdp_regret=mean("v_opt") - mean("v_qmdp"), info=mean("info"), map_diff=mean("map_diff"), uncertain=mean("uncertain"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.15, 0.25])
    ap.add_argument("--pairs", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.eps, a.pairs, a.jobs = 4, [0.15], [2], 4
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, dict(eps=e, pairs=p, H=8 if a.quick else 12)) for e in a.eps for p in a.pairs for s in range(a.seeds)]
    with ProcessPoolExecutor(a.jobs, max_tasks_per_child=1) as ex:
        res = list(ex.map(assess, jobs))
    skipped = sum(r is None for r in res)
    for r, (s, kw) in zip(res, jobs):
        if r is not None:
            r.update(eps=kw["eps"], pairs=kw["pairs"])
    res = [r for r in res if r is not None]
    print(f"{len(res)} mazes solved, {skipped} skipped (over {CAP} reachable beliefs for a goal)")
    res.sort(key=lambda r: -r["map_regret"])
    (out / "tune.json").write_text(json.dumps(res, indent=1))
    print("pairs eps  seed  V*     map regret  qmdp regret  info   map_diff  uncertain")
    for r in res[:15]:
        print(f"{r['pairs']}     {r['eps']:.2f} {r['seed']:4d}  {r['v_opt']:.3f}  {r['map_regret']:.4f}      {r['qmdp_regret']:.4f}       "
              f"{r['info']:.3f}  {r['map_diff']:.3f}     {r['uncertain']:.3f}")
    allr = np.array([r["map_regret"] for r in res])
    print("all mazes: map regret median", np.median(allr), "90th pct", np.quantile(allr, 0.9))


if __name__ == "__main__":
    main()
