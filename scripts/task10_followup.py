"""TASK10 follow-up: rule out under-training.

Functional equivalence is the premise of both claims, so any model that did not reach the
Bayes floor is retrained with `--long` steps. For experiment 2 this is the load-bearing
control: a `frozen` model below c_crit must fail *because of the ceiling*, so the
below-boundary models are retrained for 3x as long and must still fail, while `none` and
`learn` models at the same gain must still succeed.

    .venv/bin/python scripts/task10_followup.py [--long 30000] [--jobs 8]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from goalgeo.analysis import to_jsonable
from run_task10 import ROUTING, analyse_factor, analyse_routing, run_factor, run_routing, write_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results10"); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--long", type=int, default=30000); ap.add_argument("--kl-tol", type=float, default=0.003)
    ap.add_argument("--lr-control", type=float, default=3e-3, help="second learning rate for the frozen boundary models")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = Path(args.out); t0 = time.time()
    R = json.loads((out / "results10.json").read_text()); ctx = mp.get_context("spawn")
    kw = dict(ROUTING)

    redo_r = [r for r in R.get("routing_runs", []) if r["kl"] >= args.kl_tol]
    if redo_r:
        todo = [(r["label"], kw[r["label"]], r["seed"], args.long, False, args.device, str(out)) for r in redo_r]
        print(f"[routing] retraining {len(todo)} runs at {args.long} steps", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            new = list(ex.map(run_routing, todo))
        keep = {(r["label"], r["seed"]) for r in new}
        R["routing_runs"] = [r for r in R["routing_runs"] if (r["label"], r["seed"]) not in keep] + new
        for r in new:
            print(f"  {r['label']:8s} s{r['seed']} kl={r['kl']:.4f} cut={r['phi_L1{t,t+1}']:.4f} "
                  f"t={r['phi_L1{t}']:.4f} t+1={r['phi_L1{t+1}']:.4f}", flush=True)
        R["routing"] = analyse_routing(R["routing_runs"], args.kl_tol)

    # experiment 2: retrain every model that missed the floor, plus every below-boundary model
    fr = R.get("factor_runs", [])
    c_crit = {0.4: 0.1373, 0.2: 0.0530}
    redo_f = [r for r in fr if (not r["converged"]) or r["kl"] >= args.kl_tol or r["gain"] < c_crit[r["delta"]]]
    if redo_f:
        todo = [(r["norm"], r["gain"], r["delta"], r["seed"], args.long, False, args.device, str(out)) for r in redo_f]
        print(f"[factor] retraining {len(todo)} runs at {args.long} steps", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
            new = list(ex.map(run_factor, todo))
        keep = {(r["norm"], r["gain"], r["delta"], r["seed"]) for r in new}
        R["factor_runs"] = [r for r in fr if (r["norm"], r["gain"], r["delta"], r["seed"]) not in keep] + new
        for r in sorted(new, key=lambda r: (r["delta"], r["norm"], r["gain"])):
            print(f"  {r['label']:18s} s{r['seed']} kl={r['kl']:.4f} klrel={r['kl_relevant']:.4f} "
                  f"C={r['C']:.3f}/{r['C_required']:.3f} ceiling={r['ceiling']:.2f} D={r['D']:.2f} "
                  f"cos={r['cos_theta']:.3f} conv={r['converged']:.0f}", flush=True)
        R["factor"] = analyse_factor(R["factor_runs"])
    # an optimiser control: if the frozen models fail because of the ceiling and not because
    # small readout gradients stall training, a 3x learning rate must not rescue them.
    ctrl = [(n, c, d, sd) for n in ("frozen",) for c, d in ((0.1, 0.4), (0.2, 0.4), (0.3, 0.4), (0.04, 0.2), (0.07, 0.2)) for sd in (0,)]
    todo = [(n, c, d, sd, args.long, False, args.device, str(out), args.lr_control) for n, c, d, sd in ctrl]
    print(f"[lr control] {len(todo)} frozen runs at lr={args.lr_control:g}, {args.long} steps", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
        new = list(ex.map(run_factor, todo))
    for r in new:
        print(f"  {r['label']:22s} s{r['seed']} kl={r['kl']:.4f} C={r['C']:.3f}/{r['C_required']:.3f} conv={r['converged']:.0f}", flush=True)
    R["factor_runs"] = R.get("factor_runs", []) + new
    R["factor"] = analyse_factor(R["factor_runs"])
    R["followup"] = {"long_steps": args.long, "lr_control": args.lr_control, "n_retrained_routing": len(redo_r), "n_retrained_factor": len(redo_f)}

    write_tables(R, out); (out / "results10.json").write_text(json.dumps(to_jsonable(R), indent=1))
    from goalgeo.plotting10 import make_figures
    make_figures(R, out)
    for exp in ("routing", "factor"):
        if exp in R:
            print(f"\n{exp}:"); [print(f"  {'HELD  ' if v else 'FAILED'} {k}") for k, v in R[exp]["predictions"].items()]
    print(f"done in {time.time()-t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
