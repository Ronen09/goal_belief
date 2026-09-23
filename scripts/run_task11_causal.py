"""TASK11 part 2: causal tests of the decoded posterior on the round-12 models
(docs/task11_causal_theory.md). No training.

    .venv/bin/python scripts/run_task11_causal.py            # ~15 min on 64 CPU workers
    .venv/bin/python scripts/run_task11_causal.py --quick    # 2 models, few pairs
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import belief_train as BT, beliefcausal as BC, beliefprobe as BP, latentgoal as LG
from goalgeo.analysis import to_jsonable

KINDS = ("iid", "channel")
T_TRANSPLANT = (6, 12)
EQUAL_SETS = {"iid": ("iid_exact",), "channel": ("chan_equal_b", "chan_equal_joint")}


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_pairs(out: Path, n_a: int, n_b: int, n_c: int):
    for kind in KINDS:
        env = LG.make_env(kind, 4); P = {}
        for t in T_TRANSPLANT:
            P[f"A{t}_XA"], P[f"A{t}_XB"] = BC.pairs_transplant(env, t, n_a, seed=500 + t)
        for mode in EQUAL_SETS[kind]:
            P[f"B_{mode}_H1"], P[f"B_{mode}_H2"], P[f"B_{mode}_J1"] = BC.pairs_equal(env, n_b, seed=600, mode=mode)
        P["B_random_H1"], P["B_random_H2"], P["B_random_J1"] = BC.pairs_random(env, n_b, seed=700)
        P["C_H1"], P["C_H2"], P["C_C"] = BC.pairs_same_action(env, n_c, seed=800)
        np.savez_compressed(out / f"pairs_{kind}.npz", **P)
        _log(f"pairs {kind}: " + ", ".join(f"{k} {v.shape[0]}" for k, v in P.items() if k.endswith(("XA", "H1"))))


_E = {}


def _eval(kind):
    if kind not in _E:
        _E[kind] = BP.EvalSet(kind, 4)
    return _E[kind]


def _series(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def worker(args):
    job, model_path, pairs_path = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(job.seed + 99)
    env = LG.make_env(job.kind, 4); E = _eval(job.kind); P = np.load(pairs_path)
    net = BT.load_net(job, model_path)
    rec = {"name": job.name, "arch": job.arch, "kind": job.kind, "objective": job.objective, "seed": job.seed,
           "A": {}, "B": {}, "C": {}}
    if job.arch == "gru":
        S = BC.gru_states(net, E.X)
        maps = {"h": BC.fit_map(S, E)}
        run_full = lambda X, t: BC.gru_outputs(net, X, t)                          # noqa: E731
        sites_A = {"h": (lambda X, t: BC.gru_states(net, X)[:, t],
                         lambda XA, t, h: BC.gru_from(net, h, XA[:, t + 1:]))}
        ladder_sites = {"h": S[:, 1:].reshape(-1, S.shape[-1])}
    else:
        R = BC.tfm_residuals(net, E.X)
        with torch.no_grad():
            U = net.ln_f(torch.as_tensor(R[-1], dtype=torch.float32)).double().numpy()
        maps = {f"res{s}": BC.fit_map(R[s], E) for s in (1, 2)}
        run_full = lambda X, t: BC.tfm_outputs(net, X, t)                          # noqa: E731
        sites_A = {f"res{s}": ((lambda s: lambda X, t: BC.tfm_residuals(net, X)[s][:, t])(s),
                               (lambda s: lambda XA, t, h: BC.tfm_patch(net, XA, t, s, h))(s)) for s in (1, 2)}
        ladder_sites = {"res0": R[0], "res1": R[1], "res2": R[2], "u": U}
        ladder_sites = {k: v[:, 1:].reshape(-1, v.shape[-1]) for k, v in ladder_sites.items()}
    for t in T_TRANSPLANT:
        XA, XB = P[f"A{t}_XA"], P[f"A{t}_XB"]
        for site, (get_state, from_state) in sites_A.items():
            r = BC.test_transplant(lambda X: run_full(X, t), lambda h: from_state(XA, t, h),
                                   lambda X: get_state(X, t), maps[site], env, job.objective, XA, XB, t, rng)
            rec["A"][f"t{t}_{site}"] = _series(r)
    for mode in EQUAL_SETS[job.kind] + ("random",):
        H1, H2, J1 = P[f"B_{mode}_H1"], P[f"B_{mode}_H2"], P[f"B_{mode}_J1"]
        rec["B"][mode] = _series(BC.test_equal(lambda X: run_full(X, H1.shape[1] - 1), env, job.objective, H1, H2, J1,
                                               np.random.default_rng(1)))
    rec["C"]["flip"] = BC.test_flip(lambda X: run_full(X, P["C_H1"].shape[1] - 1), env, job.objective,
                                    P["C_H1"], P["C_H2"], P["C_C"])
    rec["C"]["ladder"] = BC.ladder(ladder_sites, E)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results11/causal")
    ap.add_argument("--models", default="results11/models")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_a, n_b, n_c = (100, 40, 100) if a.quick else (1500, 500, 1500)
    t0 = time.time(); build_pairs(out, n_a, n_b, n_c); _log(f"pairs {time.time() - t0:.0f}s")
    seeds = range(1) if a.quick else range(4)
    objs = ("goal",) if a.quick else LG.OBJECTIVES
    jobs = [BT.Job(arch, kind, 4, o, s) for arch in ("gru", "tfm") for kind in KINDS for o in objs for s in seeds]
    conv = {r["name"]: r.get("converged") for r in json.load(open(Path(a.models).parent / "results11.json"))["runs"]}
    tasks = [(j, str(Path(a.models) / f"{j.name}.pt"), str(out / f"pairs_{j.kind}.npz")) for j in jobs]
    with get_context("spawn").Pool(min(a.workers, len(tasks))) as pool:
        recs = pool.map(worker, tasks, chunksize=1)
    for r in recs:
        r["converged"] = conv.get(r["name"])
    json.dump(to_jsonable({"runs": recs}), open(out / "causal.json", "w"))
    _log(f"{len(recs)} models -> {out / 'causal.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
