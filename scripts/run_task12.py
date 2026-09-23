"""TASK12 (round 13): the full filter state (docs/task12_filter_theory.md).

Trains the hidden-size bottleneck GRUs, then measures full-state recoverability, matched
two-coordinate interventions and the equivalence hierarchy on the round-12 channel models and the
bottleneck models.

    .venv/bin/python scripts/run_task12.py                 # ~15 min, CPU workers
    .venv/bin/python scripts/run_task12.py --measure-only
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
from goalgeo import belief_train as BT, beliefcausal as BC, beliefprobe as BP, filterstate as FS, latentgoal as LG
from goalgeo.analysis import to_jsonable

R12 = Path("results11/models")
HIDDEN = (2, 3, 4, 5, 6, 7, 8, 12, 16, 32)


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def bottleneck_jobs(seeds=range(3)):
    return [BT.Job("gru", kind, 4, "goal", s, extra={"hidden": n}) for kind in ("iid", "channel") for n in HIDDEN for s in seeds]


def _train(args):
    job, path = args
    state, hist = BT.train_gru(job, 15000)
    torch.save({"state": state, "hist": torch.as_tensor(hist)}, path)
    return job.name


def build_pairs(path: Path, quick: bool):
    env = LG.make_env("channel", 4); n_a, n_e = (100, 40) if quick else (1500, 500)
    P = {}
    for t in (6, 12):
        P[f"A{t}_XA"], P[f"A{t}_XB"] = BC.pairs_transplant(env, t, n_a, seed=900 + t)
    P["Eb_H1"], P["Eb_H2"], P["Eb_J1"] = BC.pairs_equal(env, n_e, seed=600, mode="chan_equal_b")
    P["Em_H1"], P["Em_H2"], P["Em_J1"] = FS.pairs_equal_marginals(env, n_e, seed=601)
    P["Ej_H1"], P["Ej_H2"], P["Ej_J1"] = FS.pairs_equal_full(env, n_e, seed=602, pool=300000 if quick else 1000000)
    P["Ej3_H1"], P["Ej3_H2"], P["Ej3_J1"] = FS.pairs_equal_full(env, n_e, seed=602, pool=300000 if quick else 1000000, min_positions=3)
    P["R_H1"], P["R_H2"], P["R_J1"] = BC.pairs_random(env, n_e, seed=700)
    P["Ex_H1"], P["Ex_H2"], P["Ex_J1"], P["Ex_J2"] = FS.pairs_cross_length(env, n_e, seed=603)
    P["Xr_H1"], P["Xr_H2"], P["Xr_J1"], P["Xr_J2"] = FS.pairs_cross_random(env, n_e, seed=604)
    np.savez_compressed(path, **P)
    _log("pairs: " + ", ".join(f"{k[:-3]} {v.shape[0]}" for k, v in P.items() if k.endswith(("_H1", "_XA"))))


_F = {}


def _full():
    if "F" not in _F:
        _F["F"] = FS.FullEval()
    return _F["F"]


def _eval(kind):
    if kind not in _F:
        _F[kind] = BP.EvalSet(kind, 4)
    return _F[kind]


def _ser(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def worker(args):
    job, path, pairs_path, full_tests = args
    torch.set_num_threads(1); rng = np.random.default_rng(job.seed + 5)
    net = BT.load_net(job, path)
    env = LG.make_env(job.kind, 4)
    rec = {"name": job.name if path else f"{job.arch}_{job.kind}_init_s{job.seed}", "arch": job.arch, "kind": job.kind,
           "objective": job.objective if path else "init", "seed": job.seed, "hidden": job.extra.get("hidden", 64)}
    if job.kind == "iid":                                     # bottleneck control: goal block only
        E = _eval("iid"); S = BP.gru_sites(net, E.X)
        rec["recover"] = {"IID": {"goal": BP.measure_site(S["h"], E)["IID_r2y"]}}
        rec["behaviour"] = BP.behaviour(S["p"], E, job.objective)
        return rec
    F = _full()
    if job.arch == "tfm":
        S = BP.tfm_sites(net, F.E.X, device="cpu")
        rec["recover"] = {s: FS.recover(S[s], F) for s in ("res1", "res2", "u")}
        rec["behaviour"] = BP.behaviour(S["p"], F.E, job.objective) if path else None
        return rec
    S = BP.gru_sites(net, F.E.X)
    rec["recover"] = FS.recover(S["h"], F)
    if not path:
        return rec
    rec["behaviour"] = BP.behaviour(S["p"], F.E, job.objective)
    if not full_tests:
        pass
    P = np.load(pairs_path)
    states = BC.gru_states(net, F.E.X)
    bmap = BC.BeliefMap(states[:, 1:].reshape(-1, states.shape[-1]), F.Y)
    run_full = lambda X, t: BC.gru_outputs(net, X, t)                                       # noqa: E731
    if full_tests:
        rec["I"] = {}
        for t in (6, 12):
            XA, XB = P[f"A{t}_XA"], P[f"A{t}_XB"]
            rec["I"][f"t{t}"] = _ser(FS.test_matched(lambda X: run_full(X, t), lambda h: BC.gru_from(net, h, XA[:, t + 1:]),
                                                     lambda X: BC.gru_states(net, X)[:, t], bmap, env, job.objective, XA, XB, t, rng))
    rec["E"] = {}
    for m in (("Eb", "Em", "Ej", "Ej3", "R") if full_tests else ("Ej", "Ej3", "R")):
        H1, H2, J1 = P[f"{m}_H1"], P[f"{m}_H2"], P[f"{m}_J1"]
        rec["E"][m] = _ser(BC.test_equal(lambda X: run_full(X, 12), env, job.objective, H1, H2, J1, np.random.default_rng(1)))

    def from_pair(H1, H2, c):
        h1 = BC.gru_states(net, H1)[:, -1]; h2 = BC.gru_states(net, H2)[:, -1]
        return BC.gru_from(net, h1, c), BC.gru_from(net, h2, c)
    for m in ("Ex", "Xr"):
        rec["E"][m] = _ser(FS.test_from_states(from_pair, env, job.objective, P[f"{m}_H1"], P[f"{m}_H2"], P[f"{m}_J1"], P[f"{m}_J2"],
                                               np.random.default_rng(2)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results12")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    t0 = time.time(); ctx = get_context("spawn")
    bjobs = bottleneck_jobs(range(1) if a.quick else range(3))
    if a.quick:
        bjobs = [j for j in bjobs if j.extra["hidden"] in (3, 8)]
    if not a.measure_only:
        with ctx.Pool(min(a.workers, len(bjobs))) as pool:
            pool.map(_train, [(j, str(out / "models" / f"{j.name}.pt")) for j in bjobs], chunksize=1)
        _log(f"trained {len(bjobs)} bottleneck GRUs ({time.time() - t0:.0f}s)")
    build_pairs(out / "pairs.npz", a.quick)
    seeds = range(1) if a.quick else range(4)
    objs = ("goal",) if a.quick else ("goal", "act_soft", "next_obs", "act_hard")
    tasks = [(BT.Job("gru", "channel", 4, o, s), R12 / f"gru_channel_K4_{o}_s{s}.pt", out / "pairs.npz", True) for o in objs for s in seeds]
    tasks += [(BT.Job("gru", "channel", 4, "goal", s), None, None, False) for s in seeds]                       # untrained
    tasks += [(BT.Job("tfm", "channel", 4, o, s), R12 / f"tfm_channel_K4_{o}_s{s}.pt", None, False) for o in objs for s in seeds]
    tasks += [(j, out / "models" / f"{j.name}.pt", out / "pairs.npz", False) for j in bjobs]
    with ctx.Pool(min(a.workers, len(tasks))) as pool:
        recs = pool.map(worker, tasks, chunksize=1)
    conv = {r["name"]: r.get("converged") for r in json.load(open("results11/results11.json"))["runs"]}
    for r in recs:
        if r.get("behaviour"):
            r["converged"] = conv.get(r["name"], r["behaviour"]["kl"] < 0.01 if r["objective"] != "act_hard" else None)
    base = FS.baselines(_full())
    json.dump(to_jsonable({"runs": recs, "baselines": base}), open(out / "results12.json", "w"))
    _log(f"{len(recs)} measured -> {out / 'results12.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
