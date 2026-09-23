"""TASK11 (round 12): a hidden goal. Train transformers and GRUs under four objectives on two
evidence processes, then probe every representation site for affine recoverability of the
posterior log-odds (docs/task11_theory.md).

    .venv/bin/python scripts/run_task11.py                  # train + measure, ~25 min
    .venv/bin/python scripts/run_task11.py --measure-only   # re-measure from results11/models
    .venv/bin/python scripts/run_task11.py --quick --out results11/_smoke
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # workers: one BLAS thread each
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import belief_train as BT, beliefprobe as BP, latentgoal as LG, tfm as Tf
from goalgeo.analysis import to_jsonable

KINDS = ("iid", "channel")


def job_grid(quick: bool):
    s_main, s_ho, s_k = (range(1), range(1), range(1)) if quick else (range(4), range(3), range(3))
    J = []
    for arch in ("tfm", "gru"):
        for kind in KINDS:
            for obj in LG.OBJECTIVES:
                J += [BT.Job(arch, kind, 4, obj, s) for s in s_main]
                J += [BT.Job(arch, kind, 4, obj, s, netho=True) for s in s_ho]
            for K in (3, 5):
                for obj in ("goal", "act_hard"):
                    J += [BT.Job(arch, kind, K, obj, s) for s in s_k]
    return J


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _gru_worker(args):
    job, steps, path = args
    state, hist = BT.train_gru(job, steps)
    torch.save({"state": state, "hist": torch.as_tensor(hist)}, path)
    return job.name, float(hist[-1])


def train_all(jobs, out: Path, tfm_steps: int, gru_steps: int, workers: int):
    mdir = out / "models"; mdir.mkdir(parents=True, exist_ok=True)
    grus = [(j, gru_steps, str(mdir / f"{j.name}.pt")) for j in jobs if j.arch == "gru"]
    ctx = get_context("spawn")
    with ctx.Pool(min(workers, len(grus))) as pool:
        async_res = pool.map_async(_gru_worker, grus)            # CPU, while the GPU trains the stacks
        for K in sorted({j.K for j in jobs if j.arch == "tfm"}):
            stack = [j for j in jobs if j.arch == "tfm" and j.K == K]
            t0 = time.time()
            nets, hist = BT.train_tfm_stack(stack, tfm_steps,
                                            log=lambda s, l: _log(f"tfm K={K} step {s}: mean loss {l.mean():.4f}"))
            for net, j, h in zip(nets, stack, hist.T):
                torch.save({"state": net.state_dict(), "out_mask": net.out_mask, "hist": torch.as_tensor(h)}, mdir / f"{j.name}.pt")
            _log(f"tfm K={K}: {len(stack)} models in {time.time() - t0:.0f}s")
        for name, last in async_res.get():
            pass
    _log(f"gru: {len(grus)} models done")


# ---- measurement -------------------------------------------------------------------------
_E = {}


def _eval(kind, K):
    if (kind, K) not in _E:
        _E[(kind, K)] = BP.EvalSet(kind, K)
    return _E[(kind, K)]


def load_net(job: BT.Job, path: Path | None):
    """path None -> the untrained network of that seed."""
    env = LG.make_env(job.kind, job.K)
    if job.arch == "gru":
        return BT.make_gru(job, None if path is None else torch.load(path)["state"])
    net = Tf.CausalTransformer(n_vocab=env.V, T=LG.T_OBS + 1, seed=job.seed, final_norm="learn", **BT.TFM_KW)
    mask = torch.full((env.V,), BT.NEG); mask[:LG.out_dim(env, job.objective)] = 0.0
    if path is not None:
        ck = torch.load(path); net.load_state_dict(ck["state"]); mask = ck["out_mask"]
    net.out_mask = mask
    return net.eval()


def _measure_worker(args):
    job, path = args
    torch.set_num_threads(1)
    E = _eval(job.kind, job.K)
    net = load_net(job, None if path is None else Path(path))
    S = BP.gru_sites(net, E.X) if job.arch == "gru" else BP.tfm_sites(net, E.X, device="cpu")
    rec = {"name": job.name if path else f"{job.arch}_{job.kind}_K{job.K}_init_s{job.seed}", "arch": job.arch,
           "kind": job.kind, "K": job.K, "objective": job.objective if path else "init", "seed": job.seed,
           "netho": job.netho, "sites": {k: BP.measure_site(v, E) for k, v in S.items() if k != "p"}}
    if path:
        rec["behaviour"] = BP.behaviour(S["p"], E, job.objective)
        rec["final_loss"] = float(torch.load(path)["hist"][-1])
    return rec


def converged(rec):
    b = rec["behaviour"]
    return b["act_agree"] >= 0.99 if rec["objective"] == "act_hard" else b["kl"] < 0.01


def measure_all(jobs, out: Path, workers: int):
    mdir = out / "models"
    tasks = [(j, str(mdir / f"{j.name}.pt")) for j in jobs]
    inits = {}
    for j in jobs:                                   # untrained controls: 4 seeds per (arch, env, K)
        for s in range(4):
            inits[(j.arch, j.kind, j.K, s)] = BT.Job(j.arch, j.kind, j.K, "goal", s)
    tasks += [(j, None) for j in inits.values()]
    ctx = get_context("spawn")
    with ctx.Pool(workers) as pool:
        recs = pool.map(_measure_worker, tasks, chunksize=1)
    for r in recs:
        if r["objective"] != "init":
            r["converged"] = converged(r)
    base = {f"{kind}_K{K}": BP.baselines(_eval(kind, K)) for kind in KINDS for K in sorted({j.K for j in jobs})}
    json.dump(to_jsonable({"runs": recs, "baselines": base}), open(out / "results11.json", "w"), indent=1)
    _log(f"measured {len(recs)} networks -> {out / 'results11.json'}")
    return recs, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results11")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--tfm-steps", type=int, default=20000)
    ap.add_argument("--gru-steps", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs = job_grid(a.quick)
    if a.quick:
        a.tfm_steps, a.gru_steps = min(a.tfm_steps, 300), min(a.gru_steps, 300)
    _log(f"{len(jobs)} jobs ({sum(j.arch == 'tfm' for j in jobs)} transformers)")
    t0 = time.time()
    if not a.measure_only:
        train_all(jobs, out, a.tfm_steps, a.gru_steps, a.workers)
    _log(f"training phase {time.time() - t0:.0f}s"); t0 = time.time()
    measure_all(jobs, out, a.workers)
    _log(f"measurement phase {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
