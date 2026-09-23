"""TASK13 (round 14): windowed transformers with and without a recurrent carry
(rounds/r14_window_carry/THEORY.md).

    .venv/bin/python rounds/r14_window_carry/run.py                  # ~25 min on CPU workers
    .venv/bin/python rounds/r14_window_carry/run.py --measure-only
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from goalgeo import beliefcausal as BC, beliefprobe as BP, filterstate as FS, latentgoal as LG, wtfm as W
from goalgeo.analysis import to_jsonable

WINDOWS = (1, 2, 4, 8, 25)
PAIRS = {"iid": "rounds/r12_hidden_goal/causal/pairs_iid.npz", "channel": "rounds/r13_filter_state/pairs.npz"}
N_PAIRS = 1000


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def name(kind, l, carry, seed):
    return f"wtfm_{kind}_l{l}_{'carry' if carry else 'nocarry'}_s{seed}"


def make(kind, l, carry, seed):
    env = LG.make_env(kind, 4)
    return W.WindowTransformer(env.V, env.K, l, carry, seed=seed)


def train(args):
    kind, l, carry, seed, steps, path = args
    torch.set_num_threads(1)
    env = LG.make_env(kind, 4)
    X, _, _ = LG.sample(env, 20000, seed=seed + 101)
    Y = torch.as_tensor(LG.targets(env, X, "goal"), dtype=torch.float32); X = torch.as_tensor(X)
    net = make(kind, l, carry, seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 7); hist = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), 128))
        loss = -(Y[idx] * F.log_softmax(net.forward_all(X[idx]), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append(loss.item())
    torch.save({"state": net.state_dict(), "hist": torch.as_tensor(hist)}, path)
    return path


@torch.no_grad()
def states(net, X, batch=2000):
    zs, us, rs = [], [], []
    for i in range(0, len(X), batch):
        z, u, r = net.run(torch.as_tensor(X[i:i + batch]), keep=True)
        zs.append(torch.softmax(z, -1).double().numpy()); us.append(u.double().numpy()); rs.append(r[:, :, 1].double().numpy())
    return np.concatenate(zs), np.concatenate(us), np.concatenate(rs)          # p, u, res1: [n, T, .]


def _ser(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


def measure(args):
    kind, l, carry, seed, path = args
    torch.set_num_threads(1)
    env = LG.make_env(kind, 4)
    net = make(kind, l, carry, seed); net.load_state_dict(torch.load(path)["state"]); net.eval()
    Fe = FS.FullEval() if kind == "channel" else None
    E = Fe.E if Fe else BP.EvalSet("iid", 4)
    p, u, r1 = states(net, E.X)
    b = E.b_full
    klpos = (b * (np.log(np.clip(b, 1e-300, None)) - np.log(np.clip(p, 1e-300, None)))).sum(-1).mean(0)   # [T+1]
    rec = {"kind": kind, "window": l, "carry": carry, "seed": seed, "kl": float(klpos[1:].mean()), "kl_pos": klpos.tolist()}
    U = u[:, 1:].reshape(-1, u.shape[-1])
    rec["r2_goal"] = BP.r2(E.y, BP.cv_pred(U, E.y, E.fold))
    if Fe:
        rec["recover_u"] = FS.recover(U, Fe)
        rec["recover_res1"] = FS.recover(r1[:, 1:].reshape(-1, r1.shape[-1]), Fe)
    bmap = BC.BeliefMap(U, Fe.Y if Fe else E.y)
    P = np.load(PAIRS[kind]); rec["steer"] = {}
    for t in (6, 12):
        XA, XB = P[f"A{t}_XA"][:N_PAIRS], P[f"A{t}_XB"][:N_PAIRS]
        XS = np.concatenate([XB[:, :t + 1], XA[:, t + 1:]], 1)
        pA, uA, _ = states(net, XA); pB, uB, rB = states(net, XB); pS, _, _ = states(net, XS)
        pA, pS = pA[:, t:], pS[:, t:]
        JB = LG.filter_joint(env, XB)[:, t]
        tgt = FS.full(JB) if Fe else LG.log_odds(JB.sum(-1))
        XAt = torch.as_tensor(XA)
        edits = {"swap_u": net.run_edited(XAt, t, u_new=uB[:, t]),
                 "probe_u": net.run_edited(XAt, t, u_new=bmap.probe_e(uA[:, t], tgt)),
                 "swap_res1": net.run_edited(XAt, t, res1_new=rB[:, t])}
        # per-offset mean divergences to the genuine-B future; the gap closed is pooled over offsets
        # in the tables (a per-offset ratio is 0/0 wherever the unedited future already equals B's)
        rec["steer"][f"t{t}"] = {k: BC.js(v, pS).mean(0).tolist() for k, v in edits.items()}
        rec["steer"][f"t{t}"]["base"] = BC.js(pA, pS).mean(0).tolist()
        rec["steer"][f"t{t}"]["identity_check"] = float(np.abs(net.run_edited(XAt, t) - pA).max())
    if Fe:
        rec["equal"] = {}
        for m in ("Ex", "Xr"):
            H1, H2, J1, J2 = P[f"{m}_H1"], P[f"{m}_H2"], P[f"{m}_J1"], P[f"{m}_J2"]
            rng = np.random.default_rng(2); dm, db = [], []
            for _ in range(8):
                c = BC.sample_continuation(env, J1, 8, rng)
                p1, _, _ = states(net, np.concatenate([H1, c], 1)); p2, _, _ = states(net, np.concatenate([H2, c], 1))
                p1, p2 = p1[:, H1.shape[1] - 1:], p2[:, H2.shape[1] - 1:]
                i1 = BC.ideal(env, BC.filter_continue(env, J1, c), "goal"); i2 = BC.ideal(env, BC.filter_continue(env, J2, c), "goal")
                dm.append(BC.js(p1, p2)[:, 1:].mean()); db.append(BC.js(i1, i2)[:, 1:].mean())
            rec["equal"][m] = {"model": float(np.mean(dm)), "bayes": float(np.mean(db))}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rounds/r14_window_carry")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args()
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    seeds = range(1) if a.quick else range(3)
    wins = (1, 4) if a.quick else WINDOWS
    steps = 200 if a.quick else a.steps
    jobs = [(k, l, c, s) for k in ("iid", "channel") for l in wins for c in (False, True) for s in seeds]
    ctx = get_context("spawn"); t0 = time.time()
    if not a.measure_only:
        with ctx.Pool(min(a.workers, len(jobs))) as pool:
            pool.map(train, [j + (steps, str(out / "models" / f"{name(*j)}.pt")) for j in jobs], chunksize=1)
        _log(f"trained {len(jobs)} models ({time.time() - t0:.0f}s)")
    with ctx.Pool(min(a.workers, len(jobs))) as pool:
        recs = pool.map(measure, [j + (str(out / "models" / f"{name(*j)}.pt"),) for j in jobs], chunksize=1)
    json.dump(to_jsonable({"runs": recs}), open(out / "results13.json", "w"))
    _log(f"measured {len(recs)} -> {out / 'results13.json'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
