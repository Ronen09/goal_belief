"""TASK15 (round 16): inducing recurrence by random K/V dropout on history
(docs/task15_incentive_theory.md).

    .venv/bin/python scripts/run_task15.py                 # ~30 min (GPU for plain, CPU workers for carry)
    .venv/bin/python scripts/run_task15.py --measure-only
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import beliefcausal as BC, filterstate as FS, kvprior as KP, latentgoal as LG, wtfm as W
from goalgeo.analysis import to_jsonable

PS = (0.0, 0.5, 0.75, 0.9, 0.97, 1.0)
TS = (4, 8, 16, 23)
T = 24
N_PAIRS = 1000


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pname(p, s):
    return f"plain_L4_p{p}_s{s}"


def cname(p, s):
    return f"carry_p{p}_s{s}"


def make_carry(seed):
    env = LG.make_env("channel", 4)
    return W.WindowTransformer(env.V, env.M, T + 1, True, seed=seed)


def train_carry(args):
    p, seed, steps, path = args
    torch.set_num_threads(1)
    env = LG.make_env("channel", 4)
    X, Y = KP.lm_data(env, 100000, T, seed + 1000); X, Y = torch.as_tensor(X), torch.as_tensor(Y - 1)
    net = make_carry(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 7); gen = torch.Generator().manual_seed(seed + 11); hist = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), 128))
        z = net.forward_all(X[idx], drop=p, gen=gen)
        loss = F.cross_entropy(z.reshape(-1, z.shape[-1]), Y[idx].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            hist.append(loss.item())
    torch.save({"state": net.state_dict(), "hist": torch.as_tensor(hist)}, path)
    return path


def pairs(env, t):
    XA, _ = KP.lm_data(env, 20000, T, seed=900 + t); XB, _ = KP.lm_data(env, 20000, T, seed=950 + t)
    JA = LG.filter_joint(env, XA)[:, t]; JB = LG.filter_joint(env, XB)[:, t]
    keep = np.flatnonzero(np.linalg.norm(FS.full(JA) - FS.full(JB), axis=1) >= 2.0)[:N_PAIRS]
    XA, XB, JA, JB = XA[keep], XB[keep], JA[keep], JB[keep]
    x = XA[:, t + 1]
    oA = KP.centred_log(BC.ideal(env, KP.update(env, JA, x), "next_obs")); oB = KP.centred_log(BC.ideal(env, KP.update(env, JB, x), "next_obs"))
    return XA, XB, x, oA, oB


def history_visible(n, t, p, seed):
    """Query t+1's view of positions 0..t under dropout: t always visible, older kept w.p. 1 - p."""
    g = np.random.default_rng(seed); v = g.random((n, t + 1)) >= p; v[:, t] = True
    return v


def cells_plain(net, XA, XB, x, t, vis):
    L = len(net.blocks)
    RA, RB = KP.residuals(net, XA[:, :t + 1]), KP.residuals(net, XB[:, :t + 1])
    A = [r for r in RA[:L]]; B = [r for r in RB[:L]]
    splice = {"R1": A, "R2": [A[0]] + [np.concatenate([A[i][:, :t], B[i][:, t:t + 1]], 1) for i in range(1, L)],
              "R3": [np.concatenate([B[i][:, :t], A[i][:, t:t + 1]], 1) for i in range(L)], "R4": B}
    return {k: KP.forward_query(net, src, x, t + 1, visible=vis)[2] for k, src in splice.items()}


@torch.no_grad()
def cells_carry(net, XA, XB, x, t, vis):
    cA, uA = net.prefix(XA, t); cB, uB = net.prefix(XB, t)
    sp = lambda first, last: [first[i][:t] + [last[i][t]] for i in range(len(first))]     # noqa: E731
    cfg = {"R1": (cA, uA), "R2": (sp(cA, cB), uB), "R3": (sp(cB, cA), uA), "R4": (cB, uB)}
    V = None if vis is None else torch.as_tensor(np.concatenate([vis, np.ones((len(vis), 1), bool)], 1))
    out = {}
    for k, (c, u) in cfg.items():
        z, _, _ = net.advance(torch.as_tensor(x), c, u, V)
        out[k] = torch.softmax(z, -1).double().numpy()
    return out


def measure(args):
    fam, p, seed, path = args
    torch.set_num_threads(1)
    env = LG.make_env("channel", 4)
    if fam == "plain":
        net = KP.make_net(KP.LMSpec(4, T, seed)); net.load_state_dict(torch.load(path)["state"]); net.eval()
    else:
        net = make_carry(seed); net.load_state_dict(torch.load(path)["state"]); net.eval()
    X, _ = KP.lm_data(env, 3000, T, seed=12345)
    exact = LG.next_obs(env, X)
    if fam == "plain":
        pf = KP.predictive(net, X)
        pd = KP.predictive_masked(net, X, KP.drop_visible(p, len(X), T + 1, gen=torch.Generator().manual_seed(3)).numpy())
    else:
        with torch.no_grad():
            pf = torch.softmax(net.forward_all(torch.as_tensor(X)), -1).double().numpy()
            pd = torch.softmax(net.forward_all(torch.as_tensor(X), drop=p, gen=torch.Generator().manual_seed(3)), -1).double().numpy()
    rec = {"family": fam, "p": p, "seed": seed, "kl_full": float(BC.kl(exact[:, 1:], pf[:, 1:]).mean()),
           "kl_drop": float(BC.kl(exact[:, 1:], pd[:, 1:]).mean()), "t": {}}
    for t in TS:
        XA, XB, x, oA, oB = pairs(env, t)
        rec["t"][str(t)] = {}
        for regime, vis in (("full", None), ("drop", history_visible(len(x), t, p, 100 + t))):
            outs = (cells_plain if fam == "plain" else cells_carry)(net, XA, XB, x, t, vis)
            rec["t"][str(t)][regime] = {k: KP.along(KP.centred_log(v), oA, oB)[0] for k, v in outs.items()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results15")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--ps", type=float, nargs="*", help="override the dropout grid (follow-up runs)")
    ap.add_argument("--plain-ps", type=float, nargs="*", help="dropout grid for the plain family if different")
    a = ap.parse_args()
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    seeds = range(1) if a.quick else range(3); ps = (0.0, 1.0) if a.quick else tuple(a.ps or PS)
    pps = tuple(a.plain_ps) if a.plain_ps is not None else ps
    steps = 200 if a.quick else a.steps
    t0 = time.time(); ctx = get_context("spawn")
    if not a.measure_only:
        pool = ctx.Pool(min(a.workers, len(ps) * len(seeds)))
        carry_jobs = pool.map_async(train_carry, [(p, s, steps, str(out / "models" / f"{cname(p, s)}.pt")) for p in ps for s in seeds])
        specs = [KP.LMSpec(4, T, s) for p in pps for s in seeds]; drops = [p for p in pps for s in seeds]
        nets, _ = KP.train_lm_stack(specs, steps, n_pool=5000 if a.quick else 100000, drop=drops,
                                    log=lambda s, l: _log(f"plain step {s}: loss {l.mean():.4f}"))
        for net, p, s in zip(nets, drops, [s for p in pps for s in seeds]):
            torch.save({"state": net.state_dict()}, out / "models" / f"{pname(p, s)}.pt")
        _log(f"plain trained ({time.time() - t0:.0f}s)")
        carry_jobs.get(); pool.close(); pool.join()
        _log(f"carry trained ({time.time() - t0:.0f}s)")
    tasks = [(f, p, s, str(out / "models" / f"{(pname if f == 'plain' else cname)(p, s)}.pt"))
             for f in ("plain", "carry") for p in (pps if f == "plain" else ps) for s in seeds]
    with ctx.Pool(min(a.workers, len(tasks))) as pool:
        recs = pool.map(measure, tasks, chunksize=1)
    json.dump(to_jsonable({"runs": recs}), open(out / "results15.json", "w"))
    _log(f"measured {len(recs)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
