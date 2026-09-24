# rounds/r17_read_cost/run.py
"""TASK16 (round 17): pricing recomputation with a learned read gate (rounds/r17_read_cost/THEORY.md).

    .venv/bin/python rounds/r17_read_cost/run.py                 # ~1 h (GPU for plain, CPU workers for carry)
    .venv/bin/python rounds/r17_read_cost/run.py --measure-only
    .venv/bin/python rounds/r17_read_cost/run.py --quick --out rounds/r17_read_cost/_smoke/run
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
from goalgeo import beliefcausal as BC, filterstate as FS, kvprior as KP, latentgoal as LG, readgate as RG, wtfm as W
from goalgeo.analysis import to_jsonable

CS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)
TS = (4, 8, 16, 23)
T = 24
N_PAIRS = 1000
N_EVAL = 3000


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pname(c, s):
    return f"plain_L4_c{c}_s{s}"


def cname(c, s):
    return f"carry_c{c}_s{s}"


def make_carry(seed):
    env = LG.make_env("channel", 4)
    return W.WindowTransformer(env.V, env.M, T + 1, True, seed=seed, gated=True)


def train_carry(args):
    c, seed, steps, path = args
    torch.set_num_threads(1)
    env = LG.make_env("channel", 4)
    X, Y = KP.lm_data(env, 100000, T, seed + 1000); X, Y = torch.as_tensor(X), torch.as_tensor(Y - 1)
    net = make_carry(seed); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed + 7); gen = torch.Generator().manual_seed(seed + 11); hist = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), 128))
        z, p = net.forward_gated(X[idx], gate="train", gen=gen)
        ce = F.cross_entropy(z.reshape(-1, z.shape[-1]), Y[idx].reshape(-1))
        loss = ce + c * RG.penalty(p)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            hist.append([ce.item(), p[:, 2:].mean().item()])
    torch.save({"state": net.state_dict(), "hist": torch.as_tensor(hist)}, path)
    return path


def pairs(env, t):
    """Round 16's transplant pairs: full filter states >= 2 apart at position t."""
    XA, _ = KP.lm_data(env, 20000, T, seed=900 + t); XB, _ = KP.lm_data(env, 20000, T, seed=950 + t)
    JA = LG.filter_joint(env, XA)[:, t]; JB = LG.filter_joint(env, XB)[:, t]
    keep = np.flatnonzero(np.linalg.norm(FS.full(JA) - FS.full(JB), axis=1) >= 2.0)[:N_PAIRS]
    XA, XB, JA, JB = XA[keep], XB[keep], JA[keep], JB[keep]
    x = XA[:, t + 1]
    oA = KP.centred_log(BC.ideal(env, KP.update(env, JA, x), "next_obs")); oB = KP.centred_log(BC.ideal(env, KP.update(env, JB, x), "next_obs"))
    return XA, XB, x, oA, oB


def cells_plain(net, XA, XB, x, t, gate):
    """Predictive at t+1 and the mean gate decision per layer, per cell."""
    L = len(net.blocks)
    A = KP.residuals(net, XA[:, :t + 1], gate=gate)[:L]; B = KP.residuals(net, XB[:, :t + 1], gate=gate)[:L]   # sources under the same regime
    splice = {"R1": A, "R2": [A[0]] + [np.concatenate([A[i][:, :t], B[i][:, t:t + 1]], 1) for i in range(1, L)],
              "R3": [np.concatenate([B[i][:, :t], A[i][:, t:t + 1]], 1) for i in range(L)], "R4": B}
    out = {}
    for k, src in splice.items():
        _, _, p, _, g = KP.forward_query_g(net, src, x, t + 1, gate=gate)
        out[k] = (p, g.mean(0))
    return out


@torch.no_grad()
def cells_carry(net, XA, XB, x, t, gate):
    cA, uA = net.prefix(XA, t, gate); cB, uB = net.prefix(XB, t, gate)                   # sources under the same regime
    sp = lambda first, last: [first[i][:t] + [last[i][t]] for i in range(len(first))]     # noqa: E731
    cfg = {"R1": (cA, uA), "R2": (sp(cA, cB), uB), "R3": (sp(cB, cA), uA), "R4": (cB, uB)}
    out = {}
    for k, (c, u) in cfg.items():
        z, _, _, p = net.advance_g(torch.as_tensor(x), c, u, None, gate)
        out[k] = (torch.softmax(z, -1).double().numpy(), (p > 0.5).double().mean(0).numpy())
    return out


def selectivity(p_open, J):
    """p_open [n, T+1, L] deterministic open probabilities; J [n, T+1, K, C] exact joint filter.
    Entropy and movement of the exact filter at open vs closed (block, position) pairs, u >= 8."""
    g = p_open > 0.5                                                     # [n, T+1, L]
    n, T1 = J.shape[:2]; Jf = J.reshape(n, T1, -1)                       # joint over (G, c) flattened to 8
    H = -(Jf * np.log(np.clip(Jf, 1e-12, None))).sum(-1)                 # [n, T+1] entropy of the exact filter
    mv = np.zeros_like(H); mv[:, 1:] = BC.kl(Jf[:, 1:], Jf[:, :-1])      # KL(J_u || J_{u-1}): how far the belief moved
    sl = slice(8, None)
    go, Hs, ms = g[:, sl], np.broadcast_to(H[:, sl, None], g[:, sl].shape), np.broadcast_to(mv[:, sl, None], g[:, sl].shape)
    n_open, n_closed = int(go.sum()), int((~go).sum())
    split = lambda v: (float(v[go].mean()) - float(v[~go].mean())) if n_open and n_closed else float("nan")   # noqa: E731
    return {"entropy_split": split(Hs), "movement_split": split(ms), "n_open": n_open, "n_closed": n_closed,
            "rate_pos": g.mean((0, 2)).tolist(), "rate_layer": g.mean((0, 1)).tolist()}


def measure(args):
    fam, c, seed, path = args
    torch.set_num_threads(1)
    env = LG.make_env("channel", 4)
    if fam == "plain":
        net = KP.make_net(KP.LMSpec(4, T, seed), gated=True); net.load_state_dict(torch.load(path)["state"]); net.eval()
    else:
        net = make_carry(seed); net.load_state_dict(torch.load(path)["state"]); net.eval()
    X, _ = KP.lm_data(env, N_EVAL, T, seed=12345)
    J = LG.filter_joint(env, X); exact = LG.next_obs(env, X, J)
    with torch.no_grad():
        if fam == "plain":
            p_own = KP.predictive(net, X, gate="own"); p_open = KP.predictive(net, X, gate="open")
            gates = net.residuals_g(torch.as_tensor(X), gate="own")[1].double().numpy()
        else:
            z_own, gates = net.forward_gated(torch.as_tensor(X), gate="own"); z_open, _ = net.forward_gated(torch.as_tensor(X), gate="open")
            p_own = torch.softmax(z_own, -1).double().numpy(); p_open = torch.softmax(z_open, -1).double().numpy()
            gates = gates.double().numpy()
    rec = {"family": fam, "c": c, "seed": seed,
           "kl_own": float(BC.kl(exact[:, 1:], p_own[:, 1:]).mean()), "kl_open": float(BC.kl(exact[:, 1:], p_open[:, 1:]).mean()),
           "read_rate": float((gates[:, 2:] > 0.5).mean()), "t": {}}
    rec.update(selectivity(gates, J))
    for t in TS:
        XA, XB, x, oA, oB = pairs(env, t)
        rec["t"][str(t)] = {}
        for regime in ("open", "own"):
            outs = (cells_plain if fam == "plain" else cells_carry)(net, XA, XB, x, t, regime)
            rec["t"][str(t)][regime] = {k: {"lam": KP.along(KP.centred_log(v), oA, oB)[0], "gate": g.tolist()} for k, (v, g) in outs.items()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rounds/r17_read_cost")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cs", type=float, nargs="*", help="override the price grid (follow-up runs)")
    a = ap.parse_args()
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    seeds = range(1) if a.quick else range(3); cs = (0.0, 0.3) if a.quick else tuple(a.cs or CS)
    steps = 200 if a.quick else a.steps
    t0 = time.time(); ctx = get_context("spawn")
    if not a.measure_only:
        pool = ctx.Pool(min(a.workers, len(cs) * len(seeds)))
        carry_jobs = pool.map_async(train_carry, [(c, s, steps, str(out / "models" / f"{cname(c, s)}.pt")) for c in cs for s in seeds])
        specs = [KP.LMSpec(4, T, s) for c in cs for s in seeds]; costs = [c for c in cs for s in seeds]
        nets, _ = KP.train_lm_stack(specs, steps, n_pool=5000 if a.quick else 100000, cost=costs,
                                    log=lambda s, l: _log(f"plain step {s}: loss {l.mean():.4f}"))
        for net, c, s in zip(nets, costs, [s for c in cs for s in seeds]):
            torch.save({"state": net.state_dict()}, out / "models" / f"{pname(c, s)}.pt")
        _log(f"plain trained ({time.time() - t0:.0f}s)")
        carry_jobs.get(); pool.close(); pool.join()
        _log(f"carry trained ({time.time() - t0:.0f}s)")
    tasks = [(f, c, s, str(out / "models" / f"{(pname if f == 'plain' else cname)(c, s)}.pt"))
             for f in ("plain", "carry") for c in cs for s in seeds]
    with ctx.Pool(min(a.workers, len(tasks))) as pool:
        recs = pool.map(measure, tasks, chunksize=1)
    json.dump(to_jsonable({"runs": recs}), open(out / "results16.json", "w"))
    _log(f"measured {len(recs)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
