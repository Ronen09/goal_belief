"""TASK14 (round 15): prior vs recomputation through the K/V cache (docs/task14_prior_theory.md).

    .venv/bin/python scripts/run_task14.py                  # train 12 next-token transformers + measure
    .venv/bin/python scripts/run_task14.py --measure-only
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import beliefcausal as BC, beliefprobe as BP, filterstate as FS, kvprior as KP, latentgoal as LG
from goalgeo.analysis import to_jsonable

TS = {24: (2, 4, 8, 16, 23), 64: (2, 4, 8, 16, 32, 48, 63)}
N_PAIRS = 1000


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def measure(net, spec: KP.LMSpec):
    env = LG.make_env("channel", 4); L, T = spec.L, spec.T
    X, _ = KP.lm_data(env, 4000, T, seed=12345)
    J = LG.filter_joint(env, X)
    exact = LG.next_obs(env, X, J); p = KP.predictive(net, X)
    kl = BC.kl(exact[:, 1:], p[:, 1:]).mean(0)
    rec = {"name": spec.name, "L": L, "T": T, "seed": spec.seed, "kl": float(kl.mean()), "kl_pos": kl.tolist(), "t": {}}
    R = KP.residuals(net, X)
    with torch.no_grad():
        U = net.ln_f(torch.as_tensor(R[L], dtype=torch.float32, device=net.pos.device)).double().cpu().numpy()
    Y = FS.full(J[:, 1:].reshape(-1, 4, 2))
    flat = lambda A: A[:, 1:].reshape(-1, A.shape[-1])                         # noqa: E731
    maps = {f"res{i}": BC.BeliefMap(flat(R[i]), Y) for i in range(1, L + 1)}
    maps["u"] = BC.BeliefMap(flat(U), Y)
    rec["probe_r2"] = {k: float(BP.r2(Y, m.z(flat(R[int(k[3:])]) if k != "u" else flat(U)))) for k, m in maps.items()}
    prior0 = np.broadcast_to(env.prior[:, None] * env.c0[None, :], (N_PAIRS, 4, 2)).copy()
    for t in TS[T]:
        XA, _ = KP.lm_data(env, 20000, T, seed=900 + t); XB, _ = KP.lm_data(env, 20000, T, seed=950 + t)
        JA = LG.filter_joint(env, XA)[:, t]; JB = LG.filter_joint(env, XB)[:, t]
        keep = np.flatnonzero(np.linalg.norm(FS.full(JA) - FS.full(JB), axis=1) >= 2.0)[:N_PAIRS]
        XA, XB, JA, JB = XA[keep], XB[keep], JA[keep], JB[keep]
        x = XA[:, t + 1]
        Ap, Bp, Pp = KP.update(env, JA, x), KP.update(env, JB, x), KP.update(env, prior0[:len(x)], x)
        zA, zB, zP = FS.full(Ap), FS.full(Bp), FS.full(Pp)
        oA, oB = KP.centred_log(BC.ideal(env, Ap, "next_obs")), KP.centred_log(BC.ideal(env, Bp, "next_obs"))
        RA, RB = KP.residuals(net, XA[:, :t + 2]), KP.residuals(net, XB[:, :t + 1])
        srcA = [r[:, :t + 1] for r in RA[:L]]; srcB = [r[:, :t + 1] for r in RB[:L]]
        fullB = FS.full(JB)

        def with_t(src, rows):
            out = [s.copy() for s in src]
            for i, v in rows.items():
                out[i][:, t] = v
            return out
        cells = {
            "R1_baseline": (srcA, None),
            "R2_swap": (with_t(srcA, {i: srcB[i][:, t] for i in range(1, L)}), None),
            "R2_probe": (with_t(srcA, {i: maps[f"res{i}"].probe_e(srcA[i][:, t], fullB) for i in range(1, L)}), None),
            "R3_corrupt": ([np.concatenate([srcB[i][:, :t], srcA[i][:, t:t + 1]], 1) for i in range(L)], None),
            "R3_mask": (srcA, np.arange(t + 1) == t),
            "R4_consistent": (srcB, None),
        }
        out = {}
        for name, (src, vis) in cells.items():
            res, u, pq, attn = KP.forward_query(net, src, x, t + 1, visible=vis)
            sites = {f"res{i + 1}": res[i] for i in range(L)}; sites["u"] = u
            c = {}
            for s, h in sites.items():
                lam, perp = KP.along(maps[s].z(h), zA, zB)
                c[s] = {"lambda": lam, "perp": perp}
            zu = maps["u"].z(u)
            c["retention"] = float(1 - np.linalg.norm(zu - zA, axis=1).mean() / np.linalg.norm(zP - zA, axis=1).mean())
            lo, po = KP.along(KP.centred_log(pq), oA, oB)
            c["output"] = {"lambda": lo, "perp": po}
            if name == "R1_baseline":
                c["attn_on_t"] = [float(a[:, t].mean()) for a in attn]
            out[name] = c
        rec["t"][str(t)] = out
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results14")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); (out / "models").mkdir(parents=True, exist_ok=True)
    groups = [(L, T) for L in (2, 4) for T in (24, 64)]
    seeds = range(1) if a.quick else range(3)
    steps = 300 if a.quick else a.steps
    t0 = time.time()
    if not a.measure_only:
        for L, T in groups:
            specs = [KP.LMSpec(L, T, s) for s in seeds]
            nets, hist = KP.train_lm_stack(specs, steps, n_pool=5000 if a.quick else 100000,
                                           log=lambda s, l: _log(f"L={L} T={T} step {s}: loss {l.mean():.4f}"))
            for net, sp, h in zip(nets, specs, hist.T):
                torch.save({"state": net.state_dict(), "hist": torch.as_tensor(h)}, out / "models" / f"{sp.name}.pt")
            _log(f"trained L={L} T={T} ({time.time() - t0:.0f}s)")
    recs = []
    for L, T in groups:
        for s in seeds:
            sp = KP.LMSpec(L, T, s); net = KP.make_net(sp)
            net.load_state_dict(torch.load(out / "models" / f"{sp.name}.pt")["state"]); net = net.cuda().eval()
            recs.append(measure(net, sp)); _log(f"measured {sp.name}: KL {recs[-1]['kl']:.4f}")
    json.dump(to_jsonable({"runs": recs}), open(out / "results14.json", "w"))
    _log(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
