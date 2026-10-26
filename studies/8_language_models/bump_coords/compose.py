"""Part A of PLAN.md: composition of the pairwise goal maps. For every ordered triple (A, B, C) and site, the direct
prediction M(A→C) x_A against the composed M(B→C) M(A→B) x_A on the test histories under A: R² between the two, R²
of each against the state under C, and the causal switch to C's decision when installed in the prompt under A (the
natural state under C the ceiling, a random vector of the natural change's norm the floor); round trips and three-
cycles back to A. Writes compose.json.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_coords .venv/bin/python studies/8_language_models/bump_coords/compose.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as C


def triples(K=C.K):
    return [(a, b, c) for a in range(K) for b in range(K) for c in range(K) if len({a, b, c}) == 3]


def summarise(x):
    """[mean, lo, hi] of a list / array (nans dropped)."""
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    return [float(x.mean()) if len(x) else float("nan")] + C.boot_mean(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None); ap.add_argument("--batch", type=int, default=64); a = ap.parse_args(); t0 = time.time()
    out = Path(a.out) if a.out else C.HERE; out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache) if a.cache else (out / "cache" if a.quick else C.CACHE)
    states, maps = C.load_cache(cache); swap, sites = states["swap"], tuple(states["sites"])
    free, goals, worlds, sols = C.worlds(); M = C.Model(C.RUN / "model", worlds, a.device, a.batch, sites); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    hists, steps = swap["hists_h1"], swap["steps_test"]; nat = {g: swap["L_h1"][g].argmax(1) for g in range(C.K)}; n = len(hists)
    T = triples()[:2] if a.quick else triples()
    pairs = sorted({p for (A, B, Cc) in T for p in ((A, B), (B, Cc), (A, Cc))})
    res = dict(n_pairs=n, sites=list(sites), by_site={}, best_site=None)
    for si, s in enumerate(sites):
        W = lambda A, B: maps[(s, A, B)]["W"]; x = lambda g: swap["X_h1"][g][:, si].double()
        run = lambda A, new: M.patched(hists, A, s, new)
        nat_sw, rnd_sw, single = {}, {}, {}
        g = torch.Generator().manual_seed(3)
        for (A, B) in pairs:
            xA, xB = x(A), x(B); nat_sw[(A, B)] = C.switch(run(A, xB), nat[A], nat[B])
            rnd = torch.randn(xA.shape, generator=g, dtype=torch.float64); rnd = xA + rnd / rnd.norm(dim=1, keepdim=True) * (xB - xA).norm(dim=1, keepdim=True)
            rnd_sw[(A, B)] = C.switch(run(A, rnd), nat[A], nat[B]); single[(A, B)] = C.switch(run(A, C.apply_map(W(A, B), xA, steps)), nat[A], nat[B])
        P(f"site {s}: natural swaps {np.nanmean([v[0] for v in nat_sw.values()]):.3f}, single maps {np.nanmean([v[0] for v in single.values()]):.3f}, random {np.nanmean([v[0] for v in rnd_sw.values()]):.3f}")
        rows, per_d, per_c = [], [], []
        for (A, B, Cc) in T:
            xA, xC = x(A), x(Cc); direct = C.apply_map(W(A, Cc), xA, steps); composed = C.apply_map(W(B, Cc), C.apply_map(W(A, B), xA, steps), steps)
            swd, pd = C.switch(run(A, direct), nat[A], nat[Cc]); swc, pc = C.switch(run(A, composed), nat[A], nat[Cc]); per_d.append(pd); per_c.append(pc)
            rows.append(dict(A=A, B=B, C=Cc, r2_composed_vs_direct=C.r2(composed, direct), r2_direct_vs_xC=C.r2(direct, xC), r2_composed_vs_xC=C.r2(composed, xC),
                             norm_ratio=float(((composed - xA).norm(dim=1) / (direct - xA).norm(dim=1).clamp(min=1e-9)).mean()),
                             switch_direct=swd, switch_composed=swc, switch_natural=nat_sw[(A, Cc)][0], switch_random=rnd_sw[(A, Cc)][0], n_dep=int(len(pd))))
        rts, cyc = [], []
        for (A, B) in pairs:
            xA, xB = x(A), x(B); rt = C.apply_map(W(B, A), C.apply_map(W(A, B), xA, steps), steps)
            rts.append(dict(A=A, B=B, r2_vs_xA=C.r2(rt, xA), keep_round=C.keep(run(A, rt), nat[A], nat[B])[0], keep_xA=C.keep(run(A, xA), nat[A], nat[B])[0],
                            keep_xB=C.keep(run(A, xB), nat[A], nat[B])[0], switch_AB_full=single[(A, B)][0], switch_natural=nat_sw[(A, B)][0]))
        for (A, B, Cc) in T:
            xA = x(A); cy = C.apply_map(W(Cc, A), C.apply_map(W(B, Cc), C.apply_map(W(A, B), xA, steps), steps), steps)
            cyc.append(dict(A=A, B=B, C=Cc, r2_vs_xA=C.r2(cy, xA), keep_cycle=C.keep(run(A, cy), nat[A], nat[B])[0]))
        pd, pc = np.concatenate(per_d), np.concatenate(per_c)
        summary = dict(r2_composed_vs_direct=summarise([r["r2_composed_vs_direct"] for r in rows]), r2_direct_vs_xC=summarise([r["r2_direct_vs_xC"] for r in rows]),
                       r2_composed_vs_xC=summarise([r["r2_composed_vs_xC"] for r in rows]), switch_direct=[float(pd.mean())] + C.boot_mean(pd), switch_composed=[float(pc.mean())] + C.boot_mean(pc),
                       switch_ratio=[float(pc.mean() / pd.mean()) if pd.mean() > 0 else float("nan")] + C.boot_ratio(pc, pd), switch_natural=summarise([r["switch_natural"] for r in rows]),
                       switch_random=summarise([r["switch_random"] for r in rows]), keep_round=summarise([r["keep_round"] for r in rts]), keep_xA=summarise([r["keep_xA"] for r in rts]),
                       keep_xB=summarise([r["keep_xB"] for r in rts]), r2_round=summarise([r["r2_vs_xA"] for r in rts]), keep_cycle=summarise([r["keep_cycle"] for r in cyc]), r2_cycle=summarise([r["r2_vs_xA"] for r in cyc]))
        res["by_site"][str(s)] = dict(triples=rows, round_trips=rts, cycles=cyc, summary=summary)
        P(f"site {s}: composed vs direct R² {summary['r2_composed_vs_direct'][0]:.3f}; switch direct {summary['switch_direct'][0]:.3f} composed {summary['switch_composed'][0]:.3f} (ratio {summary['switch_ratio'][0]:.3f}); round trip keeps {summary['keep_round'][0]:.3f} (x_A {summary['keep_xA'][0]:.3f}, x_B {summary['keep_xB'][0]:.3f}); cycle keeps {summary['keep_cycle'][0]:.3f}")
        res["best_site"] = int(max(res["by_site"], key=lambda k: res["by_site"][k]["summary"]["switch_natural"][0]))
        (out / "compose.json").write_text(json.dumps(res, indent=1))
    P("done")


if __name__ == "__main__":
    main()
