"""Collection (PLAN.md, "The model, sites and data"): the decision token's residuals at the four sites under the four
goals for the goal-swap study's fit decisions and test pairs and for the probe pool, their natural logits, and the
twelve ridge maps per site (λ tuned on a held-out quarter of the fit decisions, the default λ = 1e-2 kept alongside).
Writes cache/states.pt and cache/maps.pt (not committed).

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_coords .venv/bin/python studies/8_language_models/bump_coords/collect.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch", type=int, default=32); a = ap.parse_args(); t0 = time.time()
    out = Path(a.out) if a.out else C.HERE; cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
    n_fit, n_test, n_probe, sites = (64, 16, 64, (23,)) if a.quick else (None, 1024, 8192, C.SITES)
    free, goals, worlds, sols = C.worlds(); M = C.Model(C.RUN / "model", worlds, a.device, a.batch, sites); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)

    # the swap pool: decisions along the optimum, deduplicated by history and split by history (no history in both halves)
    dec = C.swap_decisions(worlds, sols, 64 if a.quick else 1024); fidx, i1 = C.split_swap_pool(dec, 0.25, n_test); n_distinct = len(fidx) + len(i1); fidx = fidx[:n_fit] if n_fit else fidx
    hist = lambda idx: [(dec[i]["acts"], dec[i]["obss"]) for i in idx]; steps = lambda idx: torch.tensor([dec[i]["step"] for i in idx])
    swap = dict(dec_fit_idx=fidx, dec_test_i1=i1, steps_fit=steps(fidx), steps_test=steps(i1), hists_fit=hist(fidx), hists_h1=hist(i1), X_fit={}, X_h1={}, L_fit={}, L_h1={},
                n_decisions=len(dec), n_distinct=n_distinct)
    for g in range(C.K):
        swap["L_fit"][g], swap["X_fit"][g] = M.record(swap["hists_fit"], g); swap["L_h1"][g], swap["X_h1"][g] = M.record(swap["hists_h1"], g)
        P(f"swap pool under goal {g}: {len(fidx)} fit histories, {len(i1)} test histories recorded (of {len(dec)} decisions, {swap['n_distinct']} distinct histories)")

    # the probe pool
    probe = C.probe_decisions(worlds, n_probe); probe["X"], probe["L"] = {}, {}
    for g in range(C.K): probe["L"][g], probe["X"][g] = M.record(probe["hists"], g)
    P(f"probe pool: {len(probe['hists'])} decisions over held-out-by-set halves ({probe['dropped']} dropped as ended under some goal) recorded")
    torch.save(dict(swap=swap, probe=probe, sites=sites), cache / "states.pt")

    # the maps
    maps = {}; summary = {}
    for si, s in enumerate(sites):
        r2s = []
        for A in range(C.K):
            for B in range(C.K):
                if A == B: continue
                XA, XB, st = swap["X_fit"][A][:, si].double(), swap["X_fit"][B][:, si].double(), swap["steps_fit"]
                lam, by_lam = C.tune_lambda(XA, st, XB); W = C.ridge(XA, st, XB, lam); W0 = C.ridge(XA, st, XB, 1e-2)
                maps[(s, A, B)] = dict(W=W, lam=lam, r2_heldout=by_lam[lam], r2_by_lambda=by_lam, W_default=W0); r2s.append(by_lam[lam])
        summary[s] = dict(mean_heldout_r2=float(np.mean(r2s)), lams=[maps[(s, A, B)]["lam"] for A in range(C.K) for B in range(C.K) if A != B])
        P(f"site {s}: 12 maps, mean held-out state R² {np.mean(r2s):.3f}, λ chosen {sorted(set(summary[s]['lams']))}")
    torch.save(maps, cache / "maps.pt"); (out / "collect.json").write_text(json.dumps(dict(sites=list(sites), n_fit=len(fidx), n_test=len(i1), n_decisions=len(dec), n_distinct=swap["n_distinct"], n_probe=len(probe["hists"]), probe_dropped=probe["dropped"], maps={str(k): v for k, v in summary.items()}), indent=1))
    P("done")


if __name__ == "__main__":
    main()
