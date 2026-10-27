"""Part 0 of PLAN.md: the goal subspace of a from-scratch four-goal model (tfm_120000 / gru_120000), by the
coordinates study's rank analysis at the model's sites (the transformer's block entrances 1–3 and final residual; the
GRU's hidden state at the decision and one and two tokens before). Records the pool's states under every goal
(cache/states_<kind>.pt), fits the twelve ridge goal maps per site on the fit histories, takes the per-step-centred
goal differences' singular directions per pair and pooled, and runs the causal sweep on the test histories'
goal-dependent decisions (top-k, pooled, bottom-k, three random draws); the effective causal rank per pair and the
registered rank (the median pooled k_c rounded up to the grid), the registered site (the non-linear site with the
largest natural goal swap, at or above the floor). Writes subspace_<kind>.json and cache/subspaces_<kind>.pt.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_channel .venv/bin/python studies/8_language_models/bump_channel/subspace_small.py --kind tfm
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as CH

CC = CH.CC
VARIANTS = ("top", "pooled", "bottom", "random0", "random1", "random2")


def pairs_of():
    return [(A, B) for A in range(CH.K) for B in range(CH.K) if A != B]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=["tfm", "gru"], required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--batch", type=int, default=2048); a = ap.parse_args(); t0 = time.time(); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    out = Path(a.out) if a.out else CH.HERE; cache = out / "cache"; cache.mkdir(parents=True, exist_ok=True)
    KS = (1, 4, CH.D_SMALL) if a.quick else CH.KS_SMALL; linear_site = 4 if a.kind == "tfm" else 0
    free, goals, worlds, sols = CC.worlds(); pool = CH.pool(worlds, sols, a.quick); M = CH.load_small(a.kind, worlds[0].H, a.device); M.batch = a.batch; sites = M.sites
    hf, ht, sf, st = pool["hists_fit"], pool["hists_test"], pool["steps_fit"], pool["steps_test"]
    swap = dict(sites=sites, dec_fit_idx=pool["fit_idx"], dec_test_i1=pool["test_idx"], steps_fit=sf, steps_test=st, hists_fit=hf, hists_h1=ht, keys_fit=pool["keys_fit"], keys_test=pool["keys_test"], sizes_test=pool["sizes_test"],
                X_fit={}, X_h1={}, L_fit={}, L_h1={})
    for g in range(CH.K):
        swap["L_fit"][g], swap["X_fit"][g] = M.record(hf, g); swap["L_h1"][g], swap["X_h1"][g] = M.record(ht, g)
    torch.save(swap, cache / f"states_{a.kind}.pt"); P(f"{a.kind}: {len(hf)} fit and {len(ht)} test histories recorded under {CH.K} goals at sites {sites}")
    nat = {g: swap["L_h1"][g].argmax(1) for g in range(CH.K)}; pairs = pairs_of(); dep_idx = {(A, B): (nat[A] != nat[B]).nonzero().squeeze(1) for (A, B) in pairs}
    groups = lambda A, B: dep_idx[(A, B)].numpy()

    def run_dep(A, B, s, new):
        idx = dep_idx[(A, B)]
        if len(idx) == 0: return float("nan"), np.zeros(0)
        L = M.patched([ht[i] for i in idx.tolist()], A, s, new[idx]); per = (L.argmax(1) == nat[B][idx]).double().numpy(); return float(per.mean()), per

    res = dict(kind=a.kind, sites=list(sites), ks=list(KS), linear_site=linear_site, n_fit=len(hf), n_test=len(ht), by_site={}); subspaces = {}
    for si, s in enumerate(sites):
        xf = lambda g: swap["X_fit"][g][:, si].double(); x = lambda g: swap["X_h1"][g][:, si].double()
        natural = {(A, B): run_dep(A, B, s, x(B)) for (A, B) in pairs}; nat_all = np.concatenate([natural[p][1] for p in pairs]); nat_g = np.concatenate([groups(*p) for p in pairs])
        Dc, spec, Vh, off = {}, {}, {}, {}
        for (A, B) in pairs:
            Dm = xf(B) - xf(A); Dcen = CC._centre(Dm, sf); Dc[(A, B)] = Dcen; off[(A, B)] = float(1 - (Dcen ** 2).sum() / (Dm ** 2).sum()); spec[(A, B)] = CC.spectrum_ranks(Dm, sf); Vh[(A, B)] = CC.basis(Dcen, None)
        pooled = torch.cat(list(Dc.values())); Vh_pool = CC.basis(pooled, None); spec_pool = CC.spectrum_ranks(pooled, None)
        rows, per_full, g_full, per_pool = [], [], [], {k: [] for k in KS}
        for (A, B) in pairs:
            xA = x(A); lam, by_lam = CC.tune_lambda(xf(A), sf, xf(B)); W = CC.ridge(xf(A), sf, xf(B), lam); full = CC.apply_map(W, xA, st); sw_full, pf = run_dep(A, B, s, full)
            proj = {"top": lambda k: CC.projector(Vh[(A, B)], slice(0, k)), "pooled": lambda k: CC.projector(Vh_pool, slice(0, k)), "bottom": lambda k: CC.projector(Vh[(A, B)], slice(CH.D_SMALL - k, None)),
                    "random0": lambda k: CC.random_projector(CH.D_SMALL, k, 0), "random1": lambda k: CC.random_projector(CH.D_SMALL, k, 1), "random2": lambda k: CC.random_projector(CH.D_SMALL, k, 2)}
            sw = {v: {} for v in VARIANTS}
            for v in VARIANTS:
                for k in KS:
                    sw[v][k], per = run_dep(A, B, s, xA + (full - xA) @ proj[v](k))
                    if v == "pooled": per_pool[k].append(per)
            defined = sw_full >= CH.FLOOR; k_c = CC.causal_rank(sw["top"], sw_full, KS) if defined else None; k_cp = CC.causal_rank(sw["pooled"], sw_full, KS) if defined else None
            rows.append(dict(A=A, B=B, **spec[(A, B)], offset_norm_share=off[(A, B)], r2_heldout=by_lam[lam], lam=lam, switch_full=sw_full, switch_natural=natural[(A, B)][0], n_dep=int(len(pf)),
                             switch_by_k={v: {str(k): sw[v][k] for k in KS} for v in VARIANTS}, k_c=k_c, k_c_pooled=k_cp))
            per_full.append(pf); g_full.append(groups(A, B))
            P(f"  site {s} {A}->{B}: natural {natural[(A, B)][0]:.3f} full {sw_full:.3f}; pooled-k " + " ".join(f"{k}:{sw['pooled'][k]:.2f}" for k in KS) + f"; k_c {k_c} (pooled {k_cp}); k50/90/99 {spec[(A, B)]['k50']}/{spec[(A, B)]['k90']}/{spec[(A, B)]['k99']}")
        kcp = [r["k_c_pooled"] for r in rows]; k_reg = CH.registered_k(kcp, KS); pf_all, gf = np.concatenate(per_full), np.concatenate(g_full)
        ratio_at_k = ([float(np.concatenate(per_pool[k_reg]).mean() / pf_all.mean())] + CC.boot_ratio(np.concatenate(per_pool[k_reg]), pf_all, groups=gf)) if (k_reg is not None and pf_all.mean() > 0) else None
        summary = dict(natural=[float(nat_all.mean())] + CC.boot_mean(nat_all, groups=nat_g), switch_full=[float(pf_all.mean())] + CC.boot_mean(pf_all, groups=gf), k_c_pooled=kcp, k_c_pooled_median_raw=float(np.median([k for k in kcp if k is not None])) if any(k is not None for k in kcp) else None,
                       k_registered=k_reg, rank_at_k_ratio=ratio_at_k, k90_median=float(np.median([r["k90"] for r in rows])), k90_pooled=spec_pool["k90"], offset_share_mean=float(np.mean([r["offset_norm_share"] for r in rows])), linear=s == linear_site)
        res["by_site"][str(s)] = dict(pairs=rows, pooled=dict(k50=spec_pool["k50"], k90=spec_pool["k90"], k99=spec_pool["k99"]), summary=summary)
        subspaces[s] = dict(Vh=Vh_pool.float(), k=k_reg, k_c_pooled=kcp, natural=summary["natural"][0])
        P(f"site {s}: natural goal swap {summary['natural'][0]:.3f}, full map {summary['switch_full'][0]:.3f}, k_c pooled {kcp} -> registered k {k_reg}, pooled at k / full {ratio_at_k}, k90 median {summary['k90_median']}")
    cands = [s for s in sites if s != linear_site and subspaces[s]["natural"] >= CH.FLOOR and subspaces[s]["k"] is not None]
    reg = max(cands, key=lambda s: subspaces[s]["natural"]) if cands else None; res["registered_site"] = reg; subspaces["registered_site"] = reg
    torch.save(subspaces, cache / f"subspaces_{a.kind}.pt"); (out / f"subspace_{a.kind}.json").write_text(json.dumps(res, indent=1))
    P(f"registered site {reg}" + (f" (k = {subspaces[reg]['k']})" if reg is not None else " (none: no non-linear site at or above the floor with a defined rank)"))


if __name__ == "__main__":
    main()
