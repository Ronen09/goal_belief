"""Part B of PLAN.md: the rank of the goal's effect. Per site and ordered goal pair, the per-history goal difference
Δ(h) = x(h, B) − x(h, A) over the fit decisions with its per-step mean removed: its singular spectrum (reconstruction
ranks k50 / k90 / k99, the pooled spectrum over all pairs), and the causal sweep: the full map's prediction with its
change restricted to the top-k singular directions (k in KS), installed in the prompt under A on the test pairs'
goal-dependent decisions; the effective causal rank k_c is the smallest k that switches ≥ 0.9 × the full map. Controls:
bottom-k, k random directions (three draws), the pooled subspace, the map's own singular directions. Writes rank.json
and cache/rank_subspaces.pt (the singular directions, for part C).

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_coords .venv/bin/python studies/8_language_models/bump_coords/rank.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as C

ALL_VARIANTS = ("top", "bottom", "random0", "random1", "random2", "pooled", "mapsvd")
CHEAP_VARIANTS = ("top", "pooled")


def rank_k_state(xA, full_pred, P):
    """x_A + P(M x_A − x_A): the full map's change restricted to the projector P's subspace."""
    return xA + (full_pred - xA) @ P


def pairs_of(K=C.K):
    return [(A, B) for A in range(K) for B in range(K) if A != B]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--n_control_sites", type=int, default=2); a = ap.parse_args(); t0 = time.time()
    out = Path(a.out) if a.out else C.HERE; out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache) if a.cache else (out.parent / "collect" / "cache" if a.quick else C.CACHE)      # quick: the smoke collect's cache
    states, maps = C.load_cache(cache); swap, sites = states["swap"], tuple(states["sites"]); KS = (1, 8, 896) if a.quick else C.KS
    free, goals, worlds, sols = C.worlds(); M = C.Model(C.RUN / "model", worlds, a.device, a.batch, sites); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    hists, steps, steps_fit = swap["hists_h1"], swap["steps_test"], swap["steps_fit"]; nat = {g: swap["L_h1"][g].argmax(1) for g in range(C.K)}
    pairs = pairs_of()
    dep_idx = {(A, B): (nat[A] != nat[B]).nonzero().squeeze(1) for (A, B) in pairs}

    def run_dep(A, B, s, new):
        """Patched run under A on the pair's goal-dependent decisions only; -> (switch to B's decision, per-decision 0/1)."""
        idx = dep_idx[(A, B)]
        if len(idx) == 0: return float("nan"), np.zeros(0)
        L = M.patched([hists[i] for i in idx.tolist()], A, s, new[idx]); per = (L.argmax(1) == nat[B][idx]).double().numpy(); return float(per.mean()), per

    # the natural goal swap per site, to choose the control sites
    natural = {}
    for si, s in enumerate(sites):
        natural[s] = {(A, B): run_dep(A, B, s, swap["X_h1"][B][:, si].double()) for (A, B) in pairs}
        P(f"site {s}: natural goal swap {np.nanmean([v[0] for v in natural[s].values()]):.3f} on {int(np.mean([len(dep_idx[p]) for p in pairs]))} goal-dependent decisions per pair")
    control_sites = () if a.quick else tuple(sorted(sites, key=lambda s: -np.nanmean([v[0] for v in natural[s].values()]))[: a.n_control_sites])
    P(f"control variants at sites {control_sites}")
    res = dict(sites=list(sites), ks=list(KS), control_sites=list(control_sites), by_site={}); subspaces = {}
    for si, s in enumerate(sites):
        variants = ALL_VARIANTS if s in control_sites else CHEAP_VARIANTS
        xf = lambda g: swap["X_fit"][g][:, si].double(); x = lambda g: swap["X_h1"][g][:, si].double()
        Dc, spec, offset_share, Vh = {}, {}, {}, {}
        for (A, B) in pairs:
            Dm = xf(B) - xf(A); Dcen = C._centre(Dm, steps_fit); Dc[(A, B)] = Dcen
            offset_share[(A, B)] = float(1 - (Dcen ** 2).sum() / (Dm ** 2).sum()); spec[(A, B)] = C.spectrum_ranks(Dm, steps_fit); Vh[(A, B)] = C.basis(Dcen, None)
        pooled_D = torch.cat(list(Dc.values())); Vh_pool = C.basis(pooled_D, None); spec_pool = C.spectrum_ranks(pooled_D, None)
        rows, per_full, per_32, per_pool_kc, per_full_kc = [], [], [], [], []
        for (A, B) in pairs:
            xA = x(A); W = maps[(s, A, B)]["W"]; full = C.apply_map(W, xA, steps); sw_full, pf = run_dep(A, B, s, full)
            Vm = C.basis(W[: C.D] - torch.eye(C.D, dtype=torch.float64), None)
            proj = {"top": lambda k: C.projector(Vh[(A, B)], slice(0, k)), "bottom": lambda k: C.projector(Vh[(A, B)], slice(C.D - k, None)),
                    "random0": lambda k: C.random_projector(C.D, k, 0), "random1": lambda k: C.random_projector(C.D, k, 1), "random2": lambda k: C.random_projector(C.D, k, 2),
                    "pooled": lambda k: C.projector(Vh_pool, slice(0, k)), "mapsvd": lambda k: C.projector(Vm, slice(0, k))}
            sw, per = {v: {} for v in variants}, {v: {} for v in variants}
            for v in variants:
                for k in KS: sw[v][k], per[v][k] = run_dep(A, B, s, rank_k_state(xA, full, proj[v](k)))
            k_c = C.causal_rank(sw["top"], sw_full, KS); k_cp = C.causal_rank(sw["pooled"], sw_full, KS); k_cm = C.causal_rank(sw["mapsvd"], sw_full, KS) if "mapsvd" in sw else None
            rows.append(dict(A=A, B=B, **spec[(A, B)], offset_norm_share=offset_share[(A, B)], switch_full=sw_full, switch_natural=natural[s][(A, B)][0], n_dep=int(len(pf)),
                             switch_by_k={v: {str(k): sw[v][k] for k in KS} for v in variants}, k_c=k_c, k_c_pooled=k_cp, k_c_mapsvd=k_cm))
            per_full.append(pf)
            if 32 in KS: per_32.append(per["top"][32])
            if k_c is not None: per_pool_kc.append(per["pooled"][k_c]); per_full_kc.append(pf)
            subspaces[(s, A, B)] = dict(Vh=Vh[(A, B)].float(), k_c=k_c)
            P(f"  site {s} {A}->{B}: full {sw_full:.3f} (natural {natural[s][(A, B)][0]:.3f}); top-k " + " ".join(f"{k}:{sw['top'][k]:.2f}" for k in KS) + f"; k_c {k_c} (pooled {k_cp}); k50/90/99 {spec[(A, B)]['k50']}/{spec[(A, B)]['k90']}/{spec[(A, B)]['k99']}")
        kcs = [r["k_c"] for r in rows if r["k_c"] is not None]
        overl = []
        for i, (A, B) in enumerate(pairs):
            for (A2, B2) in pairs[i + 1:]:
                k1, k2 = subspaces[(s, A, B)]["k_c"], subspaces[(s, A2, B2)]["k_c"]
                if k1 is None or k2 is None: continue
                k = min(k1, k2); P1 = C.projector(Vh[(A, B)], slice(0, k)); P2 = C.projector(Vh[(A2, B2)], slice(0, k))
                overl.append([A, B, A2, B2, k, C.subspace_overlap(P1, P2, k), C.subspace_overlap(P1, C.random_projector(C.D, k, 7), k)])
        pf_all = np.concatenate(per_full); p32 = np.concatenate(per_32) if per_32 else np.zeros(0)
        ppk, pfk = (np.concatenate(per_pool_kc), np.concatenate(per_full_kc)) if per_pool_kc else (np.zeros(0), np.zeros(0))
        summary = dict(k_c_median=float(np.median(kcs)) if kcs else None, k_c_range=[int(min(kcs)), int(max(kcs))] if kcs else None, k_c_defined=len(kcs),
                       rank32_ratio=([float(p32.mean() / pf_all.mean())] + C.boot_ratio(p32, pf_all)) if len(p32) else None,
                       pooled_at_kc_ratio=([float(ppk.mean() / pfk.mean())] + C.boot_ratio(ppk, pfk)) if len(ppk) else None,
                       switch_full=[float(pf_all.mean())] + C.boot_mean(pf_all), k90_median=float(np.median([r["k90"] for r in rows])), k99_median=float(np.median([r["k99"] for r in rows])),
                       offset_share_mean=float(np.mean([r["offset_norm_share"] for r in rows])), overlap_mean=float(np.mean([o[5] for o in overl])) if overl else None,
                       overlap_random_mean=float(np.mean([o[6] for o in overl])) if overl else None)
        res["by_site"][str(s)] = dict(variants=list(variants), pairs=rows, pooled=spec_pool, overlap_pairs=overl, summary=summary)
        P(f"site {s}: k_c median {summary['k_c_median']} range {summary['k_c_range']}; rank-32 / full {summary['rank32_ratio']}; pooled at k_c / full {summary['pooled_at_kc_ratio']}; k90 median {summary['k90_median']}; overlap {summary['overlap_mean']} (random {summary['overlap_random_mean']})")
        (out / "rank.json").write_text(json.dumps(res, indent=1)); torch.save(subspaces, cache / "rank_subspaces.pt")
    P("done")


if __name__ == "__main__":
    main()
