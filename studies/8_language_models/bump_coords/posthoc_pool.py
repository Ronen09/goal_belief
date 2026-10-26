"""Post hoc (after the registered part C was seen): the registered maps, fitted on the swap pool (decisions along the
optimum's trajectories), do not transfer to the probe pool (shortest histories, uniform over sets), so the pull-back
test was run through maps that are not maps there. Here the maps are refitted on the probe pool's fit half (same ridge
recipe, λ tuned on a quarter) and the A-probe's cells are re-scored on its test half: the map's R² on the probe pool
against the registered maps' and the identity, the restoration by the inverse map and by the offset, the bump probe,
and the causal check (the pull-back installed under B). Writes posthoc_pool.json.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_coords .venv/bin/python studies/8_language_models/bump_coords/posthoc_pool.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as C
from rank import pairs_of
from preserve import fit, apply_probe, restoration, r2_cells


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--probe_steps", type=int, default=800); a = ap.parse_args(); t0 = time.time()
    out = Path(a.out) if a.out else C.HERE; out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache) if a.cache else (out.parent / "collect" / "cache" if a.quick else C.CACHE)
    states, maps = C.load_cache(cache); probe, sites = states["probe"], tuple(states["sites"]); psteps = 50 if a.quick else a.probe_steps
    free, goals, worlds, sols = C.worlds(); M = C.Model(C.RUN / "model", worlds, a.device, a.batch, sites); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    masks, steps, trn, tst = probe["masks"], probe["steps"], probe["fit"], probe["test"]; pb = C.pbump_target(worlds[0], masks); hists_te = [h for h, t in zip(probe["hists"], tst.tolist()) if t]
    nat = {g: probe["L"][g].argmax(1) for g in range(C.K)}; pairs = pairs_of()[:1] if a.quick else pairs_of()
    res = dict(sites=list(sites), n_fit=int(trn.sum()), n_test=int(tst.sum()), by_site={})
    for si, s in enumerate(sites):
        X = {g: probe["X"][g][:, si] for g in range(C.K)}; Xd = {g: X[g].double() for g in range(C.K)}
        probes = {g: dict(set=fit(X[g], masks, trn, "logistic", psteps, dev=a.device), bump=fit(X[g], pb, trn, "ridge", psteps, dev=a.device)) for g in range(C.K)}
        rows, pool = [], dict(drop=[], num_back=[], den=[], num_off=[], bump_drop=[], bump_back=[], chk=[], chk_nat=[], r2_pool=[], r2_reg=[], r2_id=[])
        for (A, B) in pairs:
            # maps on the probe pool: B -> A (the pull-back) fitted on the fit half
            lam, by_lam = C.tune_lambda(Xd[B][trn], steps[trn], Xd[A][trn]); W = C.ridge(Xd[B][trn], steps[trn], Xd[A][trn], lam)
            XA, XB, st = Xd[A][tst], Xd[B][tst], steps[tst]; back = C.apply_map(W, XB, st); r2_pool = C.r2(back, XA)
            r2_reg = C.r2(C.apply_map(maps[(s, B, A)]["W"], XB, st), XA); r2_id = C.r2(XB, XA)
            Dm = Xd[B][trn] - Xd[A][trn]; sf = steps[trn]; off = torch.stack([Dm[sf == v].mean(0) if (sf == v).any() else Dm.mean(0) for v in range(C.H)]); x_off = XB - off[st.clamp(max=C.H - 1)]
            net, mu, sd = probes[A]["set"]; bnet, bmu, bsd = probes[A]["bump"]
            cells = dict(A=XA, B=XB, back=back, off=x_off)
            jac = {k: C.jaccard(apply_probe(net, mu, sd, v), masks[tst])[1] for k, v in cells.items()}; jac["BB"] = C.jaccard(apply_probe(*probes[B]["set"], XB), masks[tst])[1]
            bmp = {k: r2_cells(apply_probe(bnet, bmu, bsd, v), pb[tst]) for k, v in cells.items()}; bmp["BB"] = r2_cells(apply_probe(*probes[B]["bump"], XB), pb[tst])
            drop = jac["A"] - jac["B"]; drop_ci = C.boot_mean(drop); dropped = drop_ci[0] > 0
            natB, natA = nat[B][tst], nat[A][tst]; dep = (natB != natA).nonzero().squeeze(1)
            def check(new):
                if len(dep) == 0: return float("nan"), np.zeros(0)
                L = M.patched([hists_te[i] for i in dep.tolist()], B, s, torch.as_tensor(new)[dep]); per = (L.argmax(1) == natA[dep]).double().numpy(); return float(per.mean()), per
            chk, cp = check(back); chn, cn = check(XA)
            bd = (bmp["B"][1] - bmp["A"][1]) / bmp["A"][2]; bb = (bmp["back"][1] - bmp["A"][1]) / bmp["A"][2]
            row = dict(A=A, B=B, lam=lam, r2_map_probe_pool=r2_pool, r2_registered_map_probe_pool=r2_reg, r2_identity=r2_id, j_A=float(jac["A"].mean()), j_B=float(jac["B"].mean()), j_back=float(jac["back"].mean()),
                       j_off=float(jac["off"].mean()), j_BB=float(jac["BB"].mean()), drop=[float(drop.mean())] + drop_ci,
                       restoration=([restoration(jac["A"].mean(), jac["B"].mean(), jac["back"].mean())] + C.boot_ratio(jac["back"] - jac["B"], drop)) if dropped else None,
                       over_offset=([float((jac["back"] - jac["off"]).mean() / drop.mean())] + C.boot_ratio(jac["back"] - jac["off"], drop)) if dropped else None,
                       b_A=bmp["A"][0], b_B=bmp["B"][0], b_back=bmp["back"][0], b_off=bmp["off"][0], b_BB=bmp["BB"][0], bump_drop=[float(bd.mean())] + C.boot_mean(bd), bump_drop_back=[float(bb.mean())] + C.boot_mean(bb),
                       check_switch_back=chk, check_switch_natural=chn)
            rows.append(row); pool["drop"].append(drop); pool["bump_drop"].append(bd); pool["bump_back"].append(bb); pool["chk"].append(cp); pool["chk_nat"].append(cn); pool["r2_pool"].append(r2_pool); pool["r2_reg"].append(r2_reg); pool["r2_id"].append(r2_id)
            if dropped: pool["num_back"].append(jac["back"] - jac["B"]); pool["den"].append(drop); pool["num_off"].append(jac["back"] - jac["off"])
            P(f"  site {s} {A}->{B}: map R² on the probe pool {r2_pool:.3f} (registered {r2_reg:.3f}, identity {r2_id:.3f}); Jaccard A {row['j_A']:.3f} B {row['j_B']:.3f} back {row['j_back']:.3f} off {row['j_off']:.3f} BB {row['j_BB']:.3f}; restoration {row['restoration']}; bump A {row['b_A']:.2f} B {row['b_B']:.2f} back {row['b_back']:.2f}; check back {chk:.3f} natural {chn:.3f}")
        cat = lambda k: np.concatenate(pool[k]) if pool[k] else np.zeros(0); d, nb, de, no = cat("drop"), cat("num_back"), cat("den"), cat("num_off")
        summary = dict(r2_map_probe_pool=float(np.mean(pool["r2_pool"])), r2_registered_map_probe_pool=float(np.mean(pool["r2_reg"])), r2_identity=float(np.mean(pool["r2_id"])),
                       drop=[float(d.mean())] + C.boot_mean(d), restoration=([float(nb.mean() / de.mean())] + C.boot_ratio(nb, de)) if len(de) else None,
                       over_offset=([float(no.mean() / de.mean())] + C.boot_ratio(no, de)) if len(de) else None, bump_drop=[float(cat("bump_drop").mean())] + C.boot_mean(cat("bump_drop")),
                       bump_drop_back=[float(cat("bump_back").mean())] + C.boot_mean(cat("bump_back")), check_switch_back=[float(cat("chk").mean())] + C.boot_mean(cat("chk")), check_switch_natural=[float(cat("chk_nat").mean())] + C.boot_mean(cat("chk_nat")))
        res["by_site"][str(s)] = dict(pairs=rows, summary=summary)
        P(f"site {s}: map R² probe pool {summary['r2_map_probe_pool']:.3f} (registered {summary['r2_registered_map_probe_pool']:.3f}, identity {summary['r2_identity']:.3f}); drop {summary['drop'][0]:.3f}; restoration {summary['restoration']}; over offset {summary['over_offset']}; bump drop {summary['bump_drop'][0]:.3f} after pull-back {summary['bump_drop_back'][0]:.3f}; check back {summary['check_switch_back'][0]:.3f} natural {summary['check_switch_natural'][0]:.3f}")
        (out / "posthoc_pool.json").write_text(json.dumps(res, indent=1))
    P("done")


if __name__ == "__main__":
    main()
