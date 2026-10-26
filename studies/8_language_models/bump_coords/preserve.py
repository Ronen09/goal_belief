"""Part C of PLAN.md: shared belief preservation. At each site and under each goal, a linear soft set probe (140
logistic outputs, Jaccard and exact recovery) and a linear bump probe (P(bump | forward), ridge, R²) fitted on the
probe pool's fit sets, with chance floors from shuffled labels. For each ordered goal pair the A-probe is scored on
held-out sets under A (baseline), under B unchanged (drop), under B pulled back through the fitted inverse map M(B→A)
(restoration), under B with the per-step mean offset removed only (control), under B pulled back through the
rank-k_c inverse map, and the B-probe under B (ceiling); a causal check installs the pull-backs in the prompt under B;
the set probe's row space against the causal goal subspace. The probe fitter is the policy study's recipe
(bump_policy/probes.py: standardise on the fit rows, LBFGS, L2 1e-4), copied because that module's imports collide
with this study's. Writes preserve.json.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_coords .venv/bin/python studies/8_language_models/bump_coords/preserve.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common as C
from rank import rank_k_state, pairs_of


def fit(X, Y, tr, kind, steps=800, seed=0, dev="cuda"):
    """A linear probe from X to Y fitted by LBFGS on the rows tr; kind 'logistic' (multi-label) or 'ridge'.
    Returns (net, mu, sd): apply with apply_probe."""
    torch.manual_seed(seed); X = X.float().to(dev); mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-3); Z = (X - mu) / sd; Y = Y.to(dev)
    net = nn.Linear(Z.shape[1], Y.shape[1]).to(dev)
    lf = {"logistic": lambda p, y: nn.functional.binary_cross_entropy_with_logits(p, y.float()), "ridge": lambda p, y: ((p - y.float()) ** 2).mean()}[kind]
    opt = torch.optim.LBFGS(net.parameters(), max_iter=steps, line_search_fn="strong_wolfe", history_size=20)
    with torch.enable_grad():
        def closure():
            opt.zero_grad(); l = lf(net(Z[tr]), Y[tr]) + 1e-4 * sum((p ** 2).sum() for p in net.parameters()); l.backward(); return l
        opt.step(closure)
    return net.eval(), mu, sd


@torch.no_grad()
def apply_probe(net, mu, sd, X):
    dev = mu.device; return net(((torch.as_tensor(X).float().to(dev) - mu) / sd)).cpu()


def restoration(j_A, j_B, j_back):
    return (j_back - j_B) / (j_A - j_B)


def probe_rows(net, sd):
    """The probe's weight rows in state coordinates (through the standardisation)."""
    return net.weight.detach().cpu() / torch.as_tensor(sd).cpu()


def r2_cells(pred, y):
    """(R², per-decision squared error, total variance per decision) for bootstrapping R² differences."""
    pred, y = torch.as_tensor(pred).double().squeeze(1), torch.as_tensor(y).double().squeeze(1); se = ((pred - y) ** 2).numpy(); sst = float(((y - y.mean()) ** 2).mean())
    return float(1 - se.mean() / sst), se, sst


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None); ap.add_argument("--subspaces", default=None); ap.add_argument("--batch", type=int, default=128); ap.add_argument("--probe_steps", type=int, default=800); a = ap.parse_args(); t0 = time.time()
    out = Path(a.out) if a.out else C.HERE; out.mkdir(parents=True, exist_ok=True); cache = Path(a.cache) if a.cache else (out.parent / "collect" / "cache" if a.quick else C.CACHE)      # quick: the smoke collect's cache
    states, maps = C.load_cache(cache); subspaces = torch.load(Path(a.subspaces) if a.subspaces else cache / "rank_subspaces.pt", weights_only=False)
    swap, probe, sites = states["swap"], states["probe"], tuple(states["sites"]); psteps = 50 if a.quick else a.probe_steps
    free, goals, worlds, sols = C.worlds(); M = C.Model(C.RUN / "model", worlds, a.device, a.batch, sites); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    masks, steps, trn, tst = probe["masks"], probe["steps"], probe["fit"], probe["test"]; pb = C.pbump_target(worlds[0], masks); hists_te = [h for h, t in zip(probe["hists"], tst.tolist()) if t]
    nat = {g: probe["L"][g].argmax(1) for g in range(C.K)}; pairs = pairs_of()[:1] if a.quick else pairs_of()
    rng = np.random.default_rng(0); perm = torch.as_tensor(rng.permutation(int(trn.sum()))); tr_idx = trn.nonzero().squeeze(1)
    M_shuf, pb_shuf = masks.clone(), pb.clone(); M_shuf[tr_idx] = masks[tr_idx][perm]; pb_shuf[tr_idx] = pb[tr_idx][perm]            # labels shuffled among the fit decisions
    res = dict(sites=list(sites), n_fit=int(trn.sum()), n_test=int(tst.sum()), by_site={})
    for si, s in enumerate(sites):
        X = {g: probe["X"][g][:, si].float() for g in range(C.K)}; Xf = {g: swap["X_fit"][g][:, si].double() for g in range(C.K)}; sf = swap["steps_fit"]
        probes, by_goal = {}, []
        for g in range(C.K):
            net, mu, sd = fit(X[g], masks, trn, "logistic", psteps, dev=a.device); bnet, bmu, bsd = fit(X[g], pb, trn, "ridge", psteps, dev=a.device)
            fnet, fmu, fsd = fit(X[g], M_shuf, trn, "logistic", psteps, dev=a.device); fb, fbmu, fbsd = fit(X[g], pb_shuf, trn, "ridge", psteps, dev=a.device)
            probes[g] = dict(set=(net, mu, sd), bump=(bnet, bmu, bsd))
            Pm = apply_probe(net, mu, sd, X[g][tst]); j, _ = C.jaccard(Pm, masks[tst]); jf, _ = C.jaccard(apply_probe(fnet, fmu, fsd, X[g][tst]), masks[tst])
            by_goal.append(dict(g=g, set_exact=float(((Pm > 0) == masks[tst]).all(1).double().mean()), set_jaccard=j, set_jaccard_floor=jf,
                                bump_r2=r2_cells(apply_probe(bnet, bmu, bsd, X[g][tst]), pb[tst])[0], bump_r2_floor=r2_cells(apply_probe(fb, fbmu, fbsd, X[g][tst]), pb[tst])[0]))
            P(f"site {s} goal {g}: set exact {by_goal[-1]['set_exact']:.3f} Jaccard {j:.3f} (floor {jf:.3f}); bump R² {by_goal[-1]['bump_r2']:.3f} (floor {by_goal[-1]['bump_r2_floor']:.3f})")
        rows, pool = [], dict(drop=[], num_back=[], den=[], num_off=[], num_kc=[], den_kc=[], bump_drop=[], overlap=[], overlap_rnd=[], chk_full=[], chk_kc=[], chk_nat=[])
        for (A, B) in pairs:
            net, mu, sd = probes[A]["set"]; bnet, bmu, bsd = probes[A]["bump"]; XA, XB, st = X[A][tst].double(), X[B][tst].double(), steps[tst]
            full_back = C.apply_map(maps[(s, B, A)]["W"], XB, st)
            Dm = Xf[B] - Xf[A]; off = torch.stack([Dm[sf == v].mean(0) if (sf == v).any() else Dm.mean(0) for v in range(C.H)]); x_off = XB - off[st.clamp(max=C.H - 1)]
            sub = subspaces[(s, B, A)]; k_c = sub["k_c"]
            back_kc = rank_k_state(XB, full_back, C.projector(sub["Vh"].double(), slice(0, k_c))) if k_c is not None else None
            cells = dict(A=XA, B=XB, back=full_back, off=x_off, kc=back_kc)
            jac = {k: C.jaccard(apply_probe(net, mu, sd, v), masks[tst])[1] for k, v in cells.items() if v is not None}; jac["BB"] = C.jaccard(apply_probe(*probes[B]["set"], XB), masks[tst])[1]
            bmp = {k: r2_cells(apply_probe(bnet, bmu, bsd, v), pb[tst]) for k, v in cells.items() if v is not None}; bmp["BB"] = r2_cells(apply_probe(*probes[B]["bump"], XB), pb[tst])
            drop = jac["A"] - jac["B"]; drop_ci = C.boot_mean(drop); dropped = drop_ci[0] > 0
            row = dict(A=A, B=B, k_c=k_c, j_A=float(jac["A"].mean()), j_B=float(jac["B"].mean()), j_back=float(jac["back"].mean()), j_off=float(jac["off"].mean()),
                       j_back_kc=float(jac["kc"].mean()) if "kc" in jac else None, j_BB=float(jac["BB"].mean()), drop=[float(drop.mean())] + drop_ci,
                       restoration=([restoration(jac["A"].mean(), jac["B"].mean(), jac["back"].mean())] + C.boot_ratio(jac["back"] - jac["B"], drop)) if dropped else None,
                       restoration_kc=([restoration(jac["A"].mean(), jac["B"].mean(), jac["kc"].mean())] + C.boot_ratio(jac["kc"] - jac["B"], drop)) if (dropped and "kc" in jac) else None,
                       over_offset=([float((jac["back"] - jac["off"]).mean() / drop.mean())] + C.boot_ratio(jac["back"] - jac["off"], drop)) if dropped else None,
                       b_A=bmp["A"][0], b_B=bmp["B"][0], b_back=bmp["back"][0], b_off=bmp["off"][0], b_back_kc=bmp["kc"][0] if "kc" in bmp else None, b_BB=bmp["BB"][0])
            bd = (bmp["B"][1] - bmp["A"][1]) / bmp["A"][2]; row["bump_drop"] = [float(bd.mean())] + C.boot_mean(bd)                     # R²_A − R²_B per decision
            # the causal check: the pull-backs installed in the prompt under B, switch to A's decision
            natB, natA = nat[B][tst], nat[A][tst]
            def check(new):
                dep = (natB != natA).nonzero().squeeze(1)
                if len(dep) == 0: return float("nan"), np.zeros(0)
                L = M.patched([hists_te[i] for i in dep.tolist()], B, s, torch.as_tensor(new)[dep]); per = (L.argmax(1) == natA[dep]).double().numpy(); return float(per.mean()), per
            row["check_switch_full"], cf = check(full_back); row["check_switch_natural"], cn = check(XA); row["check_switch_kc"], ck = check(back_kc) if back_kc is not None else (None, np.zeros(0))
            # the probe's row space against the causal goal subspace of (A, B)
            subAB = subspaces[(s, A, B)]
            if subAB["k_c"] is not None:
                Q, _ = torch.linalg.qr(probe_rows(net, sd).double().T); Pp = Q @ Q.T; Pk = C.projector(subAB["Vh"].double(), slice(0, subAB["k_c"]))
                row["probe_overlap_kc"] = C.subspace_overlap(Pp, Pk, subAB["k_c"]); row["probe_overlap_random"] = C.subspace_overlap(Pp, C.random_projector(C.D, subAB["k_c"], 11), subAB["k_c"])
            else: row["probe_overlap_kc"] = row["probe_overlap_random"] = None
            rows.append(row); pool["drop"].append(drop); pool["bump_drop"].append(bd); pool["chk_full"].append(cf); pool["chk_nat"].append(cn); pool["chk_kc"].append(ck)
            if dropped: pool["num_back"].append(jac["back"] - jac["B"]); pool["den"].append(drop); pool["num_off"].append(jac["back"] - jac["off"])
            if dropped and "kc" in jac: pool["num_kc"].append(jac["kc"] - jac["B"]); pool["den_kc"].append(drop)
            if row["probe_overlap_kc"] is not None: pool["overlap"].append(row["probe_overlap_kc"]); pool["overlap_rnd"].append(row["probe_overlap_random"])
            P(f"  site {s} {A}->{B}: Jaccard A {row['j_A']:.3f} B {row['j_B']:.3f} back {row['j_back']:.3f} off {row['j_off']:.3f} kc {row['j_back_kc']} BB {row['j_BB']:.3f}; drop {row['drop'][0]:.3f} restoration {row['restoration']}; bump R² A {row['b_A']:.3f} B {row['b_B']:.3f} back {row['b_back']:.3f}; check full {row['check_switch_full']:.3f} nat {row['check_switch_natural']:.3f}; overlap {row['probe_overlap_kc']}")
        cat = lambda k: np.concatenate(pool[k]) if pool[k] else np.zeros(0)
        d, nb, de, no, nk, dk, bdp = cat("drop"), cat("num_back"), cat("den"), cat("num_off"), cat("num_kc"), cat("den_kc"), cat("bump_drop")
        summary = dict(drop=[float(d.mean())] + C.boot_mean(d), restoration=([float(nb.mean() / de.mean())] + C.boot_ratio(nb, de)) if len(de) else None,
                       over_offset=([float(no.mean() / de.mean())] + C.boot_ratio(no, de)) if len(de) else None, restoration_kc=([float(nk.mean() / dk.mean())] + C.boot_ratio(nk, dk)) if len(dk) else None,
                       bump_drop=[float(bdp.mean())] + C.boot_mean(bdp), pairs_dropped=int(len(pool["den"])), overlap=[float(np.mean(pool["overlap"]))] + C.boot_mean(pool["overlap"]) if pool["overlap"] else None,
                       overlap_random=float(np.mean(pool["overlap_rnd"])) if pool["overlap_rnd"] else None, check_switch_full=[float(cat("chk_full").mean())] + C.boot_mean(cat("chk_full")),
                       check_switch_kc=([float(cat("chk_kc").mean())] + C.boot_mean(cat("chk_kc"))) if len(cat("chk_kc")) else None, check_switch_natural=[float(cat("chk_nat").mean())] + C.boot_mean(cat("chk_nat")))
        res["by_site"][str(s)] = dict(probes_by_goal=by_goal, pairs=rows, summary=summary)
        P(f"site {s}: drop {summary['drop']}; restoration {summary['restoration']}; over offset {summary['over_offset']}; bump drop {summary['bump_drop']}; check {summary['check_switch_full'][0]:.3f} (natural {summary['check_switch_natural'][0]:.3f}); overlap {summary['overlap']} (random {summary['overlap_random']})")
        (out / "preserve.json").write_text(json.dumps(res, indent=1))
    P("done")


if __name__ == "__main__":
    main()
