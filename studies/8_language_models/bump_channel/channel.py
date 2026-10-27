"""Parts 1–3 of PLAN.md on one model: same-step pairs of held-out histories with different goal-free sets under every
goal; (part 2) the natural history difference at each site restricted to the goal subspace, its complement, the
bottom and random directions, the logit readout's Jacobian subspace and the belief effect's own principal directions,
and the sweep over the rank; (part 1) the belief swapped in at an upstream site, the change it induces downstream
restricted the same way; (part 3) every through ratio split by whether the receiving history's set is a singleton;
the overlaps between the subspaces and the registered scorecard. Writes channel_<model>.json.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_channel .venv/bin/python studies/8_language_models/bump_channel/channel.py --model llm
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as CH

CC = CH.CC
THRU, REJECT, COMP_MAX, CTRL_MAX, DISTINCT_MAX = 0.65, 0.40, 0.35, 0.20, 0.50


def _json(o):
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor): return o.tolist()
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", choices=["llm", "tfm", "gru"], required=True); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cache", default=None, help="the cache holding states_<kind>.pt and subspaces_<kind>.pt (from-scratch models)"); ap.add_argument("--batch", type=int, default=None); ap.add_argument("--n_pairs", type=int, default=None); ap.add_argument("--n_jac", type=int, default=None)
    a = ap.parse_args(); t0 = time.time(); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    out = Path(a.out) if a.out else CH.HERE; out.mkdir(parents=True, exist_ok=True); kind = a.model; n_pairs = a.n_pairs or (16 if a.quick else CH.N_PAIRS); n_jac = a.n_jac or (8 if a.quick else 512)
    free, goals, worlds, sols = CC.worlds()
    if kind == "llm":
        swap = CH.load_states("llm"); pool = CH.pool(worlds, sols); assert list(pool["test_idx"]) == list(swap["dec_test_i1"]) and list(pool["fit_idx"]) == list(swap["dec_fit_idx"])
        keys_fit, keys_test, sizes_test = pool["keys_fit"], pool["keys_test"], pool["sizes_test"]; sites_all = CH.SITES_LLM; D = 896; KS = (1, 4, D) if a.quick else CH.KS_LLM
        M = CH.LLM(None, worlds, a.device, a.batch or 32); sites = (19, 23) if a.quick else sites_all; k_by = dict(CH.K_LLM); linear = None
        in_out = [(19, 23)] if a.quick else [(15, 21), (15, 23), (19, 21), (19, 23)]; reg_site, reg_in = 23, 19
        Vh_goal = {}
        for s in sites:
            si = sites_all.index(s); Vh_goal[s], spec = CH.goal_basis({g: swap["X_fit"][g][:, si] for g in range(CH.K)}, swap["steps_fit"]); P(f"goal basis at {s}: pooled k50/90/99 {spec['k50']}/{spec['k90']}/{spec['k99']}, k = {k_by[s]}")
    else:
        cache = Path(a.cache) if a.cache else CH.CACHE; swap = torch.load(cache / f"states_{kind}.pt", weights_only=False); sub = torch.load(cache / f"subspaces_{kind}.pt", weights_only=False)
        keys_fit, keys_test, sizes_test = swap["keys_fit"], swap["keys_test"], swap["sizes_test"]; sites_all = tuple(swap["sites"]); sites = sites_all; D = CH.D_SMALL; KS = (1, 4, D) if a.quick else CH.KS_SMALL
        M = CH.load_small(kind, worlds[0].H, a.device); M.batch = a.batch or 2048; k_by = {s: sub[s]["k"] for s in sites}; Vh_goal = {s: sub[s]["Vh"].double() for s in sites}; linear = 4 if kind == "tfm" else 0
        in_out = [(s1, s2) for s1 in sites for s2 in sites if s1 < s2]; reg_site = sub["registered_site"]; reg_in = max([s1 for (s1, s2) in in_out if s2 == reg_site], default=None) if reg_site is not None else None
        P(f"{kind}: sites {sites}, k {k_by}, registered site {reg_site}")
    hf, ht, sf, st = swap["hists_fit"], swap["hists_h1"], swap["steps_fit"], swap["steps_test"]; X_fit, X_h1, L_h1 = swap["X_fit"], swap["X_h1"], swap["L_h1"]
    sidx = lambda s: sites_all.index(s)

    # the pairs and the history-dependent cells
    pairs = CH.draw_pairs(st, keys_test, n_pairs, seed=1); I = torch.tensor([p[0] for p in pairs]); Jx = torch.tensor([p[1] for p in pairs]); known_pair = np.asarray(sizes_test)[I.numpy()] == 1
    nat_i = {g: L_h1[g][I].argmax(1) for g in range(CH.K)}; nat_j = {g: L_h1[g][Jx].argmax(1) for g in range(CH.K)}; dep = {g: (nat_i[g] != nat_j[g]).nonzero().squeeze(1) for g in range(CH.K)}
    cells_g = np.concatenate([np.full(len(dep[g]), g) for g in range(CH.K)]); cells_p = np.concatenate([dep[g].numpy() for g in range(CH.K)]); groups = I.numpy()[cells_p]; known = known_pair[cells_p]
    dep_share = [float(len(dep[g]) / n_pairs) for g in range(CH.K)]; hists_I = [ht[i] for i in I.tolist()]
    P(f"{n_pairs} pairs x {CH.K} goals: {len(cells_p)} history-dependent cells (share by goal {[round(x, 2) for x in dep_share]}), {int(known.sum())} at known states")

    def run_cells(site, cand):
        """cand: {g: [n_pairs, D]} installed at `site` in h1's run under g on the history-dependent cells; -> per-cell 0/1 to h2's decision."""
        per = []
        for g in range(CH.K):
            idx = dep[g]
            if len(idx) == 0: continue
            L = M.patched([hists_I[t] for t in idx.tolist()], g, site, cand[g][idx]); per.append((L.argmax(1) == nat_j[g][idx]).double().numpy())
        return np.concatenate(per) if per else np.zeros(0)

    # subspaces per site: goal (registered k), history, readout
    bases, projs, overlaps = {}, {}, {}
    rng_j = np.random.default_rng(3)
    for s in sites:
        si = sidx(s); k = k_by[s]
        Vh_hist = CH.history_basis({g: X_fit[g][:, si] for g in range(CH.K)}, sf, keys_fit, n_pairs=min(CH.N_PAIRS, len(hf)), seed=2)
        pick = rng_j.choice(len(hf), min(n_jac, len(hf)), replace=False); J = torch.cat([M.jacobian([hf[i] for i in pick], g, s) for g in range(CH.K)]); Vh_read = CH.readout_basis(J)
        bases[s] = dict(goal=Vh_goal[s], hist=Vh_hist, read=Vh_read)
        if k is None: P(f"site {s}: no registered k; only the sweep is run"); projs[s] = {}; overlaps[s] = None; continue
        pr = CH.projectors(Vh_goal[s], k, D); pr["readout"] = CH.projector(Vh_read, k); pr["history"] = CH.projector(Vh_hist, k); projs[s] = pr
        overlaps[s] = dict(goal_readout=CH.overlap(pr["goal"], pr["readout"], k), goal_history=CH.overlap(pr["goal"], pr["history"], k), history_readout=CH.overlap(pr["history"], pr["readout"], k), random=CH.overlap(pr["goal"], pr["random0"], k), k=k)
        P(f"site {s}: k {k}; overlap goal/readout {overlaps[s]['goal_readout']:.3f}, goal/history {overlaps[s]['goal_history']:.3f}, history/readout {overlaps[s]['history_readout']:.3f}, random {overlaps[s]['random']:.3f}")

    def variants(s, x1, delta, extra):
        """{name: {g: candidate state}} for the registered projectors, the sweep and the extras (natural / full / floor)."""
        V = dict(extra)
        for name, Pm in projs[s].items(): V[name] = {g: CH.candidate(x1[g], delta[g], Pm) for g in range(CH.K)}
        for k in KS: V[f"k{k}"] = {g: CH.candidate(x1[g], delta[g], CH.projector(bases[s]["goal"], k)) for g in range(CH.K)}
        return V

    def measure(s, V, ref_name, delta):
        per = {name: run_cells(s, cand) for name, cand in V.items()}; ref = per[ref_name]; res = {}
        dstack = torch.cat([delta[g][dep[g]] for g in range(CH.K)])
        for name in V:
            r = CH.through(per[name], ref, groups); r["known"] = CH.split_known(per[name], ref, groups, known)
            if name in projs[s]: r["norm_share"] = CH.norm_share(dstack, projs[s][name])
            elif name.startswith("k"): r["norm_share"] = CH.norm_share(dstack, CH.projector(bases[s]["goal"], int(name[1:])))
            res[name] = r
        return res, per

    res = dict(kind=kind, sites=list(sites), k={str(s): k_by[s] for s in sites}, ks=list(KS), n_pairs=n_pairs, n_cells=int(len(cells_p)), n_known=int(known.sum()), dep_share_by_goal=dep_share, linear_site=linear,
               overlap={str(s): overlaps[s] for s in sites}, part2={}, part1={}, registered=None)
    # part 2: the natural history difference at each site
    for s in sites:
        si = sidx(s); x1 = {g: X_h1[g][I, si].double() for g in range(CH.K)}; x2 = {g: X_h1[g][Jx, si].double() for g in range(CH.K)}; delta = {g: x2[g] - x1[g] for g in range(CH.K)}
        gen = torch.Generator().manual_seed(3); floor = {}
        for g in range(CH.K):
            r = torch.randn(delta[g].shape, generator=gen, dtype=torch.float64); floor[g] = x1[g] + r / r.norm(dim=1, keepdim=True) * delta[g].norm(dim=1, keepdim=True)
        V = variants(s, x1, delta, dict(natural=x2, floor=floor)); res["part2"][str(s)], _ = measure(s, V, "natural", delta)
        r2 = res["part2"][str(s)]; P(f"part 2 site {s}: natural {r2['natural']['switch']:.3f}; " + " ".join(f"{n} {r2[n]['ratio']:.2f}" if r2[n]['ratio'] is not None else f"{n} -" for n in ("goal", "comp", "bottom", "random0", "readout", "history", "floor") if n in r2) + "; sweep " + " ".join(f"{k}:{r2[f'k{k}']['ratio']:.2f}" if r2[f'k{k}']['ratio'] is not None else f"{k}:-" for k in KS))
        (out / f"channel_{kind}.json").write_text(json.dumps(res, indent=1, default=_json))
    # part 1: the upstream swap, measured downstream
    induced = {}
    for s_in in sorted(set(s1 for s1, _ in in_out)):
        si = sidx(s_in); induced[s_in] = {}
        for g in range(CH.K):
            z, Xt = M.patched_record(hists_I, g, s_in, X_h1[g][Jx, si]); induced[s_in][g] = (z, Xt)
        P(f"upstream swap at {s_in} recorded under {CH.K} goals")
    for (s_in, s_out) in in_out:
        so = sidx(s_out); x1 = {g: X_h1[g][I, so].double() for g in range(CH.K)}; xt = {g: induced[s_in][g][1][:, so].double() for g in range(CH.K)}; delta = {g: xt[g] - x1[g] for g in range(CH.K)}
        per_in = np.concatenate([(induced[s_in][g][0].argmax(1) == nat_j[g])[dep[g]].double().numpy() for g in range(CH.K) if len(dep[g])])
        V = variants(s_out, x1, delta, dict(full=xt)); r1, per1 = measure(s_out, V, "full", delta); r1["switch_in"] = dict(switch=float(per_in.mean()), switch_ci=CC.boot_mean(per_in, groups=groups), n=int(len(per_in))); r1["full_vs_upstream"] = CH.through(per1["full"], per_in, groups)
        res["part1"][f"{s_in}->{s_out}"] = r1
        P(f"part 1 {s_in}->{s_out}: upstream switch {per_in.mean():.3f}, full reproduces {r1['full_vs_upstream']['ratio']}; " + " ".join(f"{n} {r1[n]['ratio']:.2f}" if r1[n]['ratio'] is not None else f"{n} -" for n in ("goal", "comp", "bottom", "random0", "readout", "history") if n in r1))
        (out / f"channel_{kind}.json").write_text(json.dumps(res, indent=1, default=_json))
    # the registered scorecard
    if reg_site is not None and k_by.get(reg_site) is not None:
        p2 = res["part2"][str(reg_site)]; p1 = res["part1"].get(f"{reg_in}->{reg_site}") if reg_in is not None else None
        crit = lambda r, lo=None, hi=None: None if r is None or r.get("ratio") is None else dict(value=r["ratio"], ci=r["ratio_ci"], holds=bool((lo is None or r["ratio_ci"][0] > lo) and (hi is None or r["ratio_ci"][1] < hi)))
        thru_d = crit(p2["goal"], lo=THRU); thru_u = crit(p1["goal"], lo=THRU) if (kind == "llm" and p1 is not None) else None
        comp = [crit(p2["comp"], hi=COMP_MAX)] + ([crit(p1["comp"], hi=COMP_MAX)] if p1 is not None else []); ctrl = [crit(r[n], hi=CTRL_MAX) for r in ([p2] + ([p1] if p1 is not None else [])) for n in ("bottom", "random0", "random1", "random2")]
        both = lambda cs: None if any(c is None for c in cs) else dict(values=[c["value"] for c in cs], holds=all(c["holds"] for c in cs))
        ov = overlaps[reg_site]["goal_readout"]; kn = p2["goal"]["known"]
        res["registered"] = dict(site=reg_site, k=k_by[reg_site], s_in=reg_in, THRU_D=thru_d, THRU_U=thru_u, COMP=both(comp), CTRL=both(ctrl), DISTINCT=dict(value=ov, holds=bool(ov < DISTINCT_MAX)),
                                 KNOWN=dict(known=kn["known"]["ratio"], uncertain=kn["uncertain"]["ratio"], diff_ci=kn["diff_ci"], holds=bool(kn["diff_ci"] is not None and kn["diff_ci"][0] > 0)),
                                 reject=bool(thru_d is not None and thru_d["ci"][1] < REJECT), part1_decision_bearing=kind == "llm", degenerate=bool(k_by[reg_site] >= D // 2))     # a rank at or above half the space is no subspace: the criteria are vacuous there
        P(f"registered site {reg_site} (k {k_by[reg_site]}, upstream {reg_in}): " + ", ".join(f"{c} {res['registered'][c]}" for c in ("THRU_D", "THRU_U", "COMP", "CTRL", "DISTINCT", "KNOWN", "reject")))
    else: P("no registered site: the scorecard is undefined for this model")
    (out / f"channel_{kind}.json").write_text(json.dumps(res, indent=1, default=_json)); P("done")


if __name__ == "__main__":
    main()
