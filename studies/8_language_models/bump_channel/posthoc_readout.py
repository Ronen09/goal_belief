"""Post hoc (after the review; not registered in PLAN.md): is the goal subspace's carrying of the belief effect
explained by its overlap with the logit readout? C = the goal subspace (k directions) at a site of the fine-tuned
model, R = the top-k_R right-singular directions of the logit Jacobian; C⊥R = orth((I − P_R) C), C∥R = orth(P_R C).
The belief effect (part 2: the natural history difference; part 1: the change induced at 23 by the belief swapped in
at 19) restricted to C, R, C⊥R, C∥R, C⊥R ∪ R, random subspaces of C⊥R's dimension anywhere and inside R's
complement (three draws each), installed as channel.py installs them. Writes posthoc_readout.json.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_channel .venv/bin/python studies/8_language_models/bump_channel/posthoc_readout.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import common as CH
from channel import _json

CC = CH.CC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--quick", action="store_true"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--kR", type=int, nargs="*", default=(4, 8, 16)); ap.add_argument("--n_jac", type=int, default=None); ap.add_argument("--n_pairs", type=int, default=None); a = ap.parse_args(); t0 = time.time(); P = lambda *s: print(*s, f"{time.time() - t0:.0f}s", flush=True)
    out = Path(a.out) if a.out else CH.HERE; out.mkdir(parents=True, exist_ok=True); n_pairs = a.n_pairs or (16 if a.quick else CH.N_PAIRS); n_jac = a.n_jac or (8 if a.quick else 512); kRs = (4,) if a.quick else tuple(a.kR)
    free, goals, worlds, sols = CC.worlds(); swap = CH.load_states("llm"); pool = CH.pool(worlds, sols); assert list(pool["test_idx"]) == list(swap["dec_test_i1"])
    sites_all, D = CH.SITES_LLM, 896; sites = (23,) if a.quick else (23, 21); M = CH.LLM(None, worlds, a.device, a.batch)
    hf, ht, sf, st = swap["hists_fit"], swap["hists_h1"], swap["steps_fit"], swap["steps_test"]; X_fit, X_h1, L_h1 = swap["X_fit"], swap["X_h1"], swap["L_h1"]; sidx = lambda s: sites_all.index(s)
    pairs = CH.draw_pairs(st, pool["keys_test"], n_pairs, seed=1); I = torch.tensor([p[0] for p in pairs]); Jx = torch.tensor([p[1] for p in pairs])
    nat_i = {g: L_h1[g][I].argmax(1) for g in range(CH.K)}; nat_j = {g: L_h1[g][Jx].argmax(1) for g in range(CH.K)}; dep = {g: (nat_i[g] != nat_j[g]).nonzero().squeeze(1) for g in range(CH.K)}
    cells_p = np.concatenate([dep[g].numpy() for g in range(CH.K)]); groups = I.numpy()[cells_p]; hists_I = [ht[i] for i in I.tolist()]; P(f"{n_pairs} pairs, {len(cells_p)} history-dependent cells (as channel.py)")

    def run_cells(site, cand):
        per = []
        for g in range(CH.K):
            idx = dep[g]
            if len(idx) == 0: continue
            L = M.patched([hists_I[t] for t in idx.tolist()], g, site, cand[g][idx]); per.append((L.argmax(1) == nat_j[g][idx]).double().numpy())
        return np.concatenate(per) if per else np.zeros(0)

    def measure(site, x1, delta, extra, projs):
        V = dict(extra); [V.__setitem__(name, {g: CH.candidate(x1[g], delta[g], Pm) for g in range(CH.K)}) for name, Pm in projs.items()]
        per = {name: run_cells(site, cand) for name, cand in V.items()}; ref = per["natural" if "natural" in per else "full"]; dstack = torch.cat([delta[g][dep[g]] for g in range(CH.K)])
        res = {}
        for name in V:
            r = CH.through(per[name], ref, groups)
            if name in projs: r["norm_share"] = CH.norm_share(dstack, projs[name])
            res[name] = r
        return res

    rng_j = np.random.default_rng(3); res = dict(kR=list(kRs), n_pairs=n_pairs, n_cells=int(len(cells_p)), part2={}, part1={})
    induced = None
    for s in sites:
        si = sidx(s); k = CH.K_LLM[s]; Vh_goal, _ = CH.goal_basis({g: X_fit[g][:, si] for g in range(CH.K)}, sf); C = Vh_goal[:k].T
        pick = rng_j.choice(len(hf), min(n_jac, len(hf)), replace=False); J = torch.cat([M.jacobian([hf[i] for i in pick], g, s) for g in range(CH.K)]); Vh_read = CH.readout_basis(J); P(f"site {s}: goal basis (k {k}) and readout basis")
        x1 = {g: X_h1[g][I, si].double() for g in range(CH.K)}; x2 = {g: X_h1[g][Jx, si].double() for g in range(CH.K)}; delta = {g: x2[g] - x1[g] for g in range(CH.K)}
        if s == 23:
            if induced is None:
                induced = {g: M.patched_record(hists_I, g, 19, X_h1[g][Jx, sidx(19)])[1][:, si].double() for g in range(CH.K)}; P("upstream swap at 19 recorded")
            delta1 = {g: induced[g] - x1[g] for g in range(CH.K)}
        res["part2"][str(s)] = {}; res["part1"][str(s)] = {}
        for kR in kRs:
            perp, par = CH.orth_split(C, Vh_read, kR); kp, kpa = perp.shape[1], par.shape[1]; PR = CH.projector(Vh_read, kR); PC = C @ C.T
            projs = dict(C=PC, R=PR, C_perp=perp @ perp.T, C_par=par @ par.T, C_perp_plus_R=(lambda Q: Q @ Q.T)(CH._orth(torch.cat([perp, Vh_read[:kR].T], 1))))
            for i in range(3):
                projs[f"random_full{i}"] = CC.random_projector(D, max(kp, 1), 10 + i); Q = CH.random_in_complement(Vh_read, kR, D, max(kp, 1), 20 + i); projs[f"random_comp{i}"] = Q @ Q.T
            meta = dict(k=k, k_perp=kp, k_par=kpa, overlap_C_R=CH.overlap(PC, PR, k))
            r2 = measure(s, x1, delta, dict(natural=x2), projs); res["part2"][str(s)][str(kR)] = dict(**meta, variants=r2)
            P(f"part 2 site {s} kR {kR}: C⊥R dim {kp}, C∥R dim {kpa}, overlap {meta['overlap_C_R']:.3f}; " + " ".join(f"{n} {r2[n]['ratio']:.2f}" if r2[n]["ratio"] is not None else f"{n} -" for n in ("C", "R", "C_perp", "C_par", "C_perp_plus_R", "random_full0", "random_comp0")))
            if s == 23:
                r1 = measure(s, x1, delta1, dict(full={g: induced[g] for g in range(CH.K)}), projs); res["part1"][str(s)][str(kR)] = dict(**meta, variants=r1)
                P(f"part 1 19->{s} kR {kR}: " + " ".join(f"{n} {r1[n]['ratio']:.2f}" if r1[n]["ratio"] is not None else f"{n} -" for n in ("C", "R", "C_perp", "C_par", "C_perp_plus_R", "random_full0", "random_comp0")))
            (out / "posthoc_readout.json").write_text(json.dumps(res, indent=1, default=_json))
    P("done")


if __name__ == "__main__":
    main()
