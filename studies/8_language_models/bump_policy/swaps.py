"""Part C of PLAN.md: same-set and different-set swaps of the decision token's residual (C1) and membership edits
along the linear set probe (C2), the 100k model. Writes swaps.json.

    PYTHONPATH=/workspace/goal_belief:. .venv/bin/python studies/8_language_models/bump_policy/swaps.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from common import HERE, RUNS, Runner, world, load_sets, boot_mean

C1_LAYERS = [8, 12, 16, 18, 20, 22]
N_PAIRS, N_EDIT = 1024, 2048


def probe_read(weights, l, x):
    """The set probe's logits at layer l for residuals x [n, d]."""
    p = weights[l]; return ((x - p["mu"].double()) / p["sd"].double()) @ p["W"].double().T + p["b"].double()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--model", default="llm_100000"); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); S = load_sets(); n = len(S["keys"]); masks = S["masks"]; Q = S["Q"].double(); V = Q.max(1).values; opt = Q >= V[:, None] - 1e-9; size = masks.sum(1).numpy()
    hp = HERE / "histories.pt"
    if hp.exists(): hist = torch.load(hp)
    else:
        from policy_table import collect_histories; hist = collect_histories(w, sol, S)
    R = Runner(str(RUNS / a.model / "model"), w, a.device); rng = np.random.default_rng(0); res = {}
    # ---- C1
    multi = [i for i in range(n) if len(hist[S["keys"][i]]) >= 2]; same = [(i, int(rng.integers(1, len(hist[S["keys"][i]])))) for i in rng.choice(multi, N_PAIRS, replace=False)]
    hA = [(S["acts"][i], S["obss"][i]) for i, _ in same]; hB = [hist[S["keys"][i]][j] for i, j in same]
    diff = []
    while len(diff) < N_PAIRS:
        i = int(rng.integers(n)); cand = np.nonzero(size == size[i])[0]; j = int(cand[rng.integers(len(cand))])
        if j != i: diff.append((i, j))
    hC = [(S["acts"][i], S["obss"][i]) for i, _ in diff]; hD = [(S["acts"][j], S["obss"][j]) for _, j in diff]
    zA, rA = R.run(hA, record=C1_LAYERS, pos=[-1]); zB, rB = R.run(hB, record=C1_LAYERS, pos=[-1]); zC, rC = R.run(hC, record=C1_LAYERS, pos=[-1]); zD, rD = R.run(hD, record=C1_LAYERS, pos=[-1])
    dA, dB, dC, dD = zA.argmax(1), zB.argmax(1), zC.argmax(1), zD.argmax(1)
    res["c1"] = dict(same_set_natural_disagree=float((dA != dB).double().mean()), diff_set_natural_disagree=float((dC != dD).double().mean()), layers=[])
    print(f"C1: same-set pairs' natural decisions differ in {float((dA != dB).double().mean()):.3f}, different-set {float((dC != dD).double().mean()):.3f} {time.time() - t0:.0f}s", flush=True)
    for l in C1_LAYERS:
        zp, _ = R.run(hA, patch=[(l, -1, rB[l][:, 0])]); dp = zp.argmax(1); chg = (dp != dA); toB = (dp == dB)[dA != dB]
        zq, _ = R.run(hC, patch=[(l, -1, rD[l][:, 0])]); dq = zq.argmax(1); inf = dC != dD; toD = (dq == dD)[inf]
        tr = (((zq - zC) * (zD - zC)).sum(1) / ((zD - zC) ** 2).sum(1).clamp(min=1e-9))[inf].mean()
        row = dict(layer=l, same_changed=float(chg.double().mean()), same_changed_ci=boot_mean(chg.numpy()), same_to_donor=float(toB.double().mean()) if len(toB) else None, diff_to_donor=float(toD.double().mean()), diff_to_donor_ci=boot_mean(toD.numpy()), diff_transfer=float(tr))
        res["c1"]["layers"].append(row); print(f"  layer {l}: same-set swap changes the decision {row['same_changed']:.3f} (to the donor's where they differ {row['same_to_donor']}); different-set swap to the donor's {row['diff_to_donor']:.3f}, transfer {row['diff_transfer']:.2f} {time.time() - t0:.0f}s", flush=True)
    (HERE / "swaps.json").write_text(json.dumps(res, indent=1))
    # ---- C2
    weights = torch.load(HERE / "probe_weights.pt"); pj = json.loads((HERE / "probes.json").read_text()); best = int(pj["best_layer"]["set_exact"]); layers = sorted(set([best - 2, best, best + 2, 20, 22]) & set(weights))
    pick = np.random.default_rng(0).choice(n, 8192, replace=False); fitset = set(pick[np.arange(8192) % 8 != 0].tolist())        # the probes' fit sets (probes.py)
    near = S["nearest"].numpy(); nidx = S["near_idx"].numpy()
    pool = [i for i in range(n) if i not in fitset and 1 <= near[i] <= 2]; rng.shuffle(pool)
    tab, _ = (torch.load(HERE / f"table_{a.model}.pt")["logits"], None) if (HERE / f"table_{a.model}.pt").exists() else R.run([(S["acts"][i], S["obss"][i]) for i in range(n)])
    actm = tab.argmax(1).numpy(); pairs = [(i, int(nidx[i])) for i in pool if actm[i] != actm[nidx[i]]][:N_EDIT]
    hS = [(S["acts"][i], S["obss"][i]) for i, _ in pairs]; hT = [(S["acts"][j], S["obss"][j]) for _, j in pairs]
    MS = masks[[i for i, _ in pairs]]; MT = masks[[j for _, j in pairs]]; D = MS ^ MT; optT = opt[[j for _, j in pairs]]; optS = opt[[i for i, _ in pairs]]
    rec_layers = sorted(set(layers) | {best, min(best + 2, 24)})
    zS, rS = R.run(hS, record=rec_layers, pos=[-1]); zT, rT = R.run(hT, record=rec_layers, pos=[-1]); dS, dT = zS.argmax(1), zT.argmax(1)
    print(f"C2: {len(pairs)} pairs at Hamming ≤ 2 with disjoint optimal sets and different model decisions; natural decisions differ in {float((dS != dT).double().mean()):.3f}; model optimal on S {float(optS[torch.arange(len(pairs)), dS].double().mean()):.3f}, on S' {float(optT[torch.arange(len(pairs)), dT].double().mean()):.3f} {time.time() - t0:.0f}s", flush=True)
    res["c2"] = dict(n_pairs=len(pairs), best_set_layer=best, layers=[], natural_differ=float((dS != dT).double().mean()))
    for l in layers:
        p = weights[l]; W, b, mu, sd = p["W"].double(), p["b"].double(), p["mu"].double(), p["sd"].double(); x = rS[l][:, 0].double(); z = (x - mu) / sd; logit = z @ W.T + b
        edits = {}
        # the minimum-norm change (in standardised coordinates) that mirrors the probe's logits for the differing states
        dz = torch.zeros_like(z); dctl = torch.zeros_like(z)
        for r in range(len(pairs)):
            idx = torch.nonzero(D[r]).squeeze(1); Wd = W[idx]; target = -logit[r, idx]; dz[r] = Wd.T @ torch.linalg.solve(Wd @ Wd.T + 1e-6 * torch.eye(len(idx), dtype=torch.float64), target - logit[r, idx])
            other = torch.as_tensor(rng.choice(np.nonzero(~D[r].numpy())[0], len(idx), replace=False)); Wo = W[other]; v = Wo.T @ torch.linalg.solve(Wo @ Wo.T + 1e-6 * torch.eye(len(idx), dtype=torch.float64), -2 * logit[r, other]); dctl[r] = v / v.norm().clamp(min=1e-9) * dz[r].norm()
        dx = dz * sd; dxc = dctl * sd; gr = torch.Generator().manual_seed(l); rnd = torch.randn(dx.shape, generator=gr, dtype=torch.float64); rnd = rnd / rnd.norm(dim=1, keepdim=True) * dx.norm(dim=1, keepdim=True)
        nat_norm = (rT[l][:, 0].double() - x).norm(dim=1)
        variants = {"edit": x + dx, "edit_x2": x + 2 * dx, "edit_x4": x + 4 * dx, "control_same_subspace": x + dxc, "random": x + rnd, "natural_swap": rT[l][:, 0].double()}
        for name, xn in variants.items():
            rl = [ll for ll in rec_layers if ll > l]; zp, rp = R.run(hS, record=rl, pos=[-1], patch=[(l, -1, xn.float())]); dp = zp.argmax(1)
            row = dict(to_model_T=float((dp == dT).double().mean()), to_model_T_ci=boot_mean((dp == dT).numpy()), into_opt_T=float(optT[torch.arange(len(pairs)), dp].double().mean()), changed=float((dp != dS).double().mean()), norm_ratio=float((xn - x).norm(dim=1).mean() / nat_norm.mean()))
            # does the set probe at a later layer read S' now?
            for ll in rl[:1]:
                pr = probe_read(weights, ll, rp[ll][:, 0].double()) > 0; row[f"probe_L{ll}_reads_T"] = float((pr == MT).all(1).double().mean()); row[f"probe_L{ll}_reads_S"] = float((pr == MS).all(1).double().mean()); row[f"probe_L{ll}_diff_states_T"] = float((pr[D] == MT[D]).double().mean())
            edits[name] = row
        res["c2"]["layers"].append(dict(layer=l, probe_exact_here=float(((logit > 0) == MS).all(1).double().mean()), edits=edits))
        print(f"  layer {l} (probe exact here {res['c2']['layers'][-1]['probe_exact_here']:.2f}): " + " | ".join(f"{k}: to S'-decision {v['to_model_T']:.3f}, into π*(S') {v['into_opt_T']:.3f}, norm {v['norm_ratio']:.2f}" + (f", later probe reads S' {v[[kk for kk in v if kk.endswith('reads_T')][0]]:.2f}" if any(kk.endswith('reads_T') for kk in v) else "") for k, v in edits.items()) + f" {time.time() - t0:.0f}s", flush=True)
        (HERE / "swaps.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
