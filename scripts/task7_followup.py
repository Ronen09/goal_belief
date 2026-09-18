"""TASK7 follow-up on the saved models: (1) cross-model represented-subspace overlap at full and
reduced rank versus Euclidean RSA agreement, per family; (2) finite (non-linearised) future
logit displacement against the linearised ||J dh||, to diagnose the D_future / D_F instability."""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr, invariants as Iv
sys.path.insert(0, str(Path(__file__).resolve().parent)); from run_task7 import conditions, eval_indices, EPS_KL

out = Path("results7"); R = json.load(open(out / "results7.json"))
conv = {f"{r['label']}_s{r['seed']}" for r in R["runs"] if r["metrics"]["kl"] < EPS_KL and r["cond"]["delta"] == 0.4 and r["family"] != "horizon"}
m = H4.make_hmm4(1.0, 0.4, 1); ev = Pr.make_eval(m, n=2000, T=48); sub, dec = eval_indices(ev["X"].size, False)
P, anchors = Pr.positions(m, ev["Z"]); rng = np.random.default_rng(0)
Z = ev["Z"]; aA = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["P"]]; aB = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["Q"]]
aA = aA[rng.choice(len(aA), 200, replace=False)]; jB = aB[rng.integers(0, len(aB), 200)]
win = torch.as_tensor(np.stack([ev["X"][i, t + 1:t + 2] for i, t in aA]))
M = {}
for C in conditions(False):
    for s in C["seeds"]:
        name = f"{C['label']}_s{s}"
        if name not in conv: continue
        c = C["cond"]; net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=s, gain=c["gain"], out_scale=c["init_scale"])
        net.load_state_dict(torch.load(out / "models" / f"{name}.pt"))
        with torch.no_grad():
            Hs = net.states(torch.as_tensor(ev["X"])).numpy().astype(np.float64)
            hA = torch.as_tensor(Hs[aA[:, 0], aA[:, 1]], dtype=torch.float32); hB = torch.as_tensor(Hs[jB[:, 0], jB[:, 1]], dtype=torch.float32)
            e = net.emb(win); zA = net.out(net.gru(e, hA[None].contiguous())[0][:, -1]).numpy(); zB = net.out(net.gru(e, hB[None].contiguous())[0][:, -1]).numpy()
        dz = (zA - zB) - (zA - zB).mean(1, keepdims=True)
        J, _ = Iv.future_jacobian(net, ev["X"], Hs, aA, 1)
        dh = Hs[P["cueA"]].mean(0) - Hs[P["cueB"]].mean(0); Jd = J @ dh; Jd = Jd - Jd.mean(1, keepdims=True)
        dh_pair = Hs[aA[:, 0], aA[:, 1]] - Hs[jB[:, 0], jB[:, 1]]; Jdp = np.einsum("mvh,mh->mv", J, dh_pair); Jdp = Jdp - Jdp.mean(1, keepdims=True)
        Hsub = Hs.reshape(-1, 64)[sub]; U, sv, _ = np.linalg.svd(Hsub - Hsub.mean(0), full_matrices=False)
        M[name] = {"family": C["family"], "seed": s, "U": U, "RDM": Pr.G.rdm(Hsub), "D_future_lin": float(np.linalg.norm(Jd, axis=1).mean()),
                   "D_future_lin_pairs": float(np.linalg.norm(Jdp, axis=1).mean()), "D_future_finite": float(np.linalg.norm(dz, axis=1).mean())}
        print(name, {k: round(v, 3) for k, v in M[name].items() if k.startswith("D_")}, flush=True)
def fam_pairs(f):
    names = [n for n, v in M.items() if v["family"] == f or (f != "base" and v["family"] == "base" and v["seed"] < 3)]
    return [(a, b) for a, b in itertools.combinations(names, 2) if not (M[a]["family"] == "base" and M[b]["family"] == "base") or f == "base"]
iu = np.triu_indices(len(sub), 1); ks = [4, 8, 16, 32, 64]
res = {"models": {n: {k: v for k, v in d.items() if k.startswith("D_")} for n, d in M.items()}, "overlap": {}}
from scipy.stats import spearmanr
for f in ("base", "gain", "lr", "init"):
    pairs = fam_pairs(f); row = {"n_pairs": len(pairs)}
    for k in ks:
        row[f"overlap_top{k}"] = float(np.mean([np.linalg.norm(M[a]["U"][:, :k].T @ M[b]["U"][:, :k], "fro") ** 2 / k for a, b in pairs]))
        row[f"overlap_top{k}_random"] = k / len(sub)
    row["rsa_between"] = float(np.mean([spearmanr(M[a]["RDM"][iu], M[b]["RDM"][iu]).correlation for a, b in pairs]))
    res["overlap"][f] = row; print(f, {k: round(v, 3) for k, v in row.items()})
vals = np.array([[d["D_future_lin"], d["D_future_lin_pairs"], d["D_future_finite"]] for d in M.values()])
res["cv"] = {"D_future_lin": float(vals[:, 0].std() / vals[:, 0].mean()), "D_future_lin_pairs": float(vals[:, 1].std() / vals[:, 1].mean()), "D_future_finite": float(vals[:, 2].std() / vals[:, 2].mean())}
print("CV", res["cv"])
json.dump(res, open(out / "followup.json", "w"), indent=1)
