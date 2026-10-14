"""Post hoc (after B1 was seen): is the set code of the single-goal study a general code or a lookup over the optimum's
few sets? The set probe at layers 12–24 on (i) the optimum's decisions split by episode (the single-goal study's
split: held-out decisions, seen sets), (ii) the same split by set (held-out sets), (iii) decisions from the training
data's episodes (ε = 0.2) split by set, (iv) B1's uniform sets with shortest histories (from probes.json). The 100k
model. Writes posthoc_probe.json.

    PYTHONPATH=/workspace/goal_belief:. .venv/bin/python studies/8_language_models/bump_policy/posthoc_probe.py
"""

from __future__ import annotations

import argparse, json, time

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ
from common import HERE, RUNS, Runner, world, load_sets, key
from probes import fit, exact_share
import sys; sys.path.insert(0, str(HERE.parent / "bump_maze")); from zero_shot import optimum_decisions   # noqa: E402

LAYERS = [12, 16, 18, 20, 24]


def scores(P, M):
    pred = P > 0; aucs = []
    for p, m in zip(P.numpy(), M.numpy()):
        if m.any() and (~m).any(): pos, neg = p[m], p[~m]; aucs.append(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))
    return dict(exact=exact_share(P, M), jaccard=float(((pred & M).sum(1) / (pred | M).sum(1).clamp(min=1)).mean()), auc=float(np.mean(aucs)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--model", default="llm_100000"); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); R = Runner(str(RUNS / a.model / "model"), w, a.device); rng = np.random.default_rng(0); res = {}
    # (i)/(ii) the optimum's decisions, as in the single-goal probe (1024 episodes, seed 4000; 8192 decisions)
    dec_all = optimum_decisions(w, sol, 1024, 4000); ep_ids = []; e = BMZ.Env(w, 1024, torch.Generator().manual_seed(4000))
    for t in range(w.H):
        al = torch.nonzero(~e.done).squeeze(1).tolist(); ep_ids += al; a_opt, _ = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); e.step(torch.as_tensor(a_opt))
        if e.done.all(): break
    pick = rng.choice(len(dec_all), 8192, replace=False); dec = [dec_all[i] for i in pick]; ep = np.array([ep_ids[i] for i in pick]); M = torch.stack([torch.as_tensor(d["mask"]) for d in dec]); ks = [key(d["mask"]) for d in dec]
    uniq = sorted(set(ks)); sid = {k: i for i, k in enumerate(uniq)}; sidx = np.array([sid[k] for k in ks]); set_te = np.isin(sidx, rng.choice(len(uniq), len(uniq) // 4, replace=False))
    _, rec = R.run([(d["acts"], d["obss"]) for d in dec], record=LAYERS, pos=[-1])
    for name, te in (("optimum decisions, split by episode", torch.as_tensor(ep % 8 == 0)), ("optimum decisions, split by set", torch.as_tensor(set_te))):
        res[name] = dict(n_sets=len(uniq), n_test=int(te.sum()), layers={})
        for l in LAYERS:
            P, _ = fit(rec[l][:, 0], M, ~te, te, "logistic"); res[name]["layers"][str(l)] = scores(P, M[te])
        print(name + ": " + " | ".join(f"L{l} exact {v['exact']:.3f} jaccard {v['jaccard']:.2f} auc {v['auc']:.3f}" for l, v in res[name]["layers"].items()) + f" {time.time() - t0:.0f}s", flush=True)
    # (iii) decisions from the training data's episodes, split by set
    data = torch.load(HERE.parent / "bump_maze" / "data" / "train_100000.pt"); A, O, Ln = data["acts"], data["obss"], data["length"]
    rows = rng.choice(len(Ln), 8192, replace=False); dec3 = []
    for i in rows:
        t = int(rng.integers(0, int(Ln[i]))); acts, obss = A[i, :t].tolist(), O[i, :t].tolist(); m = np.zeros(w.n_states, dtype=bool); m[w.start_states] = True
        for a_, o_ in zip(acts, obss): m = BMZ.step_belief(w, m, a_, o_)
        dec3.append(dict(acts=acts, obss=obss, mask=m))
    M3 = torch.stack([torch.as_tensor(d["mask"]) for d in dec3]); ks3 = [key(d["mask"]) for d in dec3]; uniq3 = sorted(set(ks3)); sid3 = {k: i for i, k in enumerate(uniq3)}; s3 = np.array([sid3[k] for k in ks3]); te3 = torch.as_tensor(np.isin(s3, rng.choice(len(uniq3), len(uniq3) // 8, replace=False)))
    _, rec3 = R.run([(d["acts"], d["obss"]) for d in dec3], record=LAYERS, pos=[-1]); name = "training-data decisions, split by set"; res[name] = dict(n_sets=len(uniq3), n_test=int(te3.sum()), layers={})
    for l in LAYERS:
        P, _ = fit(rec3[l][:, 0], M3, ~te3, te3, "logistic"); res[name]["layers"][str(l)] = scores(P, M3[te3])
    print(name + ": " + " | ".join(f"L{l} exact {v['exact']:.3f} jaccard {v['jaccard']:.2f} auc {v['auc']:.3f}" for l, v in res[name]["layers"].items()) + f" {time.time() - t0:.0f}s", flush=True)
    # the same training-data decisions split by episode (seen sets allowed)
    te3e = torch.as_tensor(np.arange(len(dec3)) % 8 == 0); name = "training-data decisions, split by episode"; res[name] = dict(n_sets=len(uniq3), n_test=int(te3e.sum()), layers={})
    for l in LAYERS:
        P, _ = fit(rec3[l][:, 0], M3, ~te3e, te3e, "logistic"); res[name]["layers"][str(l)] = scores(P, M3[te3e])
    print(name + ": " + " | ".join(f"L{l} exact {v['exact']:.3f} jaccard {v['jaccard']:.2f} auc {v['auc']:.3f}" for l, v in res[name]["layers"].items()) + f" {time.time() - t0:.0f}s", flush=True)
    (HERE / "posthoc_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
