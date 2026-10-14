"""Post hoc (after part A was seen): the model's optimality on the same training-visited sets reached by three kinds of
history, one from the training data itself, the shortest, and a random walk; and what same-set disagreements go with
(the difference in length, the last action, the last observation). The 100k model. Writes posthoc_histories.json.

    PYTHONPATH=/workspace/goal_belief:. .venv/bin/python studies/8_language_models/bump_policy/posthoc_histories.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ
from common import HERE, RUNS, Runner, world, load_sets, boot_mean, key, size_bins


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--model", default="llm_100000"); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); S = load_sets(); n = len(S["keys"]); idx = {k: i for i, k in enumerate(S["keys"])}; Q = S["Q"].numpy(); V = Q.max(1); opt = Q >= V[:, None] - 1e-9
    cnt_opt = S["cnt_opt"].numpy(); cnt100 = S["cnt_train_100000"].numpy()
    data = torch.load(HERE.parent / "bump_maze" / "data" / "train_100000.pt"); A, O, Ln = data["acts"], data["obss"], data["length"]; N = len(Ln)
    train_hist = {}; f = BMZ.Filter(w, N)
    for t in range(w.H):
        alive = np.nonzero((Ln > t).numpy())[0]; pk = np.packbits(f.mask.numpy()[alive], axis=1)
        for j, i in zip(range(len(alive)), alive):
            k = pk[j].tobytes()
            if k not in train_hist and k in idx: train_hist[k] = (A[i, :t].tolist(), O[i, :t].tolist())
        f.update(A[:, t], O[:, t])
    print(f"training histories for {len(train_hist)} sets {time.time() - t0:.0f}s", flush=True)
    hist = torch.load(HERE / "histories.pt"); R = Runner(str(RUNS / a.model / "model"), w, a.device)
    sets = [i for i in range(n) if cnt100[i] > 0 and S["keys"][i] in train_hist]
    kinds = {"training-data history": [train_hist[S["keys"][i]] for i in sets], "shortest history": [(S["acts"][i], S["obss"][i]) for i in sets]}
    rw = []
    for i in sets:
        hs = [h for h in hist[S["keys"][i]][1:] if h != train_hist[S["keys"][i]]]; rw.append(hs[0] if hs else None)
    has_rw = np.array([h is not None for h in rw]); kinds["random-walk history"] = [h if h is not None else (S["acts"][i], S["obss"][i]) for h, i in zip(rw, sets)]
    acts = {}; res = dict(n_sets=len(sets), kinds={})
    groups = {"optimum-visited": cnt_opt[sets] > 0, "training-visited, not by the optimum": cnt_opt[sets] == 0}
    for kname, hs in kinds.items():
        z, _ = R.run(hs); ac = z.argmax(1).numpy(); acts[kname] = ac; ok = opt[sets, ac]; valid = has_rw if kname.startswith("random") else np.ones(len(sets), dtype=bool)
        res["kinds"][kname] = dict(optimal=float(ok[valid].mean()), optimal_ci=boot_mean(ok[valid]), by_group={g: float(ok[valid & m].mean()) for g, m in groups.items()}, mean_length=float(np.mean([len(h[0]) for h, v in zip(hs, valid) if v])))
        print(f"{kname}: optimal {ok[valid].mean():.3f} (" + ", ".join(f"{g} {ok[valid & m].mean():.3f}" for g, m in groups.items()) + f"); mean length {res['kinds'][kname]['mean_length']:.1f} {time.time() - t0:.0f}s", flush=True)
    ag = {}
    names = list(kinds)
    for i_ in range(3):
        for j_ in range(i_ + 1, 3):
            v = has_rw if "random" in (names[i_] + names[j_]) else np.ones(len(sets), dtype=bool); ag[f"{names[i_]} vs {names[j_]}"] = float((acts[names[i_]] == acts[names[j_]])[v].mean())
    res["agreement"] = ag; print("agreement between kinds:", ag, flush=True)
    # same-set disagreement against history differences (training-data history vs shortest)
    at, ash = acts["training-data history"], acts["shortest history"]; ht, hsh = kinds["training-data history"], kinds["shortest history"]
    dl = np.array([len(x[0]) - len(y[0]) for x, y in zip(ht, hsh)]); last_a_same = np.array([x[0][-1] == y[0][-1] if x[0] and y[0] else True for x, y in zip(ht, hsh)]); dis = at != ash
    by_dl = {}
    for lo, hi, lab in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 100, "> 10")):
        m = (dl >= lo) & (dl <= hi)
        if m.sum() >= 30: by_dl[lab] = dict(n=int(m.sum()), disagree=float(dis[m].mean()))
    res["disagreement_by_length_difference"] = by_dl; res["disagreement_by_last_action"] = dict(same=float(dis[last_a_same].mean()), different=float(dis[~last_a_same].mean()))
    # when they disagree, which one is optimal
    both = dis & opt[sets, at] & opt[sets, ash]; only_t = dis & opt[sets, at] & ~opt[sets, ash]; only_s = dis & ~opt[sets, at] & opt[sets, ash]; neither = dis & ~opt[sets, at] & ~opt[sets, ash]
    res["when_disagreeing"] = dict(n=int(dis.sum()), both_optimal=float(both.sum() / max(dis.sum(), 1)), only_training_optimal=float(only_t.sum() / max(dis.sum(), 1)), only_shortest_optimal=float(only_s.sum() / max(dis.sum(), 1)), neither=float(neither.sum() / max(dis.sum(), 1)))
    print("disagreement by length difference:", by_dl, "| by last action:", res["disagreement_by_last_action"], "| when disagreeing:", res["when_disagreeing"], flush=True)
    (HERE / "posthoc_histories.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
