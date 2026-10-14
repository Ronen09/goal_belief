"""Part A of PLAN.md: set-sufficiency over histories (A1), the model's action on every reachable set against the
optimum (A2) with the transfer readouts, and the exact value of the policy table (A3). Writes table.json and
table_<model>.pt (logits per set).

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_policy/policy_table.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ
from common import HERE, RUNS, Runner, TfmRunner, world, load_sets, key, size_bins, boot_mean, SIZES

MAX_H = 4


def collect_histories(w, sol, S, n_ep=100000, seed=7):
    """Up to MAX_H distinct histories per set from random-action and ε = 0.5 episodes (the shortest history first)."""
    hist = {k: [(a, o)] for k, a, o in zip(S["keys"], S["acts"], S["obss"])}; need = set(hist)
    for pol, sd in (("random", seed), ("eps", seed + 1)):
        gen = torch.Generator().manual_seed(sd); e = BMZ.Env(w, n_ep, gen)
        for t in range(w.H):
            alive = torch.nonzero(~e.done).squeeze(1); mk = e.filter.mask.numpy(); pk = np.packbits(mk[alive.numpy()], axis=1)
            for j, i in enumerate(alive.tolist()):
                k = pk[j].tobytes(); hs = hist.get(k)
                if hs is not None and len(hs) < MAX_H:
                    h = ([int(x[i]) for x in e.acts], [int(x[i]) for x in e.obss])
                    if h not in hs: hs.append(h)
            a_opt, _ = BMZ.optimal_action(w, sol, mk); rnd = torch.randint(3, (n_ep,), generator=gen)
            a = rnd if pol == "random" else torch.where(torch.rand(n_ep, generator=gen) < 0.5, rnd, torch.as_tensor(a_opt))
            e.step(a)
            if e.done.all(): break
    return hist


def policy_value(w, sol, S, actions):
    """V^π over all sets for the table π (actions indexed as S['keys']), by iteration on the solver's transitions."""
    n = len(S["keys"]); sidx = np.array([sol["sets"][k] for k in S["keys"]]); back = {si: i for i, si in enumerate(sidx)}
    rows, cols, probs, goal = [], [], [], np.zeros(n)
    for i in range(n):
        for p, j in sol["trans"][sidx[i]][int(actions[i])]:
            if j < 0: goal[i] += p
            else: rows.append(i); cols.append(back[j]); probs.append(p)
    P = torch.sparse_coo_tensor(torch.tensor([rows, cols]), torch.tensor(probs, dtype=torch.float64), (n, n)).coalesce(); g = torch.tensor(goal)
    V = torch.zeros(n, dtype=torch.float64)
    for _ in range(20000):
        Vn = g + w.gamma * torch.sparse.mm(P, V[:, None]).squeeze(1)
        if (Vn - V).abs().max() < 1e-12: V = Vn; break
        V = Vn
    return V.numpy()


def fit_readout(X, y, wt, hidden, seed=0, dev="cuda"):
    torch.manual_seed(seed); X = X.float().to(dev); y = y.to(dev); wt = wt.float().to(dev); wt = wt / wt.sum()
    net = (torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 3)) if hidden else torch.nn.Linear(X.shape[1], 3)).to(dev)
    o = torch.optim.LBFGS(net.parameters(), max_iter=500, line_search_fn="strong_wolfe", history_size=20)
    def closure():
        o.zero_grad(); l = (torch.nn.functional.cross_entropy(net(X), y, reduction="none") * wt).sum() + 1e-5 * sum((p ** 2).sum() for p in net.parameters()); l.backward(); return l
    o.step(closure)
    with torch.no_grad(): return net(X).argmax(1).cpu()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--models", nargs="*", default=["llm_100000", "llm_10000", "tfm_100000"]); ap.add_argument("--n_ep", type=int, default=100000); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); S = load_sets(); n = len(S["keys"]); masks = S["masks"].numpy(); size = masks.sum(1); Q = S["Q"].numpy(); V = Q.max(1); opt = Q >= V[:, None] - 1e-9; margin = np.sort(Q, 1)[:, -1] - np.sort(Q, 1)[:, -2]
    cnt_opt = S["cnt_opt"].numpy(); cnt100 = S["cnt_train_100000"].numpy(); cnt10 = S["cnt_train_10000"].numpy(); start = int(np.argmax(masks.sum(1)))  # the full start set is the largest
    assert S["depth"][start] == 0
    hist = collect_histories(w, sol, S, a.n_ep); nh = np.array([len(hist[k]) for k in S["keys"]])
    print(f"histories per set: " + ", ".join(f"{k}: {int((nh == k).sum())}" for k in range(1, MAX_H + 1)) + f" {time.time() - t0:.0f}s", flush=True)
    res = dict(n_sets=n, histories_per_set={str(k): int((nh == k).sum()) for k in range(1, MAX_H + 1)}, models={})
    bins = size_bins(size); groups = {"optimum-visited": cnt_opt > 0, "training-visited (100k), not by the optimum": (cnt100 > 0) & (cnt_opt == 0), "never visited": cnt100 == 0}
    for name in a.models:
        kind = name.split("_")[0]; run = RUNS / name
        if kind == "llm": R = Runner(str(run / "model"), w, a.device)
        else: R = TfmRunner(str(run / "model.pt"), w, a.device)
        z0, _ = R.run([(S["acts"][i], S["obss"][i]) for i in range(n)]); act0 = z0.argmax(1).numpy()
        ok = opt[np.arange(n), act0]; regret = (V - Q[np.arange(n), act0]) / np.maximum(V, 1e-9)
        groups_m = dict(groups); 
        if name == "llm_10000": groups_m = {"optimum-visited": cnt_opt > 0, "training-visited (10k), not by the optimum": (cnt10 > 0) & (cnt_opt == 0), "never visited (10k)": cnt10 == 0}
        m = dict(table={g: dict(n=int(msk.sum()), optimal=float(ok[msk].mean()), optimal_ci=boot_mean(ok[msk]), regret=float(regret[msk].mean()), by_size={b: float(ok[msk & bm].mean()) if (msk & bm).any() else None for b, bm in bins.items()},
                          near_tie=float(ok[msk & (margin < 0.01)].mean()) if (msk & (margin < 0.01)).any() else None, clear=float(ok[msk & (margin >= 0.01)].mean()) if (msk & (margin >= 0.01)).any() else None) for g, msk in groups_m.items()},
                 all=dict(optimal=float(ok.mean()), regret=float(regret.mean())))
        print(f"{name}: optimal on all sets {ok.mean():.3f}; " + "; ".join(f"{g} {v['optimal']:.3f} (regret {v['regret']:.3f}, n {v['n']})" for g, v in m["table"].items()) + f" {time.time() - t0:.0f}s", flush=True)
        # transfer readouts on the never-visited sets: agreement of the model's action with the optimum and with the readouts
        seen = cnt100 > 0 if name != "llm_10000" else cnt10 > 0; unseen = ~seen; X = torch.as_tensor(masks).float(); y = torch.as_tensor(Q.argmax(1)); wt = torch.as_tensor((cnt100 if name != "llm_10000" else cnt10).astype(float))
        agree = {}
        for rname, hid in (("linear", 0), ("mlp256", 256)):
            pr = fit_readout(X, y, wt, hid).numpy(); agree[rname] = dict(readout_optimal_unseen=float(opt[np.arange(n), pr][unseen].mean()), model_agrees_unseen=float((pr == act0)[unseen].mean()), model_agrees_seen=float((pr == act0)[seen].mean()))
        m["transfer"] = dict(model_optimal_unseen=float(ok[unseen].mean()), readouts=agree); print(f"  never-visited: model optimal {ok[unseen].mean():.3f}; agrees with linear readout {agree['linear']['model_agrees_unseen']:.3f} (readout optimal {agree['linear']['readout_optimal_unseen']:.3f}), with MLP {agree['mlp256']['model_agrees_unseen']:.3f} ({agree['mlp256']['readout_optimal_unseen']:.3f})", flush=True)
        # A1: set-sufficiency on the sets with several histories
        multi = [i for i in range(n) if nh[i] >= 2]; extra = [(i, h) for i in multi for h in hist[S["keys"][i]][1:]]
        ze, _ = R.run([h for _, h in extra]); acte = ze.argmax(1).numpy(); zall = {i: [z0[i]] for i in multi}; aall = {i: [int(act0[i])] for i in multi}
        for (i, _), z, ac in zip(extra, ze, acte): zall[i].append(z); aall[i].append(int(ac))
        same = np.array([len(set(aall[i])) == 1 for i in multi]); multi_arr = np.array(multi)
        # the between-set share of logit variance over all (set, history) logits
        Zs = torch.stack([z for i in multi for z in zall[i]]); grp = torch.tensor([i for i in multi for _ in zall[i]]); mu = Zs.mean(0); tot = ((Zs - mu) ** 2).sum()
        means = torch.zeros(n, 3).index_add_(0, grp, Zs) / torch.bincount(grp, minlength=n).clamp(min=1)[:, None]; between = ((means[grp] - mu) ** 2).sum()
        rng = np.random.default_rng(0); base = []
        for _ in range(len(multi)):
            i = multi_arr[rng.integers(len(multi_arr))]; cand = np.nonzero(size == size[i])[0]; j = cand[rng.integers(len(cand))]
            if j != i: base.append(act0[i] == act0[j])
        suf = {g: dict(n=int((msk[multi_arr]).sum()), same=float(same[msk[multi_arr]].mean()) if msk[multi_arr].any() else None, same_ci=boot_mean(same[msk[multi_arr]])) for g, msk in groups_m.items()}
        suf["all"] = dict(n=len(multi), same=float(same.mean()), same_ci=boot_mean(same), baseline_same_size=float(np.mean(base)), between_set_variance_share=float(between / tot))
        m["sufficiency"] = suf; print(f"  set-sufficiency: same action on every history {same.mean():.3f} of {len(multi)} sets (baseline {np.mean(base):.3f}; between-set variance share {float(between / tot):.3f}); " + "; ".join(f"{g} {v['same']:.3f} ({v['n']})" for g, v in suf.items() if g != "all" and v["same"] is not None), flush=True)
        # A3: the exact value of the table (shortest-history action; majority over histories where several)
        maj = act0.copy()
        for i in multi: maj[i] = np.bincount(aall[i], minlength=3).argmax()
        for lab, acts_ in (("shortest", act0), ("majority", maj)):
            Vp = policy_value(w, sol, S, acts_); ratio = Vp / np.maximum(V, 1e-9)
            m[f"value_{lab}"] = dict(start=float(Vp[start]), start_optimal=float(V[start]), start_ratio=float(Vp[start] / V[start]), mean_ratio_all=float(ratio.mean()), mean_ratio_ci=boot_mean(ratio), mean_ratio_unseen=float(ratio[unseen].mean()), lost=float((ratio < 0.5).mean()), lost_unseen=float((ratio < 0.5)[unseen].mean()), by_size={b: float(ratio[bm].mean()) for b, bm in bins.items()})
            print(f"  value of the table ({lab}): start {Vp[start]:.4f} / {V[start]:.4f} = {Vp[start] / V[start]:.3f}; mean V^π/V* {ratio.mean():.3f} (never-visited {ratio[unseen].mean():.3f}); lost (< 0.5) {float((ratio < 0.5).mean()):.3f} {time.time() - t0:.0f}s", flush=True)
        torch.save(dict(logits=z0, actions=torch.as_tensor(act0), majority=torch.as_tensor(maj), multi=torch.as_tensor(multi_arr), extra=extra, extra_logits=ze), HERE / f"table_{name}.pt")
        res["models"][name] = m; (HERE / "table.json").write_text(json.dumps(res, indent=1))
        del R; torch.cuda.empty_cache()
    torch.save(hist, HERE / "histories.pt")


if __name__ == "__main__":
    main()
