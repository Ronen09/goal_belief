"""Design probe for the policy study (no network): the space of reachable belief sets of the single-goal bump maze
with a shortest history to each; coverage of that space by the optimum's episodes and by the training data; Q-margins
and ties; the ceiling for a readout linear in set membership (and an MLP); the pool of nearby reachable sets with
different optimal actions, for the edit test. Writes design.md / design.json / sets.pt (one history per set).

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_policy/design.py
"""

from __future__ import annotations

import json, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze"))
from zero_shot import SEED, SIZE, LOOPS, optimum_decisions          # noqa: E402

key = lambda m: np.packbits(m).tobytes()


def enumerate_sets(w, sol):
    """BFS over reachable sets from the start set: for each set its shortest history (acts, obss) and depth."""
    full = np.zeros(w.n_states, dtype=bool); full[w.start_states] = True
    hist = {key(full): ([], [])}; q = deque([full]); depth = {key(full): 0}
    while q:
        m = q.popleft(); k = key(m); acts, obss = hist[k]
        for a in range(3):
            for o in ((0,) if a < 2 else (0, 1)):
                if a == 2:
                    cons = m & (w.bump == bool(o))
                    if o == 0: cons = cons & ~w.at_goal[w.nxt[:, 2]]
                    if not cons.any(): continue
                nm = BMZ.step_belief(w, m, a, o); nk = key(nm)
                if nk not in hist and nm.any():
                    hist[nk] = (acts + [a], obss + [o]); depth[nk] = depth[k] + 1; q.append(nm)
    return hist, depth


def fit_readout(X, y, opt, w_, hidden=0, steps=500, seed=0, dev="cuda"):
    """Multinomial readout from X [n, d] to the label y [n] (weights w_), scored as in-optimal-set accuracy (weighted)."""
    torch.manual_seed(seed); X = X.float().to(dev); y = y.to(dev); w_ = w_.float().to(dev); w_ = w_ / w_.sum()
    net = (torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 3)) if hidden else torch.nn.Linear(X.shape[1], 3)).to(dev)
    o = torch.optim.LBFGS(net.parameters(), max_iter=steps, line_search_fn="strong_wolfe", history_size=20)
    def closure():
        o.zero_grad(); l = (torch.nn.functional.cross_entropy(net(X), y, reduction="none") * w_).sum() + 1e-5 * sum((p ** 2).sum() for p in net.parameters()); l.backward(); return l
    o.step(closure)
    with torch.no_grad(): pred = net(X).argmax(1).cpu()
    return float((opt[torch.arange(len(pred)), pred].double() * w_.cpu().double()).sum())


def main():
    t0 = time.time(); w = BMZ.make(SEED, SIZE, LOOPS); sol = BMZ.solve(w)
    hist, depth = enumerate_sets(w, sol); keys = list(hist); idx = {k: i for i, k in enumerate(keys)}
    masks = np.stack([np.unpackbits(np.frombuffer(k, dtype=np.uint8))[: w.n_states].astype(bool) for k in keys]); size = masks.sum(1); dep = np.array([depth[k] for k in keys])
    assert set(keys) == set(sol["sets"]), (len(keys), len(sol["sets"]))
    Q = np.stack([sol["Q"][sol["sets"][k]] for k in keys]); V = Q.max(1); opt = Q >= V[:, None] - 1e-9; srt = np.sort(Q, 1); margin = srt[:, -1] - srt[:, -2]
    n = len(keys); print(f"{n} reachable sets (solver {len(sol['sets'])}), depth ≤ 39: {int((dep <= 39).sum())}, max depth {dep.max()} {time.time() - t0:.0f}s", flush=True)
    bins = (("1", 1, 1), ("2-4", 2, 4), ("5-16", 5, 16), (">16", 17, 10 ** 9))
    by_size = {b: int(((size >= lo) & (size <= hi)).sum()) for b, lo, hi in bins}
    ties = opt.sum(1); tie_share = float((ties > 1).mean()); near = float((margin < 0.01).mean())
    # coverage: the optimum's episodes and the training data
    dec = optimum_decisions(w, sol, 4096, 4000); opt_keys = [key(d["mask"]) for d in dec]; opt_set = set(opt_keys)
    cnt_opt = np.zeros(n, dtype=np.int64)
    for k in opt_keys: cnt_opt[idx[k]] += 1
    data = torch.load(HERE.parent / "bump_maze" / "data" / "train_100000.pt"); A, O, Ln = data["acts"], data["obss"], data["length"]
    cnt_train = {10000: np.zeros(n, dtype=np.int64), 100000: np.zeros(n, dtype=np.int64)}; unknown = 0
    for lo, hi, nm in ((0, 10000, 10000), (0, 100000, 100000)):
        if nm == 100000: base = cnt_train[10000].copy()
        rng_ = range(10000, 100000) if nm == 100000 else range(0, 10000); N = len(rng_); f = BMZ.Filter(w, N); a_ = A[list(rng_)]; o_ = O[list(rng_)]; l_ = Ln[list(rng_)]
        for t in range(w.H):
            alive = (l_ > t).numpy(); mk = f.mask.numpy()
            pk = np.packbits(mk[alive], axis=1)
            for row in pk:
                k = row.tobytes(); j = idx.get(k)
                if j is None: unknown += 1
                else: cnt_train[nm][j] += 1
            f.update(a_[:, t], o_[:, t])
        if nm == 100000: cnt_train[nm] += base
    cov = {"optimum_4096_episodes": int((cnt_opt > 0).sum()), "train_10000": int((cnt_train[10000] > 0).sum()), "train_100000": int((cnt_train[100000] > 0).sum())}
    cov_dec = {"share of optimum decisions on sets never in train_10000": float(cnt_opt[cnt_train[10000] == 0].sum() / cnt_opt.sum()), "share on sets never in train_100000": float(cnt_opt[cnt_train[100000] == 0].sum() / cnt_opt.sum())}
    print(f"coverage of {n} sets: optimum {cov['optimum_4096_episodes']}, train 10k {cov['train_10000']}, train 100k {cov['train_100000']} (masks not among reachable sets: {unknown}) {time.time() - t0:.0f}s", flush=True)
    # same set at different steps along the optimum
    steps_by_set = {}
    for d, k in zip(dec, opt_keys): steps_by_set.setdefault(k, set()).add(d["step"])
    multi_step = float(np.mean([len(v) > 1 for v in steps_by_set.values()]))
    # the linear ceiling: the optimal action from membership, uniform over sets / weighted by training frequency / by the optimum's
    X = torch.as_tensor(masks).float(); y = torch.as_tensor(Q.argmax(1)); optT = torch.as_tensor(opt); ones = torch.ones(n)
    ceil = {}
    for wname, wt in (("uniform over sets", ones), ("train_100000 frequency", torch.as_tensor(cnt_train[100000]).float()), ("optimum frequency", torch.as_tensor(cnt_opt).float())):
        ceil[wname] = dict(linear=fit_readout(X, y, optT, wt), mlp256=fit_readout(X, y, optT, wt, hidden=256), constant=float(max((optT[:, a].double() * (wt / wt.sum()).double()).sum() for a in range(3))))
        print(f"readout ceiling, {wname}: linear {ceil[wname]['linear']:.3f} mlp {ceil[wname]['mlp256']:.3f} constant {ceil[wname]['constant']:.3f} {time.time() - t0:.0f}s", flush=True)
    # transfer: readouts fitted on the training-visited sets (train frequency), scored on the never-visited sets (uniform)
    seen = torch.as_tensor(cnt_train[100000] > 0); unseen = ~seen
    def transfer(hidden):
        torch.manual_seed(0); dev = "cuda"; Xd = X.to(dev); yd = y.to(dev); wt = torch.as_tensor(cnt_train[100000]).float().to(dev); wt = wt / wt.sum()
        net = (torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 3)) if hidden else torch.nn.Linear(X.shape[1], 3)).to(dev)
        o = torch.optim.LBFGS(net.parameters(), max_iter=500, line_search_fn="strong_wolfe", history_size=20)
        def closure():
            o.zero_grad(); l = (torch.nn.functional.cross_entropy(net(Xd), yd, reduction="none") * wt).sum() + 1e-5 * sum((p ** 2).sum() for p in net.parameters()); l.backward(); return l
        o.step(closure)
        with torch.no_grad(): pred = net(Xd).argmax(1).cpu()
        ok = optT[torch.arange(n), pred]; return dict(seen=float(ok[seen].double().mean()), unseen=float(ok[unseen].double().mean()), unseen_by_size={b: float(ok[unseen & torch.as_tensor((size >= lo) & (size <= hi))].double().mean()) for b, lo, hi in bins})
    ceil["transfer"] = dict(linear=transfer(0), mlp256=transfer(256), n_seen=int(seen.sum()), n_unseen=int(unseen.sum()))
    print(f"transfer to never-visited sets (uniform): linear {ceil['transfer']['linear']['unseen']:.3f} (seen {ceil['transfer']['linear']['seen']:.3f}), mlp {ceil['transfer']['mlp256']['unseen']:.3f} (seen {ceil['transfer']['mlp256']['seen']:.3f}) {time.time() - t0:.0f}s", flush=True)
    # nearest reachable set with a disjoint optimal set (Hamming over states), per set
    P = torch.as_tensor(np.packbits(masks, axis=1)).cuda(); optc = optT.cuda(); nearest = torch.full((n,), 10 ** 6, dtype=torch.long); near_idx = torch.full((n,), -1, dtype=torch.long)
    pop = torch.tensor([bin(i).count("1") for i in range(256)], dtype=torch.int16).cuda()
    for b in range(0, n, 512):
        x = P[b:b + 512]; ham = pop[(x[:, None, :] ^ P[None]).long()].sum(-1)                      # [b, n]
        disjoint = ~(optc[b:b + 512][:, None, :] & optc[None]).any(-1); ham = torch.where(disjoint, ham, torch.full_like(ham, 10 ** 6))
        v, j = ham.min(1); nearest[b:b + 512] = v.cpu(); near_idx[b:b + 512] = j.cpu()
    nearest = nearest.numpy(); near_hist = {int(k): int((nearest == k).sum()) for k in range(1, 9)}; near_hist["> 8"] = int(((nearest > 8) & (nearest < 10 ** 6)).sum())
    print(f"nearest reachable set with a disjoint optimal set: Hamming 1: {near_hist[1]}, 2: {near_hist[2]}, ≤ 4: {int(((nearest <= 4)).sum())}, ≤ 8: {int((nearest <= 8).sum())} of {n} {time.time() - t0:.0f}s", flush=True)
    # histories per set: the shortest; save
    torch.save(dict(keys=keys, masks=torch.as_tensor(masks), acts=[hist[k][0] for k in keys], obss=[hist[k][1] for k in keys], depth=torch.as_tensor(dep), Q=torch.as_tensor(Q), cnt_opt=torch.as_tensor(cnt_opt), cnt_train_10000=torch.as_tensor(cnt_train[10000]), cnt_train_100000=torch.as_tensor(cnt_train[100000]), nearest=torch.as_tensor(nearest), near_idx=near_idx), HERE / "sets.pt")
    res = dict(n_sets=n, within_horizon=int((dep <= 39).sum()), max_depth=int(dep.max()), by_size=by_size, tie_share=tie_share, near_tie_share_0_01=near, mean_margin=float(margin.mean()), coverage=cov, coverage_decisions=cov_dec, same_set_several_steps=multi_step,
               n_optimum_decisions=len(dec), n_optimum_sets_several_steps=len(steps_by_set), readout_ceiling=ceil, nearest_disjoint=near_hist,
               by_size_coverage={b: dict(sets=int(((size >= lo) & (size <= hi)).sum()), train_100000=int(((size >= lo) & (size <= hi) & (cnt_train[100000] > 0)).sum()), optimum=int(((size >= lo) & (size <= hi) & (cnt_opt > 0)).sum()), ties=float((ties[(size >= lo) & (size <= hi)] > 1).mean())) for b, lo, hi in bins})
    (HERE / "design.json").write_text(json.dumps(res, indent=1))
    L = ["# Bump policy: design probe", "", f"Generated by `design.py` (no network). The single-goal bump maze (seed {SEED}, {SIZE} × {SIZE}, {LOOPS} loops; `../bump_maze`). Every reachable belief set by breadth-first search from the start set, with a shortest history to each (`sets.pt`).", "",
         f"* **{n} reachable sets** (the solver's count), {res['within_horizon']} within the 40-move horizon (max depth {res['max_depth']}); by size 1 / 2–4 / 5–16 / > 16: " + " / ".join(str(by_size[b]) for b, _, _ in bins) + ".",
         f"* **Ties**: {tie_share:.3f} of sets have more than one optimal action (by size: " + " / ".join(f"{res['by_size_coverage'][b]['ties']:.2f}" for b, _, _ in bins) + f"); margin between the best and second-best action below 0.01: {near:.3f}; mean margin {margin.mean():.3f}.",
         f"* **Coverage**: the optimum's 4 096 episodes ({len(dec)} decisions) visit {cov['optimum_4096_episodes']} sets; the training data (ε = 0.2 random actions) visits {cov['train_10000']} sets at 10 000 episodes and {cov['train_100000']} at 100 000. By size (sets / in train 100k / visited by the optimum): " + "; ".join(f"{b}: {res['by_size_coverage'][b]['sets']} / {res['by_size_coverage'][b]['train_100000']} / {res['by_size_coverage'][b]['optimum']}" for b, _, _ in bins) + f". Of the optimum's decisions, {cov_dec['share of optimum decisions on sets never in train_10000']:.3f} are on sets the 10k data never visits, {cov_dec['share on sets never in train_100000']:.3f} the 100k data.",
         f"* **The same set at several steps**: along the optimum, {multi_step:.2f} of visited sets occur at more than one step (the optimum is stationary over sets; a readout that uses the step is not a function of the set).",
         "* **Readout ceilings** (the optimal action from the 140-dimensional membership vector, in-optimal-set accuracy):", "", "| weighting | linear | MLP (256) | constant |", "|---|---|---|---|"]
    for wname, c in ceil.items():
        if wname != "transfer": L.append(f"| {wname} | {c['linear']:.3f} | {c['mlp256']:.3f} | {c['constant']:.3f} |")
    tr_ = ceil["transfer"]; L += ["", f"* **Transfer**: readouts fitted on the {tr_['n_seen']} sets the 100k training data visits (weighted by frequency), scored uniformly on the {tr_['n_unseen']} never-visited sets: linear {tr_['linear']['unseen']:.3f} (on the seen sets {tr_['linear']['seen']:.3f}; by size on unseen " + " / ".join(f"{tr_['linear']['unseen_by_size'][b]:.2f}" for b, _, _ in bins) + f"), MLP {tr_['mlp256']['unseen']:.3f} (seen {tr_['mlp256']['seen']:.3f}; by size " + " / ".join(f"{tr_['mlp256']['unseen_by_size'][b]:.2f}" for b, _, _ in bins) + ")."]
    L += ["", f"* **Edit pairs**: for each set, the nearest reachable set (Hamming distance over the 140 states) whose optimal actions are disjoint from its own: at distance 1: {near_hist[1]}, 2: {near_hist[2]}, 3: {near_hist[3]}, 4: {near_hist[4]}, 5–8: {sum(near_hist[k] for k in range(5, 9))}, > 8: {near_hist['> 8']}."]
    (HERE / "design.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
