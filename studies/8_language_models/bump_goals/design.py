"""Design probe of the bump maze with several goals: goal placement, the exact optimum per goal, references, and
the exact additive ceiling (the best rule argmax h[goal-free belief set] + g[goal, step] against the optimal action
under every goal). Writes design.md / design.json.

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_goals/design.py
"""

from __future__ import annotations

import json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
SEED, SIZE, LOOPS, K, N_EP = 0, 9, 4, 4, 1024


def worlds_and_solutions():
    free = BMZ.layout(SEED, SIZE, LOOPS); goals = BMZ.pick_goals(free, K)
    worlds = [BMZ.World(free, g) for g in goals]; sols = [BMZ.solve(w) for w in worlds]
    return free, goals, worlds, sols


def map_with_goals(w, goals):
    rows = w.ascii(mark_goal=False).split("\n")
    for i, (x, y) in enumerate(goals): rows[y] = rows[y][:x] + str(i) + rows[y][x + 1:]
    return "\n".join(rows)


def rollout(w, sol, policy, N, seed):
    gen = torch.Generator().manual_seed(seed); e = BMZ.Env(w, N, gen); ret = torch.zeros(N); L = torch.full((N,), w.H)
    for t in range(w.H):
        if policy == "optimal": a = torch.as_tensor(BMZ.optimal_action(w, sol, e.filter.mask.numpy())[0])
        elif policy == "qmdp": a = BMZ.qmdp_action(w, e.filter.posterior)
        else: a = torch.randint(3, (N,), generator=gen)
        was = e.done.clone(); r, _ = e.step(a); ret += r; L[~was & e.done] = t + 1
        if e.done.all(): break
    return dict(ret=float(ret.mean()), success=float((ret > 0).float().mean()), length=float(L.float().mean()))


def decisions_all_goals(worlds, sols, n_ep, seed):
    """Histories from the optimum under each goal; for every decision the goal-free set key, the step, and the
    optimal set under every goal whose filtered set is non-empty."""
    K = len(worlds); dec = []; key = lambda m: np.packbits(m).tobytes()
    for gi, (w, sol) in enumerate(zip(worlds, sols)):
        gen = torch.Generator().manual_seed(seed + gi); e = BMZ.Env(w, n_ep, gen); alive_hist = []
        for t in range(w.H):
            alive_hist.append(~e.done.clone()); a = torch.as_tensor(BMZ.optimal_action(w, sol, e.filter.mask.numpy())[0]); e.step(a)
            if e.done.all(): break
        T = len(e.acts); common = BMZ.Filter(w, n_ep, remove_goal=False); filters = [BMZ.Filter(w2, n_ep) for w2 in worlds]
        for t in range(T):
            cm = common.mask.numpy(); masks = [f.mask.numpy() for f in filters]
            for i in torch.nonzero(alive_hist[t]).squeeze(1).tolist():
                opts = {}
                for g2 in range(K):
                    m = masks[g2][i]
                    if m.any():
                        q = sols[g2]["Q"][sols[g2]["sets"][key(m)]]; opts[g2] = q >= q.max() - 1e-9
                dec.append(dict(key=key(cm[i]), step=t, goal=gi, opts=opts))
            common.update(e.acts[t], e.obss[t])
            for f in filters: f.update(e.acts[t], e.obss[t])
    return dec


def main():
    t0 = time.time(); free, goals, worlds, sols = worlds_and_solutions()
    res = dict(seed=SEED, size=SIZE, loops=LOOPS, K=K, goals=[list(g) for g in goals], free_cells=worlds[0].n_cells, states=worlds[0].n_states, map=map_with_goals(worlds[0], goals), per_goal=[])
    for gi, (w, sol) in enumerate(zip(worlds, sols)):
        row = dict(goal=list(goals[gi]), belief_sets=len(sol["masks"]), V_uniform=float(sol["V"][0]), oracle=float(np.mean(w.gamma ** (w.dist[w.start_states] - 1))), oracle_length=float(np.mean(w.dist[w.start_states])))
        for pol in ("optimal", "qmdp", "random"): row[pol] = rollout(w, sol, pol, 2048, 0)
        res["per_goal"].append(row); print(f"goal {gi} {goals[gi]}: sets {row['belief_sets']}, optimum {row['optimal']}, oracle {row['oracle']:.3f} {time.time() - t0:.0f}s", flush=True)
    dec = decisions_all_goals(worlds, sols, N_EP, 100)
    flat = [(d["key"], g2, d["step"], o) for d in dec for g2, o in d["opts"].items()]
    full = [d for d in dec if len(d["opts"]) == K]
    goal_dep = float(np.mean([not all((o == d["opts"][0]).all() for o in d["opts"].values()) for d in full]))
    disjoint = float(np.mean([any(not (d["opts"][a] & d["opts"][b]).any() for a in range(K) for b in range(a + 1, K)) for d in full]))
    print(f"{len(dec)} decisions, {len(flat)} (decision, goal) pairs, {len(full)} with every goal defined; optimal action depends on the goal in {goal_dep:.3f}, some pair of goals disjoint in {disjoint:.3f} {time.time() - t0:.0f}s", flush=True)
    ceil = BMZ.additive_ceiling(flat, K, worlds[0].H, device="cuda" if torch.cuda.is_available() else "cpu")
    print("additive ceiling", ceil, f"{time.time() - t0:.0f}s", flush=True)
    res.update(decisions=len(dec), pairs=len(flat), goal_dependent=goal_dep, disjoint_pair=disjoint, ceiling=ceil)
    (HERE / "design.json").write_text(json.dumps(res))
    L = ["# Bump maze with several goals: design probe", "", f"Generated by `design.py`. The bump maze (seed {SEED}, {SIZE} × {SIZE}, {LOOPS} loops, {worlds[0].n_cells} free cells, {worlds[0].n_states} states) with {K} goals placed by farthest-point sampling in move distance (digits on the map); one goal per episode, the start uniform over the other free cells and the four headings.", "", "```", res["map"], "```", "",
         "| goal | belief sets | exact optimum: return / success / length | oracle: return / length | QMDP: return / success | random: return / success |", "|---|---|---|---|---|---|"]
    for gi, r in enumerate(res["per_goal"]):
        L.append(f"| {gi} at (x {r['goal'][0]}, y {r['goal'][1]}) | {r['belief_sets']} | {r['optimal']['ret']:.3f} / {r['optimal']['success']:.3f} / {r['optimal']['length']:.1f} | {r['oracle']:.3f} / {r['oracle_length']:.1f} | {r['qmdp']['ret']:.3f} / {r['qmdp']['success']:.3f} | {r['random']['ret']:.3f} / {r['random']['success']:.3f} |")
    L += ["", f"**The additive ceiling.** {len(dec)} decisions from the optimum's episodes under every goal ({N_EP} per goal), each scored under every goal whose filtered belief is non-empty ({len(flat)} decision-goal pairs, {ceil['sets']} distinct goal-free belief sets). "
          f"The optimal action depends on the goal in {goal_dep:.3f} of decisions (some pair of goals with disjoint optimal sets in {disjoint:.3f}). The best additive rule argmax h[set] + g[goal, step] is optimal in **{ceil['additive']:.3f}** of the pairs; "
          f"the best goal-blind rule h[set] in {ceil['goal_blind']:.3f}; the best history-blind rule g[goal, step] in {ceil['history_blind']:.3f}. (Fit by gradient on the fitted decisions themselves, so an upper bound for this sample; the maze10 maze's exact ceiling was 0.963.)"]
    (HERE / "design.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
