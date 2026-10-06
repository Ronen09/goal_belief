"""The active-maze experiment, solver-only screen (no network). Two changes to the maze-belief task, crossed:
    prefix   passive prefix of 0-4 random moves before the goal is shown (current) / none: the goal is shown with the
             first symbol and every move is the agent's own
    spawns   four corridor cells (current). Every non-goal cell was planned too; its exact graph does not fit
             (see the random-spawns experiment), which is why the maze10 experiment drops the solver
For each variant, on the exact belief graph:
  * how much information seeking pays: loss of QMDP (ignores the value of information) and of acting on the most likely
    cell, against the Bayes-optimal value at the reveal; the share of decisions on the optimal path where QMDP's move
    is not optimal;
  * how additive the task is: the hard-cases experiment's fit of H(history) + G(goal) to the solver's own Q*.

    .venv/bin/python studies/7_information_seeking/active_maze/screen.py            # writes screen.json, screen.md
"""

from __future__ import annotations

import importlib.util, json, time
from pathlib import Path

import numpy as np

from goalgeo import mazegraph as MG

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hcscreen", HERE.parent.parent / "6_hard_cases_and_tasks" / "hard_cases" / "screen.py")
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
GOALS = [(1, 5), (2, 6), (1, 10)]
SPAWNS = {"4 corridor cells": SC.X_START}                                 # all non-goal cells: the exact graph does not fit (random-spawns experiment)


def policy_actions(g, kind):
    """Action per decision node for a belief-based baseline policy (0 for prefix nodes, unused)."""
    m = g.maze
    act = np.zeros(len(g.k), dtype=int)
    for gi, goal in enumerate(m.goals):
        Qm = m.mdp_values(goal)                                                             # [H + 1, n, 4]
        for k in range(1, m.H + 1):
            sel = np.nonzero((g.goal == gi) & (g.k == k))[0]
            b = g.belief[sel]
            act[sel] = (b @ Qm[k] if kind == "qmdp" else Qm[k][b.argmax(1)]).argmax(1)
    return act


def evaluate(g, act):
    V = np.zeros(len(g.k) + 1)
    for k in range(1, g.maze.H + 1):
        sel = np.nonzero(g.k == k)[0]
        a = act[sel]
        V[sel] = g.r[sel, a] + g.maze.gamma * (g.p_obs[sel, a] * V[g.nxt[sel, a]]).sum(-1)
    return V


def reveal_weights(g, max_prefix):
    """Probability of each prefix node at the reveal: prefix length uniform on 0..max_prefix, uniform random moves."""
    m = g.maze
    w, lev = np.zeros(g.n_prefix), np.zeros(g.n_prefix)
    p0 = m.prior() @ m.E
    for o in range(m.n_sym):
        if g.start[o] >= 0:
            lev[g.start[o]] += p0[o]
    for L in range(max_prefix + 1):
        w += lev / (max_prefix + 1)
        if L == max_prefix:
            break
        ids = np.nonzero(lev)[0]
        nxt_lev = np.zeros(g.n_prefix)
        for a in range(4):
            for o in range(m.n_sym):
                tgt = g.nxt[ids, a, o]
                ok = tgt >= 0
                np.add.at(nxt_lev, tgt[ok], lev[ids][ok] * 0.25 * g.p_obs[ids, a, o][ok])
        lev = nxt_lev
    return w


def info_value(g, max_prefix):
    m = g.maze
    K = len(m.goals)
    aq = policy_actions(g, "qmdp")
    Vq, Vm = evaluate(g, aq), evaluate(g, policy_actions(g, "mls"))
    w = reveal_weights(g, max_prefix)
    pre = np.nonzero(w)[0]
    tot = {n: sum((w[pre] * V[g.reveal[pre, gi]]).sum() for gi in range(K)) / K for n, V in (("opt", g.V), ("qmdp", Vq), ("mls", Vm))}
    # the optimal policy's own visitation (first optimal action)
    opt = g.optimal()
    vis = np.zeros(len(g.k))
    for gi in range(K):
        np.add.at(vis, g.reveal[pre, gi], w[pre] / K)
    wrong = mass = 0.0
    for k in range(m.H, 0, -1):
        sel = np.nonzero((g.k == k) & (vis > 0))[0]
        if not len(sel):
            continue
        a, p = g.Q[sel].argmax(1), vis[sel]
        mass += p.sum()
        wrong += (p * ~opt[sel, aq[sel]]).sum()
        for o in range(m.n_sym):
            tgt = g.nxt[sel, a, o]
            ok = tgt >= 0
            np.add.at(vis, tgt[ok], p[ok] * g.p_obs[sel, a, o][ok])
    return dict(v_star=float(tot["opt"]), loss_qmdp=float(1 - tot["qmdp"] / tot["opt"]), loss_mls=float(1 - tot["mls"] / tot["opt"]),
                qmdp_wrong_on_optimal_path=float(wrong / mass))


def main():
    rows, t0 = {}, time.time()
    build = MG.build
    for sname, starts in SPAWNS.items():
        for pname, mp in (("prefix 0-4", 4), ("no prefix", 0)):
            maze = SC.make(SC.X, GOALS, starts)
            g = build(maze, max_prefix=mp)
            r = info_value(g, mp)
            SC.MG.build = lambda maze_, max_prefix, g=g: g                                  # score() builds its graph: reuse this one
            r.update(SC.score(maze, max_prefix=mp))
            SC.MG.build = build
            key = f"{pname}, spawn: {sname}"
            rows[key] = r
            print(f"{key}: nodes {r['nodes']} V* {r['v_star']:.3f} | loss qmdp {r['loss_qmdp']:.3f} mls {r['loss_mls']:.3f} qmdp wrong on optimal path {r['qmdp_wrong_on_optimal_path']:.3f} | "
                  f"goal-dependent {r['dep_share']:.3f} additive share {r['additive_share']:.3f} additive move fails {r['add_fail_dep']:.3f} unsolvable {r['unsolvable_dep']:.3f} "
                  f"loss/decision {r['add_loss_per_decision']:.4f} | {time.time() - t0:.0f}s", flush=True)
            (HERE / "screen.json").write_text(json.dumps(rows, indent=1))
    L = ["# Active maze: solver-only screen", "", "Generated by `screen.py`. The maze-belief maze and goals, 12 moves, noise 0.4. Values at the reveal, averaged over goals "
         "(and over prefix lengths 0–4 with uniform random prefix moves). Loss = (V* − V_π) / V*. Additive columns: the hard-cases experiment's fit of H + G to the "
         "solver's own Q*, on histories from the solver with 20 % random moves.", "",
         "| task | graph nodes | V* | loss: QMDP | loss: most likely cell | QMDP wrong on the optimal path | goal-dependent decisions | additive share of Q*'s goal dependence | additive move fails (goal-dependent) | additively unsolvable | additive loss per decision |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        L.append(f"| {k} | {r['nodes']} | {r['v_star']:.3f} | {r['loss_qmdp']:.3f} | {r['loss_mls']:.3f} | {r['qmdp_wrong_on_optimal_path']:.3f} | {r['dep_share']:.3f} | "
                 f"{r['additive_share']:.3f} | {r['add_fail_dep']:.3f} | {r['unsolvable_dep']:.3f} | {r['add_loss_per_decision']:.4f} |")
    (HERE / "screen.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
