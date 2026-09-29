"""Round 18: the four checks made before freezing the task, all from the exact filter and solver.

1. Effective dimension of the reachable location posterior, for 2, 4 and 6 start cells.
2. Matched evidence: a passive prefix (actions imposed, no goal shown, no termination), then the goal is revealed.
   The posterior at the reveal is the goal-free one, identical for every goal.
3. For each goal, at the reveal: does the belief change the optimal action (action relevance) and does it change
   the value and the expected arrival time (predictive relevance)?
4. Three baselines with separate definitions: most likely cell taken as certain; belief-weighted fully observed
   values; planning without future observations.
"""

from __future__ import annotations

import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from goalgeo import mazebelief as MB

HERE = Path(__file__).resolve().parent
PREFIX = (0, 1, 2, 3, 4)
CAP = 3_000_000


def dimension(args):
    starts, H = args
    m = MB.cross_maze(starts=starts, H=H)
    out = dict(starts=starts)
    for L in (0, 2, 4):
        P = MB.prefix_beliefs(m, L)
        B, w = np.array([p[0] for p in P]), np.array([p[1] for p in P])
        arm = B[:, [i for i, (r, c) in enumerate(m.cells) if c <= 4]].sum(1)      # mass on the left arm
        off = B - np.where(np.arange(m.n)[None] < 0, 0, 0)
        out[f"prefix{L}"] = dict(beliefs=len(P), supports=len({tuple((b > 1e-9).tolist()) for b in B}),
                                 **MB.effective_dimension(B, w),
                                 arm_only_r2=float(1 - _resid(B, w, arm)))
    return out


def _resid(B, w, arm):
    """Share of the weighted variance of the beliefs left after regressing every coordinate on P(left arm)."""
    w = w / w.sum()
    X = B - w @ B
    a = arm - w @ arm
    beta = (w * a) @ X / max((w * a) @ a, 1e-15)
    R = X - a[:, None] * beta[None]
    return (w[:, None] * R ** 2).sum() / max((w[:, None] * X ** 2).sum(), 1e-15)


def arrival(m, goal, S, b, k):
    """Expected number of moves to the goal under the optimal policy, given that it is reached, and P(reached)."""
    memo = {}

    def rec(b, k):
        if k == 0:
            return 0.0, 0.0
        kk = MB.key(b, k)
        if kk in memo:
            return memo[kk]
        a = int(S.q(b, k).argmax())
        r, q, z = m.move(b, a, goal)
        p, t = r, r * 1.0
        if z > 1e-12 and k > 1:
            for o in range(m.n_sym):
                b2, zo = m.observe(q, o)
                if zo > 1e-12:
                    p2, t2 = rec(b2, k - 1)
                    p += z * zo * p2; t += z * zo * (t2 + p2)
        memo[kk] = (p, t)
        return p, t
    p, t = rec(b, k)
    return p, (t / p if p > 0 else np.nan)


def per_goal(args):
    gi, H, quick = args
    m = MB.cross_maze(H=H)
    g = m.goals[gi]
    S = MB.Solver(m, g, cap=CAP)
    pol = dict(map=MB.map_policy(m, g), qmdp=MB.qmdp_policy(m, g), open_loop=MB.open_loop_policy(m, g))
    Sp = {k: MB.Solver(m, g, p, cap=CAP) for k, p in pol.items()}
    rows = []
    for L in (PREFIX[:2] if quick else PREFIX):
        P = MB.prefix_beliefs(m, L)
        w = np.array([p[1] for p in P]); w = w / w.sum()
        V = np.array([S.v(p[0], H) for p in P])
        A = np.array([S.q(p[0], H) >= S.q(p[0], H).max() - MB.TIE for p in P])
        arr = np.array([arrival(m, g, S, p[0], H) for p in P])
        base = {k: float(w @ np.array([s.v(p[0], H) for p in P])) for k, s in Sp.items()}
        first = A.argmax(1)
        counts = np.array([w[first == a].sum() for a in range(4)])
        rows.append(dict(prefix=L, beliefs=len(P), v_opt=float(w @ V), v_sd=float(np.sqrt(w @ (V - w @ V) ** 2)),
                         v_min=float(V.min()), v_max=float(V.max()),
                         arrival_mean=float(np.nansum(w * arr[:, 1])), arrival_sd=float(np.sqrt(np.nansum(w * (arr[:, 1] - np.nansum(w * arr[:, 1])) ** 2))),
                         p_reach=float(w @ arr[:, 0]),
                         action_share={MB.ACTION_NAMES[a]: float(counts[a]) for a in range(4)},
                         action_entropy=float(-(counts[counts > 0] * np.log2(counts[counts > 0])).sum()),
                         common_action=bool(A.all(0).any()),
                         regret={k: float(w @ V) - v for k, v in base.items()}))
    return dict(goal=gi + 1, cell=int(g), rows=rows, solver_beliefs=len(S.memo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    H = 6 if a.quick else 12
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        dims = list(ex.map(dimension, [(s, H) for s in (2, 4, 6)]))
        goals = list(ex.map(per_goal, [(g, H, a.quick) for g in range(3)]))
    res = dict(H=H, dimension=dims, goals=goals)
    (out / "checks.json").write_text(json.dumps(res, indent=1))
    m = MB.cross_maze(H=H)
    L = ["# Round 18: checks before freezing the task", "", "Generated by `rounds/r18_maze_belief/checks.py` from the exact filter and solver.", "",
         "```", MB.draw(m), "```", "", f"Cells are numbered; letter = symbol (L: landmark); * = goal cell. Start cells: {m.starts}. "
         f"eps = {m.eps}, gamma = {m.gamma}, H = {m.H} moves after the goal is revealed.", "",
         "## 1. Effective dimension of the reachable posterior (goal-free filter, random prefix actions)", "",
         "| start cells | prefix | distinct posteriors | distinct supports | participation ratio | components for 95 % | variance explained by P(left arm) alone | leading spectrum |",
         "|---|---|---|---|---|---|---|---|"]
    for d in dims:
        for k in ("prefix0", "prefix2", "prefix4"):
            x = d[k]
            L.append(f"| {d['starts']} | {k[6:]} | {x['beliefs']} | {x['supports']} | {x['participation']:.2f} | {x['n95']} | {x['arm_only_r2']:.3f} | {x['spectrum']} |")
    L += ["", "## 2–4. At the reveal, by goal and prefix length (4 start cells)", "",
          "Beliefs are weighted by their probability under random prefix actions. Regret: optimal value minus the baseline's exact value.", "",
          "| goal | prefix | posteriors | V* mean | V* sd over beliefs | V* range | arrival time mean | arrival sd over beliefs | first action shares | one action optimal for every belief | most likely cell | fully observed average | no future observations |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in goals:
        for r in g["rows"]:
            sh = " ".join(f"{k}{v:.2f}" for k, v in r["action_share"].items() if v > 0.005)
            L.append(f"| G{g['goal']} | {r['prefix']} | {r['beliefs']} | {r['v_opt']:.3f} | {r['v_sd']:.3f} | {r['v_min']:.2f}–{r['v_max']:.2f} | {r['arrival_mean']:.2f} | "
                     f"{r['arrival_sd']:.2f} | {sh} | {'yes' if r['common_action'] else 'no'} | {r['regret']['map']:.3f} | {r['regret']['qmdp']:.3f} | {r['regret']['open_loop']:.3f} |")
    (out / "checks.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
