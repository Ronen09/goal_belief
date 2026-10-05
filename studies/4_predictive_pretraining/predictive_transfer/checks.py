"""The predictive-transfer experiment: solver-only checks made before any model of this experiment was trained.

    .venv/bin/python studies/4_predictive_pretraining/predictive_transfer/checks.py            # writes checks.json, checks.md

1. Predictive equivalence. Two goal-free posteriors are k-step equivalent if they give the same distribution of the
   next k symbols for every sequence of k moves. A k-step prediction objective requires no more than the class at a
   token. Per goal: the share of held-out decisions at the reveal whose class contains posteriors with another
   optimal action set, and the regret of a policy that knows only the class (it takes the action best on average
   over the class, weighted by how often each posterior occurs).
2. Ceilings for the representation measures: affine decoding of the posterior from the exact k-step predictions.
3. Scale of regret at the reveal for reference policies.
All on the pair-types experiment's bank: fit side for anything fitted, held-out side for every number.
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazegraph as MG, mazemeasure as MS, mazepred as PR
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
R18 = HERE.parent.parent / "3_reward_trained_agents" / "maze_belief"
sys.path.insert(0, str(R18))
import train as T18                                                  # noqa: E402

BANK_SEED, N_BANK = 220022, 200000                                   # the pair-types experiment's bank


def main():
    t = T18.tables("cuda")
    g = MG.cached(MB.cross_maze(), R18 / "cache" / "graph.npz")
    maze = g.maze
    bank, fit = MS.make_bank(t, BANK_SEED, N_BANK)
    node = bank.pre_node[torch.arange(len(fit), device=t.dev), bank.prefix].cpu().numpy()
    L, test = bank.prefix.cpu().numpy(), ~fit.cpu().numpy()
    npre = g.n_prefix
    freq = np.bincount(node[~test], minlength=npre).astype(float)    # how often each posterior occurs (fit side)
    Qr = np.stack([g.Q[g.reveal[:, k]] for k in range(3)], 1)        # [prefix nodes, goal, 4]
    V = Qr.max(-1)
    opt = Qr >= V[..., None] - 1e-6
    bel_t = torch.tensor(g.belief[:npre], dtype=torch.float64, device=t.dev)
    nd, Lt = node[test], L[test]
    out = dict(prefix_nodes=int(npre), visited_nodes=int((np.bincount(node, minlength=npre) > 0).sum()),
               test_histories=int(test.sum()), fit_histories=int((~test).sum()), by_length_test={int(l): int((Lt == l).sum()) for l in range(5)},
               k_step={})
    for k in (1, 2):
        Pk = PR.predictive(t, bel_t, k).flatten(1).cpu().numpy()
        Pk = np.round(Pk, 9)
        _, cls = np.unique(np.round(Pk, 6), axis=0, return_inverse=True); cls = cls.reshape(-1)
        row = dict(classes=int(len(np.unique(cls[nd]))), dims=int(Pk.shape[1]), rank=int(np.linalg.matrix_rank(Pk - Pk.mean(0), tol=1e-6 * np.linalg.norm(Pk - Pk.mean(0), 2))), goals={})
        for j in range(3):
            num = np.zeros((cls.max() + 1, 4)); np.add.at(num, cls, freq[:, None] * Qr[:, j])
            a_cls = num.argmax(1)[cls]
            reg = V[:, j] - Qr[np.arange(npre), j, a_cls]
            det = np.ones(npre, bool)
            for c in np.unique(cls):
                m = cls == c
                det[m] = (opt[m, j] == opt[m, j][0]).all()
            row["goals"][f"G{j + 1}"] = dict(undetermined_share=float(1 - det[nd].mean()), undetermined_share_L2_4=float(1 - det[nd][Lt >= 2].mean()),
                                             regret_class_policy=float(reg[nd].mean()), regret_class_policy_L2_4=float(reg[nd][Lt >= 2].mean()))
            if k == 1:
                out.setdefault("determined_1step", {})[f"G{j + 1}"] = det.tolist()        # used by the measures
        # affine ceiling: posterior decoded from the exact k-step prediction
        X = torch.tensor(Pk, device=t.dev)
        Bt = torch.tensor(g.belief[:npre], device=t.dev)
        nfit, ntest = torch.tensor(node[~test], device=t.dev), torch.tensor(nd, device=t.dev)
        pr = Affine(X[nfit], Bt[nfit])
        row["posterior_r2_affine_from_prediction"] = MS.belief_scores(pr(X[ntest]), Bt[ntest])["r2"]
        out["k_step"][k] = row
    out["regret_reference"] = {}
    for j in range(3):
        bel = g.belief[:npre]
        Qm = maze.mdp_values(maze.goals[j])[maze.H]
        a_map, a_qmdp = Qm[bel.argmax(1)].argmax(1), (bel @ Qm).argmax(1)
        a_goal = (freq[:, None] * Qr[:, j]).sum(0).argmax()
        r = lambda a: float((V[nd, j] - Qr[nd, j][np.arange(len(nd)), a[nd] if np.ndim(a) else np.full(len(nd), a)]).mean())
        out["regret_reference"][f"G{j + 1}"] = dict(uniform=float((V[nd, j] - Qr[nd, j].mean(1)).mean()), goal_only=r(a_goal), most_likely_cell=r(a_map),
                                                   belief_weighted_mdp=r(a_qmdp), value=float(V[nd, j].mean()))
    (HERE / "checks.json").write_text(json.dumps(out))
    md = ["# The predictive-transfer experiment: solver-only checks", "", "Generated by `checks.py` before any model was trained. Held-out side of the pair-types experiment's bank "
          f"({out['test_histories']} histories, prefix lengths 0–4 equally often), each under every goal; decision at the reveal (12 moves left).", "",
          "## Predictive equivalence", "",
          "| prediction | values | rank of the centred predictions | classes met | undetermined decisions G1 / G2 / G3 | same, prefix 2–4 | regret of the class policy G1 / G2 / G3 | posterior R² from the exact prediction (affine) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, row in out["k_step"].items():
        gg = row["goals"]
        md.append(f"| {k}-step | {row['dims']} | {row['rank']} | {row['classes']} | " + " / ".join(f"{gg[x]['undetermined_share']:.2f}" for x in gg) + " | " +
                  " / ".join(f"{gg[x]['undetermined_share_L2_4']:.2f}" for x in gg) + " | " + " / ".join(f"{gg[x]['regret_class_policy']:.4f}" for x in gg) +
                  f" | {row['posterior_r2_affine_from_prediction']:.3f} |")
    md += ["", "## Regret at the reveal of reference policies", "", "| goal | optimal value | uniform | goal only | most likely cell | belief-weighted fully observed |", "|---|---|---|---|---|---|"]
    for x, r in out["regret_reference"].items():
        md.append(f"| {x} | {r['value']:.3f} | {r['uniform']:.4f} | {r['goal_only']:.4f} | {r['most_likely_cell']:.4f} | {r['belief_weighted_mdp']:.4f} |")
    (HERE / "checks.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
