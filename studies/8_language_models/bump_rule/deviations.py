"""Where the agent's deviations from the optimum sit, on its own 20 000 greedy episodes (the same collection as rule.py):
per belief set the agent's majority action and whether it is optimal (a systematic deviation) or the deviation is
occasional (the same set, different histories, different actions on its own distribution); the sets that carry the
deviations; the known states (set size 1) where the agent's majority move is wrong, drawn on the map.
Writes deviations_<name>.json and prints a summary.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from goalgeo import bumpmaze as BMZ
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from rule import Sets, collect, gru_policy, opt_policy, N_EP, SPLIT_SEED
from battery import load_model
from common import world
from train import test_starts


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--model", default=None); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
    w, sol = world(); sets = Sets(w, sol); dev = a.device
    model = load_model("gru", (HERE / a.model).resolve(), w.H, dev) if a.model else None; pol = gru_policy(model) if model else opt_policy(sets, dev)
    D = collect(w, sets, pol, N_EP, SPLIT_SEED, dev); act, si, st = D["act"], D["set"], D["step"]; n = len(act)
    optimal = sets.opt[si, act]; devi = ~optimal
    # per set
    U, inv = np.unique(si, return_inverse=True); cnt = np.zeros((len(U), 3), dtype=np.int64); np.add.at(cnt, (inv, act), 1)
    maj = cnt.argmax(1); maj_opt = sets.opt[U, maj]; tot = cnt.sum(1); mixed = (cnt > 0).sum(1) > 1
    dev_per_set = np.zeros(len(U), dtype=np.int64); np.add.at(dev_per_set, inv, devi.astype(np.int64))
    systematic = dev_per_set[~maj_opt].sum(); occasional = dev_per_set[maj_opt].sum()
    order = np.argsort(-dev_per_set); cum = np.cumsum(dev_per_set[order]) / max(devi.sum(), 1)
    n50, n90 = int((cum < 0.5).sum() + 1), int((cum < 0.9).sum() + 1)
    # on-distribution history dependence: sets (>= 20 decisions) with more than one action; among them, both actions optimal (a tie) or not
    big = tot >= 20; mixed_big = mixed & big
    tie = np.array([bool(sets.opt[U[i]][cnt[i] > 0].all()) for i in range(len(U))])
    res = dict(name=a.name, decisions=n, deviations=int(devi.sum()), deviation_share=float(devi.mean()), sets_visited=len(U),
               systematic_share=float(systematic / max(devi.sum(), 1)), occasional_share=float(occasional / max(devi.sum(), 1)),
               sets_with_wrong_majority=int((~maj_opt).sum()), sets_for_half_of_deviations=n50, sets_for_90pct=n90,
               big_sets=int(big.sum()), big_sets_mixed=int(mixed_big.sum()), big_sets_mixed_tie=int((mixed_big & tie).sum()), big_sets_mixed_not_tie=int((mixed_big & ~tie).sum()),
               decisions_in_mixed_not_tie=float(tot[mixed_big & ~tie].sum() / n))
    # known states
    known = sets.known[si]; res["known"] = dict(decisions=int(known.sum()), deviations=int((devi & known).sum()), deviation_share=float(devi[known].mean()))
    wrong = [i for i in range(len(U)) if sets.known[U[i]] and not maj_opt[i]]
    rows = []
    for i in wrong:
        s = int(np.nonzero(sol["masks"][U[i]])[0][0]); c, d = divmod(s, 4); x, y = w.cells[c]
        rows.append(dict(cell=[x, y], heading=BMZ.DIR_NAMES[d], n=int(tot[i]), agent="LRF"[maj[i]], optimal="".join("LRF"[k] for k in range(3) if sets.opt[U[i], k]), gap=float((sets.V[U[i]] - sets.Q[U[i], maj[i]]) / sets.V[U[i]]), share_of_known_deviations=float(dev_per_set[i] / max((devi & known).sum(), 1))))
    rows.sort(key=lambda r: -r["n"]); res["known_wrong"] = rows
    # the top deviating sets overall
    top = []
    for i in order[:12]:
        top.append(dict(set=int(U[i]), size=int(sets.size[U[i]]), n=int(tot[i]), deviations=int(dev_per_set[i]), agent_freq=(cnt[i] / tot[i]).round(2).tolist(), optimal="".join("LRF"[k] for k in range(3) if sets.opt[U[i], k]), majority_optimal=bool(maj_opt[i]), mean_step=float(st[inv == i].mean()), gap_of_majority=float((sets.V[U[i]] - sets.Q[U[i], maj[i]]) / sets.V[U[i]])))
    res["top_sets"] = top
    # map of known-state wrong moves
    g = [list(r) for r in w.ascii().split("\n")]
    for r in rows: x, y = r["cell"]; g[y][x] = "x" if g[y][x] in ".x" else g[y][x]
    res["map"] = "\n".join("".join(r) for r in g)
    (HERE / f"deviations_{a.name}.json").write_text(json.dumps(res, indent=1))
    print(f"{a.name}: {n} decisions, {res['deviations']} deviations ({res['deviation_share']:.3f}); systematic (set's majority wrong) {res['systematic_share']:.2f}, occasional {res['occasional_share']:.2f}; "
          f"{res['sets_with_wrong_majority']} of {len(U)} sets wrong by majority; {n50} sets carry half, {n90} 90 %; big sets {big.sum()}: mixed {mixed_big.sum()} (tie {int((mixed_big & tie).sum())}, not tie {int((mixed_big & ~tie).sum())}, {res['decisions_in_mixed_not_tie']:.3f} of decisions)")
    print(f"  known states: {res['known']['decisions']} decisions, deviation share {res['known']['deviation_share']:.3f}; wrong-majority known states {len(rows)}: " + "; ".join(f"({r['cell'][0]},{r['cell'][1]}) {r['heading'][0]} n={r['n']} {r['agent']} vs {r['optimal']} gap {r['gap']:.2f}" for r in rows[:10]))
    print("  top sets: " + "; ".join(f"#{t['set']} size {t['size']} n={t['n']} dev {t['deviations']} freq {t['agent_freq']} opt {t['optimal']} step {t['mean_step']:.1f} gap {t['gap_of_majority']:.2f}" for t in top[:6]))
    print(res["map"])


if __name__ == "__main__":
    main()
