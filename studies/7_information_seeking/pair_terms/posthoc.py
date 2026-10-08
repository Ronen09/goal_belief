"""Post hoc (after the registered run). (1) Is H on two-cell decisions a function of the pair alone? MLPs on
[s1, s2 one-hot, w, step] against the full belief b, and with the tail of the belief (its mass and entropy outside the
two cells) added. (2) The behaviour contrast, posed properly: where a common shortest-path move exists and the
likelier cell also has shortest-path moves that are not common, how often the model takes a common one, against the
chance share |common| / |likelier's moves|. (3) The same per-pair: for pairs seen ≥ 30 times, how well a table by pair
and w predicts H (ridge on pair one-hot ⊗ [1, w]).

    .venv/bin/python studies/7_information_seeking/pair_terms/posthoc.py      # writes posthoc.json, posthoc.md
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "maze10"))
import measure as MS, train as TR                                         # noqa: E402
import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

HS = load("hs_run", HERE.parent / "H_simplex" / "run.py")
PT = load("pt_run", HERE / "run.py")
MIN_PAIR = 30


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len, n = len(t.goal_cell), t.H, t.n
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, HS.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, HS.N_EP, gen); fgoal, fcell = e.goal, e.cell
    popt, reach = HS.cell_candidates(t); opt = PT.opt_moves(t)
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = HS.decisions(t, net, rf, K), HS.decisions(t, net, r, K)
        h_cells, cover, _ = PT.cell_profiles(df, n)
        for dd in (df, dt): dd["cover"] = cover
        tf, tt = PT.two_cell(df, t, H_len, h_cells, popt, opt), PT.two_cell(dt, t, H_len, h_cells, popt, opt)
        S = lambda z: HS.step_onehot(z["st"], H_len)
        oh = lambda x: torch.nn.functional.one_hot(x, n).double()
        pair = lambda z: torch.cat([S(z), oh(z["s1"]), oh(z["s2"]), z["w"][:, None]], 1)
        def tail(z):
            b = z["b"].clone(); b.scatter_(1, torch.stack([z["s1"], z["s2"]], 1), 0)
            mass = b.sum(1, keepdim=True); q = b / mass.clamp(min=1e-9)
            return torch.cat([mass, -(q * q.clamp(min=1e-30).log()).sum(1, keepdim=True)], 1)
        Y, Yt, ym = tf["H"], tt["H"], tf["H"].mean(0)
        va = df["ep"][tf["idx"]] % 10 == 0
        F = lambda X, Yf, v: HS.fit_mlp(X, Yf, v, std_floor=1e-2)                           # one-hot inputs: no blow-up on rare columns
        row = {"mlp_pair": HS.r2(F(pair(tf), Y, va)(pair(tt)), Yt, ym),
               "mlp_pair+tail": HS.r2(F(torch.cat([pair(tf), tail(tf)], 1), Y, va)(torch.cat([pair(tt), tail(tt)], 1)), Yt, ym),
               "mlp_b": HS.r2(F(torch.cat([S(tf), tf["b"]], 1), Y, va)(torch.cat([S(tt), tt["b"]], 1)), Yt, ym),
               "mlp_b+pair": HS.r2(F(torch.cat([S(tf), tf["b"], pair(tf)], 1), Y, va)(torch.cat([S(tt), tt["b"], pair(tt)], 1)), Yt, ym)}
        # landmark seeking: is the move a shortest-path move toward the nearest landmark (from the likelier cell / the true cell)?
        land = list(t.maze.landmarks)
        dl = torch.stack([torch.tensor(BM.distances(t.maze.nxt, l), device=dev) for l in land]).float()        # [L, n]
        dln = dl[:, t.nxt_cell]                                                                   # [L, n, 4]
        toward = (dln <= dln.min(-1, keepdim=True).values + 1e-6)                                 # moves toward each landmark
        near = dl.argmin(0)                                                                       # nearest landmark per cell
        tow_near = toward[near, torch.arange(n, device=dev)]                                      # [n, 4] toward the nearest landmark
        mv_all = dt["lc"].argmax(-1)[torch.arange(len(dt["st"]), device=dev), dt["goal"]]
        topc = dt["b"].argmax(1)
        lm = lambda cells, mv_: tow_near[cells].gather(1, mv_[:, None]).squeeze(1).double()
        topv = dt["b"].topk(2, 1); two = (topv.values.sum(1) >= PT.MASS) & (topv.values[:, 1] >= PT.MINOR); one = topv.values[:, 0] >= PT.CERTAIN
        on_land = torch.zeros(n, dtype=torch.bool, device=dev); on_land[land] = True
        row["landmark"] = dict(toward_from_likelier_two=float(lm(topc, mv_all)[two].mean()), toward_from_likelier_one=float(lm(topc, mv_all)[one].mean()),
                               toward_from_true_two=float(lm(dt["cell"], mv_all)[two].mean()), toward_from_true_one=float(lm(dt["cell"], mv_all)[one].mean()),
                               toward_from_true_all=float(lm(dt["cell"], mv_all).mean()),
                               chance_from_true_two=float(tow_near[dt["cell"]][two].double().mean()), chance_from_true_one=float(tow_near[dt["cell"]][one].double().mean()))
        # per-pair table (frequent ordered pairs)
        pid = lambda z: z["s1"] * n + z["s2"]
        cnt = torch.bincount(pid(tf), minlength=n * n)
        freq = cnt >= MIN_PAIR
        keep_f, keep_t = freq[pid(tf)], freq[pid(tt)]
        ids = torch.nonzero(freq).squeeze(1); remap = torch.full((n * n,), -1, device=dev, dtype=torch.long); remap[ids] = torch.arange(len(ids), device=dev)
        def ptab(z, m):
            o = torch.nn.functional.one_hot(remap[pid(z)[m]], len(ids)).double()
            return torch.cat([S({"st": z["st"][m]}), o, o * z["w"][m][:, None]], 1)
        mp = HS.Ridge(ptab(tf, keep_f), Y[keep_f], va[keep_f])
        row["pair_table_r2"] = HS.r2(mp(ptab(tt, keep_t)), Yt[keep_t], Y[keep_f].mean(0))
        row["pair_table_mlp_b_r2"] = HS.r2(HS.fit_mlp(torch.cat([S(tf), tf["b"]], 1)[keep_f], Y[keep_f], va[keep_f])(torch.cat([S(tt), tt["b"]], 1)[keep_t]), Yt[keep_t], Y[keep_f].mean(0))
        row["frequent_pairs"], row["frequent_share_test"] = int(freq.sum()), float(keep_t.double().mean())
        # behaviour, posed properly
        mv = dt["lc"][tt["idx"]].argmax(-1)[torch.arange(len(tt["idx"]), device=dev), tt["goal"]]
        gi = tt["goal"]; ar = torch.arange(len(gi), device=dev)
        both, lik = tt["both_goal"][ar, gi], tt["likelier_goal"][ar, gi]
        ex = both.sum(1) > 0
        extra = ex & (lik.sum(1) > both.sum(1))                                                 # the likelier cell has non-common shortest-path moves too
        took_common = both.gather(1, mv[:, None]).squeeze(1) > 0
        took_lik = lik.gather(1, mv[:, None]).squeeze(1) > 0
        chance = (both.sum(1) / lik.sum(1).clamp(min=1))
        row["hedge"] = dict(decisions=int(extra.sum()), takes_common=float(took_common[extra].double().mean()), takes_likelier_any=float(took_lik[extra].double().mean()),
                            chance_common_given_likelier=float(chance[extra].mean()),
                            takes_common_given_likelier=float((took_common[extra & took_lik]).double().mean()) if (extra & took_lik).any() else None)
        res[f"seed{s}"] = row
        print(f"seed{s}: landmark {row['landmark']}")
        print(f"seed{s}: mlp pair {row['mlp_pair']:.3f} pair+tail {row['mlp_pair+tail']:.3f} b {row['mlp_b']:.3f} b+pair {row['mlp_b+pair']:.3f} | pair table {row['pair_table_r2']:.3f} (mlp b on the same {row['pair_table_mlp_b_r2']:.3f}; "
              f"{row['frequent_pairs']} pairs, {row['frequent_share_test']:.2f} of decisions) | hedge: {row['hedge']}", flush=True)
    (HERE / "posthoc.json").write_text(json.dumps(res))
    mm = lambda f, nd=3: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([f(res[f"seed{s}"]) for s in range(6)])
    L = ["# Pair terms: post hoc", "", "Generated by `posthoc.py` after the registered run; medians (min–max) over the six models; held-out R² on two-cell test decisions.", "",
         "| model of H on two-cell decisions | R² |", "|---|---|",
         f"| MLP on the pair alone: s1, s2 (one-hot), w, step | {mm(lambda r: r['mlp_pair'])} |",
         f"| MLP on the pair + the tail of the belief (mass and entropy outside the two cells) | {mm(lambda r: r['mlp_pair+tail'])} |",
         f"| MLP on the full belief b | {mm(lambda r: r['mlp_b'])} |",
         f"| MLP on b + the pair | {mm(lambda r: r['mlp_b+pair'])} |",
         f"| a table by ordered pair ⊗ [1, w] + step, pairs seen ≥ {MIN_PAIR} times ({mm(lambda r: r['frequent_share_test'], 2)} of decisions) | {mm(lambda r: r['pair_table_r2'])} |",
         f"| MLP on b on the same decisions | {mm(lambda r: r['pair_table_mlp_b_r2'])} |", "",
         "**Hedging, posed properly.** Where a shortest-path move from both cells exists and the likelier cell also has shortest-path moves that are not common "
         f"({mm(lambda r: r['hedge']['decisions'], 0)} decisions): the model takes a common move in {mm(lambda r: r['hedge']['takes_common'], 2)}, any likelier-cell move in "
         f"{mm(lambda r: r['hedge']['takes_likelier_any'], 2)}; given a likelier-cell move, it is a common one in {mm(lambda r: r['hedge']['takes_common_given_likelier'], 2)} against a chance share of "
         f"{mm(lambda r: r['hedge']['chance_common_given_likelier'], 2)}.", "",
         "**Landmark seeking.** The share of moves that are shortest-path moves toward the nearest landmark, from the true cell: two-cell decisions "
         f"{mm(lambda r: r['landmark']['toward_from_true_two'], 2)} (chance, a uniformly random move: {mm(lambda r: r['landmark']['chance_from_true_two'], 2)}), near-certain decisions "
         f"{mm(lambda r: r['landmark']['toward_from_true_one'], 2)} (chance {mm(lambda r: r['landmark']['chance_from_true_one'], 2)}), all decisions {mm(lambda r: r['landmark']['toward_from_true_all'], 2)}; "
         f"from the likelier cell: two-cell {mm(lambda r: r['landmark']['toward_from_likelier_two'], 2)}, near-certain {mm(lambda r: r['landmark']['toward_from_likelier_one'], 2)}."]
    (HERE / "posthoc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
