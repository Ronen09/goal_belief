"""Post hoc, second pass: which nonlinearity of the belief H uses. Ridge fits only (held-out R² as in tables.md):
    quadratic        [b, b ⊗ b] (pairwise cell interactions, 55 + 1540 features)
    log b, piecewise log b by entropy bin, log b + entropy bins
    information      affine (or log b) + the expected information gain of every move EIG(b, a) and the entropy
    top cells        one-hot of the most likely and second most likely cells, their probabilities, and their product
                     with the affine term (a "commit between two" code)
    MLP on log b     the nonlinear ceiling in log space (is log b the right coordinate?)

    .venv/bin/python studies/7_information_seeking/H_simplex/posthoc2.py      # writes posthoc2.json, posthoc2.md
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR          # noqa: E402

ENT_BINS = 8


def eig_of(t, d):
    """Expected information gain of every move at each decision: entropy(b) − expected entropy after the move and the
    next symbol (the maze10 experiment's measure), from the posterior alone."""
    b = d["b"].float()
    out = []
    for a in range(4):
        p = torch.zeros_like(b).scatter_add_(1, t.nxt_cell[:, a][None].expand(len(b), -1), b)          # moved belief (goal cells kept: goal-free)
        w = p[:, None, :] * t.E.T[None]
        z = w.sum(-1)
        out.append((z * BM.entropy(w / z[..., None].clamp(min=1e-30))).sum(-1))
    ee = torch.stack(out, 1)
    return (d["ent"].float()[:, None] - ee).double()


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len = len(t.goal_cell), t.H
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    iu = torch.triu_indices(t.n, t.n, device=dev)
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        for d in (df, dt):
            d["eig"] = eig_of(t, d)
        va = df["ep"] % 10 == 0
        S = lambda d: R.step_onehot(d["st"], H_len)
        Y, Yt, ym = df["H"], dt["H"], df["H"].mean(0)
        fit = lambda f: R.r2(R.Ridge(f(df), Y, va)(f(dt)), Yt, ym)
        B = lambda d: d["b"]
        LB = lambda d: (d["b"] + 1e-3).log()
        EB = lambda d: torch.nn.functional.one_hot((d["ent"] / 4.0 * ENT_BINS).long().clamp(max=ENT_BINS - 1), ENT_BINS).double()
        quad = lambda d: (d["b"][:, :, None] * d["b"][:, None, :])[:, iu[0], iu[1]]
        top = lambda d: (lambda o: torch.cat([torch.nn.functional.one_hot(o[:, 0], t.n).double(), torch.nn.functional.one_hot(o[:, 1], t.n).double(),
                                              d["b"].gather(1, o[:, :2]).double()], 1))(d["b"].topk(2, 1).indices)
        row = {
            "affine": fit(lambda d: torch.cat([S(d), B(d)], 1)),
            "quadratic": fit(lambda d: torch.cat([S(d), B(d), quad(d)], 1)),
            "log_b": fit(lambda d: torch.cat([S(d), LB(d)], 1)),
            "log_b+entropy_bins": fit(lambda d: torch.cat([S(d), LB(d), EB(d)], 1)),
            "log_b_piecewise": fit(lambda d: torch.cat([S(d), (LB(d)[:, :, None] * EB(d)[:, None, :]).flatten(1), EB(d)], 1)),
            "affine+eig": fit(lambda d: torch.cat([S(d), B(d), d["eig"], d["ent"][:, None]], 1)),
            "log_b+eig": fit(lambda d: torch.cat([S(d), LB(d), d["eig"], d["ent"][:, None]], 1)),
            "eig_only": fit(lambda d: torch.cat([S(d), d["eig"], d["ent"][:, None]], 1)),
            "top2": fit(lambda d: torch.cat([S(d), top(d)], 1)),
            "affine+top2": fit(lambda d: torch.cat([S(d), B(d), top(d)], 1)),
            "log_b+top2": fit(lambda d: torch.cat([S(d), LB(d), top(d)], 1)),
            "log_b+quadratic": fit(lambda d: torch.cat([S(d), LB(d), quad(d)], 1)),
        }
        row["mlp_log_b"] = R.r2(R.fit_mlp(torch.cat([S(df), LB(df)], 1), Y, va)(torch.cat([S(dt), LB(dt)], 1)), Yt, ym)
        row["mlp_b"] = R.r2(R.fit_mlp(torch.cat([S(df), B(df)], 1), Y, va)(torch.cat([S(dt), B(dt)], 1)), Yt, ym)
        # how concentrated the beliefs are along the episodes (context for the top-cell codes)
        pm = dt["b"].max(1).values
        row["pmax_quantiles"] = [float(q) for q in torch.quantile(pm, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=pm.dtype, device=dev))]
        res[f"seed{s}"] = row
        print(f"seed{s}: " + " ".join(f"{k} {v:.3f}" for k, v in row.items() if k != "pmax_quantiles"), flush=True)
    (HERE / "posthoc2.json").write_text(json.dumps(res))
    keys = [k for k in res["seed0"] if k != "pmax_quantiles"]
    mm = lambda k: (lambda xs: f"{np.median(xs):.3f} ({min(xs):.3f}–{max(xs):.3f})")([res[f"seed{s}"][k] for s in range(6)])
    L = ["# H on the belief simplex: post hoc, second pass", "", "Generated by `posthoc2.py`. Held-out R² of ridge fits (the MLPs as in `tables.md`); medians (min–max) over the six models.", "",
         "| features | R² |", "|---|---|"] + [f"| {k} | {mm(k)} |" for k in keys] + ["",
         "Most likely cell's probability along the test episodes, quantiles 10/25/50/75/90 % (seed medians): " +
         ", ".join(f"{np.median([res[f'seed{s}']['pmax_quantiles'][i] for s in range(6)]):.2f}" for i in range(5)) + "."]
    (HERE / "posthoc2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
