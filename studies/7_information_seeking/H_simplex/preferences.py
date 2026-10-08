"""How much the nonlinear (pair) part of H changes action preferences. H is split into its affine-in-b part (ridge, fit
episodes) and the rest; the quadratic fit stands for "affine + pair terms". On test decisions, by belief type
(near-certain: max b >= 0.9; two-cell: top two >= 0.85 and minor >= 0.15; diffuse: the rest):
    top action of H vs of its affine part (and quadratic vs affine): share changed;
    the decision with the goal bias: argmax(H + G) vs argmax(affine + G) under the episode's goal: share changed;
    size: mean |H - affine| over actions relative to H's spread; total variation between softmax(H + G) and softmax(affine + G);
    margin: among changed decisions, H's margin between its top two actions.

    .venv/bin/python studies/7_information_seeking/H_simplex/preferences.py      # writes preferences.md / .json
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


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len, n = len(t.goal_cell), t.H, t.n
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    iu = torch.triu_indices(n, n, device=dev)
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        va = df["ep"] % 10 == 0
        S = lambda d: R.step_onehot(d["st"], H_len)
        quad = lambda d: (d["b"][:, :, None] * d["b"][:, None, :])[:, iu[0], iu[1]]
        aff = R.Ridge(torch.cat([S(df), df["b"]], 1), df["H"], va)
        qd = R.Ridge(torch.cat([S(df), df["b"], quad(df)], 1), df["H"], va)
        Ha = aff(torch.cat([S(dt), dt["b"]], 1)); Hq = qd(torch.cat([S(dt), dt["b"], quad(dt)], 1)); H = dt["H"]
        G = R.goal_bias(df, K, H_len)
        Gt = G[dt["goal"], dt["st"].clamp(max=R.SB - 1)]                                        # [m, 4] the episode's goal bias
        top = dt["b"].topk(2, 1)
        kinds = {"near-certain": top.values[:, 0] >= 0.9, "two-cell": (top.values.sum(1) >= 0.85) & (top.values[:, 1] >= 0.15)}
        kinds["diffuse"] = ~(kinds["near-certain"] | kinds["two-cell"]); kinds["all"] = torch.ones_like(kinds["diffuse"])
        spread = H.std()
        tv = lambda A, B: 0.5 * (torch.softmax(A, 1) - torch.softmax(B, 1)).abs().sum(1)
        margin = lambda A: (lambda v: v[:, 0] - v[:, 1])(A.topk(2, 1).values)
        row = {}
        for k, m in kinds.items():
            f = lambda x: float(x[m].double().mean())
            chg_top = H.argmax(1) != Ha.argmax(1); chg_dec = (H + Gt).argmax(1) != (Ha + Gt).argmax(1)
            row[k] = dict(share=float(m.double().mean()), decisions=int(m.sum()),
                          top_changed=f(chg_top), decision_changed=f(chg_dec),
                          top_changed_quad_vs_affine=f(Hq.argmax(1) != Ha.argmax(1)), decision_changed_quad_vs_affine=f((Hq + Gt).argmax(1) != (Ha + Gt).argmax(1)),
                          top_changed_H_vs_quad=f(H.argmax(1) != Hq.argmax(1)), decision_changed_H_vs_quad=f((H + Gt).argmax(1) != (Hq + Gt).argmax(1)),
                          rel_size=f((H - Ha).abs().mean(1) / spread), rel_size_quad_part=f((Hq - Ha).abs().mean(1) / spread),
                          tv_policy=f(tv(H + Gt, Ha + Gt)), tv_policy_quad_vs_affine=f(tv(Hq + Gt, Ha + Gt)),
                          margin_H_all=f(margin(H)), margin_H_where_decision_changed=float(margin(H + Gt)[m & chg_dec].double().mean()) if (m & chg_dec).any() else None,
                          margin_affine_where_decision_changed=float(margin(Ha + Gt)[m & chg_dec].double().mean()) if (m & chg_dec).any() else None,
                          r2_affine=R.r2(Ha[m], H[m], df["H"].mean(0)), r2_quad=R.r2(Hq[m], H[m], df["H"].mean(0)))
        res[f"seed{s}"] = row
        print(f"seed{s}: " + " | ".join(f"{k}: share {v['share']:.2f} top chg {v['top_changed']:.2f} dec chg {v['decision_changed']:.2f} (quad-aff {v['decision_changed_quad_vs_affine']:.2f}, H-quad {v['decision_changed_H_vs_quad']:.2f}) tv {v['tv_policy']:.2f} size {v['rel_size']:.2f}" for k, v in row.items()), flush=True)
    (HERE / "preferences.json").write_text(json.dumps(res))
    mm = lambda k, f, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([res[f"seed{s}"][k][f] for s in range(6) if res[f"seed{s}"][k][f] is not None])
    L = ["# How much the nonlinear part of H changes action preferences", "",
         "Generated by `preferences.py`. H split into its affine-in-b part (ridge on the fit episodes) and the rest; the quadratic fit (b and pairwise products) stands for affine + pair terms. "
         "Test decisions of the six maze10 models, medians (min–max). 'Decision' = argmax(H + G) under the episode's goal; 'top' = argmax H (goal-free). "
         "Size = mean |H − affine| over actions relative to H's overall spread (one standard deviation); TV = total variation between softmax(H + G) and softmax(affine + G).", "",
         "| belief | share of decisions | R² affine / quadratic | top action changed by the rest | decision changed by the rest | of which by the pair terms (quadratic vs affine) | beyond the pair terms (H vs quadratic) | size of the rest | TV of the policy | H's margin where the decision changed (affine's) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("near-certain", "two-cell", "diffuse", "all"):
        L.append(f"| {k} | {mm(k, 'share')} | {mm(k, 'r2_affine')} / {mm(k, 'r2_quad')} | {mm(k, 'top_changed')} | {mm(k, 'decision_changed')} | {mm(k, 'decision_changed_quad_vs_affine')} | {mm(k, 'decision_changed_H_vs_quad')} | "
                 f"{mm(k, 'rel_size')} | {mm(k, 'tv_policy')} | {mm(k, 'margin_H_where_decision_changed')} ({mm(k, 'margin_affine_where_decision_changed')}) |")
    (HERE / "preferences.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
