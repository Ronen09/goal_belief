"""Round 44: is H a mixture of round 43's leading candidates, and how does it weight the three goals? Round 23's reward
models, frozen.

    .venv/bin/python rounds/r44_H_mixture/run.py            # writes results.json

Per-goal quantities, one value per action, for the posterior at the reveal:
    opt_g = 1[a optimal under g],   Q_g = Q*(b, g, a),   M_g = Q_MDP(b, g, a) (fully observed reachability)
Prediction of H (round 43's: held-out histories of length >= 1, R² against H's variance):
    single       popt, meanMDP, meanQ (flexible 4 x 4, as round 43)
    mix          popt + meanMDP + meanQ together (flexible)
    weighted_q   for each quantity q: H ~ alpha[a, L] + sum_g beta_g q_g (one slope per goal: 3 parameters); goal weights
                 w_g = beta_g / sum beta; the same with one shared slope (uniform weights)
    weighted     all nine: H ~ alpha + sum_{g, q} beta_{g,q} q_g; its composite C = sum beta q_g is a 4-number code
Causal edits at the goal token entering block 2 (round 43's editor and pairs): popt, meanMDP, mix, weighted (the composite),
weighted_uniform (each quantity's goal weights forced equal), Htab, and mix rotated.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import torch

from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R43 = load("r43run", ROUNDS / "r43_what_is_H" / "run.py")
R32, R42, R27, R30, M26 = R43.R32, R43.R42, R43.R27, R43.R30, R43.M26
SITE = ("resid", 2)
QUANTS = ("opt", "Q", "MDP")


def slope_fit(H, feats, ctx, fit, test, var):
    """H ~ alpha[a, L] + sum_j beta_j F_j (F_j [nh, 4]); returns beta, held-out R², and the composite sum_j beta_j F_j."""
    nL = ctx.t.max_prefix + 1
    def demean(X, rows):
        mu = torch.stack([X[rows][ctx.L[rows] == l].mean(0) for l in range(nL)])
        return X - mu[ctx.L]
    Y = demean(H, fit)
    Xs = [demean(F, fit) for F in feats]
    A = torch.stack([X[fit].reshape(-1) for X in Xs], 1)
    beta = torch.linalg.lstsq(A, Y[fit].reshape(-1, 1)).solution.squeeze(1)
    P = sum(b * X for b, X in zip(beta, Xs))
    r2 = float(1 - ((Y[test] - P[test]) ** 2).sum() / var)
    comp = sum(b * F for b, F in zip(beta, feats))
    return beta, r2, comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    nL = t.max_prefix + 1
    pairs = R27.Pairs(ctx)
    m = pairs.main
    C = R43.candidates(t, ctx)
    Cc = R43.centre(C)
    ci = {k: i for i, k in enumerate(R43.CANDS)}
    QM, _ = R43.qmdp(t)
    b = t.belief[ctx.node].double()
    per_goal = dict(opt=ctx.opt.double(), Q=ctx.Q.double(), MDP=torch.einsum("hs,sga->hga", b, QM))   # [nh, 3, 4]
    per_goal = {k: R43.centre(v) for k, v in per_goal.items()}
    bel = b
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    npost = int(pid.max()) + 1
    fit_all = torch.nonzero(ctx.fit).squeeze(1)
    fit, test = torch.nonzero(ctx.fit & (ctx.L >= 1)).squeeze(1), torch.nonzero(ctx.test & (ctx.L >= 1)).squeeze(1)
    L1 = torch.nn.functional.one_hot(ctx.L, nL).double()
    kinds = ("none", "whole", "popt", "meanMDP", "mix", "weighted", "weighted_uniform", "Htab", "rotated")
    R32.KINDS = kinds
    res = dict(runs={})
    for s in a.seeds:
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        gen = torch.Generator(device=dev); gen.manual_seed(4400 + s)
        all_h = torch.arange(len(ctx.L), device=dev)
        lg = R42.logits(net, ctx, all_h)
        H = lg.mean(1)
        Ht = H[test]
        var = ((Ht - Ht.mean(0)) ** 2).sum()
        flex = lambda F: float(1 - ((Ht - Affine(torch.cat([F, L1], 1)[fit], H[fit])(torch.cat([F, L1], 1)[test])) ** 2).sum() / var)
        single = {k: flex(Cc[:, ci[k]]) for k in ("popt", "meanMDP", "meanQ")}
        mixF = torch.cat([Cc[:, ci["popt"]], Cc[:, ci["meanMDP"]], Cc[:, ci["meanQ"]]], 1)
        row = dict(checkpoint=ck, single=single, mix=flex(mixF), mix_popt_meanMDP=flex(torch.cat([Cc[:, ci["popt"]], Cc[:, ci["meanMDP"]]], 1)))
        # goal weights per quantity
        row["weights"] = {}
        uni_comp = []
        for q in QUANTS:
            feats = [per_goal[q][:, g] for g in range(3)]
            beta, r2, comp = slope_fit(H, feats, ctx, fit, test, var)
            _, r2u, compu = slope_fit(H, [sum(feats) / 3], ctx, fit, test, var)
            row["weights"][q] = dict(beta=beta.tolist(), w=(beta / beta.sum()).tolist(), r2=r2, r2_uniform=r2u)
            uni_comp.append(compu)
        feats9 = [per_goal[q][:, g] for q in QUANTS for g in range(3)]
        beta9, r2_9, comp9 = slope_fit(H, feats9, ctx, fit, test, var)
        _, r2_9u, comp9u = slope_fit(H, uni_comp, ctx, fit, test, var)
        row["weighted"] = dict(beta=beta9.tolist(), r2=r2_9, r2_uniform=r2_9u,
                               goal_share={q: [float(beta9[QUANTS.index(q) * 3 + g] / beta9[QUANTS.index(q) * 3:(QUANTS.index(q) + 1) * 3].sum()) for g in range(3)] for q in QUANTS})
        # H's own posterior table (ceiling), as round 43
        sH = torch.zeros(npost, 4, dtype=torch.float64, device=dev).index_add_(0, pid[fit_all], H[fit_all])
        nH = torch.bincount(pid[fit_all], minlength=npost).double()[:, None]
        Htab = (sH + R43.SHRINK * H[fit_all].mean(0)) / (nH + R43.SHRINK)
        row["Htab"] = flex(R43.centre(Htab[pid]))
        # causal edits
        codes = dict(popt=Cc[:, ci["popt"]], meanMDP=Cc[:, ci["meanMDP"]], mix=mixF, weighted=R43.centre(comp9), weighted_uniform=R43.centre(comp9u),
                     Htab=R43.centre(Htab[pid]), maxQ=Cc[:, ci["maxQ"]])
        Z = R43.states(net, ctx, all_h, SITE)
        ed = R43.Editor(Z, codes, ctx, gen)
        base_edits = ed.edits
        def edits(a_, b_, g):
            e = base_edits(a_, b_, g)
            mixv = (codes["mix"][b_] - codes["mix"][a_]) @ ed.W["mix"]
            e["rotated"] = [ed.Z[a_, g] + mixv @ Q for Q in ed.Q]
            e.pop("maxQ", None)
            return e
        ed.edits = edits
        R32.SITE = SITE
        R32.outputs.check = 0.0
        ra, rb = m["r"], m["d"]
        P, num, den = R32.outputs(net, ctx, ed, ra, rb)
        dj, sm = pairs.disjoint(ra, rb), pairs.same(ra, rb)
        row["edits"] = dict(change=R32.summarise(P, num, den, ctx, ra, rb, dj), preserve=R32.preserve(P, ctx, ra, rb, sm), goal_dependent=R32.goal_dependent(P, ctx, ra, rb))
        res["runs"][f"seed{s}"] = row
        ch = row["edits"]["change"]
        print(f"seed{s} {ck} R²: " + " ".join(f"{k} {v:.2f}" for k, v in single.items()) + f" mix {row['mix']:.2f} weighted9 {r2_9:.2f} (uniform {r2_9u:.2f}) Htab {row['Htab']:.2f} | weights: " +
              " ".join(f"{q} " + ",".join(f"{x:.2f}" for x in row['weights'][q]['w']) + f" (R² {row['weights'][q]['r2']:.2f} uni {row['weights'][q]['r2_uniform']:.2f})" for q in QUANTS) +
              " | edit donor-optimal: " + " ".join(f"{k} {ch[k]['donor_optimal']:.3f}" for k in kinds) + f" | {time.time() - t0:.0f}s", flush=True)
        (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
