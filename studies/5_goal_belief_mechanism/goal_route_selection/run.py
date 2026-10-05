"""The goal-route-selection experiment: does the goal change which route the reward policy takes its evidence from? Fixed history pairs, all three
goals, the observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/goal_route_selection/run.py            # writes results.json

Two routes into the decision. The decision at the goal token is a function of the residual stream entering block 1 at
the prefix tokens (the interface; goal-free) and at the goal token (the direct route: the goal token's own block 0,
which reads the raw tokens). Per pair (recipient A, donor B) and goal g:

    none        the recipient's run
    hybrid      the donor's prefix with goal g (the reference)
    interface   prefix states entering block 1 from the hybrid run, the goal token's own state kept
    encoding    prefix states h_A,i + E (b_B,i - b_A,i) (the policy-belief-edit experiment's edit)
    direct      the goal token's state entering block 1 from the hybrid run, the prefix states kept
    both        both replaced (= hybrid exactly; a check)

Reliance on a route: the projection of its logit change onto the hybrid's, a = <l_k - l_none, l_hyb - l_none> / |l_hyb - l_none|^2
(log-probabilities, centred). Direct-route adequacy, defined without any patch: an affine probe of the goal token's
state entering block 1 (fitted per goal on the fit side) to the exact action values; the route is adequate for a cell if
the probe's greedy action is optimal for both histories under that goal.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazeedit as ED, mazemodel as MM
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R28 = load("r28run", ROUNDS / "4_predictive_pretraining" / "policy_belief_edit" / "run.py")
R27, M26, R22 = R28.R27, R28.M26, R28.R22
L_IF, CHUNK, MIN_TV = 1, 8192, 0.1
ROUTES = ("interface", "encoding", "direct", "both")


def centred(lg):
    lp = torch.log_softmax(lg, -1)
    return lp - lp.mean(-1, keepdim=True)


@torch.no_grad()
def goal_states(net, ctx, hist, g):
    """The goal token's state entering block 1, and the last prefix token's, for histories `hist` under goal g."""
    L = ctx.L[hist]
    out_g, out_p = [], []
    for s in range(0, len(hist), CHUNK):
        h, LL = hist[s:s + CHUNK], L[s:s + CHUNK]
        st = ED.states(net, ctx.data.seq(h, LL, g), L_IF)
        n = torch.arange(len(h), device=h.device)
        out_g.append(st[n, 1 + LL]); out_p.append(st[n, LL])
    return torch.cat(out_g), torch.cat(out_p)


class Probes:
    """Per goal: affine probes from the goal token's state (direct route) and from the last prefix token's state
    (interface, goal-free) to the exact action values; greedy correctness for every history."""

    def __init__(self, net, ctx):
        nh = len(ctx.L)
        all_h = torch.arange(nh, device=ctx.t.dev)
        fit = ctx.fit
        self.direct_ok = torch.zeros(nh, 3, dtype=torch.bool, device=ctx.t.dev)
        self.interface_ok = torch.zeros_like(self.direct_ok)
        self.acc = {}
        for g in range(3):
            G, Pf = goal_states(net, ctx, all_h, g)
            Y = ctx.Q[:, g].double()
            for name, X, ok in (("direct", G, self.direct_ok), ("interface", Pf, self.interface_ok)):
                pr = Affine(X[fit], Y[fit])
                a = pr(X).argmax(1)
                ok[:, g] = ctx.opt[all_h, g][all_h, a]
                self.acc[f"{name}_G{g + 1}"] = float(ok[~fit, g].float().mean())


@torch.no_grad()
def cells(net, ctx, itf, a, b):
    """Per (pair, goal): reliance on each route, behavioural difference, logit-difference norm, and the greedy
    actions. Returns dict of [n, goals] tensors."""
    data, t = ctx.data, ctx.t
    L = ctx.L[a]
    out = {k: torch.zeros(len(a), 3, device=t.dev) for k in ("tv", "dnorm") + tuple(f"a_{r}" for r in ROUTES) + tuple(f"moved_{r}" for r in ROUTES)}
    greedy = {k: torch.zeros(len(a), 3, dtype=torch.long, device=t.dev) for k in ("none", "hybrid") + ROUTES}
    for s in range(0, len(a), CHUNK):
        sl = slice(s, s + CHUNK)
        aa, bb, LL = a[sl], b[sl], L[sl]
        gp = 1 + LL
        n = torch.arange(len(aa), device=t.dev)
        for g in range(3):
            tA, tH = data.seq(aa, LL, g), data.seq(bb, LL, g)
            sA, sH = ED.states(net, tA, L_IF), ED.states(net, tH, L_IF)
            mask = ED.prefix_mask(tA, gp)
            hi, pi = torch.nonzero(mask, as_tuple=True)
            bA, bB = t.belief[ctx.bank.pre_node[aa[hi], pi]], t.belief[ctx.bank.pre_node[bb[hi], pi]]
            enc = (sA[mask].double() + (bB - bA).double() @ itf.E).float()

            def run(prefix=None, goal_state=None):
                def f(x):
                    x = x.clone()
                    if prefix is not None:
                        x[mask] = prefix
                    if goal_state is not None:
                        x[n, gp] = goal_state
                    return x
                return net(tA, patch={("resid", L_IF): f})[0][n, gp]

            l0 = net(tA)[0][n, gp]; lh = net(tH)[0][n, gp]
            lg = dict(interface=run(prefix=sH[mask]), encoding=run(prefix=enc), direct=run(goal_state=sH[n, gp]), both=run(prefix=sH[mask], goal_state=sH[n, gp]))
            c0, ch = centred(l0), centred(lh)
            d = ch - c0
            dn = (d ** 2).sum(-1)
            p0, ph = l0.softmax(-1), lh.softmax(-1)
            tvh = R27.tv(p0, ph)
            out["tv"][sl, g], out["dnorm"][sl, g] = tvh, dn.sqrt()
            greedy["none"][sl, g], greedy["hybrid"][sl, g] = l0.argmax(-1), lh.argmax(-1)
            for r, x in lg.items():
                out[f"a_{r}"][sl, g] = ((centred(x) - c0) * d).sum(-1) / dn.clamp(min=1e-9)
                out[f"moved_{r}"][sl, g] = 1 - R27.tv(x.softmax(-1), ph) / tvh.clamp(min=1e-6)
                greedy[r][sl, g] = x.argmax(-1)
    return out, greedy


def oracle_gap(ctx, a, b):
    """Cost of acting on the other history's optimal action, averaged over the two directions: [n, goals]."""
    QA, QB = ctx.Q[a], ctx.Q[b]
    aA, aB = QA.argmax(-1, keepdim=True), QB.argmax(-1, keepdim=True)
    return 0.5 * ((ctx.V[a] - QA.gather(-1, aB).squeeze(-1)) + (ctx.V[b] - QB.gather(-1, aA).squeeze(-1)))


def within_pair_ols(y, covs, pair):
    """OLS of y on covariates after removing each pair's mean (pair fixed effects). Returns coefficients."""
    uniq, inv = torch.unique(pair, return_inverse=True)
    def dm(v):
        s = torch.zeros(len(uniq), dtype=torch.float64, device=v.device).index_add_(0, inv, v.double())
        c = torch.zeros(len(uniq), dtype=torch.float64, device=v.device).index_add_(0, inv, torch.ones_like(v, dtype=torch.float64))
        return v.double() - (s / c)[inv]
    X = torch.stack([dm(c) for c in covs.values()], 1)
    beta = torch.linalg.lstsq(X, dm(y)[:, None]).solution.squeeze(1)
    return {k: float(v) for k, v in zip(covs, beta)}


def analyse(c, greedy, ctx, probes, a, b):
    """Per-goal descriptives and within-pair regressions, on eligible cells (TV(none, hybrid) >= MIN_TV)."""
    n = len(a)
    elig = c["tv"] >= MIN_TV
    X = probes.direct_ok[a] & probes.direct_ok[b]                                   # direct route adequate for this cell
    Y = probes.interface_ok[a] & probes.interface_ok[b]
    gap = oracle_gap(ctx, a, b)
    dj = ~(ctx.opt[a] & ctx.opt[b]).any(-1)
    pair = torch.arange(n, device=a.device)[:, None].expand(n, 3)
    goal = torch.arange(3, device=a.device)[None].expand(n, 3)
    multi = elig.sum(1) >= 2                                                        # pairs with at least two eligible goals
    sel = elig & multi[:, None]
    res = dict(n_cells=int(elig.sum()), n_within=int(sel.sum()), n_pairs_within=int(multi.sum()),
               both_check=float((c["a_both"][elig] - 1).abs().max()) if elig.any() else None)
    pg = {}
    for g in range(3):
        m = elig[:, g]
        pg[f"G{g + 1}"] = dict(n=int(m.sum()), **{k: float(c[k][m, g].mean()) for k in c if k.startswith(("a_", "moved_"))},
                               tv=float(c["tv"][m, g].mean()), gap=float(gap[m, g].mean()), direct_adequate=float(X[m, g].float().mean()),
                               interface_adequate=float(Y[m, g].float().mean()), change_cells=float(dj[m, g].float().mean()))
        for lab, mm in (("direct_adequate", m & X[:, g]), ("direct_inadequate", m & ~X[:, g])):
            pg[f"G{g + 1}"][lab + "_a_interface"] = float(c["a_interface"][mm, g].mean()) if mm.any() else None
    res["per_goal"] = pg
    # pairs eligible under all three goals: the same pairs, goal by goal
    all3 = elig.all(1)
    res["all_three"] = dict(n=int(all3.sum()), **{f"G{g + 1}": {k: float(c[k][all3, g].mean()) for k in ("a_interface", "a_encoding", "a_direct")} for g in range(3)})
    f = lambda v: v[sel]
    covs = dict(direct_adequate=f(X).double(), tv=f(c["tv"]), dnorm=f(c["dnorm"]), gap=f(gap))
    covs_g = dict(covs, G2=f(goal == 1).double(), G3=f(goal == 2).double())
    covs_y = dict(covs, interface_adequate=f(Y).double())
    res["regression"] = {}
    for r in ("interface", "encoding", "direct"):
        y = f(c[f"a_{r}"])
        res["regression"][r] = dict(primary=within_pair_ols(y, covs, f(pair)), with_goal=within_pair_ols(y, covs_g, f(pair)),
                                    goal_only=within_pair_ols(y, dict(tv=f(c["tv"]), dnorm=f(c["dnorm"]), gap=f(gap), G2=f(goal == 1).double(), G3=f(goal == 2).double()), f(pair)),
                                    with_interface_adequacy=within_pair_ols(y, covs_y, f(pair)))
    # matched bins: within goal pairs inside the same pair, the goal with the inadequate direct route against the adequate one
    d_if = []
    for g1 in range(3):
        for g2 in range(3):
            if g1 != g2:
                m = sel[:, g1] & sel[:, g2] & ~X[:, g1] & X[:, g2]
                if m.any():
                    d_if.append((c["a_interface"][m, g1] - c["a_interface"][m, g2], c["tv"][m, g1] - c["tv"][m, g2], gap[m, g1] - gap[m, g2]))
    if d_if:
        dd = [torch.cat([x[i] for x in d_if]) for i in range(3)]
        close = (dd[1].abs() < 0.1) & (dd[2].abs() < 0.02)
        res["contrast"] = dict(n=int(len(dd[0])), diff_interface=float(dd[0].mean()), diff_tv=float(dd[1].mean()), diff_gap=float(dd[2].mean()),
                               n_matched=int(close.sum()), diff_interface_matched=float(dd[0][close].mean()) if close.any() else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    res = dict(runs={})
    for s in a.seeds:
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        gen = torch.Generator(device=dev); gen.manual_seed(2900 + s)
        itf = R28.Interface(net, ctx, gen)
        probes = Probes(net, ctx)
        row = dict(checkpoint=ck, probe_accuracy=probes.acc)
        for name, pr in (("main", pairs.main), ("onestep", pairs.onestep)):
            c, greedy = cells(net, ctx, itf, pr["r"], pr["d"])
            row[name] = analyse(c, greedy, ctx, probes, pr["r"], pr["d"])
        res["runs"][f"seed{s}"] = row
        m = row["main"]
        print(f"seed{s} probes " + " ".join(f"{k} {v:.2f}" for k, v in probes.acc.items()) + f" | both-check {m['both_check']:.1e} | a_interface by goal " +
              " ".join(f"{g} {m['per_goal'][g]['a_interface']:.2f}" for g in m["per_goal"]) + f" | β direct-adequate {m['regression']['interface']['primary']['direct_adequate']:.3f} | {time.time() - t0:.0f}s", flush=True)
        (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
