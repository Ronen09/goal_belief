"""Round 38: does swapping only the goal token's self value in block 0 move the goal x belief interaction at the MLP
input to the other goal's, while the shared belief component stays fixed? Round 23's reward models, frozen.

    .venv/bin/python rounds/r38_self_value_interaction/run.py            # writes results.json
    .venv/bin/python rounds/r38_self_value_interaction/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

Sites at the goal token: u = ln2(m), block 0's MLP input (primary); m, the residual stream before that layer norm; z,
the residual stream after block 0's MLP (round 32's site, where the goal x belief interaction is large).

Decomposition, from natural states on fit-side histories under all goals (b: posterior; L: prefix length):
    c[g, L]     mean state per goal and length;  cbar[L] = mean over goals
    S(b)        shared code: mean of (state - c[g, L]) per posterior, shrunk by n / (n + 30)
    S_g(b)      per posterior and goal, shrunk toward S(b)
    f(b, g) = c[g, L] + S_g(b),   f(b) = cbar[L] + S(b),   I(b, g) = f(b, g) - f(b) = M(g, L) + J(b, g)
    M(g, L) = c[g, L] - cbar[L]   the goal's main effect;    J(b, g) = S_g(b) - S(b)   the goal x belief part

Swaps (held-out histories h, ordered goal pairs g1 -> g2; block 0's attention output at the goal token recomputed):
    selfv   the goal token's own value from g2, all heads (round 37)          primary
    query   the goal token's query from g2 (round 36)                        comparison
    goal    the natural run under g2                                          reference (the full change)
Per swap and site: the state change D = x_swap - x_nat is regressed, pooled over histories and pairs, on
dM = M(g2, L) - M(g1, L) and dJ = J(b, g2) - J(b, g1): coefficients alpha_M, alpha_J (1 = moved fully). Shared belief:
a goal-free affine decoder of b from the state (fit side, all goals); the L1 change of the decoded posterior under the
swap, against the natural goal change and against the decoded difference between round 27's main pairs. Also the share
of the swap's change in the shared code's subspace (top 13 principal directions of S(b)).
"""

from __future__ import annotations

import argparse, importlib.util, json, math, sys, time
from pathlib import Path

import torch

from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R37 = load("r37run", ROUNDS / "r37_self_value" / "run.py")
R36, R30, M26 = R37.R36, R37.R30, R37.M26
R27 = R36.R27
CHUNK, SHRINK = 8192, 30
SITES = ("u", "m", "z")
SWAPS = ("selfv", "query", "goal")
GOAL_PAIRS = R36.GOAL_PAIRS


@torch.no_grad()
def sites(net, tok, p, attn=None):
    """States at the goal token (u, m, z), with block 0's attention output there replaced by `attn` if given."""
    n = torch.arange(len(tok), device=tok.device)
    patch = {}
    if attn is not None:
        def f(a):
            a = a.clone(); a[n, p] = attn
            return a
        patch[("attn", 0)] = f
    rec = net(tok, patch=patch, record=True)[2]
    m = rec["mid"][0][n, p]
    return dict(u=net.blocks[0].ln2(m), m=m, z=rec["resid"][1][n, p])


class Decomposition:
    def __init__(self, Z, ctx, pid):
        """Z [nh, goals, d] natural states at one site."""
        t, dev = ctx.t, ctx.t.dev
        Z = Z.double()
        nh, _, d = Z.shape
        fit = ctx.fit
        nL = t.max_prefix + 1
        c = torch.zeros(3, nL, d, dtype=torch.float64, device=dev)
        for g in range(3):
            for l in range(nL):
                mm = fit & (ctx.L == l)
                if mm.any():
                    c[g, l] = Z[mm, g].mean(0)
        self.c, self.cbar = c, c.mean(0)
        R = Z - c[:, ctx.L].permute(1, 0, 2)
        npost = int(pid.max()) + 1
        fh = torch.nonzero(fit).squeeze(1)
        def means(key, vals, nkey):
            s = torch.zeros(nkey, d, dtype=torch.float64, device=dev).index_add_(0, key, vals)
            return s, torch.bincount(key, minlength=nkey).double()[:, None]
        s, n = means(pid[fh].repeat_interleave(3), R[fh].reshape(-1, d), npost)
        self.S = s / (n + SHRINK)
        self.Sg = []
        for g in range(3):
            sg, ng = means(pid[fh], R[fh, g], npost)
            self.Sg.append((sg + SHRINK * self.S) / (ng + SHRINK))                         # shrunk toward the shared code
        self.Sg = torch.stack(self.Sg, 1)                                                   # [npost, goals, d]
        # the shared code's subspace: top principal directions of S(b) over posteriors, weighted by their counts
        w = n.squeeze(1)
        Sc = self.S - (w[:, None] * self.S).sum(0) / w.sum()
        C = (Sc * w[:, None]).T @ Sc / w.sum()
        self.P_S = torch.linalg.eigh(C)[1].flip(1)[:, :13]
        bel = t.belief[ctx.node].double()
        self.dec = Affine(Z[fit].reshape(-1, d), bel[fit][:, None].expand(-1, 3, -1).reshape(-1, bel.shape[1]))
        self.r2_decoder = float(1 - ((self.dec(Z[ctx.test].reshape(-1, d)) - bel[ctx.test][:, None].expand(-1, 3, -1).reshape(-1, bel.shape[1])) ** 2).sum()
                                / ((bel[ctx.test] - bel[ctx.test].mean(0)) ** 2).sum() / 3)

    def dM(self, g1, g2, L):
        return (self.c[g2, L] - self.cbar[L]) - (self.c[g1, L] - self.cbar[L])

    def dJ(self, b, g1, g2):
        J = self.Sg - self.S[:, None]
        return J[b, g2] - J[b, g1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    bel = t.belief[ctx.node].double()
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    all_h = torch.arange(len(ctx.L), device=dev)
    test = torch.nonzero(ctx.test).squeeze(1)
    res = dict(runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        nat = {x: torch.zeros(len(all_h), 3, net.d, device=dev) for x in SITES}
        for g in range(3):
            for i in range(0, len(all_h), CHUNK):
                h = all_h[i:i + CHUNK]
                st = sites(net, R30.seq(ctx, h, torch.full_like(h, g)), 1 + ctx.L[h])
                for x in SITES:
                    nat[x][i:i + CHUNK, g] = st[x]
        dec = {x: Decomposition(nat[x], ctx, pid) for x in SITES}
        # sufficient statistics of the pooled regression D ~ aM dM + aJ dJ, per swap and site
        G = {(k, x): torch.zeros(2, 2, dtype=torch.float64, device=dev) for k in SWAPS for x in SITES}
        r = {(k, x): torch.zeros(2, dtype=torch.float64, device=dev) for k in SWAPS for x in SITES}
        dd = {(k, x): 0.0 for k in SWAPS for x in SITES}
        ds = {(k, x): 0.0 for k in SWAPS for x in SITES}
        db = {(k, x): [] for k in SWAPS for x in SITES}
        sizes = {x: [0.0, 0.0] for x in SITES}
        for g1, g2 in GOAL_PAIRS:
            for i in range(0, len(test), CHUNK):
                h = test[i:i + CHUNK]
                p = 1 + ctx.L[h]
                t1, t2 = R30.seq(ctx, h, torch.full_like(h, g1)), R30.seq(ctx, h, torch.full_like(h, g2))
                sw = dict(selfv=sites(net, t1, p, R37.attn0(net, t1, t2, p, (), tuple(range(net.nh)))[0]),
                          query=sites(net, t1, p, R36.attn0(net, t1, p, q_src=t2)),
                          goal={x: nat[x][h, g2] for x in SITES})
                for x in SITES:
                    x0 = nat[x][h, g1].double()
                    dMx = dec[x].dM(g1, g2, ctx.L[h])
                    dJx = dec[x].dJ(pid[h], g1, g2)
                    X = torch.stack([dMx, dJx], -1)                                         # [n, d, 2]
                    sizes[x][0] += float((dMx ** 2).sum()); sizes[x][1] += float((dJx ** 2).sum())
                    b0 = dec[x].dec(x0)
                    for k in SWAPS:
                        D = sw[k][x].double() - x0
                        G[(k, x)] += torch.einsum("ndi,ndj->ij", X, X)
                        r[(k, x)] += torch.einsum("ndi,nd->i", X, D)
                        dd[(k, x)] += float((D ** 2).sum())
                        ds[(k, x)] += float(((D @ dec[x].P_S) ** 2).sum())
                        db[(k, x)].append((dec[x].dec(sw[k][x]) - b0).abs().sum(-1))
        row = dict(checkpoint=ck, decoder_r2={x: dec[x].r2_decoder for x in SITES}, sites={})
        # scale for the decoded posterior change: decoded difference between round 27's main pairs (natural, same goal)
        mp = pairs.main
        for x in SITES:
            ref = torch.cat([(dec[x].dec(nat[x][mp["d"], g]) - dec[x].dec(nat[x][mp["r"], g])).abs().sum(-1) for g in range(3)])
            row["sites"][x] = dict(size_main_effect=sizes[x][0], size_interaction=sizes[x][1], interaction_over_main=sizes[x][1] / max(sizes[x][0], 1e-30),
                                   decoded_pair_difference=float(ref.median()))
            for k in SWAPS:
                coef = torch.linalg.solve(G[(k, x)], r[(k, x)])
                fitted = float(coef @ G[(k, x)] @ coef)
                row["sites"][x][k] = dict(alpha_M=float(coef[0]), alpha_J=float(coef[1]), explained=fitted / max(dd[(k, x)], 1e-30),
                                          decoded_change=float(torch.cat(db[(k, x)]).median()), shared_subspace_share=ds[(k, x)] / max(dd[(k, x)], 1e-30),
                                          shared_subspace_size=ds[(k, x)], size=dd[(k, x)])
        res["runs"][f"seed{s}"] = row
        print(f"seed{s} {ck} " + " | ".join(f"{x}: J/M {row['sites'][x]['interaction_over_main']:.3f} " +
                                           " ".join(f"{k} aM {row['sites'][x][k]['alpha_M']:.2f} aJ {row['sites'][x][k]['alpha_J']:.2f} db {row['sites'][x][k]['decoded_change']:.3f}" for k in SWAPS) +
                                           f" (pairs db {row['sites'][x]['decoded_pair_difference']:.3f})" for x in SITES) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
