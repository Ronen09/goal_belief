"""The belief-error experiment: does the residual r beyond the posterior act as a misplaced belief? The
observation-prediction experiment's reward models, frozen. Subspaces, arms and decision rule: PLAN.md.

    .venv/bin/python studies/5_goal_belief_mechanism/belief_error/run.py                         # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/belief_error/run.py --untrained --seeds 0   # smoke test (initial checkpoint)

d(h) = x_b(h) + r(h) at the goal token's four attention outputs (concatenated), x_b the fit-side table over (posterior, L).
S: top principal components of x_b over fit-side histories (95 % of its variance; and k = 13). On informative pairs
the real arms add r(h') - r(h), or its projection onto S or off S; each has a rotated control of the same size in the
same space. A ridge decoder from d to the exact posterior gives the decoded error e = b_hat - b.
"""

from __future__ import annotations

import argparse, importlib.util, json, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RT = load("rtrun", ROUNDS / "5_goal_belief_mechanism" / "residual_trace" / "run.py")
HM, R42 = RT.HM, RT.R42
R36, R30, M26, outputs, table, quart = RT.R36, RT.R30, RT.M26, RT.outputs, RT.table, RT.quart
CHUNK, SEED, MIN_FIT = 4096, 47, HM.MIN_FIT
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1, 10)


def ridge(X, Y, fit, gen):
    """Ridge from X to Y on fit rows; lambda (relative to the mean feature variance) chosen on a 10 % hold-out."""
    idx = fit[torch.randperm(len(fit), device=fit.device, generator=gen)]
    nv = len(idx) // 10
    va, tr = idx[:nv], idx[nv:]
    def solve(rows, lam):
        A = X[rows].T @ X[rows]
        s = lam * torch.trace(A) / A.shape[0]
        return torch.linalg.solve(A + s * torch.eye(A.shape[0], device=X.device, dtype=X.dtype), X[rows].T @ Y[rows])
    best = min(LAMBDAS, key=lambda lam: float(((X[va] @ solve(tr, lam) - Y[va]) ** 2).sum()))
    return solve(fit, best), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    nh, nL = len(ctx.L), t.max_prefix + 1
    allh = torch.arange(nh, device=dev)
    fit = torch.nonzero(ctx.fit).squeeze(1)
    Lh = ctx.L
    bel = t.belief[ctx.node].double()
    pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)[1]
    ent = -(bel * torch.log(bel.clamp(min=1e-300))).sum(-1)
    G42 = json.load(open(ROUNDS / "5_goal_belief_mechanism" / "additive_code" / "results.json"))["runs"]
    cells0 = R36.Cells(ctx)
    mixF = torch.cat([HM.R43.candidates(t, ctx)[:, HM.R43.CANDS.index(k)] for k in ("popt", "meanMDP")], 1)
    res = dict(runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        A = [("attn", l) for l in range(net.nl)]
        d = net.d
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        xs, lgs = [], []
        for g in range(3):
            xa, la = [], []
            for i in range(0, nh, 8192):
                h = allh[i:i + 8192]
                o, lg = outputs(net, ctx, h, torch.full_like(h, g))
                xa.append(torch.cat([o[c] for c in A], 1)); la.append(lg)
            xs.append(torch.cat(xa)); lgs.append(torch.cat(la))
        xbar = torch.stack(xs, 1).mean(1)
        lg_all = torch.stack(lgs, 1)
        del xs, lgs
        cntL = torch.bincount(Lh[fit], minlength=nL).clamp(min=1).float()[:, None]
        Xb = torch.zeros(nL, xbar.shape[1], device=dev).index_add_(0, Lh[fit], xbar[fit]) / cntL
        D = xbar - Xb[Lh]
        H = (lg_all.double() - lg_all.double().mean(-1, keepdim=True)).mean(1)
        order = H.argsort(-1, descending=True)
        feats = dict(top=order[:, 0], rank2=order[:, 0] * 4 + order[:, 1], mix=HM.keys_of(mixF), b=pid)
        cnt = {k: table(feats[k], Lh, nL, D, fit)[1] for k in HM.TABLES}
        xb, _ = table(pid, Lh, nL, D, fit)
        Hb, _ = table(pid, Lh, nL, H.float(), fit)
        R = D - xb
        # belief subspace S from x_b's variation over fit-side histories
        Xf = xb[fit].double()
        ev, U = torch.linalg.eigh(Xf.T @ Xf)
        ev, U = ev.flip(0), U.flip(1)
        cum = torch.cumsum(ev, 0) / ev.sum()
        k95 = int((cum < 0.95).sum()) + 1
        # decoder: d -> exact posterior (per-length means removed from both)
        bm = torch.zeros(nL, bel.shape[1], device=dev, dtype=torch.float64).index_add_(0, Lh[fit], bel[fit]) / cntL.double()
        W, lam = ridge(D.double(), bel - bm[Lh], fit, gen)
        bhat = D.double() @ W + bm[Lh]
        e = bhat - bel
        # cells, donors, informative pairs (as the residual-trace experiment, new draw)
        ok = torch.stack([cnt[k][cells0.h] >= MIN_FIT for k in HM.TABLES]).all(0)
        ch, cg = cells0.h[ok], cells0.g[ok]
        N = len(ch)
        rows = torch.arange(N, device=dev)
        cL = Lh[ch]
        test_h = torch.unique(ch)
        r2_dec = float(1 - ((bhat[test_h] - bel[test_h]) ** 2).sum() / ((bel[test_h] - bm[Lh[test_h]]) ** 2).sum())
        nat = lg_all[ch, cg].argmax(-1)
        donor = rows.clone()
        for l in range(nL):
            idx = torch.nonzero(cL == l).squeeze(1)
            if len(idx) > 1:
                donor[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
        dh = ch[donor]
        dnat = lg_all[dh, cg].argmax(-1)
        P = torch.nonzero((dh != ch) & (dnat != nat)).squeeze(1)
        h, g, hd = ch[P], cg[P], dh[P]
        an, ad = nat[P], dnat[P]
        n_ = torch.arange(len(h), device=dev)
        delta = (R[hd] - R[h]).double()

        def rand_in(B, like):
            """Random vectors in the column space of B (orthonormal), each the size of the matching row of like."""
            z = torch.randn(len(like), B.shape[1], device=dev, dtype=torch.float64, generator=gen) @ B.T
            return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * like.norm(dim=-1, keepdim=True)

        def logits(vec):
            vec = vec.float()
            out_ = []
            for i in range(0, len(h), CHUNK):
                hh, gg, vv = h[i:i + CHUNK], g[i:i + CHUNK], vec[i:i + CHUNK]
                n = torch.arange(len(hh), device=dev)
                p = 1 + Lh[hh]

                def adder(dv):
                    def f(z):
                        z = z.clone(); z[n, p] = z[n, p] + dv
                        return z
                    return f
                out_.append(net(R30.seq(ctx, hh, gg), patch={c: adder(vv[:, j * d:(j + 1) * d]) for j, c in enumerate(A)})[0][n, p])
            return torch.cat(out_)

        l0 = lg_all[h, g]
        gap = lambda l: l[n_, ad] - l[n_, an]
        row = dict(checkpoint=ck, pairs=len(P), k95=k95, decoder=dict(r2=r2_dec, ridge=lam), subspaces={})
        full = torch.eye(delta.shape[1], device=dev, dtype=torch.float64)
        results = {}
        for kname, k in (("k95", k95), ("k13", 13)):
            Us = U[:, :k]
            par = delta @ Us @ Us.T
            perp = delta - par
            Uc = torch.linalg.qr(full - Us @ Us.T)[0][:, : delta.shape[1] - k]                  # orthonormal basis of S's complement
            arms = dict(real_r=delta, real_par=par, real_perp=perp, rot_r=rand_in(full, delta), rot_par=rand_in(Us, par), rot_perp=rand_in(Uc, perp))
            fl, sh = {}, {}
            for name, vec in arms.items():
                lg = logits(vec)
                fl[name] = (lg.argmax(-1) != an).float()
                sh[name] = gap(lg) - gap(l0)
            rv = (R[test_h].double() @ Us).pow(2).sum() / R[test_h].double().pow(2).sum()
            row["subspaces"][kname] = dict(k=k, r_var_share_in_S=float(rv), xb_var_share=float(cum[k - 1]),
                                          flip={n: float(v.mean()) for n, v in fl.items()}, shift={n: float(v.mean()) for n, v in sh.items()},
                                          excess_flip={n: float((fl[f"real_{n}"] - fl[f"rot_{n}"]).mean()) for n in ("r", "par", "perp")})
            results[kname] = (fl, sh)
        # predictive: the residual-trace experiment's joint fit, with and without the decoded error's change
        fl, sh = results["k95"]
        y = (sh["real_r"] - sh["rot_r"]).double()
        Gt = torch.tensor(G42[f"seed{s}"]["G"], device=dev, dtype=torch.float64)[:, Lh[h]].permute(1, 0, 2) if not a.untrained else torch.zeros(len(h), 3, 4, device=dev, dtype=torch.float64)
        unsolv = R42.margin(ctx.Q[h].argmax(-1), Gt) <= 0
        inter = (H[h][:, None] + Gt).argmax(-1)[n_, g] != an
        margin = l0[n_, an] - l0[n_, ad]
        herr = (H[h] - Hb[h].double()).norm(dim=-1)
        etv = 0.5 * (e[hd] - e[h]).abs().sum(-1)
        z = lambda x: (x.double() - x.double().mean()) / x.double().std().clamp(min=1e-12)
        base = [torch.ones(len(h), device=dev, dtype=torch.float64), z(margin), z(unsolv), z(inter), z(g == 1), z(g == 2), z(ent[h]), z(herr), z(delta.norm(dim=-1))]
        names = ["intercept", "margin", "unsolvable", "interaction", "goal_G2", "goal_G3", "entropy", "H_error", "r_norm"]
        lsq = lambda X: torch.linalg.lstsq(X.cpu(), y.cpu()[:, None], driver="gelsd").solution.squeeze(1).tolist()
        row["joint_without"] = dict(zip(names, lsq(torch.stack(base, 1))))
        row["joint_with"] = dict(zip(names + ["decoded_error"], lsq(torch.stack(base + [z(etv)], 1))))
        row["decoded_error_tv"] = dict(mean=float((0.5 * e[test_h].abs().sum(-1)).mean()),
                                       corr_with_entropy=float(torch.corrcoef(torch.stack([0.5 * e[test_h].abs().sum(-1), ent[test_h]]))[0, 1]))
        res["runs"][f"seed{s}"] = row
        print(f"seed{s} {ck} pairs {len(P)} k95 {k95} decoder R2 {r2_dec:.3f} (ridge {lam}) | "
              + " | ".join(f"{kn}: r var in S {v['r_var_share_in_S']:.3f} excess {', '.join(f'{n} {x:.3f}' for n, x in v['excess_flip'].items())}" for kn, v in row["subspaces"].items())
              + f" | entropy coef {row['joint_without']['entropy']:.3f} -> {row['joint_with']['entropy']:.3f}, decoded error {row['joint_with']['decoded_error']:.3f} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
