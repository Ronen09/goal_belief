"""The residual-features experiment: which superseded history features carry the residual's effect? The
observation-prediction experiment's reward models, frozen. Features, arms and decision rule: PLAN.md.

    .venv/bin/python studies/5_goal_belief_mechanism/residual_features/run.py                         # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/residual_features/run.py --untrained --seeds 0   # smoke test (initial checkpoint)

d(h) = x_b(h) + r(h) at the goal token's four attention outputs. Each feature F of the prefix tokens is centred within
its (posterior, L) group on fit-side histories; x_F = ridge regression of r on the centred feature. On informative
pairs, swap x_F(h') - x_F(h) against the same vector rotated; r's own swap is the reference.
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


BE = load("berun", ROUNDS / "5_goal_belief_mechanism" / "belief_error" / "run.py")
HM, ridge = BE.HM, BE.ridge
R36, R30, M26, outputs, table = BE.R36, BE.R30, BE.M26, BE.outputs, BE.table
MM = R30.MM
CHUNK, SEED, MIN_FIT = 4096, 48, HM.MIN_FIT


def features(t, ctx):
    """Raw (uncentred) feature matrices per history: {name: [nh, p]} and the groups."""
    tok, L = ctx.bank.tok, ctx.L
    P = t.max_prefix + 1
    nh = len(L)
    rows = torch.arange(nh, device=L.device)
    ns = t.n_sym + 1                                                                       # symbol tokens 1..n_sym, 0 = none
    oh = lambda x, n: torch.nn.functional.one_hot(x, n).float()
    F = {}
    for j in range(P):
        on = j <= L
        F[f"sym@{j}"] = oh(torch.where(on, tok[:, j, MM.F_SYM], 0), ns)
        if j >= 1:
            F[f"act@{j}"] = oh(torch.where(on, tok[:, j, MM.F_ACT], 0), 5)
    for k in range(P):
        pos = L - k
        on = pos >= 0
        F[f"sym-{k}"] = oh(torch.where(on, tok[rows, pos.clamp(min=0), MM.F_SYM], 0), ns)
        if k < P - 1:
            on = pos >= 1
            F[f"act-{k}"] = oh(torch.where(on, tok[rows, pos.clamp(min=0), MM.F_ACT], 0), 5)
    keep = (torch.arange(P, device=L.device)[None] <= L[:, None]).float()
    sym, act = tok[:, :P, MM.F_SYM], tok[:, :P, MM.F_ACT]
    F["counts"] = torch.cat([((sym == v).float() * keep).sum(1, keepdim=True) for v in range(1, ns)] +
                            [((act == v).float() * keep).sum(1, keepdim=True) for v in range(1, 5)], 1)
    F["landmark"] = ((sym == t.n_sym).float() * keep).amax(1, keepdim=True)
    F["sym_counts"], F["act_counts"] = F["counts"][:, : ns - 1], F["counts"][:, ns - 1:]
    singles = [k for k in F if k not in ("sym_counts", "act_counts")]
    groups = dict(absolute=[k for k in singles if "@" in k], recency=[k for k in singles if "-" in k], counts_group=["counts", "landmark"], all=singles)
    return F, singles, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--explore", action="store_true", help="post hoc (not in PLAN.md): counts split into symbol and move counts; writes explore.json")
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
    cells0 = R36.Cells(ctx)
    mixF = torch.cat([HM.R43.candidates(t, ctx)[:, HM.R43.CANDS.index(k)] for k in ("popt", "meanMDP")], 1)
    Fraw, singles, groups = features(t, ctx)
    Fc = {k: (v - table(pid, Lh, nL, v, fit)[0]).double() for k, v in Fraw.items()}         # centred within (posterior, L)
    names = ["counts", "sym_counts", "act_counts", "act-0"] if a.explore else singles + list(groups)
    Xf = {**{k: Fc[k] for k in Fc}, **{g: torch.cat([Fc[k] for k in ks], 1) for g, ks in groups.items()}}
    res = dict(features=names, runs={})
    out = HERE / ("smoke.json" if a.untrained else "explore.json" if a.explore else "results.json")
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
        R = (D - xb).double()
        ok = torch.stack([cnt[k][cells0.h] >= MIN_FIT for k in HM.TABLES]).all(0)
        ch, cg = cells0.h[ok], cells0.g[ok]
        rows = torch.arange(len(ch), device=dev)
        cL = Lh[ch]
        nat = lg_all[ch, cg].argmax(-1)
        donor = rows.clone()
        for l in range(nL):
            idx = torch.nonzero(cL == l).squeeze(1)
            if len(idx) > 1:
                donor[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
        dh = ch[donor]
        dnat = lg_all[dh, cg].argmax(-1)
        Pp = torch.nonzero((dh != ch) & (dnat != nat)).squeeze(1)
        h, g, hd, an = ch[Pp], cg[Pp], dh[Pp], nat[Pp]
        test_h = torch.unique(ch)

        def flips(vec):
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
                out_.append(net(R30.seq(ctx, hh, gg), patch={c: adder(vv[:, j * d:(j + 1) * d]) for j, c in enumerate(A)})[0][n, p].argmax(-1))
            return (torch.cat(out_) != an).float().mean().item()

        def rot(vec):
            z = torch.randn(vec.shape, device=dev, dtype=torch.float64, generator=gen)
            return z / z.norm(dim=-1, keepdim=True).clamp(min=1e-12) * vec.norm(dim=-1, keepdim=True)

        dr = R[hd] - R[h]
        ref = dict(swap=flips(dr), rot=flips(rot(dr)))
        ref["excess"] = ref["swap"] - ref["rot"]
        row = dict(checkpoint=ck, pairs=len(Pp), r=ref, features={})
        for k in names:
            X = Xf[k][:, Xf[k][fit].var(0) > 1e-10]                                         # columns the posterior does not fix
            if X.shape[1] == 0:                                                              # fully determined by the posterior
                row["features"][k] = dict(determined_by_posterior=True, r2=0.0, ridge=None, swap=None,
                                          rot=None, excess=0.0, share=0.0)
                continue
            W, lam = ridge(X, R, fit, gen)
            xF = X @ W
            vec = xF[hd] - xF[h]
            sw, ro = flips(vec), flips(rot(vec))
            row["features"][k] = dict(r2=float(1 - ((R[test_h] - xF[test_h]) ** 2).sum() / (R[test_h] ** 2).sum()), ridge=lam, swap=sw, rot=ro,
                                      excess=sw - ro, share=(sw - ro) / ref["excess"] if ref["excess"] != 0 else None)
        res["runs"][f"seed{s}"] = row
        top = sorted(((v["share"] or 0, k) for k, v in row["features"].items() if k not in groups), reverse=True)[:4]
        print(f"seed{s} {ck} pairs {len(Pp)} r: swap {ref['swap']:.3f} rot {ref['rot']:.3f} excess {ref['excess']:.3f} | groups: "
              + " ".join(f"{gname} R2 {row['features'][gname]['r2']:.2f} share {row['features'][gname]['share'] or 0:.2f}" for gname in groups if gname in row["features"])
              + " | best singles: " + ", ".join(f"{k} {v:.2f}" for v, k in top) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
