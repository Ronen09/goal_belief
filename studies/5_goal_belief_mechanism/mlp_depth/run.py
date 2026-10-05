"""The MLP-depth experiment: the goal x belief part J at every MLP of the goal token (blocks 0-3): its bilinear description, how much each
MLP makes itself, and whether removing it from one, several or all MLP outputs breaks the decision. The observation-prediction experiment's reward
models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/mlp_depth/run.py            # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/mlp_depth/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

Per block l, at the goal token: u_l = ln2_l(m_l), the MLP input; a_l its hidden units; y_l its output. The MLP-bilinear experiment's
posterior-level tables (c, S, S_g, M, J) for each, from natural fit-side states under all goals; the analysis set is the
posteriors with >= 30 fit-side histories, weighted by count; split-half ceiling for J_y.

Description of J_y,l (the MLP-bilinear experiment's): bilinear in B_l = S_ul(b) (top-13 principal coordinates) per goal; CP ranks 1-8.
Mechanism: the MLP applied to its input rebuilt from parts ubar = cbar_ul[L], B = S_ul(b), G = M_ul(g, L): the exact
interaction and its second-order term, against J_y,l. Also |J_u|^2 / |M_u|^2 at each input (what is inherited).

Ablations (the query-swap experiment's cells), the MLP outputs at the goal token y_l -> y_l - J_y,l(b, g), natural tables, fixed:
    single l (0, 1, 2, 3); blocks 1-3; all four; shuffled-all (y_l - J_y,l(b', g), b' a random other posterior, all
    four: a same-size disruption control). Optimal under g on goal-matters and goal-neutral cells.
"""

from __future__ import annotations

import argparse, importlib.util, json, math, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R39 = load("r39run", ROUNDS / "5_goal_belief_mechanism" / "mlp_bilinear" / "run.py")
R36, R30, M26 = R39.R36, R39.R30, R39.M26
CHUNK, MIN_N, RANKS, N_MECH, SEED = 8192, 30, (1, 2, 3, 4, 6, 8), 20000, 40
ABLATIONS = {"L0": (0,), "L1": (1,), "L2": (2,), "L3": (3,), "L1-3": (1, 2, 3), "all": (0, 1, 2, 3)}


@torch.no_grad()
def natural(net, ctx, hist):
    """Per layer: u, hidden a, output y at the goal token for histories under each goal: lists of [n, goals, ...]."""
    out = {k: [[[] for _ in range(3)] for _ in range(net.nl)] for k in "uay"}
    for g in range(3):
        for s in range(0, len(hist), CHUNK):
            h = hist[s:s + CHUNK]
            n = torch.arange(len(h), device=h.device)
            p = 1 + ctx.L[h]
            rec = net(R30.seq(ctx, h, torch.full_like(h, g)), record=True)[2]
            for l, b in enumerate(net.blocks):
                u = b.ln2(rec["mid"][l][n, p])
                out["u"][l][g].append(u); out["a"][l][g].append(b.mlp[1](b.mlp[0](u))); out["y"][l][g].append(rec["mlp"][l][n, p])
    return {k: [torch.stack([torch.cat(x) for x in v[l]], 1) for l in range(net.nl)] for k, v in out.items()}


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
    bel = t.belief[ctx.node].double()
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    npost = int(pid.max()) + 1
    fit = torch.nonzero(ctx.fit).squeeze(1)
    gen0 = torch.Generator(device=dev); gen0.manual_seed(SEED)
    half = torch.rand(len(fit), device=dev, generator=gen0) < 0.5
    post_bel = torch.zeros(npost, bel.shape[1], dtype=torch.float64, device=dev); post_bel[pid] = bel
    cells = R36.Cells(ctx)
    Gm = torch.tensor([[1, 0], [0, 1], [-1, -1]], dtype=torch.float64, device=dev)
    res = dict(cells=dict(goal_matters=int((cells.cls == 0).sum()), goal_neutral=int((cells.cls == 1).sum())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        nat = natural(net, ctx, fit)
        sub = torch.randperm(len(fit), device=dev, generator=gen)[:N_MECH]
        hs = fit[sub]
        L = ctx.L[hs]
        row = dict(checkpoint=ck, layers={})
        Jtabs = []
        for l, b in enumerate(net.blocks):
            U, A, Y = nat["u"][l], nat["a"][l], nat["y"][l]
            Tu, Ta, Ty = (R39.Table(X, fit, ctx, pid, npost) for X in (U, A, Y))
            Jtabs.append(Ty.J)
            Ty1, Ty2 = R39.Table(Y[half], fit[half], ctx, pid, npost), R39.Table(Y[~half], fit[~half], ctx, pid, npost)
            keep = Ty.n >= MIN_N
            w = Ty.n[keep]
            J = Ty.J[keep]
            wsum = lambda X: (w[:, None, None] * X).sum()
            rel = float(wsum(Ty1.J[keep] * Ty2.J[keep]) / (wsum(Ty1.J[keep] ** 2) * wsum(Ty2.J[keep] ** 2)).sqrt())
            ceiling = 2 * rel / (1 + rel)
            Su = Tu.S[keep]
            Suc = Su - (w[:, None] * Su).sum(0) / w.sum()
            FB = Suc @ torch.linalg.eigh((Suc * w[:, None]).T @ Suc / w.sum())[1].flip(1)[:, :13]
            bk = post_bel[keep]
            Fb = bk - (w[:, None] * bk).sum(0) / w.sum()
            desc = dict(bilinear_b=R39.wr2(J, R39.bilinear(J, Fb, w), w), bilinear_B=R39.wr2(J, R39.bilinear(J, FB, w), w))
            for r in RANKS:
                desc[f"cp_{r}"] = R39.wr2(J, R39.cp(J, FB, Gm, w, r, gen), w)
            # mechanism from the decomposed input
            W1, b1, W2 = b.mlp[0].weight.double(), b.mlp[0].bias.double(), b.mlp[2].weight.double()
            f_mlp = lambda x: b.mlp(x.float()).double()
            ubar, B = Tu.cbar[L], Tu.S[pid[hs]]
            g2 = R39.gelu2(ubar @ W1.T + b1)
            ex, ty = [], []
            for g in range(3):
                G = Tu.M(g, L)
                ex.append(f_mlp(ubar + B + G) - f_mlp(ubar + B) - f_mlp(ubar + G) + f_mlp(ubar))
                ty.append((g2 * (B @ W1.T) * (G @ W1.T)) @ W2.T)
            def jtab(X):
                Xc = X - X.mean(1, keepdim=True)
                s_ = torch.zeros(npost, 3, X.shape[-1], dtype=torch.float64, device=dev).index_add_(0, pid[hs], Xc)
                return (s_ / torch.bincount(pid[hs], minlength=npost).double().clamp(min=1)[:, None, None])[keep]
            Jex, Jty = jtab(torch.stack(ex, 1)), jtab(torch.stack(ty, 1))
            Jsub = jtab(Y[sub].double() - Ty.c[:, L].permute(1, 0, 2))
            # hidden units
            Ja = Ta.J[keep]
            e = (W2 ** 2).sum(0) * (w[:, None, None] * Ja ** 2).sum((0, 1))
            cum = torch.cumsum(torch.sort(e, descending=True).values, 0) / e.sum()
            row["layers"][f"L{l}"] = dict(n_posteriors=int(keep.sum()), ceiling=ceiling, describe=desc,
                                          sizes=dict(J_y=float(wsum(J ** 2) / w.sum()), M_y=float(((Ty.c - Ty.cbar[None]) ** 2).sum(-1).mean()),
                                                     J_u=float(wsum(Tu.J[keep] ** 2) / w.sum()), M_u=float(((Tu.c - Tu.cbar[None]) ** 2).sum(-1).mean()),
                                                     S_y=float((w[:, None] * (Ty.S[keep] - (w[:, None] * Ty.S[keep]).sum(0) / w.sum()) ** 2).sum() / w.sum())),
                                          mechanism=dict(exact_vs_Jy=R39.wr2(Jsub, Jex, w), taylor_vs_exact=R39.wr2(Jex, Jty, w), taylor_vs_Jy=R39.wr2(Jsub, Jty, w),
                                                         exact_size=float(wsum(Jex ** 2) / wsum(Jsub ** 2))),
                                          units=dict(n_for_half=int((cum < 0.5).sum()) + 1, n_for_80=int((cum < 0.8).sum()) + 1))
        del nat
        # ablations
        perm = torch.randperm(npost, device=dev, generator=gen)
        abl = dict(ABLATIONS, **{"shuffled_all": (0, 1, 2, 3)})
        acts = {k: [] for k in ("natural",) + tuple(abl)}
        for i in range(0, len(cells.h), CHUNK):
            h, g = cells.h[i:i + CHUNK], cells.g[i:i + CHUNK]
            n = torch.arange(len(h), device=dev)
            p = 1 + ctx.L[h]
            tok = R30.seq(ctx, h, g)
            acts["natural"].append(net(tok)[0][n, p].argmax(-1))
            for k, layers in abl.items():
                patch = {}
                for l in layers:
                    src = perm[pid[h]] if k == "shuffled_all" else pid[h]
                    delta = (-Jtabs[l][src, g]).float()
                    def f(m, delta=delta):
                        m = m.clone(); m[n, p] = m[n, p] + delta
                        return m
                    patch[("mlp", l)] = f
                acts[k].append(net(tok, patch=patch)[0][n, p].argmax(-1))
        Ac = {k: torch.cat(v) for k, v in acts.items()}
        rows_ = torch.arange(len(cells.h), device=dev)
        optg = ctx.opt[cells.h, cells.g]
        row["ablation"] = {name: {k: float(optg[rows_, Ac[k]][cells.cls == c].float().mean()) for k in Ac} for c, name in ((0, "goal_matters"), (1, "goal_neutral"))}
        res["runs"][f"seed{s}"] = row
        ab = row["ablation"]["goal_matters"]
        print(f"seed{s} {ck} " + " | ".join(f"L{l}: J/M {row['layers'][f'L{l}']['sizes']['J_y'] / max(row['layers'][f'L{l}']['sizes']['M_y'], 1e-12):.2f} "
                                           f"BL {row['layers'][f'L{l}']['describe']['bilinear_B']:.2f} cp4 {row['layers'][f'L{l}']['describe']['cp_4']:.2f} "
                                           f"X {row['layers'][f'L{l}']['mechanism']['exact_vs_Jy']:.2f} T {row['layers'][f'L{l}']['mechanism']['taylor_vs_exact']:.2f}" for l in range(net.nl)) +
              " | ablation goal-matters " + " ".join(f"{k} {v:.3f}" for k, v in ab.items()) + " | goal-neutral " +
              " ".join(f"{k} {v:.3f}" for k, v in row["ablation"]["goal_neutral"].items()) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
