"""The MLP-bilinear experiment: how does block 0's MLP compute the goal x belief part J(b, g)? Bilinear and low-rank descriptions, the
MLP's own second-order term, hidden units, and a causal test. The observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/mlp_bilinear/run.py            # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/mlp_bilinear/run.py --untrained    # the ten initial checkpoints (reference); writes untrained.json

At the goal token: u = ln2(m), the MLP input; a = GELU(W1 u + b1), its hidden units (512); y = W2 a + b2, its output.
Posterior-level tables from natural fit-side states under all goals (the self-value-interaction experiment's decomposition, per site):
    c[g, L], cbar[L], S(b), S_g(b) (shrunk toward S by n / (n + 30)), M(g, L) = c[g, L] - cbar[L], J(b, g) = S_g(b) - S(b)
Analysis set: posteriors with >= 30 fit-side histories, weighted by their counts; J_y's split-half reliability (two
random halves of the fit histories, Spearman-Brown) is the noise ceiling.

Descriptions of J_y (R² over the analysis set, raw and over the ceiling):
    bilinear_b      J_y(b, g) ~ W_g b                     linear in the posterior, per goal
    bilinear_B      J_y(b, g) ~ W_g B(b)                  B: the MLP input's shared belief code S_u(b) in its top-13 principal coordinates
    cp_r            J_y(b, g) ~ sum_k (u_k . B(b)) (v_k . G(g)) w_k,  G the centred goal one-hot; r = 1, 2, 3, 4, 6, 8 (ALS)
Mechanism (no fitted parameters), per fit-side history, from the decomposed input ubar = cbar_u[L], B = S_u(b), G = M_u(g, L):
    exact           f(ubar + B + G) - f(ubar + B) - f(ubar + G) + f(ubar)   the interaction the MLP makes from the two parts
    taylor          sum_i W2_i GELU''(W1_i ubar + b1_i) (W1_i . B)(W1_i . G)   its second-order (bilinear, one term per unit) part
    tables of both, their J, against J_y
Hidden units: J_a per unit; each unit's share of the energy sum_i |W2_i|^2 |J_a_i|^2; J_y rebuilt from the top-k units.
Causal (the query-swap experiment's cells): the MLP output at the goal token y -> y - J_y(b, g) + X, X in {0 (ablate), cp_r (r = 1, 2, 4),
bilinear_B, J_y of a random other posterior (shuffled)}; optimal under g on goal-matters and goal-neutral cells.
"""

from __future__ import annotations

import argparse, importlib.util, json, math, sys, time
from pathlib import Path

import torch

from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R36 = load("r36run", ROUNDS / "5_goal_belief_mechanism" / "query_swap" / "run.py")
R30, M26 = R36.R30, R36.M26
CHUNK, SHRINK, MIN_N, RANKS, CP_SEEDS, CP_ITERS, N_MECH, SEED = 8192, 30, 30, (1, 2, 3, 4, 6, 8), 5, 300, 20000, 39
CAUSAL_RANKS = (1, 2, 4)


def gelu2(x):
    """Second derivative of the exact GELU x Phi(x)."""
    phi = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
    return phi * (2 - x ** 2)


class Table:
    """c[g, L], S, S_g, J for one quantity, from per-history values X [n, goals, d] of the histories `rows`."""

    def __init__(self, X, rows, ctx, pid, npost):
        dev = X.device
        d = X.shape[-1]
        nL = ctx.t.max_prefix + 1
        L = ctx.L[rows]
        c = torch.zeros(3, nL, d, dtype=torch.float64, device=dev)
        for g in range(3):
            for l in range(nL):
                m = L == l
                if m.any():
                    c[g, l] = X[m, g].double().mean(0)
        self.c, self.cbar = c, c.mean(0)
        R = X.double() - c[:, L].permute(1, 0, 2)
        P = pid[rows]
        n = torch.bincount(P, minlength=npost).double()
        s = torch.zeros(npost, d, dtype=torch.float64, device=dev).index_add_(0, P, R.sum(1))
        self.S = s / (3 * n + SHRINK)[:, None]
        Sg = []
        for g in range(3):
            sg = torch.zeros(npost, d, dtype=torch.float64, device=dev).index_add_(0, P, R[:, g])
            Sg.append((sg + SHRINK * self.S) / (n + SHRINK)[:, None])
        self.Sg = torch.stack(Sg, 1)
        self.J = self.Sg - self.S[:, None]                                                     # [npost, goals, d]
        self.n = n

    def M(self, g, L):
        return self.c[g, L] - self.cbar[L]


def wr2(Y, P, w):
    """Weighted R² of predictions P for Y [N, goals, d] (no mean removal: J is centred), weights w [N]."""
    return float(1 - (w[:, None, None] * (Y - P) ** 2).sum() / (w[:, None, None] * Y ** 2).sum().clamp(min=1e-30))


def bilinear(Y, F, w):
    """Per goal, weighted least squares Y[:, g] ~ F W_g (F centred). Returns predictions."""
    sw = w.sqrt()[:, None]
    P = torch.zeros_like(Y)
    for g in range(3):
        W = torch.linalg.lstsq(F * sw, Y[:, g] * sw).solution
        P[:, g] = F @ W
    return P


def cp(Y, F, Gm, w, r, gen):
    """J ~ sum_k (F u_k)(Gm v_k) w_k by weighted alternating least squares; F [N, p], Gm [3, q], Y [N, 3, d]. Best of
    CP_SEEDS random starts. Returns predictions [N, 3, d]."""
    N, p = F.shape
    q, d = Gm.shape[1], Y.shape[-1]
    sw3 = w.sqrt().repeat_interleave(3)[:, None]
    eye = lambda k: 1e-9 * torch.eye(k, dtype=torch.float64, device=Y.device)
    best, bestP = -1e9, None
    for _ in range(CP_SEEDS):
        U = torch.randn(p, r, dtype=torch.float64, device=Y.device, generator=gen)
        V = torch.randn(q, r, dtype=torch.float64, device=Y.device, generator=gen)
        for _ in range(CP_ITERS):
            a, bg = F @ U, Gm @ V                                                            # [N, r], [3, r]
            # w step: Y[n, g] ~ sum_k a[n, k] bg[g, k] W[k]
            Z = (a[:, None, :] * bg[None]).reshape(N * 3, r)
            Wk = torch.linalg.lstsq(Z * sw3, Y.reshape(N * 3, d) * sw3).solution
            # u step: linear in U: Y[n, g] ~ sum_{i,k} F[n, i] U[i, k] T[g, k]
            T = torch.einsum("gk,kd->gkd", bg, Wk)
            A = torch.einsum("n,ni,gkd,nj,gld->ikjl", w, F, T, F, T).reshape(p * r, p * r)
            bv = torch.einsum("n,ni,gkd,ngd->ik", w, F, T, Y).reshape(p * r)
            U = torch.linalg.solve(A + eye(p * r), bv).view(p, r)
            # v step: linear in V: Y[n, g] ~ sum_{i,k} Gm[g, i] V[i, k] T2[n, k]
            T2 = torch.einsum("nk,kd->nkd", F @ U, Wk)
            A = torch.einsum("gi,gj,n,nkd,nld->ikjl", Gm, Gm, w, T2, T2).reshape(q * r, q * r)
            bv = torch.einsum("n,gi,nkd,ngd->ik", w, Gm, T2, Y).reshape(q * r)
            V = torch.linalg.solve(A + eye(q * r), bv).view(q, r)
        a, bg = F @ U, Gm @ V
        Z = (a[:, None, :] * bg[None]).reshape(N * 3, r)
        Wk = torch.linalg.lstsq(Z * sw3, Y.reshape(N * 3, d) * sw3).solution
        P = torch.einsum("nk,gk,kd->ngd", a, bg, Wk)
        sc = wr2(Y, P, w)
        if sc > best:
            best, bestP = sc, P
    return bestP


@torch.no_grad()
def natural(net, ctx, hist):
    """u, hidden a, output y at the goal token for histories under each goal: [n, goals, ...]."""
    b0 = net.blocks[0]
    U, A, Y = [], [], []
    for g in range(3):
        u_, a_, y_ = [], [], []
        for s in range(0, len(hist), CHUNK):
            h = hist[s:s + CHUNK]
            n = torch.arange(len(h), device=h.device)
            rec = net(R30.seq(ctx, h, torch.full_like(h, g)), record=True)[2]
            u = b0.ln2(rec["mid"][0][n, 1 + ctx.L[h]])
            pre = b0.mlp[0](u)
            u_.append(u); a_.append(b0.mlp[1](pre)); y_.append(rec["mlp"][0][n, 1 + ctx.L[h]])
        U.append(torch.cat(u_)); A.append(torch.cat(a_)); Y.append(torch.cat(y_))
    return torch.stack(U, 1), torch.stack(A, 1), torch.stack(Y, 1)


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
    res = dict(cells=dict(goal_matters=int((cells.cls == 0).sum()), goal_neutral=int((cells.cls == 1).sum())), runs={})
    out = HERE / ("untrained.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        b0 = net.blocks[0]
        W1, b1, W2 = b0.mlp[0].weight.double(), b0.mlp[0].bias.double(), b0.mlp[2].weight.double()
        f_mlp = lambda x: b0.mlp(x.float()).double()
        U, A, Y = natural(net, ctx, fit)
        Tu, Ta, Ty = (Table(X, fit, ctx, pid, npost) for X in (U, A, Y))
        Ty1, Ty2 = Table(Y[half], fit[half], ctx, pid, npost), Table(Y[~half], fit[~half], ctx, pid, npost)
        keep = Ty.n >= MIN_N
        w = Ty.n[keep]
        J = Ty.J[keep]
        rel_r = float((w[:, None, None] * Ty1.J[keep] * Ty2.J[keep]).sum() / ((w[:, None, None] * Ty1.J[keep] ** 2).sum() * (w[:, None, None] * Ty2.J[keep] ** 2).sum()).sqrt())
        ceiling = 2 * rel_r / (1 + rel_r)
        row = dict(checkpoint=ck, n_posteriors=int(keep.sum()), reliability=rel_r, ceiling=ceiling,
                   sizes=dict(J_y=float((w[:, None, None] * J ** 2).sum() / w.sum()), M_y=float(((Ty.c - Ty.cbar[None]) ** 2).sum(-1).mean()),
                              J_u=float((w[:, None, None] * Tu.J[keep] ** 2).sum() / w.sum()), M_u=float(((Tu.c - Tu.cbar[None]) ** 2).sum(-1).mean())))
        # belief features: the posterior (centred) and the MLP input's shared code in its top-13 principal coordinates
        bk = post_bel[keep]
        Fb = bk - (w[:, None] * bk).sum(0) / w.sum()
        Su = Tu.S[keep]
        Suc = Su - (w[:, None] * Su).sum(0) / w.sum()
        evec = torch.linalg.eigh((Suc * w[:, None]).T @ Suc / w.sum())[1].flip(1)[:, :13]
        FB = Suc @ evec
        Gm = torch.tensor([[1, 0], [0, 1], [-1, -1]], dtype=torch.float64, device=dev)          # centred goal basis
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        desc = dict(bilinear_b=wr2(J, bilinear(J, Fb, w), w), bilinear_B=wr2(J, bilinear(J, FB, w), w))
        cps = {}
        for r in RANKS:
            P = cp(J, FB, Gm, w, r, gen)
            cps[r] = P
            desc[f"cp_{r}"] = wr2(J, P, w)
        row["describe"] = desc
        # mechanism: the MLP applied to the decomposed input, per fit-side history (a random subset)
        sub = torch.randperm(len(fit), device=dev, generator=gen)[:N_MECH]
        hs = fit[sub]
        L = ctx.L[hs]
        ubar = Tu.cbar[L]
        B = Tu.S[pid[hs]]
        ex, ty = [], []
        a0 = ubar @ W1.T + b1
        g2 = gelu2(a0)
        for g in range(3):
            G = Tu.M(g, L)
            ex.append(f_mlp(ubar + B + G) - f_mlp(ubar + B) - f_mlp(ubar + G) + f_mlp(ubar))
            ty.append(((g2 * (B @ W1.T) * (G @ W1.T)) @ W2.T))
        EX, TY = torch.stack(ex, 1), torch.stack(ty, 1)
        def jtab(X):
            Xc = X - X.mean(1, keepdim=True)                                                  # goal x history part per history
            Pp = pid[hs]
            s_ = torch.zeros(npost, 3, X.shape[-1], dtype=torch.float64, device=dev).index_add_(0, Pp, Xc)
            n_ = torch.bincount(Pp, minlength=npost).double()
            return (s_ / n_.clamp(min=1)[:, None, None])[keep]
        Jex, Jty = jtab(EX), jtab(TY)
        # J_y restricted to the same histories, for a like-for-like comparison
        Jy_sub = jtab(Y[sub].double() - Ty.c[:, L].permute(1, 0, 2))
        row["mechanism"] = dict(exact_vs_Jy=wr2(Jy_sub, Jex, w), taylor_vs_Jy=wr2(Jy_sub, Jty, w), taylor_vs_exact=wr2(Jex, Jty, w),
                                exact_size=float((w[:, None, None] * Jex ** 2).sum() / (w[:, None, None] * Jy_sub ** 2).sum()),
                                Jy_sub_vs_Jy=wr2(J, Jy_sub, w))
        # hidden units: energy shares and J_y rebuilt from the top-k units
        Ja = Ta.J[keep]                                                                     # [N, 3, 512]
        e = (W2 ** 2).sum(0) * (w[:, None, None] * Ja ** 2).sum((0, 1))
        order = torch.argsort(e, descending=True)
        cum = torch.cumsum(e[order], 0) / e.sum()
        units = {}
        for k in (5, 10, 20, 50, 100):
            idx = order[:k]
            units[f"top{k}_energy"] = float(cum[k - 1])
            units[f"top{k}_r2"] = wr2(J, Ja[..., idx] @ W2[:, idx].T, w)
        units["n_for_half"] = int((cum < 0.5).sum()) + 1
        units["n_for_80"] = int((cum < 0.8).sum()) + 1
        row["units"] = units
        # causal: replace J_y in the MLP output at the goal token
        Jfull = Ty.J
        repl = {"ablate": torch.zeros_like(Jfull), "bilinear_B": None}
        def scatter(Pk):
            X = torch.zeros_like(Jfull); X[keep] = Pk
            return X
        for r in CAUSAL_RANKS:
            repl[f"cp_{r}"] = scatter(cps[r])
        repl["bilinear_B"] = scatter(bilinear(J, FB, w))
        perm = torch.randperm(npost, device=dev, generator=gen)
        repl["shuffled"] = Jfull[perm]
        acts = {k: [] for k in ("natural",) + tuple(repl)}
        for i in range(0, len(cells.h), CHUNK):
            h, g = cells.h[i:i + CHUNK], cells.g[i:i + CHUNK]
            n = torch.arange(len(h), device=dev)
            p = 1 + ctx.L[h]
            tok = R30.seq(ctx, h, g)
            acts["natural"].append(net(tok)[0][n, p].argmax(-1))
            for k, X in repl.items():
                delta = (X[pid[h], g] - Jfull[pid[h], g]).float()
                def f(m, delta=delta):
                    m = m.clone(); m[n, p] = m[n, p] + delta
                    return m
                acts[k].append(net(tok, patch={("mlp", 0): f})[0][n, p].argmax(-1))
        Ac = {k: torch.cat(v) for k, v in acts.items()}
        rows_ = torch.arange(len(cells.h), device=dev)
        optg = ctx.opt[cells.h, cells.g]
        row["causal"] = {name: {k: float(optg[rows_, Ac[k]][cells.cls == c].float().mean()) for k in Ac} for c, name in ((0, "goal_matters"), (1, "goal_neutral"))}
        res["runs"][f"seed{s}"] = row
        cg = row["causal"]["goal_matters"]
        print(f"seed{s} {ck} ceiling {ceiling:.2f} | J_y/M_y {row['sizes']['J_y'] / max(row['sizes']['M_y'], 1e-12):.3f} | describe " + " ".join(f"{k} {v:.2f}" for k, v in desc.items()) +
              " | mechanism " + " ".join(f"{k} {v:.2f}" for k, v in row["mechanism"].items()) + f" | units half {units['n_for_half']} 80% {units['n_for_80']} top20 r2 {units['top20_r2']:.2f}" +
              " | causal goal-matters " + " ".join(f"{k} {v:.3f}" for k, v in cg.items()) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
