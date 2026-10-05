"""The nonlinear-belief-edit experiment: nonlinear encodings of the posterior at the attention-belief-edit experiment's site (the goal token before block 0's MLP), and edits
with them. The observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/nonlinear_belief_edit/run.py            # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/nonlinear_belief_edit/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

Site m(h, g): ("resid_mid", 0) at the goal token (attention belief edit). Encodings, fitted on the fit side under all three goals,
without action labels (b: the exact goal-free posterior; c_{g,L}: offsets per goal and prefix length):

    linear       m ~ c_{g,L} + E b                       the attention-belief-edit experiment's shared encoding
    mlp          m ~ c_{g,L} + f(b)                      f a small MLP, one for every goal (primary)
    mlp_goal     m ~ c_{g,L} + f(b, g)                   goal as an input (secondary)
    table        m ~ c_{g,L} + mu(b)                     the mean for each distinct posterior (shrunk toward f where
                                                         a posterior is rare): the most any function of b can give
    table_goal   m ~ c_{g,L} + mu_g(b)                   per posterior and goal

Edits at the goal token only, the recipient's prefix states kept (the direct-belief-edit experiment's machinery):

    none, whole, pca      as the direct-belief-edit to attention-belief-edit experiments
    linear      m(A,g) + E (b_B - b_A)
    mlp         m(A,g) + f(b_B) - f(b_A)                 the same vector under every goal
    mlp_goal    m(A,g) + f(b_B, g) - f(b_A, g)
    table       m(A,g) + mu(b_B) - mu(b_A)
    table_goal  m(A,g) + mu_g(b_B) - mu_g(b_A)
    rotated     m(A,g) + Q (f(b_B) - f(b_A))             Q a random rotation (5 draws)
    residual    m(B,g) - (mu(b_B) - mu(b_A))             the donor's difference without its posterior part
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R33 = load("r33run", ROUNDS / "5_goal_belief_mechanism" / "attention_belief_edit" / "run.py")
R32, R27, M26 = R33.R32, R33.R27, R33.M26
KINDS = ("none", "whole", "linear", "mlp", "mlp_goal", "table", "table_goal", "rotated", "pca", "residual")
HIDDEN, STEPS, BATCH, LR, SHRINK = 256, 4000, 8192, 1e-3, 30


class F(nn.Module):
    """c_{g,L} + f(b) or c_{g,L} + f(b, g)."""

    def __init__(self, nb, d, n_off, goal_input):
        super().__init__()
        self.goal_input = goal_input
        self.f = nn.Sequential(nn.Linear(nb + (3 if goal_input else 0), HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, d))
        self.c = nn.Parameter(torch.zeros(n_off, d))

    def code(self, b, g=None):
        x = torch.cat([b, nn.functional.one_hot(g, 3).float()], 1) if self.goal_input else b
        return self.f(x)

    def forward(self, b, g, off):
        return self.c[off] + self.code(b, g)


def fit_f(b, g, off, Y, n_off, goal_input, seed):
    torch.manual_seed(seed)
    net = F(b.shape[1], Y.shape[1], n_off, goal_input).to(Y.device)
    net.c.data = torch.stack([Y[off == k].mean(0) if (off == k).any() else Y.mean(0) for k in range(n_off)])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    gen = torch.Generator(device=Y.device); gen.manual_seed(seed)
    for _ in range(STEPS):
        i = torch.randint(len(Y), (BATCH,), device=Y.device, generator=gen)
        loss = ((net(b[i], g[i], off[i]) - Y[i]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net.eval()


class Nonlinear:
    """The attention-belief-edit experiment's linear encoding and subspaces (R32.Encoding at the attention-belief-edit experiment's site), plus the nonlinear encodings."""

    def __init__(self, net, ctx, gen, seed):
        R32.SITE = R33.SITE
        self.lin = lin = R32.Encoding(net, ctx, gen)
        t, dev = ctx.t, ctx.t.dev
        self.Z, self.bel = lin.Z, lin.bel
        nh = len(ctx.L)
        _, self.pid = torch.unique(torch.round(self.bel * 1e9), dim=0, return_inverse=True)
        npost = int(self.pid.max()) + 1
        fit, test = torch.nonzero(ctx.fit).squeeze(1), torch.nonzero(ctx.test).squeeze(1)
        n_off = 3 * (t.max_prefix + 1)
        rows = lambda h: (h.repeat_interleave(3), torch.arange(3, device=dev).repeat(len(h)))
        hf, gf = rows(fit)
        off = lambda h, g: g * (t.max_prefix + 1) + ctx.L[h]
        Yf = self.Z[hf, gf].float()
        bf = self.bel[hf].float()
        self.f = fit_f(bf, gf, off(hf, gf), Yf, n_off, False, seed)
        self.fg = fit_f(bf, gf, off(hf, gf), Yf, n_off, True, seed + 1)
        with torch.no_grad():
            allb = self.bel.float()
            # codes for every history (f does not depend on the goal; f_g per goal)
            self.code = self.f.code(allb).double()                                               # [nh, d]
            self.code_g = torch.stack([self.fg.code(allb, torch.full((nh,), g, device=dev)).double() for g in range(3)], 1)   # [nh, 3, d]
            # tables: the mean residual per posterior (and goal), shrunk toward the MLP where the posterior is rare
            def table(res, key, nkey):
                s = torch.zeros(nkey, res.shape[1], dtype=torch.float64, device=dev).index_add_(0, key, res)
                n = torch.bincount(key, minlength=nkey).double()[:, None]
                return s / (n + SHRINK), n.squeeze(1)
            res = Yf.double() - self.f(bf, gf, off(hf, gf)).double()
            tab, self.table_n = table(res, self.pid[hf], npost)
            self.mu = self.code + tab[self.pid]                                                  # [nh, d]
            resg = Yf.double() - self.fg(bf, gf, off(hf, gf)).double()
            tabg, _ = table(resg, self.pid[hf] * 3 + gf, npost * 3)
            self.mu_g = self.code_g + tabg[self.pid[:, None] * 3 + torch.arange(3, device=dev)[None]]   # [nh, 3, d]
            # held-out fit: R² overall and within (goal, length), every test history under every goal
            ht, gt = rows(test)
            Yt = self.Z[ht, gt]
            grp = off(ht, gt)
            cf, cfg = self.f.c.double(), self.fg.c.double()
            preds = dict(mlp=cf[grp] + self.code[ht], mlp_goal=cfg[grp] + self.code_g[ht, gt], table=cf[grp] + self.mu[ht], table_goal=cfg[grp] + self.mu_g[ht, gt])
        self.fit_stats = dict(linear=lin.fit_stats["shared"], goal_specific_linear=lin.fit_stats["goal_specific"],
                              **{k: dict(r2=R32.r2(Yt, p), r2_within_goal_length=R32.r2(Yt, p, grp)) for k, p in preds.items()},
                              posteriors=npost, posteriors_unseen=int((self.table_n == 0).sum()))
        self.P_PCA, self.Q = lin.P_PCA, lin.Q

    def edits(self, a, b, g):
        zA, zB = self.Z[a, g], self.Z[b, g]
        e_mlp = self.code[b] - self.code[a]
        e_tab = self.mu[b] - self.mu[a]
        diff = zB - zA
        return dict(none=zA, whole=zB, linear=zA + (self.bel[b] - self.bel[a]) @ self.lin.E, mlp=zA + e_mlp,
                    mlp_goal=zA + self.code_g[b, g] - self.code_g[a, g], table=zA + e_tab, table_goal=zA + self.mu_g[b, g] - self.mu_g[a, g],
                    rotated=[zA + e_mlp @ Q for Q in self.Q], pca=zA + (diff @ self.P_PCA) @ self.P_PCA.T, residual=zB - e_tab)

    def stats(self, a, b):
        """Each edit's vector against the actual difference at the site, all goals pooled."""
        diff = torch.cat([self.Z[b, g] - self.Z[a, g] for g in range(3)])
        vec = dict(linear=lambda g: (self.bel[b] - self.bel[a]) @ self.lin.E, mlp=lambda g: self.code[b] - self.code[a],
                   mlp_goal=lambda g: self.code_g[b, g] - self.code_g[a, g], table=lambda g: self.mu[b] - self.mu[a],
                   table_goal=lambda g: self.mu_g[b, g] - self.mu_g[a, g])
        out = {}
        for k, fn in vec.items():
            e = torch.cat([fn(g) for g in range(3)])
            out[k] = dict(norm_ratio=float((e.norm(dim=1) / diff.norm(dim=1).clamp(min=1e-9)).median()),
                          cosine=float(torch.nn.functional.cosine_similarity(e, diff, dim=1).mean()),
                          residual_share=float(((diff - e) ** 2).sum(1).mean() / (diff ** 2).sum(1).mean()))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    R32.KINDS = KINDS
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    m, o = pairs.main, pairs.onestep
    eq, eq_ok = R32.equivalents(ctx, m["r"])
    res = dict(counts=dict(main_pairs=len(m["r"]), onestep_pairs=len(o["r"]), equivalent_pairs=int(eq_ok.sum())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        enc = Nonlinear(net, ctx, torch.Generator(device=dev).manual_seed(3300 + s), 3400 + 10 * s)
        ra, rb = m["r"], m["d"]
        row = dict(checkpoint=ck, fit=enc.fit_stats, edit_stats=enc.stats(ra, rb))
        R32.outputs.check = 0.0
        P, num, den = R32.outputs(net, ctx, enc, ra, rb)
        dj, sm = pairs.disjoint(ra, rb), pairs.same(ra, rb)
        row["main"] = dict(change=R32.summarise(P, num, den, ctx, ra, rb, dj), preserve=R32.preserve(P, ctx, ra, rb, sm),
                           goal_dependent=R32.goal_dependent(P, ctx, ra, rb),
                           by_goal={f"G{g + 1}": R32.summarise(P, num, den, ctx, ra, rb, dj & (torch.arange(3, device=dev) == g)[None]) for g in range(3)})
        Po, numo, deno = R32.outputs(net, ctx, enc, o["r"], o["d"])
        row["onestep"] = dict(change=R32.summarise(Po, numo, deno, ctx, o["r"], o["d"], pairs.disjoint(o["r"], o["d"])))
        ie = torch.nonzero(eq_ok).squeeze(1)
        Ps = [{k: v[ie] for k, v in P.items()}] + [R32.outputs(net, ctx, enc, eq[ie, j], rb[ie])[0] for j in range(1, R32.N_EQ)]
        row["equivalent"] = dict(n_pairs=len(ie), change=R32.equivalence(Ps, ctx, rb[ie], dj[ie]))
        row["check_none_equals_natural"] = R32.outputs.check
        res["runs"][f"seed{s}"] = row
        c, gd, fs = row["main"]["change"], row["main"]["goal_dependent"], enc.fit_stats
        print(f"seed{s} {ck} R² within: " + " ".join(f"{k} {fs[k]['r2_within_goal_length']:.3f}" for k in ("linear", "mlp", "mlp_goal", "table", "table_goal")) +
              " | change donor-optimal: " + " ".join(f"{k} {c[k]['donor_optimal']:.3f}" for k in KINDS + ("hybrid",)) +
              " | goal-dep: " + " ".join(f"{k} {gd[k]['right_every_change_goal']:.3f}" for k in KINDS) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
