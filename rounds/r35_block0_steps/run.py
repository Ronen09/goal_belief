"""Round 35: the goal x belief part at each step from block 0's attention output to its MLP input, at the goal token.
Round 23's reward models, frozen.

    .venv/bin/python rounds/r35_block0_steps/run.py            # writes results.json
    .venv/bin/python rounds/r35_block0_steps/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

The steps, exactly (goal token at position p = 1 + L):

    attn     a(h, g)      block 0's attention output = o.bias + sum over heads and sources of A . v . W_O
               self       the goal token's attention to itself: its value is a function of (g, L) only
               prefix     the prefix tokens' terms: goal-free values, weights from a query that contains the goal
               head k     one head's output (self + prefix)
    mid      m = emb(g, L) + a                    rounds 33-34's site; the embedding is a function of (g, L) only
    ln       u = ln2(m) = gamma (m - mean m) / sigma(m) + beta      the MLP's input; sigma is the only nonlinear step

Representation, per step (held-out): the goal x history interaction share (round 33) and the fits of shared and
goal-conditioned encodings of the posterior: linear (c_{g,L} + E b, c_{g,L} + E_g b) and tables (the linear fit plus
the mean residual per posterior, or per posterior and goal, shrunk by n / (n + 30)). For ln also the variation of
sigma across goals for a fixed history, and the interaction of u with sigma replaced by its mean over the three goals.

Behaviour, at attn, mid and ln: edits of the goal token's state with the tables (round 34's design):
    none, whole            as before
    table                  x(A,g) + mu(b_B) - mu(b_A)                       shared
    table_goal             x(A,g) + mu_g(b_B) - mu_g(b_A)                   the recipient's goal
    table_wrong            x(A,g) + mu_g'(b_B) - mu_g'(b_A), g' != g        another goal (both averaged)
At attn and mid the tables differ only by the (g, L) offsets, so the edits must give identical outputs: a check that
nothing but the embedding lies between them.
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


R32 = load("r32run", ROUNDS / "r32_direct_belief_edit" / "run.py")
R27, R30, M26 = R32.R27, R32.R30, R32.M26
CHUNK, SHRINK = 8192, 30
EDIT_SITES = dict(attn=("attn", 0), mid=("resid_mid", 0), ln=("ln2", 0))
KINDS = ("none", "whole", "table", "table_goal", "table_wrong")


class NetLN:
    """The network with one more patch point, ("ln2", 0): block 0's layer-norm output (the MLP's input)."""

    def __init__(self, net):
        self.net = net

    def __getattr__(self, k):
        return getattr(self.net, k)

    def __call__(self, tok, patch=None, record=False):
        patch = dict(patch or {})
        fn = patch.pop(("ln2", 0), None)
        if fn is None:
            return self.net(tok, patch=patch, record=record)
        hook = self.net.blocks[0].ln2.register_forward_hook(lambda mod, inp, out: fn(out))
        try:
            return self.net(tok, patch=patch, record=record)
        finally:
            hook.remove()


@torch.no_grad()
def steps(net, ctx, hist, g):
    """Every step's state at the goal token for histories under goal g: {name: [n, d]}, plus sigma [n] and the check."""
    out = {}
    b0 = net.blocks[0]
    for s in range(0, len(hist), CHUNK):
        h = hist[s:s + CHUNK]
        n = torch.arange(len(h), device=h.device)
        p = 1 + ctx.L[h]
        tok = R30.seq(ctx, h, torch.full_like(h, g))
        rec = net(tok, record=True)[2]
        x0 = rec["resid"][0]
        a, mid = rec["attn"][0][n, p], rec["mid"][0][n, p]
        terms = b0.source_terms(rec["pattern"][0], rec["value"][0], p)                         # [n, heads, L, d]
        src = torch.arange(terms.shape[2], device=h.device)
        selfm = (src[None] == p[:, None]).float()[:, None, :, None]
        prem = (src[None] < p[:, None]).float()[:, None, :, None]
        self_t, pre_t = (terms * selfm).sum(2), (terms * prem).sum(2)                         # [n, heads, d]
        sig = (mid.var(-1, unbiased=False) + b0.ln2.eps).sqrt()
        row = dict(attn=a, self=self_t.sum(1) + b0.o.bias, prefix=pre_t.sum(1), mid=mid, ln=b0.ln2(mid), sigma=sig,
                   check_mid=(mid - x0[n, p] - a).abs().amax(-1), check_terms=(self_t.sum(1) + pre_t.sum(1) + b0.o.bias - a).abs().amax(-1))
        for k in range(net.nh):
            row[f"head{k}"] = self_t[:, k] + pre_t[:, k]
        for k, v in row.items():
            out.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in out.items()}


def interaction(Z, ctx, rows):
    """Goal x history interaction share (round 33) and its absolute size, on histories `rows`. Z [nh, goals, d]."""
    t = ctx.t
    Zr, L = Z[rows].double(), ctx.L[rows]
    cm = torch.zeros_like(Zr)
    for l in range(t.max_prefix + 1):
        m = L == l
        if m.any():
            cm[m] = Zr[m].mean(0, keepdim=True)
    Zc = Zr - cm
    inter = Zc - Zc.mean(1, keepdim=True)
    return dict(share=float((inter ** 2).sum() / (Zc ** 2).sum().clamp(min=1e-30)), size=float((inter ** 2).sum(-1).mean()), variance=float((Zc ** 2).sum(-1).mean()))


class Tables:
    """Linear and table encodings of the posterior at one step, fitted on the fit side under all goals."""

    def __init__(self, Z, ctx, pid):
        t, dev = ctx.t, ctx.t.dev
        Z = Z.double()
        self.Z = Z
        nh, d = Z.shape[0], Z.shape[-1]
        bel = t.belief[ctx.node].double()
        self.bel = bel
        npost = int(pid.max()) + 1
        fit, test = torch.nonzero(ctx.fit).squeeze(1), torch.nonzero(ctx.test).squeeze(1)
        GL = torch.nn.functional.one_hot(torch.arange(3, device=dev)[None] * (t.max_prefix + 1) + ctx.L[:, None], 3 * (t.max_prefix + 1)).double()   # [nh, 3, 15]
        L1 = torch.nn.functional.one_hot(ctx.L, t.max_prefix + 1).double()
        # shared linear: c_{g,L} + E b
        X = torch.cat([bel[:, None].expand(nh, 3, -1), GL], -1)
        lin = Affine(X[fit].reshape(-1, X.shape[-1]), Z[fit].reshape(-1, d))
        pred = lin(X.reshape(-1, X.shape[-1])).view(nh, 3, d)
        # goal-specific linear: per goal, c_{g,L} + E_g b
        predg = torch.zeros_like(Z)
        for g in range(3):
            Xg = torch.cat([bel, L1], 1)
            predg[:, g] = Affine(Xg[fit], Z[fit, g])(Xg)
        # tables: mean residual per posterior (shared) or per posterior and goal, shrunk toward the linear fit
        def table(R, key, nkey):
            s = torch.zeros(nkey, d, dtype=torch.float64, device=dev).index_add_(0, key, R)
            n = torch.bincount(key, minlength=nkey).double()[:, None]
            return s / (n + SHRINK)
        gg = torch.arange(3, device=dev)
        tab = table((Z - pred)[fit].reshape(-1, d), pid[fit].repeat_interleave(3), npost)
        tabg = table((Z - predg)[fit].reshape(-1, d), (pid[fit][:, None] * 3 + gg[None]).reshape(-1), 3 * npost)
        self.code = (bel @ lin.W[: bel.shape[1]]) + tab[pid]                                   # [nh, d]: E b + mu(b), shared
        Eg = []
        for g in range(3):
            Xg = torch.cat([bel, L1], 1)
            Eg.append(Affine(Xg[fit], Z[fit, g]).W[: bel.shape[1]])
        self.code_g = torch.stack([bel @ Eg[g] + tabg[pid * 3 + g] for g in range(3)], 1)      # [nh, 3, d]
        full = pred + tab[pid][:, None]
        fullg = predg + tabg[pid[:, None] * 3 + gg[None]]
        grp = (gg[None] * (t.max_prefix + 1) + ctx.L[test][:, None]).reshape(-1)
        Yt = Z[test].reshape(-1, d)
        r2 = lambda P: R32.r2(Yt, P[test].reshape(-1, d), grp)
        self.fit_stats = dict(linear=r2(pred), linear_goal=r2(predg), table=r2(full), table_goal=r2(fullg))

    def edits(self, a, b, g):
        zA = self.Z[a, g]
        wrong = [h for h in range(3) if h != g]
        return dict(none=zA, whole=self.Z[b, g], table=zA + self.code[b] - self.code[a], table_goal=zA + self.code_g[b, g] - self.code_g[a, g],
                    table_wrong=[zA + self.code_g[b, h] - self.code_g[a, h] for h in wrong])


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
    m = pairs.main
    ra, rb = m["r"], m["d"]
    dj, sm = pairs.disjoint(ra, rb), pairs.same(ra, rb)
    bel = t.belief[ctx.node].double()
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    all_h = torch.arange(len(ctx.L), device=dev)
    test = torch.nonzero(ctx.test).squeeze(1)
    res = dict(runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        st = [steps(net, ctx, all_h, g) for g in range(3)]
        S = {k: torch.stack([x[k] for x in st], 1) for k in st[0]}                              # [nh, goals, ...]
        row = dict(checkpoint=ck, checks=dict(mid_minus_embedding_minus_attn=float(S["check_mid"].max()), terms_sum_to_attn=float(S["check_terms"].max())))
        names = ["attn", "self", "prefix"] + [f"head{k}" for k in range(net.nh)] + ["mid", "ln"]
        row["interaction"] = {k: interaction(S[k], ctx, test) for k in names}
        # the layer norm's scale: variation across goals for a fixed history, and u with sigma fixed at its mean over goals
        sig = S["sigma"].double()
        row["sigma"] = dict(cv_across_goals=float((sig.std(1, unbiased=False) / sig.mean(1))[test].median()), mean=float(sig[test].mean()))
        b0 = net.blocks[0]
        mid = S["mid"].double()
        cen = mid - mid.mean(-1, keepdim=True)
        u_fixed = cen / sig.mean(1, keepdim=True)[..., None] * b0.ln2.weight.detach().double() + b0.ln2.bias.detach().double()
        row["interaction"]["ln_sigma_fixed"] = interaction(u_fixed, ctx, test)
        enc = {}
        row["fit"] = {}
        for k in names:
            T = Tables(S[k], ctx, pid)
            row["fit"][k] = T.fit_stats
            if k in EDIT_SITES:
                enc[k] = T
        del S, st
        row["edits"] = {}
        P_site = {}
        wnet = NetLN(net)
        for site, key in EDIT_SITES.items():
            R32.SITE = key
            R32.outputs.check = 0.0
            P, num, den = R32.outputs(wnet, ctx, enc[site], ra, rb)
            P_site[site] = P
            row["edits"][site] = dict(change=R32.summarise(P, num, den, ctx, ra, rb, dj), preserve=R32.preserve(P, ctx, ra, rb, sm),
                                      goal_dependent=R32.goal_dependent(P, ctx, ra, rb), check_none_equals_natural=R32.outputs.check)
        row["checks"]["attn_vs_mid_edits"] = max(float((P_site["attn"][k] - P_site["mid"][k]).abs().max()) for k in KINDS)
        res["runs"][f"seed{s}"] = row
        it, gd = row["interaction"], {x: row["edits"][x]["goal_dependent"] for x in EDIT_SITES}
        print(f"seed{s} {ck} checks {row['checks']} | interaction: " + " ".join(f"{k} {it[k]['share']:.3f}" for k in names + ["ln_sigma_fixed"]) +
              f" | sigma cv {row['sigma']['cv_across_goals']:.4f} | goal-dep / whole: " +
              " ".join(f"{x}: " + ",".join(f"{k} {gd[x][k]['right_every_change_goal'] / max(gd[x]['whole']['right_every_change_goal'], 1e-9):.2f}" for k in ("table", "table_goal", "table_wrong")) for x in EDIT_SITES) +
              f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
