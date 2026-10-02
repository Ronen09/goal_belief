"""Round 43: what is the history profile H? Candidate quantities per posterior, as predictors of H, as its replacement in
the additive decision, and as causal edits of the goal token's state. Round 23's reward models, frozen.

    .venv/bin/python rounds/r43_what_is_H/run.py            # writes results.json
    .venv/bin/python rounds/r43_what_is_H/run.py --untrained    # the ten initial checkpoints (prediction reference); untrained.json

H(h): the mean over goals of the centred logits at the goal token (round 42). Candidates C(b) [4], from the posterior b
at the reveal (goal-free), the exact action values Q*(b, g, a) and the fully observed values
Q_MDP(b, g, a) = sum_s b(s) gamma^d(next(s, a), goal g) (moves are deterministic; reward on entering the goal):
    maxQ      max_g Q*            meanQ     mean_g Q*            popt     share of goals for which a is optimal
    maxA      max_g (Q* - V*)     (mean_g (Q* - V*) equals meanQ once centred over actions, so it is omitted)
    maxMDP    max_g Q_MDP         meanMDP   mean_g Q_MDP         nearest  sum_s b(s) gamma^(min_g d(next(s, a), g))
Ceilings: Htab (H's own mean per posterior, fit side) and blin (affine in b).
Prediction (held-out histories, length >= 1): H (centred) ~ alpha[a, L] + C M (4 x 4; flexible) and ~ alpha + beta C
(one slope); R², top-action agreement. Decision: argmax of the fitted H plus round 42's goal bias G.
Causal: at two sites of the goal token (entering block 2; the final residual stream), the goal-free state (mean over
goals) is regressed on C (fit side): x ~ alpha_L + C W. Edit, the same vector under every goal: x(A, g) + (C(b_B) - C(b_A)) W.
Also whole, blin (W from b), Htab, and maxQ's edit rotated (5 draws). Round 27's main pairs; round 32's measures.
"""

from __future__ import annotations

import argparse, importlib.util, json, math, sys, time
from collections import deque
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
R42 = load("r42run", ROUNDS / "r42_additive_code" / "run.py")
R27, R30, M26 = R32.R27, R32.R30, R32.M26
CANDS = ("maxQ", "meanQ", "popt", "maxA", "maxMDP", "meanMDP", "nearest")       # mean_g (Q* - V*) equals meanQ after centring: omitted
SITES = (("resid", 2), ("resid", 4))
CHUNK, DRAWS, SHRINK = 8192, 5, 10


def qmdp(t):
    """Q_MDP[s, g, a] = gamma^d(next(s, a), goal g), from breadth-first distances on the maze."""
    nxt = t.nxt_cell.cpu().numpy()
    n = nxt.shape[0]
    D = torch.zeros(n, len(t.goal_cell), dtype=torch.float64)
    for gi, g in enumerate(t.goal_cell.tolist()):
        d = [math.inf] * n; d[g] = 0
        q = deque([g])
        while q:                                                                           # reverse BFS: predecessors of u
            u = q.popleft()
            for s in range(n):
                if d[s] == math.inf and u in nxt[s]:
                    d[s] = d[u] + 1; q.append(s)
        D[:, gi] = torch.tensor(d, dtype=torch.float64)
    Dn = D[torch.tensor(nxt)]                                                               # [s, a, g]
    return (t.gamma ** Dn).permute(0, 2, 1).to(t.dev), (t.gamma ** Dn.min(-1).values).to(t.dev)   # [s, g, a], [s, a]


def candidates(t, ctx):
    """C[h, cand, a] for every history (posterior at the reveal)."""
    Q, V, opt = ctx.Q.double(), ctx.V.double(), ctx.opt.double()                            # [nh, 3, 4]
    A = Q - V[..., None]
    QM, NE = qmdp(t)
    b = t.belief[ctx.node].double()                                                        # [nh, cells]
    QMb = torch.einsum("hs,sga->hga", b, QM)
    out = dict(maxQ=Q.max(1).values, meanQ=Q.mean(1), popt=opt.mean(1), maxA=A.max(1).values, meanA=A.mean(1),
               maxMDP=QMb.max(1).values, meanMDP=QMb.mean(1), nearest=b @ NE)
    return torch.stack([out[k] for k in CANDS], 1)                                         # [nh, K, 4]


def centre(x):
    return x - x.mean(-1, keepdim=True)


@torch.no_grad()
def states(net, ctx, hist, site):
    out = []
    for g in range(3):
        xs = []
        for s in range(0, len(hist), CHUNK):
            h = hist[s:s + CHUNK]
            n = torch.arange(len(h), device=h.device)
            xs.append(net(R30.seq(ctx, h, torch.full_like(h, g)), record=True)[2]["resid"][site[1]][n, 1 + ctx.L[h]])
        out.append(torch.cat(xs))
    return torch.stack(out, 1).double()                                                    # [n, 3, d]


class Editor:
    """Encodings of candidate codes in the goal-free state at one site, and edits."""

    def __init__(self, Z, codes, ctx, gen):
        """Z [nh, 3, d] states; codes {name: [nh, k]} features (fit on the fit side with per-length offsets)."""
        self.Z = Z
        fit = ctx.fit
        L1 = torch.nn.functional.one_hot(ctx.L, ctx.t.max_prefix + 1).double()
        xbar = Z.mean(1)
        self.W, self.codes = {}, codes
        for k, F in codes.items():
            X = torch.cat([F, L1], 1)
            self.W[k] = Affine(X[fit], xbar[fit]).W[: F.shape[1]]
        d = Z.shape[-1]
        self.Q = [R27.orth(torch.randn(d, d, dtype=torch.float64, device=Z.device, generator=gen)) for _ in range(DRAWS)]

    def edits(self, a, b, g):
        zA = self.Z[a, g]
        e = {k: (F[b] - F[a]) @ self.W[k] for k, F in self.codes.items()}
        out = dict(none=zA, whole=self.Z[b, g], **{k: zA + v for k, v in e.items()})
        out["rotated"] = [zA + e["maxQ"] @ Q for Q in self.Q]
        return out


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
    nL = t.max_prefix + 1
    pairs = R27.Pairs(ctx)
    m = pairs.main
    C = candidates(t, ctx)                                                                 # [nh, K, 4]
    Cc = centre(C)
    bel = t.belief[ctx.node].double()
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    npost = int(pid.max()) + 1
    fit_all = torch.nonzero(ctx.fit).squeeze(1)
    fit, test = torch.nonzero(ctx.fit & (ctx.L >= 1)).squeeze(1), torch.nonzero(ctx.test & (ctx.L >= 1)).squeeze(1)
    L1 = torch.nn.functional.one_hot(ctx.L, nL).double()
    opt = ctx.opt
    dep = torch.zeros(len(ctx.L), 3, dtype=torch.bool, device=dev)
    for g in range(3):
        for g2 in range(3):
            if g != g2:
                dep[:, g] |= ~(opt[:, g] & opt[:, g2]).any(-1)
    kinds = ("none", "whole") + CANDS + ("blin", "Htab", "rotated")
    R32.KINDS = kinds
    res = dict(candidates=CANDS, runs={})
    out = HERE / ("untrained.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        gen = torch.Generator(device=dev); gen.manual_seed(4300 + s)
        all_h = torch.arange(len(ctx.L), device=dev)
        lg = R42.logits(net, ctx, all_h)                                                     # [nh, 3, 4]
        H = lg.mean(1)
        # goal bias (round 42) from the fit side
        devf = lg - lg.mean(1, keepdim=True)
        G = torch.zeros(3, nL, 4, dtype=torch.float64, device=dev)
        for l in range(nL):
            mm = ctx.fit & (ctx.L == l)
            if mm.any():
                G[:, l] = devf[mm].mean(0)
        # H's own posterior table (ceiling), shrunk toward the overall mean per length
        sH = torch.zeros(npost, 4, dtype=torch.float64, device=dev).index_add_(0, pid[fit_all], H[fit_all])
        nH = torch.bincount(pid[fit_all], minlength=npost).double()[:, None]
        Htab = (sH + SHRINK * H[fit_all].mean(0)) / (nH + SHRINK)
        feats = {k: Cc[:, i] for i, k in enumerate(CANDS)}
        feats["blin"] = bel
        feats["Htab"] = centre(Htab[pid])
        pred = {}
        Ht = H[test]
        var = ((Ht - Ht.mean(0)) ** 2).sum()
        for k, F in feats.items():
            X = torch.cat([F, L1], 1)
            P = Affine(X[fit], H[fit])(X[test])
            pred[k] = dict(r2_flexible=float(1 - ((Ht - P) ** 2).sum() / var))
            if k in CANDS:
                # one slope: H ~ alpha[a, L] + beta C
                Xs = (F[:, :, None] * torch.eye(4, dtype=torch.float64, device=dev)).sum(1, keepdim=True)   # [n, 1, 4]
                Y = H - torch.stack([H[fit][ctx.L[fit] == l].mean(0) for l in range(nL)])[ctx.L]
                Fc = F - torch.stack([F[fit][ctx.L[fit] == l].mean(0) for l in range(nL)])[ctx.L]
                beta = float((Fc[fit] * Y[fit]).sum() / (Fc[fit] ** 2).sum().clamp(min=1e-12))
                pred[k]["r2_one_slope"] = float(1 - ((Y[test] - beta * Fc[test]) ** 2).sum() / var)
                pred[k]["beta"] = beta
                Cs = C[test, CANDS.index(k)]
                top = Cs.max(-1, keepdim=True).values
                pred[k]["top_agree"] = float(torch.gather(Cs, 1, Ht.argmax(-1, keepdim=True)).squeeze(1).ge(top.squeeze(1) - 1e-9).float().mean())
            # decision: fitted H plus the goal bias
            Hp = Affine(X[fit], H[fit])(X)
            act = (Hp[:, None] + G[:, ctx.L].permute(1, 0, 2)).argmax(-1)                 # [nh, 3]
            okk = torch.gather(opt, 2, act[..., None]).squeeze(-1)
            pred[k]["decision_goal_dependent"] = float(okk[test][dep[test]].float().mean())
        nat = torch.gather(opt, 2, lg.argmax(-1)[..., None]).squeeze(-1)
        addH = torch.gather(opt, 2, (H[:, None] + G[:, ctx.L].permute(1, 0, 2)).argmax(-1)[..., None]).squeeze(-1)
        row = dict(checkpoint=ck, prediction=pred, decision_reference=dict(natural=float(nat[test][dep[test]].float().mean()), additive_H=float(addH[test][dep[test]].float().mean())))
        # causal edits
        row["edits"] = {}
        for site in SITES:
            Z = states(net, ctx, all_h, site)
            ed = Editor(Z, feats, ctx, gen)
            R32.outputs.check = 0.0
            R32.SITE = site
            ra, rb = m["r"], m["d"]
            P, num, den = R32.outputs(net, ctx, ed, ra, rb)
            dj, sm = pairs.disjoint(ra, rb), pairs.same(ra, rb)
            row["edits"][f"{site[0]}{site[1]}"] = dict(change=R32.summarise(P, num, den, ctx, ra, rb, dj), preserve=R32.preserve(P, ctx, ra, rb, sm),
                                                      goal_dependent=R32.goal_dependent(P, ctx, ra, rb))
            del Z, ed
        res["runs"][f"seed{s}"] = row
        best = max(CANDS, key=lambda k: pred[k]["r2_flexible"])
        e2 = row["edits"]["resid2"]["change"]
        print(f"seed{s} {ck} R² flexible: " + " ".join(f"{k} {pred[k]['r2_flexible']:.2f}" for k in feats) + f" | best {best} | decision: " +
              " ".join(f"{k} {pred[k]['decision_goal_dependent']:.3f}" for k in feats) + f" (natural {row['decision_reference']['natural']:.3f}, H {row['decision_reference']['additive_H']:.3f}) | "
              f"edit resid2 donor-optimal: " + " ".join(f"{k} {e2[k]['donor_optimal']:.3f}" for k in kinds) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
