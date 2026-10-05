"""The query-swap experiment: does the goal x belief part of block 0's attention output come from the goal token's query? A causal swap,
the observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/query_swap/run.py            # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/query_swap/run.py --untrained    # the ten initial checkpoints (reference); writes untrained.json

In block 0 the goal token's input is its embedding alone, so its query q, key k and value v are functions of
(goal, prefix length) only. Block 0's attention output at the goal token of history h under goal g is recomputed with

    query     q from goal g' (k and v of every token kept): which history tokens are read, and how much
    selfkv    the goal token's own k and v from g' (q kept): what the goal token sends to itself, and how much it attracts
    both      q, k, v from g' (equals the natural output under g' exactly: a check)

Representation: per held-out history and ordered goal pair (g, g'), D_k = a_k(h, g) - a(h, g); with its mean over
histories within (g, g', L) removed (the goal's offset), the history-dependent part. The share of the full swap's part
carried by each: <D_k, D_both> / |D_both|^2, summed over histories; also on the posterior-explained part (means per
posterior, posteriors with >= 5 histories).

Behaviour: the network run with only block 0's attention output at the goal token replaced by the recomputed one,
the rest natural under g. Cells: held-out histories under ordered goal pairs, split by the solver into goal-matters
(disjoint optimal sets under g and g') and goal-neutral (identical optimal sets).
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


R32 = load("r32run", ROUNDS / "5_goal_belief_mechanism" / "direct_belief_edit" / "run.py")
R27, R30, M26 = R32.R27, R32.R30, R32.M26
CHUNK, N_HIST, MIN_POST, CELL_SEED = 8192, 12000, 5, 36
KINDS = ("query", "selfkv", "both")
GOAL_PAIRS = [(g, h) for g in range(3) for h in range(3) if g != h]


@torch.no_grad()
def attn0(net, tok, p, q_src=None, kv_src=None):
    """Block 0's attention output at positions p [n], with the goal token's q (and/or its own k, v) replaced by those
    from the token sequences q_src / kv_src (same histories, another goal)."""
    b = net.blocks[0]
    N, L, d = tok.shape[0], tok.shape[1], net.d
    n = torch.arange(N, device=tok.device)
    def qkv(t):
        return b.qkv(b.ln1(net.embed(t))).view(N, L, 3, b.h, d // b.h).permute(2, 0, 3, 1, 4)        # each [N, h, L, dh]
    q, k, v = qkv(tok)
    if q_src is not None:
        q = q.clone(); q[n, :, p] = qkv(q_src)[0][n, :, p]
    if kv_src is not None:
        _, k2, v2 = qkv(kv_src)
        k, v = k.clone(), v.clone()
        k[n, :, p], v[n, :, p] = k2[n, :, p], v2[n, :, p]
    qp = q[n, :, p]                                                                            # [N, h, dh]
    s = torch.einsum("nhd,nhld->nhl", qp, k) / math.sqrt(d // b.h)
    s = s.masked_fill(torch.arange(L, device=tok.device)[None, None] > p[:, None, None], -torch.inf)
    z = torch.einsum("nhl,nhld->nhd", s.softmax(-1), v)
    return b.o(z.reshape(N, d))


@torch.no_grad()
def swapped(net, ctx, h, g, g2):
    """Natural a(h, g) and the three recomputed outputs, for histories h under goals g with swaps from g2 (vectors)."""
    p = 1 + ctx.L[h]
    t1, t2 = R30.seq(ctx, h, g), R30.seq(ctx, h, g2)
    return dict(natural=attn0(net, t1, p), query=attn0(net, t1, p, q_src=t2), selfkv=attn0(net, t1, p, kv_src=t2), both=attn0(net, t1, p, q_src=t2, kv_src=t2),
                other=attn0(net, t2, p))


def representation(net, ctx, pid):
    """Shares of the full swap's history-dependent part carried by the query and by the goal token's own k, v."""
    test = torch.nonzero(ctx.test).squeeze(1)
    num = {k: 0.0 for k in KINDS}
    num_p = {k: 0.0 for k in KINDS}
    den, den_p, check, size = 0.0, 0.0, 0.0, 0.0
    for g, g2 in GOAL_PAIRS:
        D = {k: [] for k in KINDS}
        for s in range(0, len(test), CHUNK):
            h = test[s:s + CHUNK]
            o = swapped(net, ctx, h, torch.full_like(h, g), torch.full_like(h, g2))
            check = max(check, float((o["both"] - o["other"]).abs().max()))
            for k in KINDS:
                D[k].append((o[k] - o["natural"]).double())
        L = ctx.L[test]
        P = pid[test]
        for k in KINDS:
            X = torch.cat(D[k])
            for l in L.unique():
                m = L == l
                X[m] -= X[m].mean(0)
            D[k] = X
        # posterior-explained part: means per posterior (posteriors with >= MIN_POST histories)
        u, inv, cnt = torch.unique(P, return_inverse=True, return_counts=True)
        keep = cnt[inv] >= MIN_POST
        Dp = {}
        for k in KINDS:
            S = torch.zeros(len(u), D[k].shape[1], dtype=torch.float64, device=P.device).index_add_(0, inv, D[k])
            Dp[k] = (S / cnt[:, None].double())[inv][keep]
        for k in KINDS:
            num[k] += float((D[k] * D["both"]).sum()); num_p[k] += float((Dp[k] * Dp["both"]).sum())
        den += float((D["both"] ** 2).sum()); den_p += float((Dp["both"] ** 2).sum())
        size += float((D["both"] ** 2).sum(1).mean()) / len(GOAL_PAIRS)
    return dict(share={k: num[k] / den for k in KINDS}, share_posterior={k: num_p[k] / den_p for k in KINDS},
                posterior_part_of_full=den_p / den, size_full=size, check_both_equals_other=check)


class Cells:
    """Held-out histories under ordered goal pairs (g, g'): goal-matters (disjoint optimal sets) and goal-neutral."""

    def __init__(self, ctx):
        dev = ctx.t.dev
        g = torch.Generator(device=dev); g.manual_seed(CELL_SEED)
        pool = torch.nonzero(ctx.test & (ctx.L >= 1)).squeeze(1)
        h = pool[torch.randperm(len(pool), device=dev, generator=g)[:N_HIST]]
        rows = []
        for gr, gd in GOAL_PAIRS:
            dj = ~(ctx.opt[h, gr] & ctx.opt[h, gd]).any(-1)
            sm = (ctx.opt[h, gr] == ctx.opt[h, gd]).all(-1)
            cls = torch.where(dj, 0, torch.where(sm, 1, 2))
            rows.append(torch.stack([h, torch.full_like(h, gr), torch.full_like(h, gd), cls], 1))
        r = torch.cat(rows)
        r = r[r[:, 3] < 2]
        self.h, self.g, self.g2, self.cls = r[:, 0], r[:, 1], r[:, 2], r[:, 3]


@torch.no_grad()
def behaviour(net, ctx, cells):
    """Greedy actions with block 0's attention output at the goal token replaced, per kind."""
    out = {k: [] for k in ("natural", "other_goal") + KINDS}
    check = 0.0
    for s in range(0, len(cells.h), CHUNK):
        h, g, g2 = cells.h[s:s + CHUNK], cells.g[s:s + CHUNK], cells.g2[s:s + CHUNK]
        n = torch.arange(len(h), device=h.device)
        p = 1 + ctx.L[h]
        t1, t2 = R30.seq(ctx, h, g), R30.seq(ctx, h, g2)
        o = swapped(net, ctx, h, g, g2)
        lg = net(t1)[0][n, p]
        out["natural"].append(lg.argmax(-1)); out["other_goal"].append(net(t2)[0][n, p].argmax(-1))
        nat = None
        for k in KINDS + ("natural_recomputed",):
            val = o["natural"] if k == "natural_recomputed" else o[k]
            def f(a, val=val):
                a = a.clone(); a[n, p] = val
                return a
            lk = net(t1, patch={("attn", 0): f})[0][n, p]
            if k == "natural_recomputed":
                check = max(check, float((lk.softmax(-1) - lg.softmax(-1)).abs().max()))
            else:
                out[k].append(lk.argmax(-1))
    A = {k: torch.cat(v) for k, v in out.items()}
    rows = torch.arange(len(cells.h), device=cells.h.device)
    optg = ctx.opt[cells.h, cells.g]
    res = dict(check_recomputed_equals_natural=check)
    for c, name in ((0, "goal_matters"), (1, "goal_neutral")):
        m = cells.cls == c
        res[name] = dict(n=int(m.sum()), **{k: dict(optimal=float(optg[rows, A[k]][m].float().mean()), as_other_goal=float((A[k] == A["other_goal"])[m].float().mean()),
                                                       unchanged=float((A[k] == A["natural"])[m].float().mean())) for k in A})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    bel = t.belief[ctx.node].double()
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    cells = Cells(ctx)
    res = dict(cells=dict(goal_matters=int((cells.cls == 0).sum()), goal_neutral=int((cells.cls == 1).sum())), runs={})
    out = HERE / ("untrained.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        row = dict(checkpoint=ck, representation=representation(net, ctx, pid), behaviour=behaviour(net, ctx, cells))
        res["runs"][f"seed{s}"] = row
        r, b = row["representation"], row["behaviour"]
        print(f"seed{s} {ck} checks both=other {r['check_both_equals_other']:.1e} recomputed=natural {b['check_recomputed_equals_natural']:.1e} | share of the goal x history part: " +
              " ".join(f"{k} {r['share'][k]:.3f}" for k in KINDS) + " | posterior part: " + " ".join(f"{k} {r['share_posterior'][k]:.3f}" for k in KINDS) +
              f" (posterior part of full {r['posterior_part_of_full']:.3f}) | optimal, goal-matters: " + " ".join(f"{k} {b['goal_matters'][k]['optimal']:.3f}" for k in ("natural",) + KINDS) +
              " | goal-neutral: " + " ".join(f"{k} {b['goal_neutral'][k]['optimal']:.3f}" for k in ("natural",) + KINDS) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
