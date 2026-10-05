"""The self-value experiment: which part of the goal token's self-attention in block 0 carries the goal's identity to the decision?
The observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/self_value/run.py            # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/self_value/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

The query-swap experiment's cells (held-out histories under ordered goal pairs (g, g'), goal-matters and goal-neutral). Block 0's
attention output at the goal token is recomputed with parts of the goal token's own q/k/v from g' (functions of goal and
length only), and only that output is replaced; everything else runs naturally under g:

    selfkv              own key and value from g', all heads                  (the query-swap experiment's swap; reproduced)
    selfv               own value from g', key kept, all heads                what it sends to itself
    selfk               own key from g', value kept, all heads                how much it attends to itself
    selfkv_head{k}      own key and value from g' in head k only
    selfv_head{k}       own value from g' in head k only
    embedding_residual  block 0's attention natural; the goal embedding in the residual stream from g'
                        (m = emb(g') + a(h, g)): the complement
Also, descriptive: each head's attention weight on the goal token itself, and the norm of each head's self term.
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


R36 = load("r36run", ROUNDS / "5_goal_belief_mechanism" / "query_swap" / "run.py")
R30, M26 = R36.R30, R36.M26
CHUNK = 8192


@torch.no_grad()
def attn0(net, tok, src, p, k_heads=(), v_heads=()):
    """Block 0's attention output at positions p, with the goal token's own key (heads k_heads) and value (heads
    v_heads) taken from the sequences `src`. Also the per-head self weight and self-term norm (natural)."""
    b = net.blocks[0]
    N, L, d = tok.shape[0], tok.shape[1], net.d
    n = torch.arange(N, device=tok.device)
    qkv = lambda t: b.qkv(b.ln1(net.embed(t))).view(N, L, 3, b.h, d // b.h).permute(2, 0, 3, 1, 4)
    q, k, v = qkv(tok)
    if k_heads or v_heads:
        _, k2, v2 = qkv(src)
        k, v = k.clone(), v.clone()
        for hh in k_heads:
            k[n, hh, p] = k2[n, hh, p]
        for hh in v_heads:
            v[n, hh, p] = v2[n, hh, p]
    s = torch.einsum("nhd,nhld->nhl", q[n, :, p], k) / math.sqrt(d // b.h)
    s = s.masked_fill(torch.arange(L, device=tok.device)[None, None] > p[:, None, None], -torch.inf)
    A = s.softmax(-1)
    z = torch.einsum("nhl,nhld->nhd", A, v)
    W = b.o.weight.view(d, b.h, d // b.h)
    self_term = torch.einsum("nh,nhd,ohd->nho", A[n, :, p], v[n, :, p], W)
    return b.o(z.reshape(N, d)), A[n, :, p], self_term.norm(dim=-1)


@torch.no_grad()
def behaviour(net, ctx, cells):
    H = net.nh
    kinds = {"selfkv": (range(H), range(H)), "selfv": ((), range(H)), "selfk": (range(H), ())}
    for hh in range(H):
        kinds[f"selfkv_head{hh}"] = ((hh,), (hh,))
        kinds[f"selfv_head{hh}"] = ((), (hh,))
    acts = {k: [] for k in ("natural", "other_goal", "embedding_residual") + tuple(kinds)}
    w_self, n_self, check = [], [], 0.0
    for s in range(0, len(cells.h), CHUNK):
        h, g, g2 = cells.h[s:s + CHUNK], cells.g[s:s + CHUNK], cells.g2[s:s + CHUNK]
        n = torch.arange(len(h), device=h.device)
        p = 1 + ctx.L[h]
        t1, t2 = R30.seq(ctx, h, g), R30.seq(ctx, h, g2)
        lg = net(t1)[0][n, p]
        acts["natural"].append(lg.argmax(-1)); acts["other_goal"].append(net(t2)[0][n, p].argmax(-1))
        a_nat, A, sn = attn0(net, t1, t2, p)
        w_self.append(A); n_self.append(sn)
        def f_attn(val):
            def f(a):
                a = a.clone(); a[n, p] = val
                return a
            return f
        check = max(check, float((net(t1, patch={("attn", 0): f_attn(a_nat)})[0][n, p].softmax(-1) - lg.softmax(-1)).abs().max()))
        for k, (kh, vh) in kinds.items():
            acts[k].append(net(t1, patch={("attn", 0): f_attn(attn0(net, t1, t2, p, tuple(kh), tuple(vh))[0])})[0][n, p].argmax(-1))
        de = net.embed(t2)[n, p] - net.embed(t1)[n, p]
        def f_mid(x):
            x = x.clone(); x[n, p] = x[n, p] + de
            return x
        acts["embedding_residual"].append(net(t1, patch={("resid_mid", 0): f_mid})[0][n, p].argmax(-1))
    A = {k: torch.cat(v) for k, v in acts.items()}
    W, S = torch.cat(w_self), torch.cat(n_self)
    rows = torch.arange(len(cells.h), device=cells.h.device)
    optg = ctx.opt[cells.h, cells.g]
    res = dict(check_recomputed_equals_natural=check)
    for c, name in ((0, "goal_matters"), (1, "goal_neutral")):
        m = cells.cls == c
        res[name] = dict(n=int(m.sum()), **{k: dict(optimal=float(optg[rows, A[k]][m].float().mean()), as_other_goal=float((A[k] == A["other_goal"])[m].float().mean()),
                                                       unchanged=float((A[k] == A["natural"])[m].float().mean())) for k in A})
    res["self_weight"] = W.mean(0).tolist()
    res["self_term_norm"] = S.mean(0).tolist()
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
    cells = R36.Cells(ctx)
    res = dict(cells=dict(goal_matters=int((cells.cls == 0).sum()), goal_neutral=int((cells.cls == 1).sum())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        b = behaviour(net, ctx, cells)
        res["runs"][f"seed{s}"] = dict(checkpoint=ck, behaviour=b)
        gm = b["goal_matters"]
        print(f"seed{s} {ck} check {b['check_recomputed_equals_natural']:.1e} | goal-matters, other goal's action: " +
              " ".join(f"{k} {gm[k]['as_other_goal']:.2f}" for k in gm if k not in ("n", "natural", "other_goal")) +
              f" | natural {gm['natural']['as_other_goal']:.2f} | self weight " + " ".join(f"{x:.2f}" for x in b["self_weight"]) +
              " | self norm " + " ".join(f"{x:.2f}" for x in b["self_term_norm"]) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
