"""Round 41: what carries the decisions left after round 40's removal? Per-history removal of the goal x history
interaction from the goal token's MLP and attention outputs. Round 23's reward models, frozen.

    .venv/bin/python rounds/r41_interaction_removal/run.py            # writes results.json
    .venv/bin/python rounds/r41_interaction_removal/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

Components at the goal token: the attention output and the MLP output of each block (8). For a component k, history h,
goal g, natural output x_k(h, g):
    goal-free part     xbar_k(h) = mean over the three goals of x_k(h, .)
    goal main effect   M_k(g, L) = c_k[g, L] - cbar_k[L]     (means over fit-side histories)
    interaction        I_k(h, g) = x_k(h, g) - xbar_k(h) - M_k(g, L)
Removal replaces the output by xbar_k(h) + M_k(g, L) (fixed, from natural runs). With every component replaced, the goal
token's final state is the embedding plus a history part plus a goal part: no interaction at all.

    mlp_post       round 40: x - J_k(b, g) (posterior-level table) at all four MLPs (live subtraction)
    mlp_hist       per-history removal at all four MLPs
    attn_hist      per-history removal at all four attention outputs
    all_hist       both: no goal x history interaction anywhere at the goal token
    mlp_resid      x - (I_k(h, g) - J_k(b, g)) at all four MLPs: only the per-history deviation from the posterior-level J
                   removed (diagnostic: is round 40's harm from leaving that deviation in place?)
    shuffled       x - I_k(h', g), h' another history of the same length, at all eight (a same-size control)
    keep_<k>       all_hist except component k, which runs live (8 kinds)
Optimal under g on round 36's goal-matters and goal-neutral cells.
"""

from __future__ import annotations

import argparse, importlib.util, json, math, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R39 = load("r39run", ROUNDS / "r39_mlp_bilinear" / "run.py")
R36, R30, M26 = R39.R36, R39.R30, R39.M26
CHUNK, SEED = 4096, 41


def comps(net):
    return [("attn", l) for l in range(net.nl)] + [("mlp", l) for l in range(net.nl)]


def cname(c):
    return f"L{c[1]}.{c[0]}"


@torch.no_grad()
def outputs(net, ctx, h, g):
    """Every component's output at the goal token and the logits, for histories h under goals g (vectors)."""
    n = torch.arange(len(h), device=h.device)
    p = 1 + ctx.L[h]
    lg, _, rec = net(R30.seq(ctx, h, g), record=True)
    return {c: rec[c[0]][c[1]][n, p] for c in comps(net)}, lg[n, p]


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
    nL = t.max_prefix + 1
    cells = R36.Cells(ctx)
    res = dict(cells=dict(goal_matters=int((cells.cls == 0).sum()), goal_neutral=int((cells.cls == 1).sum())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        C = comps(net)
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        # goal main effects per component (fit side) and posterior-level J tables for the MLPs (round 40's)
        sums = {c: torch.zeros(3, nL, net.d, dtype=torch.float64, device=dev) for c in C}
        cnt = torch.bincount(ctx.L[fit], minlength=nL).double()
        Y = {l: [] for l in range(net.nl)}
        for g in range(3):
            ys = {l: [] for l in range(net.nl)}
            for i in range(0, len(fit), 8192):
                h = fit[i:i + 8192]
                o, _ = outputs(net, ctx, h, torch.full_like(h, g))
                for c in C:
                    sums[c][g].index_add_(0, ctx.L[h], o[c].double())
                for l in range(net.nl):
                    ys[l].append(o[("mlp", l)])
            for l in range(net.nl):
                Y[l].append(torch.cat(ys[l]))
        cgl = {c: sums[c] / cnt.clamp(min=1)[None, :, None] for c in C}
        M = {c: cgl[c] - cgl[c].mean(0, keepdim=True) for c in C}                                # [3, nL, d]
        Jpost = {l: R39.Table(torch.stack(Y[l], 1), fit, ctx, pid, npost).J for l in range(net.nl)}
        del Y
        kinds = ["mlp_post", "mlp_resid", "mlp_hist", "attn_hist", "all_hist", "shuffled"] + [f"keep_{cname(c)}" for c in C]
        acts = {k: [] for k in ["natural"] + kinds}
        for i in range(0, len(cells.h), CHUNK):
            h, g = cells.h[i:i + CHUNK], cells.g[i:i + CHUNK]
            n = torch.arange(len(h), device=dev)
            p = 1 + ctx.L[h]
            L = ctx.L[h]
            nat = [outputs(net, ctx, h, torch.full_like(h, gg)) for gg in range(3)]
            x = {c: torch.stack([nat[gg][0][c] for gg in range(3)], 1) for c in C}            # [n, 3, d]
            lg_nat = torch.stack([nat[gg][1] for gg in range(3)], 1)[n, g]
            acts["natural"].append(lg_nat.argmax(-1))
            xg = {c: x[c][n, g] for c in C}
            removed = {c: (x[c].mean(1) + M[c][g, L].float()) for c in C}                       # xbar(h) + M(g, L)
            inter = {c: xg[c] - removed[c] for c in C}                                           # I(h, g)
            # shuffled: another history of the same length in this chunk
            perm = torch.arange(len(h), device=dev)
            for l in range(nL):
                idx = torch.nonzero(L == l).squeeze(1)
                if len(idx) > 1:
                    perm[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
            tok = R30.seq(ctx, h, g)
            def setter(val):
                def f(z):
                    z = z.clone(); z[n, p] = val
                    return z
                return f
            def adder(delta):
                def f(z):
                    z = z.clone(); z[n, p] = z[n, p] + delta
                    return z
                return f
            mlps, attns = [c for c in C if c[0] == "mlp"], [c for c in C if c[0] == "attn"]
            patches = dict(mlp_post={c: adder(-Jpost[c[1]][pid[h], g].float()) for c in mlps},
                           mlp_resid={c: adder(-(inter[c] - Jpost[c[1]][pid[h], g].float())) for c in mlps},
                           mlp_hist={c: setter(removed[c]) for c in mlps},
                           attn_hist={c: setter(removed[c]) for c in attns},
                           all_hist={c: setter(removed[c]) for c in C},
                           shuffled={c: adder(-inter[c][perm]) for c in C})
            for k in C:
                patches[f"keep_{cname(k)}"] = {c: setter(removed[c]) for c in C if c != k}
            for k in kinds:
                acts[k].append(net(tok, patch=patches[k])[0][n, p].argmax(-1))
        Ac = {k: torch.cat(v) for k, v in acts.items()}
        rows_ = torch.arange(len(cells.h), device=dev)
        optg = ctx.opt[cells.h, cells.g]
        row = dict(checkpoint=ck, optimal={name: {k: float(optg[rows_, Ac[k]][cells.cls == c].float().mean()) for k in Ac} for c, name in ((0, "goal_matters"), (1, "goal_neutral"))})
        res["runs"][f"seed{s}"] = row
        gm = row["optimal"]["goal_matters"]
        print(f"seed{s} {ck} goal-matters: " + " ".join(f"{k} {v:.3f}" for k, v in gm.items()) + " | goal-neutral: " +
              " ".join(f"{k} {v:.3f}" for k, v in row["optimal"]["goal_neutral"].items() if not k.startswith("keep")) + f" | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
