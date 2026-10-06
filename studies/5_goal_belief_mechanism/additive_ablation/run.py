"""The additive-ablation experiment: does the decision collapse when the additive code's history part is removed, live?
The observation-prediction experiment's reward models, frozen. Kinds and decision rule: PLAN.md.

    .venv/bin/python studies/5_goal_belief_mechanism/additive_ablation/run.py                         # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/additive_ablation/run.py --untrained --seeds 0   # smoke test (initial checkpoint)

Components at the goal token: the attention and MLP output of each block (8). For history h, goal g, natural x_k(h, g):
    history part       xbar_k(h) = mean over the three goals of x_k(h, .)
    history-blind mean Xbar_k(L) = fit-side mean of xbar_k over histories of length L
    goal main effect   M_k(g, L) = c_k[g, L] - Xbar_k(L)
    interaction        I_k(h, g) = x_k(h, g) - xbar_k(h) - M_k(g, L)
F_* kinds set all eight outputs (nothing at the goal token is live); live_* kinds add a fixed vector to the outputs and
let later components recompute.
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


R41 = load("r41run", ROUNDS / "5_goal_belief_mechanism" / "interaction_removal" / "run.py")
R36, R30, M26 = R41.R36, R41.R30, R41.M26
comps, outputs = R41.comps, R41.outputs
CHUNK, SEED = 4096, 43
KINDS = ("F_additive", "F_hist_mean", "F_goal_mean", "live_noH", "live_noH_attn", "live_noI", "live_noH_rot", "live_swapH",
         "live_noG", "live_swapG")


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
    fit = torch.nonzero(ctx.fit).squeeze(1)
    nL = t.max_prefix + 1
    cells = R36.Cells(ctx)
    N = len(cells.h)
    cL = ctx.L[cells.h]
    rows = torch.arange(N, device=dev)
    optg = ctx.opt[cells.h, cells.g]                                                          # [N, 4]
    # history-blind ceiling: per (g, L) the action most often optimal on fit-side histories
    B = torch.zeros(3, nL, dtype=torch.long, device=dev)
    for l in range(nL):
        m = fit[ctx.L[fit] == l]
        if len(m):
            B[:, l] = ctx.opt[m].double().mean(0).argmax(-1)
    ceil_ok = optg[rows, B[cells.g, cL]]
    cls = cells.cls
    res = dict(cells=dict(goal_matters=int((cls == 0).sum()), goal_neutral=int((cls == 1).sum())),
               ceiling=dict(goal_matters=float(ceil_ok[cls == 0].float().mean()), goal_neutral=float(ceil_ok[cls == 1].float().mean()),
                            all=float(ceil_ok.float().mean())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        C = comps(net)
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        # fit-side means per goal and length: c_k[g, L]; Xbar_k(L) is their mean over goals
        sums = {c: torch.zeros(3, nL, net.d, dtype=torch.float64, device=dev) for c in C}
        cnt = torch.bincount(ctx.L[fit], minlength=nL).double()
        for g in range(3):
            for i in range(0, len(fit), 8192):
                h = fit[i:i + 8192]
                o, _ = outputs(net, ctx, h, torch.full_like(h, g))
                for c in C:
                    sums[c][g].index_add_(0, ctx.L[h], o[c].double())
        cgl = {c: (sums[c] / cnt.clamp(min=1)[None, :, None]).float() for c in C}
        Xb = {c: cgl[c].mean(0) for c in C}                                                      # [nL, d]
        M = {c: cgl[c] - Xb[c][None] for c in C}                                                 # [3, nL, d]
        acts = {k: [] for k in ("natural", "natural_g2", "donor_natural") + KINDS + ("live_noG_g2",)}
        valid = []
        for i in range(0, N, CHUNK):
            h, g, g2 = cells.h[i:i + CHUNK], cells.g[i:i + CHUNK], cells.g2[i:i + CHUNK]
            n = torch.arange(len(h), device=dev)
            p = 1 + ctx.L[h]
            L = ctx.L[h]
            nat = [outputs(net, ctx, h, torch.full_like(h, gg)) for gg in range(3)]
            x = {c: torch.stack([nat[gg][0][c] for gg in range(3)], 1) for c in C}              # [n, 3, d]
            lg = torch.stack([nat[gg][1] for gg in range(3)], 1)                                 # [n, 3, 4]
            xbar = {c: x[c].mean(1) for c in C}
            inter = {c: x[c][n, g] - xbar[c] - M[c][g, L] for c in C}
            # donor h': another history of the same length in this chunk, and not the same history
            perm = n.clone()
            for l in range(nL):
                idx = torch.nonzero(L == l).squeeze(1)
                if len(idx) > 1:
                    perm[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
            valid.append(h[perm] != h)
            acts["natural"].append(lg[n, g].argmax(-1)); acts["natural_g2"].append(lg[n, g2].argmax(-1))
            acts["donor_natural"].append(lg[perm, g].argmax(-1))
            dH = {c: xbar[c] - Xb[c][L] for c in C}
            rot = {}
            for c in C:
                r = torch.randn(dH[c].shape, device=dev, generator=gen)
                rot[c] = r / r.norm(dim=-1, keepdim=True) * dH[c].norm(dim=-1, keepdim=True)

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

            attns = [c for c in C if c[0] == "attn"]
            patches = dict(F_additive={c: setter(xbar[c] + M[c][g, L]) for c in C},
                           F_hist_mean={c: setter(Xb[c][L] + M[c][g, L]) for c in C},
                           F_goal_mean={c: setter(xbar[c]) for c in C},
                           live_noH={c: adder(-dH[c]) for c in C},
                           live_noH_attn={c: adder(-dH[c]) for c in attns},
                           live_noI={c: adder(-inter[c]) for c in C},
                           live_noH_rot={c: adder(-rot[c]) for c in C},
                           live_swapH={c: adder(xbar[c][perm] - xbar[c]) for c in C},
                           live_noG={c: adder(-M[c][g, L]) for c in C},
                           live_swapG={c: adder(M[c][g2, L] - M[c][g, L]) for c in C})
            tok = R30.seq(ctx, h, g)
            for k in KINDS:
                acts[k].append(net(tok, patch=patches[k])[0][n, p].argmax(-1))
            acts["live_noG_g2"].append(net(R30.seq(ctx, h, g2), patch={c: adder(-M[c][g2, L]) for c in C})[0][n, p].argmax(-1))
            if i == 0:                                                                          # zero edit reproduces natural
                assert torch.equal(net(tok, patch={c: adder(torch.zeros_like(dH[c])) for c in C})[0][n, p].argmax(-1), lg[n, g].argmax(-1))
        A = {k: torch.cat(v) for k, v in acts.items()}
        valid = torch.cat(valid)
        grp = cells.g * nL + cL

        def hist_dep(act):
            cnt_ = torch.zeros(3 * nL, 4, device=dev).index_put_((grp, act), torch.ones(N, device=dev), accumulate=True)
            return float(1 - cnt_.max(-1).values.sum() / N)

        ok = lambda act: optg[rows, act]
        row = dict(checkpoint=ck, optimal={}, history_dependence={k: hist_dep(A[k]) for k in ("natural",) + KINDS},
                   agree_ceiling={k: float((A[k] == B[cells.g, cL]).float().mean()) for k in ("natural",) + KINDS})
        for k in ("natural",) + KINDS:
            o = ok(A[k]).float()
            row["optimal"][k] = dict(goal_matters=float(o[cls == 0].mean()), goal_neutral=float(o[cls == 1].mean()), all=float(o.mean()))
        inf = valid & (A["natural"] != A["donor_natural"])
        row["swapH"] = dict(pairs=int(inf.sum()), follow=float((A["live_swapH"] == A["donor_natural"])[inf].float().mean()),
                            stay=float((A["live_swapH"] == A["natural"])[inf].float().mean()))
        gm = cls == 0
        row["goal"] = dict(dependence_natural=float((A["natural"] != A["natural_g2"])[gm].float().mean()),
                           dependence_noG=float((A["live_noG"] != A["live_noG_g2"])[gm].float().mean()),
                           swapG_follow=float((A["live_swapG"] == A["natural_g2"])[gm].float().mean()),
                           swapG_stay=float((A["live_swapG"] == A["natural"])[gm].float().mean()))
        res["runs"][f"seed{s}"] = row
        print(f"seed{s} {ck} all-cells optimal: " + " ".join(f"{k} {v['all']:.3f}" for k, v in row["optimal"].items()) +
              f" | ceiling {res['ceiling']['all']:.3f} | hist dep: " + " ".join(f"{k} {v:.3f}" for k, v in row["history_dependence"].items()) +
              f" | swapH follow {row['swapH']['follow']:.3f} stay {row['swapH']['stay']:.3f} | goal {row['goal']} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
