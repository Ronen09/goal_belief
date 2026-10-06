"""The additive-ablation experiment, post hoc (not in PLAN.md): ablations recomputed under upstream edits, and attention-only edits.

    .venv/bin/python studies/5_goal_belief_mechanism/additive_ablation/explore.py            # writes explore.json

run.py's live kinds subtract parts measured in natural runs. Once an upstream edit has changed a later component's
output, the natural part subtracted there is no longer the part present, and the edit can overshoot (e.g. subtract a
history part the component no longer makes). Here every history is run under all three goals in one batch. At each
component, in order, the parts are measured from the edited outputs themselves:
    xbar^e(h)   mean over the three goals of the edited output
    Xbar^e(L)   mean of xbar^e over the batch's histories of length L
    M^e(g, L)   mean over the batch's histories of length L under g, minus Xbar^e(L)
and the edit is applied before later components read it. Kinds:
    rec_noH     x - (xbar^e(h) - Xbar^e(L))       history main effect removed at every component, interaction kept
    rec_noI     xbar^e(h) + M^e(g, L)              interaction removed at every component (live counterpart of all_hist)
    rec_noG     x - M^e(g, L)                      goal main effect removed, interaction kept
    rec_noHI    Xbar^e(L) + M^e(g, L)              both: history-blind by construction (reference)
Attention-only kinds with run.py's fixed natural vectors, at the components the history (or the goal) enters through:
    attn_noI, attn_swapH (four attention outputs); attn0_noG, attn0_swapG (block 0's attention output only).
"""

from __future__ import annotations

import importlib.util, json, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RUN = load("aarun", HERE / "run.py")
R36, R30, M26, comps, outputs = RUN.R36, RUN.R30, RUN.M26, RUN.comps, RUN.outputs
CHUNK, SEED = 4096, 44
REC = ("rec_noH", "rec_noI", "rec_noG", "rec_noHI")
FIX = ("attn_noI", "attn_swapH", "attn0_noG", "attn0_swapG")


def rec_patch(kind, n, p3, L, nL):
    """Patch for one component acting on a [3n, T, d] batch (goal-major): parts measured from the edited outputs."""
    def f(z):
        z = z.clone()
        v = z[torch.arange(3 * n, device=z.device), p3].view(3, n, -1)
        xbar = v.mean(0)                                                                     # [n, d]
        cntL = torch.bincount(L, minlength=nL).clamp(min=1).float()[:, None]
        Xb = torch.zeros(nL, v.shape[-1], device=z.device).index_add_(0, L, xbar) / cntL     # [nL, d]
        cg = torch.stack([torch.zeros(nL, v.shape[-1], device=z.device).index_add_(0, L, v[g]) / cntL for g in range(3)])
        M = (cg - Xb[None])[:, L]                                                            # [3, n, d]
        if kind == "rec_noH":
            new = v - (xbar - Xb[L])[None]
        elif kind == "rec_noI":
            new = xbar[None] + M
        elif kind == "rec_noG":
            new = v - M
        else:
            new = (Xb[L])[None] + M
        z[torch.arange(3 * n, device=z.device), p3] = new.reshape(3 * n, -1)
        return z
    return f


def main():
    torch.set_grad_enabled(False)
    dev, t0 = "cuda", time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    fit = torch.nonzero(ctx.fit).squeeze(1)
    nL = t.max_prefix + 1
    cells = R36.Cells(ctx)
    N = len(cells.h)
    cL = ctx.L[cells.h]
    rows = torch.arange(N, device=dev)
    optg = ctx.opt[cells.h, cells.g]
    cls = cells.cls
    B = torch.zeros(3, nL, dtype=torch.long, device=dev)
    for l in range(nL):
        m = fit[ctx.L[fit] == l]
        if len(m):
            B[:, l] = ctx.opt[m].double().mean(0).argmax(-1)
    grp = cells.g * nL + cL
    res = dict(runs={})
    for s in range(10):
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        C = comps(net)
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        sums = {c: torch.zeros(3, nL, net.d, dtype=torch.float64, device=dev) for c in C}
        cnt = torch.bincount(ctx.L[fit], minlength=nL).double()
        for g in range(3):
            for i in range(0, len(fit), 8192):
                h = fit[i:i + 8192]
                o, _ = outputs(net, ctx, h, torch.full_like(h, g))
                for c in C:
                    sums[c][g].index_add_(0, ctx.L[h], o[c].double())
        cgl = {c: (sums[c] / cnt.clamp(min=1)[None, :, None]).float() for c in C}
        Xb = {c: cgl[c].mean(0) for c in C}
        M = {c: cgl[c] - Xb[c][None] for c in C}
        acts = {k: [] for k in ("natural", "natural_g2", "donor_natural", "rec_noG_g2", "attn0_noG_g2") + REC + FIX}
        valid = []
        for i in range(0, N, CHUNK):
            h, g, g2 = cells.h[i:i + CHUNK], cells.g[i:i + CHUNK], cells.g2[i:i + CHUNK]
            n = len(h)
            ar = torch.arange(n, device=dev)
            p, L = 1 + ctx.L[h], ctx.L[h]
            nat = [outputs(net, ctx, h, torch.full_like(h, gg)) for gg in range(3)]
            x = {c: torch.stack([nat[gg][0][c] for gg in range(3)], 1) for c in C}
            lg = torch.stack([nat[gg][1] for gg in range(3)], 1)
            xbar = {c: x[c].mean(1) for c in C}
            perm = ar.clone()
            for l in range(nL):
                idx = torch.nonzero(L == l).squeeze(1)
                if len(idx) > 1:
                    perm[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
            valid.append(h[perm] != h)
            acts["natural"].append(lg[ar, g].argmax(-1)); acts["natural_g2"].append(lg[ar, g2].argmax(-1))
            acts["donor_natural"].append(lg[perm, g].argmax(-1))
            # recomputed kinds: all three goals in one batch, goal-major
            h3 = h.repeat(3)
            g3 = torch.arange(3, device=dev).repeat_interleave(n)
            tok3 = R30.seq(ctx, h3, g3)
            p3 = p.repeat(3)
            for k in REC:
                lg3 = net(tok3, patch={c: rec_patch(k, n, p3, L, nL) for c in C})[0][torch.arange(3 * n, device=dev), p3].view(3, n, 4)
                acts[k].append(lg3[g, ar].argmax(-1))
                if k == "rec_noG":
                    acts["rec_noG_g2"].append(lg3[g2, ar].argmax(-1))
            if i == 0:                                                                          # the batch reproduces natural
                assert torch.equal(net(tok3)[0][torch.arange(3 * n, device=dev), p3].view(3, n, 4)[g, ar].argmax(-1), lg[ar, g].argmax(-1))

            def adder(delta):
                def f(z):
                    z = z.clone(); z[ar, p] = z[ar, p] + delta
                    return z
                return f

            attns = [c for c in C if c[0] == "attn"]
            a0 = ("attn", 0)
            inter = {c: x[c][ar, g] - xbar[c] - M[c][g, L] for c in C}
            fixp = dict(attn_noI={c: adder(-inter[c]) for c in attns},
                        attn_swapH={c: adder(xbar[c][perm] - xbar[c]) for c in attns},
                        attn0_noG={a0: adder(-M[a0][g, L])},
                        attn0_swapG={a0: adder(M[a0][g2, L] - M[a0][g, L])})
            tok = R30.seq(ctx, h, g)
            for k in FIX:
                acts[k].append(net(tok, patch=fixp[k])[0][ar, p].argmax(-1))
            acts["attn0_noG_g2"].append(net(R30.seq(ctx, h, g2), patch={a0: adder(-M[a0][g2, L])})[0][ar, p].argmax(-1))
        A = {k: torch.cat(v) for k, v in acts.items()}
        valid = torch.cat(valid)

        def hist_dep(act):
            c_ = torch.zeros(3 * nL, 4, device=dev).index_put_((grp, act), torch.ones(N, device=dev), accumulate=True)
            return float(1 - c_.max(-1).values.sum() / N)

        ks = ("natural",) + REC + FIX
        row = dict(checkpoint=ck, optimal={}, history_dependence={k: hist_dep(A[k]) for k in ks},
                   agree_ceiling={k: float((A[k] == B[cells.g, cL]).float().mean()) for k in ks})
        for k in ks:
            o = optg[rows, A[k]].float()
            row["optimal"][k] = dict(goal_matters=float(o[cls == 0].mean()), goal_neutral=float(o[cls == 1].mean()), all=float(o.mean()))
        inf = valid & (A["natural"] != A["donor_natural"])
        row["attn_swapH"] = dict(follow=float((A["attn_swapH"] == A["donor_natural"])[inf].float().mean()),
                                 stay=float((A["attn_swapH"] == A["natural"])[inf].float().mean()))
        gm = cls == 0
        row["goal"] = dict(dependence_natural=float((A["natural"] != A["natural_g2"])[gm].float().mean()),
                           dependence_rec_noG=float((A["rec_noG"] != A["rec_noG_g2"])[gm].float().mean()),
                           dependence_attn0_noG=float((A["attn0_noG"] != A["attn0_noG_g2"])[gm].float().mean()),
                           attn0_swapG_follow=float((A["attn0_swapG"] == A["natural_g2"])[gm].float().mean()))
        res["runs"][f"seed{s}"] = row
        print(f"seed{s} all-cells optimal: " + " ".join(f"{k} {v['all']:.3f}" for k, v in row["optimal"].items()) +
              " | hist dep: " + " ".join(f"{k} {v:.3f}" for k, v in row["history_dependence"].items()) +
              f" | attn_swapH {row['attn_swapH']} | goal {row['goal']} | {time.time() - t0:.0f}s", flush=True)
        (HERE / "explore.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
