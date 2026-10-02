"""Round 33: round 32's belief encoding and edits one step earlier, at the input of block 0's MLP at the goal token.
Round 23's reward models, frozen.

    .venv/bin/python rounds/r33_attention_belief_edit/run.py            # writes results.json
    .venv/bin/python rounds/r33_attention_belief_edit/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

Site: m(h, g), the residual stream at the goal token after block 0's attention and before its MLP ("resid_mid", 0).
It is the goal token's embedding, a function of (goal, prefix length) only, plus block 0's attention output, so fitting
c_{g,L} + E b at m is the same as fitting it at the attention output. Encodings, edits, pairs and measures are round 32's
(its run.py, with SITE changed). Block 0's MLP acts on each token separately, so `none`, `whole` and `hybrid` are the
same runs as round 32's.

Added here:
    interaction     the share of the state's variation (around its goal-and-length means) that changes with the goal for a
                    fixed history: 0 if the site holds a goal-free evidence code plus a goal offset; at this site and at
                    round 32's
    propagation     the shared edit at m, carried through block 0's MLP: the change it induces at round 32's site,
                    against the actual difference there and against round 32's shared and goal-specific edit vectors
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R32 = load("r32run", ROUNDS / "r32_direct_belief_edit" / "run.py")
R27, R30, M26 = R32.R27, R32.R30, R32.M26
SITE, SITE_R32 = ("resid_mid", 0), ("resid", 1)
N_PROP = 8192


def interaction(enc, ctx):
    """Share of the within-(goal, length) variation of the site's state, on held-out histories, that varies with the
    goal for a fixed history (around the per-history mean over goals)."""
    t = ctx.t
    test = torch.nonzero(ctx.test).squeeze(1)
    Z = enc.Z[test]                                                                       # [n, goals, d]
    L = ctx.L[test]
    cm = torch.zeros_like(Z)
    for l in range(t.max_prefix + 1):
        m = L == l
        if m.any():
            cm[m] = Z[m].mean(0, keepdim=True)
    Zc = Z - cm
    inter = Zc - Zc.mean(1, keepdim=True)
    return float((inter ** 2).sum() / (Zc ** 2).sum())


@torch.no_grad()
def propagation(net, ctx, enc, enc32, a, b):
    """The shared edit at m carried through block 0's MLP: induced change at round 32's site against the actual
    difference there and against round 32's edit vectors. Pooled over goals."""
    out = {k: [] for k in ("cos_actual", "cos_r32_shared", "cos_r32_goal", "norm_over_actual", "norm_over_r32_goal")}
    n = torch.arange(len(a), device=a.device)
    gp = 1 + ctx.L[a]
    db = enc.bel[b] - enc.bel[a]
    cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=1)
    for g in range(3):
        z = (enc.Z[a, g] + db @ enc.E).float()
        def f(x):
            x = x.clone(); x[n, gp] = z
            return x
        rec = net(R30.seq(ctx, a, torch.full_like(a, g)), patch={SITE: f}, record=True)[2]
        induced = rec["resid"][1][n, gp].double() - enc32.Z[a, g]
        actual = enc32.Z[b, g] - enc32.Z[a, g]
        e32, e32g = db @ enc32.E, db @ enc32.E_goal[g]
        out["cos_actual"].append(cos(induced, actual)); out["cos_r32_shared"].append(cos(induced, e32)); out["cos_r32_goal"].append(cos(induced, e32g))
        out["norm_over_actual"].append(induced.norm(dim=1) / actual.norm(dim=1).clamp(min=1e-9))
        out["norm_over_r32_goal"].append(induced.norm(dim=1) / e32g.norm(dim=1).clamp(min=1e-9))
    return {k: float(torch.cat(v).mean()) if k.startswith("cos") else float(torch.cat(v).median()) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    m, o = pairs.main, pairs.onestep
    eq, eq_ok = R32.equivalents(ctx, m["r"])
    res = dict(counts=dict(main_pairs=len(m["r"]), onestep_pairs=len(o["r"]), equivalent_pairs=int(eq_ok.sum())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        R32.SITE = SITE_R32
        enc32 = R32.Encoding(net, ctx, torch.Generator(device=dev).manual_seed(3200 + s))     # round 32's site, for comparison
        R32.SITE = SITE
        enc = R32.Encoding(net, ctx, torch.Generator(device=dev).manual_seed(3300 + s))
        ra, rb = m["r"], m["d"]
        row = dict(checkpoint=ck, fit=enc.fit_stats, edit_stats=enc.stats(ra, rb),
                   interaction=dict(site=interaction(enc, ctx), r32_site=interaction(enc32, ctx)),
                   propagation=propagation(net, ctx, enc, enc32, ra[:N_PROP], rb[:N_PROP]))
        del enc32
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
        c = row["main"]["change"]
        print(f"seed{s} {ck} R² shared {enc.fit_stats['shared']['r2_within_goal_length']:.3f} goal-specific {enc.fit_stats['goal_specific']['r2_within_goal_length']:.3f} (within) | "
              f"interaction {row['interaction']['site']:.3f} (r32 site {row['interaction']['r32_site']:.3f}) | change donor-optimal: " +
              " ".join(f"{k} {c[k]['donor_optimal']:.3f}" for k in R32.KINDS + ('hybrid',)) +
              f" | goal-dep enc {row['main']['goal_dependent']['encoding']['right_every_change_goal']:.2f} whole {row['main']['goal_dependent']['whole']['right_every_change_goal']:.2f} | "
              f"prop cos actual {row['propagation']['cos_actual']:.2f} r32 goal {row['propagation']['cos_r32_goal']:.2f} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
