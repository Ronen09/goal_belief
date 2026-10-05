"""The additive-code experiment: the additive code at the goal token. Logits ~ H(h) + G(g, L); which decisions need the interaction, and
can the solver predict them? The observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/additive_code/run.py            # writes results.json

Natural logits at the goal token, centred over actions, l(h, g). Additive code:
    G(g, L)   goal bias: mean over fit-side histories of l(h, g) - mean_g' l(h, g'), per goal and prefix length
    H(h)      history profile: mean over the three goals of l(h, .)
    additive decision: argmax_a H_a(h) + G_a(g, L)
Additive solvability (solver and G only, no network decisions): for a posterior b and length L, with a*(g) the optimal
action under each goal (the first if tied; 99.9 % unique), is there any H in R^4 with H_{a*} + G_{a*}(g) > H_a + G_a(g)
for every goal g and action a != a*(g)? These are difference constraints; the best margin
    delta*(b, L) = min over directed cycles C of the constraint graph of (-sum of the cycle's weights) / |C|
is positive exactly when some H decides optimally under all three goals. Cells: held-out histories under each goal;
goal-dependent cells have an optimal set disjoint from another goal's.
"""

from __future__ import annotations

import argparse, importlib.util, itertools, json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R30 = load("r30run", ROUNDS / "5_goal_belief_mechanism" / "goal_swap_components" / "run.py")
M26 = R30.M26
CHUNK = 8192
CYCLES = [c for k in (2, 3, 4) for c in itertools.permutations(range(4), k) if c[0] == min(c)]   # simple directed cycles on 4 nodes


@torch.no_grad()
def logits(net, ctx, hist):
    out = []
    for g in range(3):
        lg = []
        for s in range(0, len(hist), CHUNK):
            h = hist[s:s + CHUNK]
            n = torch.arange(len(h), device=h.device)
            lg.append(net(R30.seq(ctx, h, torch.full_like(h, g)))[0][n, 1 + ctx.L[h]])
        out.append(torch.cat(lg))
    l = torch.stack(out, 1).double()
    return l - l.mean(-1, keepdim=True)                                                    # [n, goals, 4]


def margin(astar, G):
    """Best additive margin delta* for optimal actions astar [n, 3] (one per goal) under goal biases G [n, 3, 4].
    Constraint for goal g, action a != a*: H[a*] - H[a] >= w = G[a] - G[a*]. Edge a* -> a with weight w; the best margin
    is min over cycles of (-sum of max weights along the cycle) / length; +inf if no cycle exists."""
    n = len(astar)
    W = torch.full((n, 4, 4), -torch.inf, dtype=torch.float64, device=G.device)
    rows = torch.arange(n, device=G.device)
    for g in range(3):
        a = astar[:, g]
        w = G[:, g] - G[rows, g, a][:, None]                                               # [n, 4]: G[a] - G[a*]
        for b in range(4):
            m = a != b
            W[rows[m], a[m], b] = torch.maximum(W[rows[m], a[m], b], w[m, b])
    best = torch.full((n,), torch.inf, dtype=torch.float64, device=G.device)
    for c in CYCLES:
        tot = torch.zeros(n, dtype=torch.float64, device=G.device)
        for i in range(len(c)):
            tot = tot + W[:, c[i], c[(i + 1) % len(c)]]
        best = torch.minimum(best, torch.where(torch.isfinite(tot), -tot / len(c), torch.full_like(tot, torch.inf)))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    nL = t.max_prefix + 1
    fit, test = torch.nonzero(ctx.fit).squeeze(1), torch.nonzero(ctx.test & (ctx.L >= 1)).squeeze(1)
    opt = ctx.opt[test]                                                                     # [n, 3, 4]
    astar = ctx.Q[test].argmax(-1)                                                          # [n, 3]
    dep = torch.zeros(len(test), 3, dtype=torch.bool, device=dev)                           # goal-dependent cells
    for g in range(3):
        for g2 in range(3):
            if g != g2:
                dep[:, g] |= ~(opt[:, g] & opt[:, g2]).any(-1)
    res = dict(cells=dict(histories=len(test), goal_dependent=int(dep.sum()), all=3 * len(test)), runs={})
    for s in a.seeds:
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        lf = logits(net, ctx, fit)
        dev_f = lf - lf.mean(1, keepdim=True)
        G = torch.zeros(3, nL, 4, dtype=torch.float64, device=dev)
        for l in range(nL):
            m = ctx.L[fit] == l
            if m.any():
                G[:, l] = dev_f[m].mean(0)
        lt = logits(net, ctx, test)
        L = ctx.L[test]
        Gt = G[:, L].permute(1, 0, 2)                                                       # [n, 3, 4]
        H = lt.mean(1)                                                                      # [n, 4]
        add = H[:, None] + Gt
        nat_a, add_a = lt.argmax(-1), add.argmax(-1)
        ok = lambda act: torch.gather(opt, 2, act[..., None]).squeeze(-1)                  # [n, 3]
        nat_ok, add_ok = ok(nat_a), ok(add_a)
        # additive share of the logits' goal dependence
        within = lt - lt.mean(1, keepdim=True)
        add_share = float(1 - ((within - (Gt - Gt.mean(1, keepdim=True))) ** 2).sum() / (within ** 2).sum())
        # solvability from the solver and G
        delta = margin(astar, Gt)
        solv = delta > 0
        need = nat_ok & ~add_ok                                                            # cells where the interaction is needed
        spoil = ~nat_ok & add_ok
        sel = lambda m: m & dep
        row = dict(checkpoint=ck, additive_share_of_goal_dependence=add_share,
                   all_cells=dict(natural=float(nat_ok.float().mean()), additive=float(add_ok.float().mean())),
                   goal_dependent=dict(natural=float(nat_ok[dep].float().mean()), additive=float(add_ok[dep].float().mean()),
                                       agree=float((nat_a == add_a)[dep].float().mean())),
                   solvable_share_histories=float(solv.float().mean()),
                   need=dict(share=float(need[dep].float().mean()), spoil=float(spoil[dep].float().mean()),
                             p_need_given_unsolvable=float(need[dep & ~solv[:, None]].float().mean()) if (dep & ~solv[:, None]).any() else None,
                             p_need_given_solvable=float(need[dep & solv[:, None]].float().mean()) if (dep & solv[:, None]).any() else None,
                             share_of_need_in_unsolvable=float((need & ~solv[:, None])[dep].float().sum() / need[dep].float().sum().clamp(min=1)),
                             unsolvable_share_of_dep_cells=float((~solv[:, None]).expand_as(dep)[dep].float().mean()),
                             natural_given_unsolvable=float(nat_ok[dep & ~solv[:, None]].float().mean()) if (dep & ~solv[:, None]).any() else None,
                             additive_given_unsolvable=float(add_ok[dep & ~solv[:, None]].float().mean()) if (dep & ~solv[:, None]).any() else None,
                             natural_given_solvable=float(nat_ok[dep & solv[:, None]].float().mean()),
                             additive_given_solvable=float(add_ok[dep & solv[:, None]].float().mean())),
                   G=G.tolist())
        # what H holds: are the goals' optimal actions (their union) H's top actions?
        union = opt.any(1)                                                                  # [n, 4]
        k = union.sum(-1)
        top = torch.argsort(H, -1, descending=True)
        rank_of = torch.argsort(top, -1)                                                    # rank of each action in H
        union_top = (torch.where(union, rank_of, torch.zeros_like(rank_of)).amax(-1) < k)
        row["H"] = dict(union_on_top=float(union_top[k < 4].float().mean()), union_size_mean=float(k.float().mean()))
        res["runs"][f"seed{s}"] = row
        nd = row["need"]
        print(f"seed{s} {ck} additive share {add_share:.3f} | goal-dependent: natural {row['goal_dependent']['natural']:.3f} additive {row['goal_dependent']['additive']:.3f} | "
              f"solvable {row['solvable_share_histories']:.3f} | need {nd['share']:.3f} spoil {nd['spoil']:.3f} P(need|unsolv) {nd['p_need_given_unsolvable']} P(need|solv) {nd['p_need_given_solvable']:.3f} "
              f"share of need in unsolvable {nd['share_of_need_in_unsolvable']:.2f} | H union on top {row['H']['union_on_top']:.3f} | {time.time() - t0:.0f}s", flush=True)
        (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
