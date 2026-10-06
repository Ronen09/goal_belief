"""The maze10 experiment: behaviour, information seeking, the additive code as a policy and as a state, decoders.
Measures and decision rule: PLAN.md. No solver: the exact filter and shortest paths are the only ground truth.

    .venv/bin/python studies/7_information_seeking/maze10/measure.py                 # writes results.json
    .venv/bin/python studies/7_information_seeking/maze10/measure.py --untrained --seeds 0      # smoke test (initial checkpoints)
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as TR                                                   # noqa: E402

N_EVAL, N_INT, SB, TOL = 8192, 4096, 25, 1e-6


# ------------------------------------------------------------------ the network as policies

def logits_goals(net, env, s):
    """Centred logits at the current decision under every goal: [N, K, 4]. The history is env's own."""
    K = len(env.t.goal_cell)
    tok = env.tok[:, : s + 2].repeat(K, 1, 1)
    tok[:, 1, MM.F_GOAL] = torch.arange(K, device=tok.device).repeat_interleave(env.N) + 1
    lg = net(tok)[0][:, s + 1].view(K, env.N, 4).transpose(0, 1).double()
    return lg - lg.mean(-1, keepdim=True)


def natural(net):
    def f(env, s):
        return net(env.tok[:, : s + 2])[0][:, s + 1].argmax(-1)
    return f


def additive(net, G, kind):
    """argmax of H + G (kind 'additive'), H alone ('goal_blind') or G alone ('history_blind')."""
    def f(env, s):
        g = G[env.goal, min(s, SB - 1)]
        if kind == "history_blind":
            return g.argmax(-1)
        H = logits_goals(net, env, s).mean(1)
        return (H + g if kind == "additive" else H).argmax(-1)
    return f


def removal(net, kind):
    """The network run under all goals with every component output at the decision token replaced, in order, by its
    parts recomputed on the edited run: noI = history part + goal part; noH / noG = that part removed."""
    comps = [(c, l) for l in range(net.nl) for c in ("attn", "mlp")]

    def f(env, s):
        K, N, p = len(env.t.goal_cell), env.N, s + 1
        w = (~env.done).double()[:, None]

        def patch(z):
            z = z.clone()
            v = z[:, p].view(K, N, -1).double()
            xbar = v.mean(0)                                                                  # history part [N, d]
            Xb = (xbar * w).sum(0) / w.sum().clamp(min=1)                                     # mean over live histories [d]
            M = (v * w[None]).sum(1) / w.sum().clamp(min=1) - Xb                              # goal part [K, d]
            new = xbar[None] + M[:, None] if kind == "noI" else v - (xbar - Xb)[None] if kind == "noH" else v - M[:, None]
            z[:, p] = new.reshape(K * N, -1).to(z.dtype)
            return z

        tok = env.tok[:, : s + 2].repeat(K, 1, 1)
        tok[:, 1, MM.F_GOAL] = torch.arange(K, device=tok.device).repeat_interleave(N) + 1
        lg = net(tok, patch={c: patch for c in comps})[0][:, p].view(K, N, 4)
        return lg[env.goal, torch.arange(N, device=tok.device)].argmax(-1)
    return f


def behaviour(fn):
    return lambda env, s: fn(env)


# ------------------------------------------------------------------ episodes with the filter's bookkeeping

@torch.no_grad()
def episodes(t, policy, goal, cell, seed, full=False):
    """Run a policy f(env, step) on fixed spawns and goals. Always: return, success, length. full: per decision the
    move, the QMDP value and the expected posterior entropy of every move, the entropy, the cell, the posterior."""
    gen = torch.Generator(device=t.dev); gen.manual_seed(seed)
    env = BM.Env(t, len(goal), gen, goal, cell)
    N, H = env.N, t.H
    z = lambda *s, dt=torch.float32: torch.zeros(N, H, *s, dtype=dt, device=t.dev)
    act, alive, rew, cells = z(dt=torch.long), z(dt=torch.bool), z(), z(dt=torch.long)
    qv, ee, en, bel = (z(4), z(4), z(), z(t.n)) if full else (None, None, None, None)
    for s in range(H):
        alive[:, s], cells[:, s] = ~env.done, env.cell
        if full:
            qv[:, s], ee[:, s], en[:, s], bel[:, s] = BM.qmdp_values(env), BM.expected_entropy(env), BM.entropy(env.belief), env.belief
        a = policy(env, s)
        act[:, s] = a
        rew[:, s] = env.step(a)
    disc = t.gamma ** torch.arange(H, device=t.dev)
    out = dict(ret=float((rew * disc).sum(1).mean()), success=float(rew.sum(1).mean()), length=float(alive.sum(1).float().mean()))
    return out, dict(tok=env.tok, act=act, alive=alive, rew=rew, cell=cells, qv=qv, ee=ee, en=en, bel=bel, goal=goal)


def seeking(t, r):
    """Deviations from QMDP and their expected information gain, on the decisions of one set of episodes."""
    a, m = r["act"][..., None], r["alive"]
    best = r["qv"].max(-1, keepdim=True).values
    dev = m & (r["qv"].gather(2, a).squeeze(2) < best.squeeze(2) - TOL)
    ig = r["en"][..., None] - r["ee"]                                                         # [N, H, 4] expected information gain
    ig_q = ig.gather(2, r["qv"].argmax(-1, keepdim=True)).squeeze(2)
    ig_a = ig.gather(2, a).squeeze(2)
    nonbest = r["qv"] < best - TOL
    ig_rand = (ig * nonbest).sum(-1) / nonbest.sum(-1).clamp(min=1)                           # a uniformly random deviation at the same state
    land = torch.zeros(t.n, dtype=torch.bool, device=t.dev); land[list(t.maze.landmarks)] = True
    on = land[r["cell"]] & m
    on[:, 0] = False
    f = lambda x, w: float(x[w].mean()) if w.any() else None
    return dict(deviation_rate=f(dev.float(), m), ig_advantage=f(ig_a - ig_q, dev), ig_advantage_random_same_state=f(ig_rand - ig_q, dev),
                more_informative=f((ig_a > ig_q + TOL).float(), dev), qmdp_value_given_up=f(best.squeeze(2) - r["qv"].gather(2, a).squeeze(2), dev),
                entropy_mean=f(r["en"], m), entropy_by_step=[f(r["en"][:, s], m[:, s]) for s in (0, 5, 10, 20, 30)],
                landmark_episodes=float(on.any(1).float().mean()))


# ------------------------------------------------------------------ decoders

def ridge_r2(X, Y, fit, test, lams=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Held-out R² of a ridge decoder; the penalty (relative to the mean feature variance) is chosen on a tenth of
    the fit decisions."""
    X = X.double(); Y = Y.double()
    mu, sd = X[fit].mean(0), X[fit].std(0).clamp(min=1e-8)
    X = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, dtype=torch.float64, device=X.device)], 1)
    idx = torch.nonzero(fit).squeeze(1)
    va, tr = idx[::10], idx[torch.arange(len(idx), device=idx.device) % 10 != 0]
    eye = torch.eye(X.shape[1], dtype=torch.float64, device=X.device)

    def solve(rows, lam):
        A = X[rows].T @ X[rows]
        return torch.linalg.solve(A + lam * torch.trace(A) / len(A) * eye, X[rows].T @ Y[rows])

    best = min(lams, key=lambda lam: float(((X[va] @ solve(tr, lam) - Y[va]) ** 2).sum()))
    P = X[test] @ solve(idx, best)
    return float(1 - ((P - Y[test]) ** 2).sum() / ((Y[test] - Y[fit].mean(0)) ** 2).sum()), P


@torch.no_grad()
def decoders(t, net, r):
    N, H = r["act"].shape
    dev, K = t.dev, len(t.goal_cell)
    store = {}
    xs = []
    for i in range(0, N, 2048):
        net(r["tok"][i:i + 2048], patch={("resid", net.nl): lambda x: (store.__setitem__("x", x), x)[1]})
        xs.append(store["x"][:, 1: H + 1])                                                    # decision tokens: positions 1..H
    x = torch.cat(xs)
    # realised discounted future visitation of every cell (the goal cell on success)
    oh = torch.nn.functional.one_hot(r["cell"], t.n).float() * r["alive"][..., None]
    occ = torch.zeros(N, H, t.n, device=dev)
    nxt = torch.zeros(N, t.n, device=dev)
    gcell = torch.nn.functional.one_hot(t.goal_cell[r["goal"]], t.n).float()
    for s in reversed(range(H)):
        nxt = oh[:, s] + t.gamma * (nxt + gcell * r["rew"][:, s, None])
        occ[:, s] = nxt
    ep, st = torch.nonzero(r["alive"], as_tuple=True)
    fit, test = ep % 2 == 0, ep % 2 == 1
    X, B, O = x[ep, st], r["bel"][ep, st], occ[ep, st]
    g1 = torch.nn.functional.one_hot(r["goal"][ep], K).float()
    s1 = torch.nn.functional.one_hot((st * 8 // H).clamp(max=7), 8).float()
    gs = (g1[:, :, None] * s1[:, None, :]).flatten(1)                                         # goal x step bin [m, 32]
    PG = torch.cat([(B[:, :, None] * gs[:, None, :]).flatten(1), gs], 1)                      # posterior x goal x step bin
    out = dict(decisions=len(ep))
    out["posterior_r2"], Pb = ridge_r2(X, B, fit, test)
    cell = r["cell"][ep, st][test]
    out["cell_accuracy_decoded"] = float((Pb.argmax(-1) == cell).float().mean())
    out["cell_accuracy_posterior"] = float((B[test].argmax(-1) == cell).float().mean())
    out["occupancy_r2_state"] = ridge_r2(X, O, fit, test)[0]
    out["occupancy_r2_posterior_goal_step"] = ridge_r2(PG, O, fit, test)[0]
    out["occupancy_r2_both"] = ridge_r2(torch.cat([X, PG], 1), O, fit, test)[0]
    out["occupancy_r2_state_goal_step"] = ridge_r2(torch.cat([X, gs], 1), O, fit, test)[0]
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoints")
    ap.add_argument("--runs", default=str(HERE / "runs" / "ppo"))
    ap.add_argument("--out", default=None, help="results file (default: results.json here)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    runs = Path(a.runs)
    task = json.load(open(runs / "task.json"))["task"]
    t = BM.Sim(BM.make(**task), dev)
    K = len(t.goal_cell)
    n_eval, n_int = (1024, 512) if a.untrained else (N_EVAL, N_INT)
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BM.Env(t, n_eval, gen)
    goal, cell = e.goal, e.cell
    gen.manual_seed(178)
    e = BM.Env(t, n_eval, gen)
    fgoal, fcell = e.goal, e.cell                                                             # separate episodes for fitting G
    res = dict(task=task, cells=t.n, references={}, runs={})
    rgen = torch.Generator(device=dev); rgen.manual_seed(5)
    for name, fn in (("oracle", BM.oracle), ("qmdp2", BM.qmdp2), ("qmdp", BM.qmdp), ("mls", BM.mls), ("random", BM.uniform(rgen))):
        s_, r = episodes(t, behaviour(fn), goal, cell, 79, full=True)
        res["references"][name] = dict(**s_, **seeking(t, r))
    print("references: " + " ".join(f"{k} {v['ret']:.3f}" for k, v in res["references"].items()), flush=True)
    out = Path(a.out) if a.out else HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        d = runs / f"seed{s}" / "ckpt"
        ck = d / "u000000.pt" if a.untrained else sorted(d.glob("u*.pt"))[-1]
        args = json.load(open(runs / "task.json"))["args"]
        net = TR.build(t.n_sym, K, t.H, args.get("d", 128), args.get("layers", 4))
        net.load_state_dict(torch.load(ck, map_location=dev))
        net = net.to(dev).eval()
        row = dict(checkpoint=ck.name)
        # A, B: behaviour and information seeking
        s_, r = episodes(t, natural(net), goal, cell, 79, full=True)
        row["natural"] = dict(**s_, **seeking(t, r))
        # the registered null: QMDP with random moves inserted at the model's deviation rate
        rate = row["natural"]["deviation_rate"]
        ngen = torch.Generator(device=dev); ngen.manual_seed(900 + s)
        def noisy(env, p=min(1.0, rate / 0.7)):
            rnd = torch.randint(4, (env.N,), device=dev, generator=ngen)
            return torch.where(torch.rand(env.N, device=dev, generator=ngen) < p, rnd, BM.qmdp(env))
        s2, r2 = episodes(t, behaviour(noisy), goal, cell, 79, full=True)
        row["noisy_qmdp"] = dict(**s2, **seeking(t, r2))
        # C: the additive code. G on separate episodes of the natural policy
        _, rf = episodes(t, natural(net), fgoal, fcell, 179)
        Gs, Gc = torch.zeros(K, SB, 4, dtype=torch.float64, device=dev), torch.zeros(SB, dtype=torch.float64, device=dev)
        lcs = {}
        for name, rr, gg in (("fit", rf, fgoal), ("test", r, goal)):
            L = []
            for st in range(t.H):
                tok = rr["tok"][:, : st + 2].repeat(K, 1, 1)
                tok[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(len(gg)) + 1
                lg = net(tok)[0][:, st + 1].view(K, len(gg), 4).transpose(0, 1).double()
                L.append(lg - lg.mean(-1, keepdim=True))
            lcs[name] = torch.stack(L, 1)                                                     # [N, H, K, 4]
        dv = lcs["fit"] - lcs["fit"].mean(2, keepdim=True)
        for st in range(t.H):
            m = rf["alive"][:, st]
            Gs[:, min(st, SB - 1)] += dv[m, st].sum(0); Gc[min(st, SB - 1)] += m.sum()
        G = Gs / Gc.clamp(min=1)[None, :, None]
        lc, m = lcs["test"], r["alive"]
        sb = torch.arange(t.H, device=dev).clamp(max=SB - 1)
        Gt = G[:, sb].permute(1, 0, 2)[None]                                                  # [1, H, K, 4]
        nat_a, add_a = lc.argmax(-1), (lc.mean(2, keepdim=True) + Gt).argmax(-1)              # [N, H, K]
        matters = m & (nat_a != nat_a[..., :1]).any(-1)                                       # the model's move depends on the goal
        dvt = lc - lc.mean(2, keepdim=True)
        row["additive_offline"] = dict(goal_matters_share=float(matters[m].float().mean()),
                                       same_action_goal_matters=float((nat_a == add_a)[matters].float().mean()),
                                       same_action_all=float((nat_a == add_a)[m].float().mean()),
                                       same_action_goal_blind=float((nat_a == lc.mean(2, keepdim=True).argmax(-1))[matters].float().mean()),
                                       additive_share=float(1 - ((dvt - (Gt - Gt.mean(2, keepdim=True))) ** 2)[m].sum() / (dvt ** 2)[m].sum()))
        row["online"] = {k: episodes(t, additive(net, G, k), goal, cell, 79)[0] for k in ("additive", "goal_blind", "history_blind")}
        # D: removals at the decision token, online (first n_int evaluation episodes)
        gi, ci = goal[:n_int], cell[:n_int]
        row["removal"] = dict(natural=episodes(t, natural(net), gi, ci, 79)[0], **{k: episodes(t, removal(net, k), gi, ci, 79)[0] for k in ("noI", "noH", "noG")},
                              goal_blind=episodes(t, additive(net, G, "goal_blind"), gi, ci, 79)[0])
        # E: decoders
        row["decoders"] = decoders(t, net, r)
        res["runs"][f"seed{s}"] = row
        rec = lambda x, blind, nat: (x - blind) / (nat - blind) if nat != blind else float('nan')
        o, rm, nt = row["online"], row["removal"], row["natural"]
        print(f"seed{s} {ck.name} return {nt['ret']:.3f} (success {nt['success']:.2f}) | deviations {nt['deviation_rate']:.3f} IG adv {nt['ig_advantage']:.4f} "
              f"(random same state {nt['ig_advantage_random_same_state']:.4f}, noisy qmdp {row['noisy_qmdp']['ig_advantage']:.4f}) | additive {o['additive']['ret']:.3f} "
              f"goal-blind {o['goal_blind']['ret']:.3f} history-blind {o['history_blind']['ret']:.3f} recovery {rec(o['additive']['ret'], o['goal_blind']['ret'], nt['ret']):.2f} "
              f"same action {row['additive_offline']['same_action_goal_matters']:.3f} | removal natural {rm['natural']['ret']:.3f} noI {rm['noI']['ret']:.3f} noH {rm['noH']['ret']:.3f} "
              f"noG {rm['noG']['ret']:.3f} | posterior R2 {row['decoders']['posterior_r2']:.3f} occupancy state {row['decoders']['occupancy_r2_state']:.3f} "
              f"posterior-goal-step {row['decoders']['occupancy_r2_posterior_goal_step']:.3f} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
