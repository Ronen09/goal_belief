"""The random-goals experiment: behaviour, information seeking, the additive code as a policy and as a state, the
posterior decoder, and the fixed bias per goal. Measures and decision rule: PLAN.md. The maze10 experiment's measures
(`../maze10/measure.py`) with the network run under all 55 goals in chunks.

    .venv/bin/python studies/7_information_seeking/random_goals/measure.py                 # writes results.json
    .venv/bin/python studies/7_information_seeking/random_goals/measure.py --untrained --seeds 0      # smoke test (initial checkpoints)
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "maze10"))
import measure as M10                                                # noqa: E402
import train as TR                                                   # noqa: E402

N_EVAL, N_FIT, N_INT, SB, MIN_BIN, CHUNK = 8192, 32768, 4096, 25, 50, 120_000
episodes, seeking, natural, behaviour, removal, ridge_r2 = M10.episodes, M10.seeking, M10.natural, M10.behaviour, M10.removal, M10.ridge_r2


def logits_goals(net, tok, p, K):
    """Centred logits at position p of each sequence under every goal: [N, K, 4], the network run in chunks."""
    N = tok.shape[0]
    out = torch.empty(N, K, 4, dtype=torch.float64, device=tok.device)
    per = max(1, CHUNK // N)
    for g0 in range(0, K, per):
        gs = torch.arange(g0, min(K, g0 + per), device=tok.device)
        tk = tok.repeat(len(gs), 1, 1)
        tk[:, 1, MM.F_GOAL] = gs.repeat_interleave(N) + 1
        lg = net(tk)[0][:, p].view(len(gs), N, 4).transpose(0, 1).double()
        out[:, gs] = lg - lg.mean(-1, keepdim=True)
    return out


def additive(net, G, kind, K):
    def f(env, s):
        g = G[env.goal, min(s, SB - 1)]
        if kind == "history_blind":
            return g.argmax(-1)
        H = logits_goals(net, env.tok[:, : s + 2], s + 1, K).mean(1)
        return (H + g if kind == "additive" else H).argmax(-1)
    return f


def fit_G(net, t, rf, K):
    """G(goal, step bin): the mean goal deviation of the centred logits on the fit episodes; a bin with fewer than
    MIN_BIN decisions takes the goal's mean over all steps."""
    dev = t.dev
    Gs, Gc = torch.zeros(K, SB, 4, dtype=torch.float64, device=dev), torch.zeros(K, SB, dtype=torch.float64, device=dev)
    N = rf["tok"].shape[0]
    for st in range(t.H):
        m = rf["alive"][:, st]
        if not m.any():
            break
        idx = torch.nonzero(m).squeeze(1)
        for i0 in range(0, len(idx), 8192):
            ii = idx[i0:i0 + 8192]
            lc = logits_goals(net, rf["tok"][ii, : st + 2], st + 1, K)                          # [n, K, 4]
            dv = lc - lc.mean(1, keepdim=True)
            Gs[:, min(st, SB - 1)] += dv.sum(0); Gc[:, min(st, SB - 1)] += len(ii)
    G_all = Gs.sum(1) / Gc.sum(1).clamp(min=1)[:, None]                                        # [K, 4] per goal over all steps
    G = Gs / Gc.clamp(min=1)[..., None]
    sparse = Gc < MIN_BIN
    G[sparse] = G_all[:, None].expand(K, SB, 4)[sparse]
    return G, G_all, dict(bins_sparse=float(sparse.float().mean()), fit_decisions=float(Gc.sum()), fit_per_goal_min=float(Gc.sum(1).min()))


@torch.no_grad()
def offline(net, t, r, G, K):
    """On the evaluation episodes: the model's move under every goal, the additive choice, the goal-blind choice."""
    dev, N = t.dev, r["tok"].shape[0]
    sb = torch.arange(t.H, device=dev).clamp(max=SB - 1)
    tot = dict(matters=0.0, same=0.0, same_blind=0.0, alive=0.0, same_all=0.0, num=0.0, den=0.0)
    for st in range(t.H):
        m = r["alive"][:, st]
        if not m.any():
            break
        idx = torch.nonzero(m).squeeze(1)
        for i0 in range(0, len(idx), 8192):
            ii = idx[i0:i0 + 8192]
            lc = logits_goals(net, r["tok"][ii, : st + 2], st + 1, K)                           # [n, K, 4]
            H = lc.mean(1, keepdim=True)
            Gt = G[:, sb[st]][None]                                                            # [1, K, 4]
            nat_a, add_a, bl_a = lc.argmax(-1), (H + Gt).argmax(-1), H.argmax(-1)
            matters = (nat_a != nat_a[:, :1]).any(-1)
            same = (nat_a == add_a)
            tot["alive"] += len(ii); tot["matters"] += float(matters.sum())
            tot["same"] += float(same[matters].float().mean(1).sum()); tot["same_blind"] += float((nat_a == bl_a)[matters].float().mean(1).sum())
            tot["same_all"] += float(same.float().mean(1).sum())                                # as maze10: per decision, averaged over the goals
            dvt = lc - H
            tot["num"] += float(((dvt - (Gt - Gt.mean(1, keepdim=True))) ** 2).sum()); tot["den"] += float((dvt ** 2).sum())
    return dict(goal_matters_share=tot["matters"] / tot["alive"], same_action_goal_matters=tot["same"] / max(tot["matters"], 1),
                same_action_all=tot["same_all"] / tot["alive"], same_action_goal_blind=tot["same_blind"] / max(tot["matters"], 1),
                additive_share=1 - tot["num"] / tot["den"])


@torch.no_grad()
def decoders(t, net, r):
    N, H = r["act"].shape
    store = {}
    xs = []
    for i in range(0, N, 2048):
        net(r["tok"][i:i + 2048], patch={("resid", net.nl): lambda x: (store.__setitem__("x", x), x)[1]})
        xs.append(store["x"][:, 1: H + 1])
    x = torch.cat(xs)
    ep, st = torch.nonzero(r["alive"], as_tuple=True)
    fit, test = ep % 2 == 0, ep % 2 == 1
    X, B = x[ep, st], r["bel"][ep, st]
    out = dict(decisions=len(ep))
    out["posterior_r2"], Pb = ridge_r2(X, B, fit, test)
    cell = r["cell"][ep, st][test]
    out["cell_accuracy_decoded"] = float((Pb.argmax(-1) == cell).float().mean())
    out["cell_accuracy_posterior"] = float((B[test].argmax(-1) == cell).float().mean())
    return out


def bias_field(t, G_all):
    """The preferred move per goal (argmax of G over all steps): the share of cells from which it is a shortest-path
    move to the goal; the best single move per goal for comparison."""
    dn = t.dist[:, t.nxt_cell]                                                                 # [K, n, 4] distance after the move
    opt = dn <= dn.min(-1, keepdim=True).values + 1e-6                                         # [K, n, 4]
    K, n = opt.shape[:2]
    keep = ~torch.eye(n, dtype=torch.bool, device=t.dev)[: K]                                  # no decision on the goal itself
    a = G_all.argmax(-1)                                                                       # [K]
    cov = opt[torch.arange(K, device=t.dev), :, a]                                             # [K, n]
    best = (opt & keep[..., None]).float().sum(1).max(-1)                                      # best single move per goal
    return dict(bias_move_optimal_share=float(cov[keep].float().mean()), best_single_move_share=float((best.values / (n - 1)).mean()),
                bias_is_best_single_move=float((a == best.indices).float().mean()))


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
    n_eval, n_fit, n_int = (1024, 2048, 512) if a.untrained else (N_EVAL, N_FIT, N_INT)
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BM.Env(t, n_eval, gen)
    goal, cell = e.goal, e.cell
    gen.manual_seed(178)
    e = BM.Env(t, n_fit, gen)
    fgoal, fcell = e.goal, e.cell                                                             # separate episodes for fitting G
    res = dict(task=task, cells=t.n, goals=K, references={}, runs={})
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
        rate = row["natural"]["deviation_rate"]
        ngen = torch.Generator(device=dev); ngen.manual_seed(900 + s)
        def noisy(env, p=min(1.0, rate / 0.7)):
            rnd = torch.randint(4, (env.N,), device=dev, generator=ngen)
            return torch.where(torch.rand(env.N, device=dev, generator=ngen) < p, rnd, BM.qmdp(env))
        s2, r2 = episodes(t, behaviour(noisy), goal, cell, 79, full=True)
        row["noisy_qmdp"] = dict(**s2, **seeking(t, r2))
        # C: the additive code. G on separate episodes of the natural policy
        _, rf = episodes(t, natural(net), fgoal, fcell, 179)
        G, G_all, ginfo = fit_G(net, t, rf, K)
        row["G_fit"] = ginfo
        row["additive_offline"] = offline(net, t, r, G, K)
        row["online"] = {k: episodes(t, additive(net, G, k, K), goal, cell, 79)[0] for k in ("additive", "goal_blind", "history_blind")}
        # D: removals at the decision token, online (first n_int evaluation episodes)
        gi, ci = goal[:n_int], cell[:n_int]
        row["removal"] = dict(natural=episodes(t, natural(net), gi, ci, 79)[0], **{k: episodes(t, removal(net, k), gi, ci, 79)[0] for k in ("noI", "noH", "noG")},
                              goal_blind=episodes(t, additive(net, G, "goal_blind", K), gi, ci, 79)[0])
        # E: the posterior decoder.  F: the fixed bias per goal
        row["decoders"] = decoders(t, net, r)
        row["bias_field"] = bias_field(t, G_all)
        row["G_all"] = G_all.cpu().tolist()
        res["runs"][f"seed{s}"] = row
        rec = lambda x, blind, nat: (x - blind) / (nat - blind) if nat != blind else float('nan')
        o, rm, nt = row["online"], row["removal"], row["natural"]
        print(f"seed{s} {ck.name} return {nt['ret']:.3f} (success {nt['success']:.2f}) | deviations {nt['deviation_rate']:.3f} IG adv {nt['ig_advantage']:.4f} "
              f"(random same state {nt['ig_advantage_random_same_state']:.4f}, noisy qmdp {row['noisy_qmdp']['ig_advantage']:.4f}) | additive {o['additive']['ret']:.3f} "
              f"goal-blind {o['goal_blind']['ret']:.3f} history-blind {o['history_blind']['ret']:.3f} recovery {rec(o['additive']['ret'], o['goal_blind']['ret'], nt['ret']):.2f} "
              f"same action {row['additive_offline']['same_action_goal_matters']:.3f} share {row['additive_offline']['additive_share']:.3f} | removal natural {rm['natural']['ret']:.3f} "
              f"noI {rm['noI']['ret']:.3f} noH {rm['noH']['ret']:.3f} noG {rm['noG']['ret']:.3f} | posterior R2 {row['decoders']['posterior_r2']:.3f} | "
              f"bias move optimal {row['bias_field']['bias_move_optimal_share']:.3f} (best single {row['bias_field']['best_single_move_share']:.3f}) | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
