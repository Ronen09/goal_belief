"""The random-spawns experiment: the additive code in each arm's models, and in the solver's own Q*. PLAN.md.

    .venv/bin/python studies/6_hard_cases_and_tasks/random_spawns/measure.py            # writes results.json

Per arm: episodes of the solver with 20 % random moves on that arm's task; every decision replayed under all three
goals through the belief graph (the hard-cases experiment's later-decisions analysis). Logits at the decision token,
centred over actions, l(h, g). G(g; prefix length, step) is the mean goal deviation on one half of the episodes;
H(h) the mean over goals; the additive decision is argmax H + G. Everything is reported on the other half.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import torch

from goalgeo import mazeaux as AX, mazemodel as MM, mazeppo as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task as TK                                                    # noqa: E402

spec = importlib.util.spec_from_file_location("r42run", HERE.parent.parent / "5_goal_belief_mechanism" / "additive_code" / "run.py")
R42 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R42)
N, EPS, SEED, STEPS = 40000, 0.2, 4545, 7


def cases(t, quick):
    """Decisions of the eps-random solver, each with its belief node under every goal."""
    dev = t.dev
    gen = torch.Generator(device=dev); gen.manual_seed(SEED)
    n_ep = 2000 if quick else N
    b = P.rollout(None, t, n_ep, gen, behaviour=P.solver_behaviour(t, EPS, gen))
    n = torch.arange(n_ep, device=dev)
    pre = b.pre_node[n, b.prefix]
    nodes = torch.full((n_ep, t.H, 3), -1, dtype=torch.long, device=dev)
    cur = torch.stack([t.reveal[pre, g] for g in range(3)], 1)
    for s in range(t.H):
        nodes[:, s] = cur
        o = b.tok[n, (b.pos[:, s] + 1).clamp(max=t.L - 1), MM.F_SYM] - 1
        a = b.act[:, s]
        ok = (cur >= 0) & (o >= 0)[:, None]
        nxt = t.node_nxt[cur.clamp(min=0), a[:, None].expand(-1, 3), o.clamp(min=0)[:, None].expand(-1, 3)]
        cur = torch.where(ok, nxt, torch.full_like(cur, -1))
    assert torch.equal(nodes[b.alive][torch.arange(int(b.alive.sum()), device=dev), b.goal[:, None].expand(-1, t.H)[b.alive]], b.node[b.alive]), "replay mismatch"
    keep = b.alive & (nodes >= 0).all(-1)
    ep, st = torch.nonzero(keep, as_tuple=True)
    nd = nodes[ep, st]
    Q = t.Q[nd].double()
    opt = Q >= Q.max(-1, keepdim=True).values - 1e-6
    dep = torch.zeros(len(ep), 3, dtype=torch.bool, device=dev)
    for g in range(3):
        for g2 in range(3):
            if g != g2:
                dep[:, g] |= ~(opt[:, g] & opt[:, g2]).any(-1)
    return dict(b=b, ep=ep, st=st, Q=Q, opt=opt, astar=Q.argmax(-1), dep=dep, pos=b.pos[ep, st], L=b.prefix[ep], fit=ep % 2 == 0, sb=st.clamp(max=STEPS - 1),
                solver_success=float((b.rew.sum(1) > 0).double().mean()), v_star=float(t.V[t.reveal[pre, b.goal]].mean()))


def additive(l, c, t):
    """The additive code for centred action scores l [m, 3, 4] (logits or Q*), measured on the test half."""
    dev_ = l - l.mean(1, keepdim=True)
    G = torch.zeros(t.max_prefix + 1, STEPS, 3, 4, dtype=torch.float64, device=l.device)
    for L in range(t.max_prefix + 1):
        for s in range(STEPS):
            m = c["fit"] & (c["L"] == L) & (c["sb"] == s)
            if m.any():
                G[L, s] = dev_[m].mean(0)
    Gt = G[c["L"], c["sb"]]
    H = l.mean(1)
    nat_a, add_a = l.argmax(-1), (H[:, None] + Gt).argmax(-1)
    ok = lambda act: torch.gather(c["opt"], 2, act[..., None]).squeeze(-1)
    nat_ok, add_ok = ok(nat_a), ok(add_a)
    uns = (R42.margin(c["astar"], Gt) <= 0)[:, None].expand_as(c["dep"])
    te = ~c["fit"]
    out = {}
    for name, m in (("all", te), ("first", te & (c["st"] == 0)), ("later", te & (c["st"] > 0))):
        d = c["dep"] & m[:, None]
        mean = lambda x, w: float(x[w].double().mean()) if w.any() else None
        out[name] = dict(goal_dependent_cells=int(d.sum()), natural=mean(nat_ok, d), additive=mean(add_ok, d), agree=mean(nat_a == add_a, d),
                         additive_share=float(1 - ((dev_ - (Gt - Gt.mean(1, keepdim=True))) ** 2)[m].sum() / (dev_ ** 2)[m].sum()),
                         unsolvable_share=mean(uns, d), natural_unsolvable=mean(nat_ok, d & uns), additive_unsolvable=mean(add_ok, d & uns),
                         natural_solvable=mean(nat_ok, d & ~uns), additive_solvable=mean(add_ok, d & ~uns),
                         natural_all_cells=float(nat_ok[m].double().mean()), additive_all_cells=float(add_ok[m].double().mean()))
    return out


@torch.no_grad()
def logits_all(net, c, dev):
    b, ep, L, pos = c["b"], c["ep"], c["L"], c["pos"]
    out = []
    for g in range(3):
        lg = []
        for s0 in range(0, len(ep), 8192):
            sl = slice(s0, s0 + 8192)
            tok = b.tok[ep[sl]].clone()
            n = torch.arange(len(tok), device=dev)
            tok[n, 1 + L[sl], MM.F_GOAL] = g + 1
            lg.append(net(tok)[0][n, pos[sl]])
        out.append(torch.cat(lg))
    l = torch.stack(out, 1).double()
    return l - l.mean(-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--arms", nargs="+", default=list(TK.SPAWNS))
    ap.add_argument("--quick", action="store_true", help="smoke test on the _smoke models")
    ap.add_argument("--out", default=None, help="directory for the results (default: this one)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    out = (Path(a.out) if a.out else HERE) / ("smoke.json" if a.quick else "results.json")
    res = {}
    for arm in a.arms:
        t = TK.tables(arm, dev, a.quick)
        c = cases(t, a.quick)
        Qc = c["Q"] - c["Q"].mean(-1, keepdim=True)
        res[arm] = dict(spawn_cells=len(TK.SPAWNS[arm]), nodes=len(t.V), decisions=len(c["ep"]), first=int((c["st"] == 0).sum()), solver_success=c["solver_success"],
                        v_star=c["v_star"], solver=additive(Qc, c, t), runs={})
        s_ = res[arm]["solver"]["all"]
        print(f"[{arm}] {len(c['ep'])} decisions | solver Q*: additive move optimal {s_['additive']:.3f} on goal-dependent cells, additive share {s_['additive_share']:.3f}, "
              f"unsolvable {s_['unsolvable_share']:.3f} | {time.time() - t0:.0f}s", flush=True)
        for s in a.seeds[:2] if a.quick else a.seeds:
            d = HERE / ("_smoke/train" if a.quick else "runs") / arm / f"seed{s}"
            ck = sorted((d / "ckpt").glob("u*.pt"))[-1]
            net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix)
            net.load_state_dict(torch.load(ck, map_location=dev))
            net = net.to(dev).eval()
            row = additive(logits_all(net, c, dev), c, t)
            log = json.load(open(d / "log.json"))["log"][-1]
            row["checkpoint"] = ck.name
            row["competence"] = {k: log[f"greedy_{k}"] for k in ("regret", "v_star", "opt_rate", "success", "ret")}
            res[arm]["runs"][f"seed{s}"] = row
            r = row["all"]
            print(f"[{arm}] seed{s} {ck.name} regret {log['greedy_regret']:.4f} | goal-dependent: natural {r['natural']:.3f} additive {r['additive']:.3f} agree {r['agree']:.3f} "
                  f"share {r['additive_share']:.3f} | unsolvable {r['unsolvable_share']:.3f}: natural {r['natural_unsolvable']} additive {r['additive_unsolvable']} | {time.time() - t0:.0f}s", flush=True)
            out.write_text(json.dumps(res))
        del t, c
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
