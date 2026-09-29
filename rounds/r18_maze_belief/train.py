"""Round 18: train several seeds of one condition side by side on the maze task. Conditions:
    ppo         reward only (the primary condition)
    frozen      PPO with the initial backbone frozen; only the policy and value heads learn
    supervised  cross-entropy to the solver's optimal action set, on histories from an epsilon-greedy solver
Saves each seed's initialisation and log-spaced checkpoints, and a log with discounted return, exact regret,
updates and environment interactions."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazegraph as MG, mazemodel as MM, mazeppo as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent


def tables(device, quick=False):
    if quick:
        return P.Tables(MG.build(MB.cross_maze(H=5), max_prefix=2), device, max_prefix=2)
    return P.Tables(MG.cached(MB.cross_maze(), HERE / "cache" / "graph.npz"), device)


def checkpoints(total, n=24):
    return sorted({0, total} | {int(round(x)) for x in np.geomspace(10, total, n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="ppo", choices=("ppo", "frozen", "supervised"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--ent-final", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_env, a.seeds = 20, 512, a.seeds[:2]
    out, dev, M = Path(a.out), a.device, len(a.seeds)
    t = tables(dev, a.quick)
    nets = []
    for s in a.seeds:
        torch.manual_seed(s)
        nets.append(MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix, layers=a.layers).to(dev))
        (out / f"seed{s}" / "ckpt").mkdir(parents=True, exist_ok=True)
    stk = P.Stacked(nets)
    params = stk.trainable(heads_only=a.cond == "frozen")
    pc = P.PPOConfig(n_env=a.n_env, lr=a.lr, ent=a.ent, ent_final=a.ent_final, epochs=a.epochs)
    opt = torch.optim.Adam(params, lr=pc.lr, eps=1e-5)
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seeds[0])
    egen = torch.Generator(device=dev)
    n_eval = 1024 if a.quick else 16384
    save_at = set(checkpoints(a.updates))
    logs, t0, inter = [[] for _ in a.seeds], time.time(), np.zeros(M)

    def evaluate(u):
        rows = {}
        for name, greedy in (("sampled", False), ("greedy", True)):
            egen.manual_seed(78)                                          # the same episodes for every model and checkpoint
            e = P.Env(t, n_eval, egen)
            pre, goal = e.prefix.repeat(M), e.goal.repeat(M)
            egen.manual_seed(79)
            rows[name] = P.summarize(P.rollout(stk, t, n_eval * M, egen, greedy=greedy, prefix=pre, goal=goal), t, M)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=u, interactions=int(inter[m]), seconds=time.time() - t0,
                                **{f"{k}_{f}": v for k in rows for f, v in rows[k][m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(args=vars(a), seed=seed, log=logs[m]), indent=1))
        torch.cuda.empty_cache()
        print(f"[{a.cond}] u={u} {time.time() - t0:.0f}s regret sampled " + " ".join(f"{x['regret']:.4f}" for x in rows["sampled"]) +
              " | greedy " + " ".join(f"{x['regret']:.4f}" for x in rows["greedy"]) + f" | V* {rows['greedy'][0]['v_star']:.3f}", flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
        if a.cond == "supervised":
            eps = torch.rand(1, device=dev, generator=gen).item() * 0.5
            b = P.rollout(None, t, pc.n_env * M, gen, behaviour=P.solver_behaviour(t, eps, gen))
            lg, _ = stk(b.tok)
            lp = torch.log_softmax(lg.gather(1, b.pos[..., None].expand(-1, -1, 4)), -1)
            q = t.Q[b.node]
            best = q >= q.max(-1, keepdim=True).values - 1e-6
            nll = -torch.logsumexp(lp.masked_fill(~best, -1e9), -1) * b.alive
            loss = (nll.view(M, -1).sum(1) / b.alive.view(M, -1).sum(1)).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            P.clip_per_model(params, 1.0); opt.step()
        else:
            b = P.rollout(stk, t, pc.n_env * M, gen)
            P.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params, t.gamma)
        inter += b.alive.view(M, -1).sum(1).cpu().numpy()


if __name__ == "__main__":
    main()
