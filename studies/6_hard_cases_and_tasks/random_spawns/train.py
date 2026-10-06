"""The random-spawns experiment: PPO on reward only, on the maze task with a chosen spawn set (task.py). The
observation-prediction experiment's reward arm: same network, initialisation per seed, settings and checkpoints.

    .venv/bin/python studies/6_hard_cases_and_tasks/random_spawns/train.py --spawn all --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazeaux as AX, mazeppo as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import task as TK                                                    # noqa: E402


def checkpoints(total, n=24):
    return sorted({0, total} | {int(round(x)) for x in np.geomspace(10, total, n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spawn", required=True, choices=tuple(TK.SPAWNS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--ent-final", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_env, a.seeds = 20, 512, a.seeds[:2]
    out, dev, M = HERE / ("_smoke" if a.quick else "runs") / a.spawn, a.device, len(a.seeds)
    t = TK.tables(a.spawn, dev, a.quick)
    nets = []
    for s in a.seeds:
        torch.manual_seed(s)
        nets.append(AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(dev))
        (out / f"seed{s}" / "ckpt").mkdir(parents=True, exist_ok=True)
    stk = AX.StackedAux(nets)
    params = stk.trainable()
    pc = P.PPOConfig(n_env=a.n_env, lr=a.lr, ent=a.ent, ent_final=a.ent_final, epochs=a.epochs)
    opt = torch.optim.Adam(params, lr=pc.lr, eps=1e-5)
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seeds[0])
    egen = torch.Generator(device=dev)
    n_eval = 1024 if a.quick else 16384
    save_at = set(checkpoints(a.updates))
    logs, t0, inter = [[] for _ in a.seeds], time.time(), np.zeros(M)

    @torch.no_grad()
    def evaluate(u):
        rows = {}
        for name, greedy in (("sampled", False), ("greedy", True)):
            egen.manual_seed(78)                                          # the same prefixes and goals for every model and checkpoint
            e = P.Env(t, n_eval, egen)
            pre, goal = e.prefix.repeat(M), e.goal.repeat(M)
            egen.manual_seed(79)
            rows[name] = P.summarize(P.rollout(stk, t, n_eval * M, egen, greedy=greedy, prefix=pre, goal=goal), t, M)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=u, interactions=int(inter[m]), seconds=time.time() - t0, **{f"{k}_{f}": v for k in rows for f, v in rows[k][m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(args=vars(a), seed=seed, log=logs[m]), indent=1))
        torch.cuda.empty_cache()
        print(f"[{a.spawn}] u={u} {time.time() - t0:.0f}s greedy regret " + " ".join(f"{x['regret']:.4f}" for x in rows["greedy"]) +
              f" | v* {rows['greedy'][0]['v_star']:.3f}", flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
        b = P.rollout(stk, t, pc.n_env * M, gen)
        AX.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params, t.gamma, 0.0, None)
        inter += b.alive.view(M, -1).sum(1).cpu().numpy()


if __name__ == "__main__":
    main()
