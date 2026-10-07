"""The random-goals experiment: PPO on reward only in the maze10 maze with every cell a possible goal (one per episode,
shown at the start; the spawn is any other cell). The maze10 experiment's network, PPO settings and evaluation.

    .venv/bin/python studies/7_information_seeking/random_goals/train.py --seeds 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM, mazeppo as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
TASK = dict(seed=0, size=10, loops=6, pairs=1, n_land=2, n_goals=4, eps=0.4, gamma=0.97, H=40, all_goals=True)


def checkpoints(total, n=24):
    return sorted({0, total} | {int(round(x)) for x in np.geomspace(10, total, n)})


def build(n_sym, n_goals, H, d=128, layers=4, heads=4):
    return MM.Net(n_sym, n_goals, H, 0, d=d, layers=layers, heads=heads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--ent-final", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    for k, v in TASK.items():
        if k != "all_goals":
            ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--out", default=None, help="default: runs/ppo (with --quick: _smoke/train)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_env, a.seeds, a.H = 10, 256, a.seeds[:2], 12
    out = Path(a.out) if a.out else HERE / ("_smoke/train" if a.quick else "runs/ppo")
    dev, M = a.device, len(a.seeds)
    task = {k: getattr(a, k, v) for k, v in TASK.items()}
    t = BM.Sim(BM.make(**task), dev)
    nets = []
    for s in a.seeds:
        torch.manual_seed(s)
        nets.append(build(t.n_sym, len(t.goal_cell), t.H, a.d, a.layers).to(dev))
        (out / f"seed{s}" / "ckpt").mkdir(parents=True, exist_ok=True)
    stk = P.Stacked(nets)
    params = stk.trainable()
    pc = P.PPOConfig(n_env=a.n_env, lr=a.lr, ent=a.ent, ent_final=a.ent_final, epochs=a.epochs)
    opt = torch.optim.Adam(params, lr=pc.lr, eps=1e-5)
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seeds[0])
    egen = torch.Generator(device=dev)
    n_eval = 512 if a.quick else 8192
    save_at = set(checkpoints(a.updates))
    logs, t0, inter = [[] for _ in a.seeds], time.time(), np.zeros(M)

    def episodes():
        egen.manual_seed(78)                                              # the same spawns and goals for every model, checkpoint and reference
        e = BM.Env(t, n_eval, egen)
        return e.goal, e.cell

    goal, cell = episodes()
    refs = {}
    for name in ("oracle", "qmdp2", "qmdp", "mls"):
        egen.manual_seed(79)
        refs[name] = BM.summarize(BM.rollout(None, t, n_eval, egen, behaviour=getattr(BM, name), goal=goal, cell=cell), t)[0]
    (out / "task.json").write_text(json.dumps(dict(task=task, cells=t.n, references=refs, args={k: v for k, v in vars(a).items() if k != "out"}), indent=1))
    print("references: " + " ".join(f"{k} {v['ret']:.3f} ({v['success']:.2f})" for k, v in refs.items()), flush=True)

    @torch.no_grad()
    def evaluate(u):
        rows = {}
        for name, greedy in (("sampled", False), ("greedy", True)):
            egen.manual_seed(79)
            rows[name] = BM.summarize(BM.rollout(stk, t, n_eval * M, egen, greedy=greedy, goal=goal.repeat(M), cell=cell.repeat(M)), t, M)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=u, interactions=int(inter[m]), seconds=time.time() - t0, **{f"{k}_{f}": v for k in rows for f, v in rows[k][m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(seed=seed, log=logs[m]), indent=1))
        torch.cuda.empty_cache()
        print(f"u={u} {time.time() - t0:.0f}s greedy return " + " ".join(f"{x['ret']:.3f}" for x in rows["greedy"]) +
              " | sampled " + " ".join(f"{x['ret']:.3f}" for x in rows["sampled"]), flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
        b = BM.rollout(stk, t, pc.n_env * M, gen)
        P.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params, t.gamma)
        inter += b.alive.view(M, -1).sum(1).cpu().numpy()


if __name__ == "__main__":
    main()
