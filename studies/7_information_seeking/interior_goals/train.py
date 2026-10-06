"""The interior-goals experiment: PPO on reward only in the larger maze with six junction goals.
    --arm single    one goal per episode (the maze10 task with the goals moved)
    --arm collect   two goals per episode, to be collected in any order (goalgeo/bigcollect.py)
The maze10 experiment's network, PPO settings and evaluation.

    .venv/bin/python studies/7_information_seeking/interior_goals/train.py --arm collect --seeds 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigcollect as BC, bigmaze as BM, mazemodel as MM, mazeppo as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
TASK = dict(seed=0, size=10, loops=6, pairs=1, n_land=2, n_goals=4, eps=0.4, gamma=0.97, H=40, junctions=6)


def checkpoints(total, n=24):
    return sorted({0, total} | {int(round(x)) for x in np.geomspace(10, total, n)})


def setup(arm, task, dev):
    """(sim, module with rollout / summarize / reference policies, network constructor) for an arm."""
    maze = BM.make(**task)
    if arm == "single":
        t = BM.Sim(maze, dev)
        return t, BM, lambda: MM.Net(t.n_sym, len(t.goal_cell), t.H, 0)
    t = BC.Sim(maze, dev)
    return t, BC, lambda: MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("single", "collect"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--ent-final", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default=None, help="default: runs/ (with --quick: _smoke/train/)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mem-fraction", type=float, default=None, help="cap this process's share of the GPU memory (to train two arms side by side)")
    a = ap.parse_args()
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    task = dict(TASK)
    if a.quick:
        a.updates, a.n_env, a.seeds, task["H"] = 10, 256, a.seeds[:2], 12
    out = (Path(a.out) if a.out else HERE / ("_smoke/train" if a.quick else "runs")) / a.arm
    dev, M = a.device, len(a.seeds)
    t, E, build = setup(a.arm, task, dev)
    nets = []
    for s in a.seeds:
        torch.manual_seed(s)
        nets.append(build().to(dev))
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
    egen.manual_seed(78)                                                  # the same spawns and goals for every model, checkpoint and reference
    e = E.Env(t, n_eval, egen)
    goal, cell = e.goal, e.cell
    refs = {}
    for name in ("oracle", "qmdp2", "qmdp", "mls"):
        egen.manual_seed(79)
        refs[name] = E.summarize(E.rollout(None, t, n_eval, egen, None, getattr(E, name), goal, cell), t)[0]
    (out / "task.json").write_text(json.dumps(dict(arm=a.arm, task=task, cells=t.n, goals=[list(t.cells[g]) for g in t.maze.goals], references=refs,
                                                   args={k: v for k, v in vars(a).items() if k != "out"}), indent=1))
    print("references: " + " ".join(f"{k} {v['ret']:.3f} ({v['success']:.2f})" for k, v in refs.items()), flush=True)

    @torch.no_grad()
    def evaluate(u):
        rows = {}
        for name, greedy in (("sampled", False), ("greedy", True)):
            egen.manual_seed(79)
            rows[name] = E.summarize(E.rollout(stk, t, n_eval * M, egen, greedy, None, goal.repeat(M), cell.repeat(M)), t, M)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=u, interactions=int(inter[m]), seconds=time.time() - t0, **{f"{k}_{f}": v for k in rows for f, v in rows[k][m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(seed=seed, log=logs[m]), indent=1))
        torch.cuda.empty_cache()
        print(f"[{a.arm}] u={u} {time.time() - t0:.0f}s greedy return " + " ".join(f"{x['ret']:.3f}" for x in rows["greedy"]) +
              " | success " + " ".join(f"{x['success']:.2f}" for x in rows["greedy"]), flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
        b = E.rollout(stk, t, pc.n_env * M, gen)
        P.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params, t.gamma)
        inter += b.alive.view(M, -1).sum(1).cpu().numpy()


if __name__ == "__main__":
    main()
