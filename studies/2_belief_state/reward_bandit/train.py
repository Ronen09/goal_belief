"""The reward-bandit experiment: PPO on reward only in the hidden-goal bandit (goalgeo/bandit.py), a transformer or a
GRU, several seeds side by side. Evaluation on fixed episodes against the exact references.

    .venv/bin/python studies/2_belief_state/reward_bandit/train.py --arch tfm --seeds 0 1 2 3 4 5
    .venv/bin/python studies/2_belief_state/reward_bandit/train.py --arch gru --seeds 0 1 2 3 4 5
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bandit as BD

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
TASK = dict(K=3, n_cue=4, T=6, clip=0.1)
N_EVAL = 16384


def checkpoints(total, n=20):
    return sorted({0, total} | {int(round(x)) for x in np.geomspace(10, total, n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=("tfm", "gru"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=600)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ent-final", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default=None, help="default: runs/<arch> (with --quick: _smoke/train/<arch>)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_env, a.seeds = 10, 256, a.seeds[:2]
    out = Path(a.out) if a.out else HERE / ("_smoke/train" if a.quick else "runs") / a.arch
    dev, M = a.device, len(a.seeds)
    s = BD.Spec(**TASK)
    t = BD.Sim(BD.Graph(s), dev)
    nets = []
    for seed in a.seeds:
        torch.manual_seed(seed)
        nets.append(BD.build(a.arch, s, a.d, a.layers).to(dev))
        (out / f"seed{seed}" / "ckpt").mkdir(parents=True, exist_ok=True)
    stk = BD.Stacked(nets)
    params = stk.trainable()
    pc = BD.PPOConfig(n_env=a.n_env, lr=a.lr, ent=a.ent, ent_final=a.ent_final, epochs=a.epochs)
    opt = torch.optim.Adam(params, lr=pc.lr, eps=1e-5)
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seeds[0])
    egen = torch.Generator(device=dev)
    n_eval = 512 if a.quick else N_EVAL
    egen.manual_seed(78)                                                  # the same goals, cues and reward draws for every model and reference
    e = BD.Env(t, n_eval, egen)
    goal, cues, u = e.goal, e.cues, e.u
    refs = {}
    for name, pol in (("optimal", BD.optimal), ("myopic", BD.myopic), ("random", BD.uniform)):
        egen.manual_seed(79)
        refs[name] = BD.summarize(BD.rollout(None, t, n_eval, egen, behaviour=pol(egen), goal=goal, cues=cues, u=u), t)[0]
    (out / "task.json").write_text(json.dumps(dict(arch=a.arch, task=TASK, states=int(t.B.shape[0] * t.B.shape[1]), references=refs,
                                                   args={k: v for k, v in vars(a).items() if k != "out"}), indent=1))
    print("references: " + " ".join(f"{k} {v['ret']:.3f} (regret {v['regret']:.3f})" for k, v in refs.items()), flush=True)
    save_at = set(checkpoints(a.updates))
    logs, t0 = [[] for _ in a.seeds], time.time()

    @torch.no_grad()
    def evaluate(upd):
        rows = {}
        for name, greedy in (("sampled", False), ("greedy", True)):
            rows[name] = BD.summarize(BD.rollout(stk, t, n_eval * M, None, greedy, goal=goal.repeat(M), cues=cues.repeat(M, 1), u=u.repeat(M, 1, 1)), t, M)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=upd, seconds=time.time() - t0, **{f"{k}_{f}": v for k in rows for f, v in rows[k][m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(seed=seed, log=logs[m]), indent=1))
        print(f"[{a.arch}] u={upd} {time.time() - t0:.0f}s greedy regret " + " ".join(f"{x['regret']:.3f}" for x in rows["greedy"]) +
              " | return " + " ".join(f"{x['ret']:.3f}" for x in rows["greedy"]) + " | optimal where information pays " +
              " ".join(f"{x['opt_where_pays']:.2f}" for x in rows["greedy"]), flush=True)

    for upd in range(a.updates + 1):
        if upd in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{upd:06d}.pt")
            evaluate(upd)
        if upd == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (upd + 1) / pc.warmup)
        b = BD.rollout(stk, t, pc.n_env * M, gen)
        BD.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * upd / a.updates, params, s.A)


if __name__ == "__main__":
    main()
