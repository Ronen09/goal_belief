"""Round 23: PPO on the maze task with a next-symbol prediction loss of coefficient --aux (0: reward only).
Round 18's train.py with the head added; settings, checkpoints and evaluation episodes are the same.

    .venv/bin/python rounds/r23_obs_prediction/train.py --aux 1.0 --seeds 0 1 2 3 4 5 6 7 8 9 --out rounds/r23_obs_prediction/runs/aux1
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazeaux as AX, mazebelief as MB, mazegraph as MG, mazeppo as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
R18 = HERE.parent / "r18_maze_belief"
sys.path.insert(0, str(R18))
import train as T18                                                  # noqa: E402


def graph(quick):
    return MG.build(MB.cross_maze(H=5), max_prefix=2) if quick else MG.cached(MB.cross_maze(), R18 / "cache" / "graph.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", type=float, default=0.0)
    ap.add_argument("--sup", default="selected", choices=("selected", "all", "one"),
                    help="round 25: targets for the head. selected: the symbol after the move taken; all / one: counterfactual symbols for four / one candidate move")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=1500)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--ent-final", type=float, default=0.003)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_env, a.seeds = 20, 512, a.seeds[:2]
    out, dev, M = Path(a.out), a.device, len(a.seeds)
    g = graph(a.quick)
    t = P.Tables(g, dev, max_prefix=2) if a.quick else P.Tables(g, dev)
    p_obs = torch.tensor(g.p_obs, dtype=torch.float32, device=dev)
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
    save_at = set(T18.checkpoints(a.updates))
    logs, t0, inter = [[] for _ in a.seeds], time.time(), np.zeros(M)

    @torch.no_grad()
    def evaluate(u):
        rows, pred = {}, None
        for name, greedy in (("sampled", False), ("greedy", True)):
            egen.manual_seed(78)                                          # the same prefixes and goals for every model and checkpoint
            e = P.Env(t, n_eval, egen)
            pre, goal = e.prefix.repeat(M), e.goal.repeat(M)
            egen.manual_seed(79)
            b = P.rollout(stk, t, n_eval * M, egen, greedy=greedy, prefix=pre, goal=goal)
            rows[name] = P.summarize(b, t, M)
            if greedy:                                                    # the head against the exact predictive distribution
                nll = torch.cat([AX.obs_nll(stk.aux(b.tok.view(M, n_eval, *b.tok.shape[1:])[:, s:s + 4096].flatten(0, 1))[2],
                                            b.tok.view(M, n_eval, *b.tok.shape[1:])[:, s:s + 4096].flatten(0, 1))[0].view(M, -1, b.tok.shape[1]) for s in range(0, n_eval, 4096)], 1)
                ex, valid = AX.exact_nll(b, p_obs)
                cnt = valid.view(M, -1).sum(1)
                pred = (nll.flatten(1).sum(1) / cnt, ex.view(M, -1).sum(1) / cnt)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=u, interactions=int(inter[m]), seconds=time.time() - t0, obs_nll=float(pred[0][m]), obs_nll_exact=float(pred[1][m]),
                                **{f"{k}_{f}": v for k in rows for f, v in rows[k][m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(args=vars(a), seed=seed, log=logs[m]), indent=1))
        torch.cuda.empty_cache()
        print(f"[aux {a.aux}] u={u} {time.time() - t0:.0f}s greedy regret " + " ".join(f"{x['regret']:.4f}" for x in rows["greedy"]) +
              " | excess nll " + " ".join(f"{float(x - y):.3f}" for x, y in zip(*pred)), flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
        cf = None
        if a.sup == "selected":
            b = P.rollout(stk, t, pc.n_env * M, gen)
        else:
            b, *cf = AX.rollout_cf(stk, t, pc.n_env * M, gen, one=a.sup == "one")
        AX.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params, t.gamma, a.aux, cf)
        inter += b.alive.view(M, -1).sum(1).cpu().numpy()


if __name__ == "__main__":
    main()
