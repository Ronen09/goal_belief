"""Round 17: train several seeds of one condition side by side. Conditions:
    ppo         reward only (the primary condition)
    frozen      PPO with the initial backbone frozen; only the policy and value heads learn
    supervised  cross-entropy to the solver's optimal action set on a mixture of exploratory and optimal histories
Saves each seed's initialisation and log-spaced checkpoints (denser where --dense says), and a log with
return, regret, updates and environment interactions kept separately."""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import navbank as B, navmodel as NM, navppo as P

torch.backends.cuda.matmul.allow_tf32 = True                         # 1.3x faster; the pilot ran without it
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
ENV = dict(n=5, H=12, c=0.025)
Q_GRID = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
HELD_VALUES = (0.7, 0.9)
HELD_COMBOS = ((0.6, 0.95), (0.95, 0.6), (0.75, 0.85), (0.85, 0.75))


def q_pairs(mode, device, which="train"):
    """Index pairs into the reliability grid. fixed: (0.8, 0.8) only. grid, train: every pair except those with
    a held-out value (0.7 or 0.9, either station) and the held-out combinations. grid, held_value / held_combo:
    those pairs. Returns (pairs [M, 2], the grid the indices refer to)."""
    if mode == "fixed":
        return torch.tensor([[0, 0]], device=device), (0.8,)
    hv = {Q_GRID.index(v) for v in HELD_VALUES}
    hc = {(Q_GRID.index(a), Q_GRID.index(b)) for a, b in HELD_COMBOS}
    every = [(i, j) for i in range(8) for j in range(8)]
    sets = dict(train=[p for p in every if p[0] not in hv and p[1] not in hv and p not in hc],
                held_value=[p for p in every if p[0] in hv or p[1] in hv], held_combo=sorted(hc), all=every)
    return torch.tensor(sets[which], device=device), Q_GRID


def checkpoints(total, n=20, dense=()):
    ks = {0, total} | {int(round(x)) for x in np.geomspace(10, total, n)}
    if dense:
        lo, hi, step = dense
        ks |= set(range(int(lo), min(int(hi), total) + 1, int(step)))
    return sorted(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="ppo", choices=("ppo", "frozen", "supervised"))
    ap.add_argument("--q", default="fixed", choices=("fixed", "grid"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--n-env", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.05)
    ap.add_argument("--ent-final", type=float, default=0.005)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dense", type=int, nargs=3, default=(), metavar=("FROM", "TO", "STEP"), help="extra checkpoints")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.n_env, a.seeds = 20, 512, a.seeds[:2]
    out = Path(a.out)
    dev, M = a.device, len(a.seeds)
    nets = []
    for s in a.seeds:
        torch.manual_seed(s)
        nets.append(NM.Net(ENV["n"] ** 2, ENV["H"], layers=a.layers).to(dev))
        (out / f"seed{s}" / "ckpt").mkdir(parents=True, exist_ok=True)
    stk = P.Stacked(nets)
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seeds[0])
    egen = torch.Generator(device=dev); egen.manual_seed(77)             # the same on-policy evaluation draws for every run
    qp, qs = q_pairs(a.q, dev)
    tab = P.Tables(ENV["n"], ENV["H"], ENV["c"], qs, dev)
    params = stk.trainable(heads_only=a.cond == "frozen")
    pc = P.PPOConfig(n_env=a.n_env, lr=a.lr, ent=a.ent, ent_final=a.ent_final, epochs=a.epochs)
    opt = torch.optim.Adam(params, lr=pc.lr, eps=1e-5)
    n_eval = 1024 if a.quick else 16384
    ecfg, estart = P.sample_configs(tab, n_eval, qp, egen)
    ecfg, estart = ecfg.repeat(M), estart.repeat(M)
    vstar = tab.V[ecfg[:n_eval], tab.H, estart[:n_eval], 0, 0].mean().item()
    save_at = set(checkpoints(a.updates, dense=a.dense))
    logs, t0, inter = [[] for _ in a.seeds], time.time(), np.zeros(M)
    pos = NM.dec_pos(torch.arange(tab.H, device=dev))

    def evaluate(u):
        egen.manual_seed(78)
        s = P.summarize_each(P.rollout(stk, tab, ecfg, estart, gen=egen), M)
        g = P.summarize_each(P.rollout(stk, tab, ecfg, estart, gen=egen, greedy=True), M)
        for m, seed in enumerate(a.seeds):
            logs[m].append(dict(update=u, interactions=int(inter[m]), seconds=time.time() - t0, v_star=vstar,
                                **{"sampled_" + k: v for k, v in s[m].items()}, **{"greedy_" + k: v for k, v in g[m].items()}))
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(args=vars(a), env=ENV, seed=seed, log=logs[m]), indent=1))
        torch.cuda.empty_cache()
        print(f"[{a.cond} {a.q}] u={u} {time.time() - t0:.0f}s regret " + " ".join(f"{x['regret']:.4f}" for x in s) +
              " | queries " + " ".join(f"{x['queries']:.2f}" for x in s), flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
        cfg, start = P.sample_configs(tab, pc.n_env * M, qp, gen)
        if a.cond == "supervised":
            eps = torch.rand(1, device=dev, generator=gen).item() * 0.5
            b = P.rollout(None, tab, cfg, start, gen=gen, behaviour=B.solver_behaviour(eps, gen))
            lg, _ = stk(b.tok, b.q)
            lp = torch.log_softmax(lg[:, pos].masked_fill(~b.legal, -1e9), -1)
            nll = -torch.logsumexp(lp.masked_fill(~b.optimal, -1e9), -1) * b.alive
            loss = (nll.view(M, -1).sum(1) / b.alive.view(M, -1).sum(1)).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            P.clip_per_model(params, 1.0); opt.step()
        else:
            b = P.rollout(stk, tab, cfg, start, gen=gen)
            P.ppo_update_stacked(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params)
        inter += b.alive.view(M, -1).sum(1).cpu().numpy()


if __name__ == "__main__":
    main()
