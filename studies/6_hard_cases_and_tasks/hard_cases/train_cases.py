"""The observation-prediction experiment's reward-only PPO from scratch, plus a supervised loss on the generated decisions (optimal-action targets,
train-side posterior triples only). --arm targeted: half of each supervised batch from the additively unsolvable cases;
control: all from ordinary decisions. Same seeds / init / settings as the observation-prediction experiment."""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, __ROOT__)
from goalgeo import mazeaux as AX, mazegraph as MG, mazeppo as P, mazemodel as MM
import importlib.util
def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
T18 = _load("t18train", __ROOT__ + "/studies/3_reward_trained_agents/maze_belief/train.py")
sys.modules["train"] = T18                                                # r23's train.py does `import train as T18`
T23 = _load("t23train", __ROOT__ + "/studies/3_reward_trained_agents/obs_prediction/train.py")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
SP = __HERE__ + ""
ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=("targeted", "control"), required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
ap.add_argument("--updates", type=int, default=1500)
ap.add_argument("--sup-coef", type=float, default=1.0)
ap.add_argument("--sup-steps", type=int, default=2)
ap.add_argument("--sup-bs", type=int, default=1024)
a = ap.parse_args()
dev, M = "cuda", len(a.seeds)
out = Path(SP) / "runs" / a.arm
t = P.Tables(T23.graph(False), dev)
d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(SP + "cache/later_cases.pt").items()}
_, tri = torch.unique(d["nodes"], dim=0, return_inverse=True)
g0 = torch.Generator(device=dev); g0.manual_seed(7)                      # the same split as finetune.py
tr = (torch.rand(int(tri.max()) + 1, device=dev, generator=g0) < 0.5)[tri]
pool_all, pool_core = torch.nonzero(tr).squeeze(1), torch.nonzero(tr & d["core"]).squeeze(1)
tgt = d["opt"].float(); tgt = tgt / tgt.sum(-1, keepdim=True)
nets = []
for s in a.seeds:
    torch.manual_seed(s)
    nets.append(AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(dev))
    (out / f"seed{s}" / "ckpt").mkdir(parents=True, exist_ok=True)
stk = AX.StackedAux(nets); params = stk.trainable()
pc = P.PPOConfig(n_env=4096, lr=1e-4, ent=0.03, ent_final=0.003, epochs=3)
opt = torch.optim.Adam(params, lr=pc.lr, eps=1e-5)
gen = torch.Generator(device=dev); gen.manual_seed(1000 + a.seeds[0])
sgen = torch.Generator(device=dev); sgen.manual_seed(2000 + a.seeds[0])
egen = torch.Generator(device=dev)
save_at = set(T18.checkpoints(a.updates)); logs, t0 = [[] for _ in a.seeds], time.time()

def sup_step():
    B = a.sup_bs
    if a.arm == "targeted":
        h = B // 2
        idx = torch.cat([pool_core[torch.randint(len(pool_core), (M, h), device=dev, generator=sgen)],
                         pool_all[torch.randint(len(pool_all), (M, B - h), device=dev, generator=sgen)]], 1).flatten()
    else:
        idx = pool_all[torch.randint(len(pool_all), (M * B,), device=dev, generator=sgen)]
    g = torch.randint(3, (M * B,), device=dev, generator=sgen)
    tok = d["tok"][idx].clone(); tok[torch.arange(M * B, device=dev), 1 + d["prefix"][idx], MM.F_GOAL] = g + 1
    lg = stk(tok)[0][torch.arange(M * B, device=dev), d["pos"][idx]]
    l = -(tgt[idx, g] * torch.log_softmax(lg, -1)).sum(-1).view(M, B).mean(1)
    opt.zero_grad(set_to_none=True); (a.sup_coef * l.sum()).backward(); P.clip_per_model(params, pc.max_grad); opt.step()
    return l.detach()

@torch.no_grad()
def evaluate(u, sl):
    egen.manual_seed(78); e = P.Env(t, 16384, egen); pre, goal = e.prefix.repeat(M), e.goal.repeat(M)
    egen.manual_seed(79); b = P.rollout(stk, t, 16384 * M, egen, greedy=True, prefix=pre, goal=goal)
    rows = P.summarize(b, t, M)
    for m, seed in enumerate(a.seeds):
        logs[m].append(dict(update=u, seconds=time.time() - t0, sup_loss=float(sl[m]) if sl is not None else None, **{f"greedy_{k}": v for k, v in rows[m].items()}))
        (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(args=vars(a), seed=seed, log=logs[m]), indent=1))
    torch.cuda.empty_cache()
    print(f"[{a.arm}] u={u} {time.time() - t0:.0f}s greedy regret " + " ".join(f"{x['regret']:.4f}" for x in rows), flush=True)

sl = None
for u in range(a.updates + 1):
    if u in save_at:
        for m, seed in enumerate(a.seeds): torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
        evaluate(u, sl)
    if u == a.updates: break
    for g_ in opt.param_groups: g_["lr"] = pc.lr * min(1, (u + 1) / pc.warmup)
    b = P.rollout(stk, t, pc.n_env * M, gen)
    AX.ppo_update(stk, opt, b, pc, pc.ent + (pc.ent_final - pc.ent) * u / a.updates, params, t.gamma, 0.0, None)
    for _ in range(a.sup_steps): sl = sup_step()
