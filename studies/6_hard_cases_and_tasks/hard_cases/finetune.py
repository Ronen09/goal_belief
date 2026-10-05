"""Fine-tune the observation-prediction experiment's reward models on the additively unsolvable cases (targeted) against the same number of steps on
ordinary decisions (control). Split by posterior triple: test cases are posteriors never trained on. Exploratory, not registered."""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, json, copy, collections, torch, importlib.util
sys.path.insert(0, __ROOT__)
from goalgeo import mazeppo as P, mazemodel as MM
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R42 = load("r42", __ROOT__ + "/studies/5_goal_belief_mechanism/additive_code/run.py"); M26 = R42.M26
SP = __HERE__ + ""
dev = "cuda"; STEPS, BS, LR = 3000, 1024, 1e-4
t = M26.T18.tables(dev)
d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(SP + "cache/later_cases.pt").items()}
m = len(d["pos"]); ar = torch.arange(m, device=dev)
_, tri = torch.unique(d["nodes"], dim=0, return_inverse=True)
g0 = torch.Generator(device=dev); g0.manual_seed(7)
train_tri = torch.rand(int(tri.max()) + 1, device=dev, generator=g0) < 0.5
tr = train_tri[tri]; te = ~tr
pool_all = torch.nonzero(tr).squeeze(1); pool_core = torch.nonzero(tr & d["core"]).squeeze(1)
print(f"train decisions {len(pool_all)} (core {len(pool_core)}), test decisions {int(te.sum())} (core {int((te & d['core']).sum())})", flush=True)
tgt = d["opt"].float(); tgt = tgt / tgt.sum(-1, keepdim=True)              # [m, 3, 4] uniform over the optimal set

def batch_tok(idx, g):
    tok = d["tok"][idx].clone()
    tok[torch.arange(len(idx), device=dev), 1 + d["prefix"][idx], MM.F_GOAL] = g + 1
    return tok

@torch.no_grad()
def all_logits(net, idx):
    out = []
    for g in range(3):
        lg = []
        for s in range(0, len(idx), 8192):
            i = idx[s:s + 8192]
            lg.append(net(batch_tok(i, torch.full_like(i, g)))[0][torch.arange(len(i), device=dev), d["pos"][i]])
        out.append(torch.cat(lg))
    l = torch.stack(out, 1).double(); return l - l.mean(-1, keepdim=True)

def evaluate(net, seed):
    net.eval()
    idx = torch.arange(m, device=dev)
    lt = all_logits(net, idx); devl = lt - lt.mean(1, keepdim=True)
    L, sb = d["prefix"], d["step"].clamp(max=6)
    G = torch.zeros(t.max_prefix + 1, 7, 3, 4, dtype=torch.float64, device=dev)
    for l in range(t.max_prefix + 1):
        for s in range(7):
            mm = tr & (L == l) & (sb == s)
            if mm.any(): G[l, s] = devl[mm].mean(0)
    Gt = G[L, sb]; H = lt.mean(1)
    ok = lambda a: torch.gather(d["opt"], 2, a[..., None]).squeeze(-1)
    nat, add = ok(lt.argmax(-1)), ok((H[:, None] + Gt).argmax(-1))
    uns = d["unsolvable_by_model"][:, seed].to(dev)                          # fixed by the original model's G
    within = lt - lt.mean(1, keepdim=True)
    r = {}
    for name, sel in [("uns", te[:, None] & d["dep"] & uns[:, None]), ("solv", te[:, None] & d["dep"] & ~uns[:, None]),
                      ("nondep", te[:, None] & ~d["dep"])]:
        r[name] = dict(nat=float(nat[sel].double().mean()), add=float(add[sel].double().mean()))
    msk = te
    r["additive_share"] = float(1 - ((within - (Gt - Gt.mean(1, keepdim=True)))[msk] ** 2).sum() / (within[msk] ** 2).sum())
    gen = torch.Generator(device=dev); gen.manual_seed(99)
    b = P.rollout(net, t, 16384, gen, greedy=True)
    r["greedy_regret"] = float(P.summarize(b, t, 1)[0]["regret"])
    return r

def finetune(net, arm, seed):
    net = copy.deepcopy(net).train()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator(device=dev); gen.manual_seed(1000 + seed)
    for step in range(STEPS):
        if arm == "targeted":
            h = BS // 2
            idx = torch.cat([pool_core[torch.randint(len(pool_core), (h,), device=dev, generator=gen)],
                             pool_all[torch.randint(len(pool_all), (BS - h,), device=dev, generator=gen)]])
        else:
            idx = pool_all[torch.randint(len(pool_all), (BS,), device=dev, generator=gen)]
        g = torch.randint(3, (BS,), device=dev, generator=gen)
        lg = net(batch_tok(idx, g))[0][torch.arange(BS, device=dev), d["pos"][idx]]
        loss = -(tgt[idx, g] * torch.log_softmax(lg, -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()

res = {}
for seed in range(10):
    base, ck = M26.load("reward", seed, t, dev, M26.R23 / "runs")
    row = {"before": evaluate(base, seed)}
    for arm in ("targeted", "control"):
        row[arm] = evaluate(finetune(base, arm, seed), seed)
    res[f"seed{seed}"] = row
    f = lambda r: f"uns nat {r['uns']['nat']:.3f} add {r['uns']['add']:.3f} | solv nat {r['solv']['nat']:.3f} | nondep {r['nondep']['nat']:.3f} | addshare {r['additive_share']:.2f} | regret {r['greedy_regret']:.4f}"
    print(f"seed{seed}\n  before   {f(row['before'])}\n  targeted {f(row['targeted'])}\n  control  {f(row['control'])}", flush=True)
    json.dump(res, open(SP + "finetune_results.json", "w"))
