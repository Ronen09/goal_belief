"""More additively-unsolvable cases: decisions after the goal reveal, each replayed under all three goals."""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[2])
import sys, collections, torch, importlib.util
sys.path.insert(0, __ROOT__)
from goalgeo import mazeppo as P, mazemodel as MM
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R42 = load("r42", __ROOT__ + "/rounds/r42_additive_code/run.py"); M26 = R42.M26
torch.set_grad_enabled(False)
dev = "cuda"; OUT = __HERE__ + "cache/later_cases.pt"
t = M26.T18.tables(dev)
N, EPS = 40000, 0.2
gen = torch.Generator(device=dev); gen.manual_seed(4545)
b = P.rollout(None, t, N, gen, behaviour=P.solver_behaviour(t, EPS, gen))
n = torch.arange(N, device=dev)
pre = b.pre_node[n, b.prefix]
# replay every decision's (action, symbol) path from the reveal under each goal
nodes = torch.full((N, t.H, 3), -1, dtype=torch.long, device=dev)
cur = torch.stack([t.reveal[pre, g] for g in range(3)], 1)
for s in range(t.H):
    nodes[:, s] = cur
    o = b.tok[n, (b.pos[:, s] + 1).clamp(max=t.L - 1), MM.F_SYM] - 1
    a = b.act[:, s]
    ok = (cur >= 0) & (o >= 0)[:, None]
    nxt = t.node_nxt[cur.clamp(min=0), a[:, None].expand(-1, 3), o.clamp(min=0)[:, None].expand(-1, 3)]
    cur = torch.where(ok, nxt, torch.full_like(cur, -1))
assert torch.equal(nodes[n, :, :][b.alive][torch.arange(int(b.alive.sum())), b.goal[:, None].expand(-1, t.H)[b.alive]], b.node[b.alive]), "replay mismatch"
keep = b.alive & (nodes >= 0).all(-1)
ep, st = torch.nonzero(keep, as_tuple=True)
nd = nodes[ep, st]                                        # [m, 3]
print(f"decisions: {int(b.alive.sum())} alive, {len(ep)} valid under all three goals; step 0: {(st==0).sum().item()}, later: {(st>0).sum().item()}")
Q = t.Q[nd].double(); V = Q.max(-1).values; opt = Q >= V[..., None] - 1e-6; astar = Q.argmax(-1)
dep = torch.zeros(len(ep), 3, dtype=torch.bool, device=dev)
for g in range(3):
    for g2 in range(3):
        if g != g2: dep[:, g] |= ~(opt[:, g] & opt[:, g2]).any(-1)
key = torch.stack([nd[:, 0], nd[:, 1], nd[:, 2]], 1)
print("distinct posterior triples:", len(torch.unique(key, dim=0)), "| goal-dependent cells:", int(dep.sum()))
bel = t.belief[nd].double().mean(1)
ent = -(bel.clamp_min(1e-12).log() * bel).sum(-1)
pmax = bel.max(-1).values
srt = Q.sort(-1, descending=True).values; qgap = (srt[..., 0] - srt[..., 1]).min(-1).values
pos = b.pos[ep, st]; L = b.prefix[ep]
fitm = (ep % 2 == 0)                                      # fit half / test half by episode
stepbin = st.clamp(max=6)                                 # G estimated per (prefix length, step bin)

def logits_all(net):
    out = []
    for g in range(3):
        tok = b.tok[ep].clone()
        tok[torch.arange(len(ep), device=dev), 1 + L, MM.F_GOAL] = g + 1
        # hide everything after the decision (causal anyway) and read the logits at the decision token
        lg = []
        for s0 in range(0, len(ep), 8192):
            sl = slice(s0, s0 + 8192)
            lg.append(net(tok[sl])[0][torch.arange(len(tok[sl]), device=dev), pos[sl]])
        out.append(torch.cat(lg))
    l = torch.stack(out, 1).double()
    return l - l.mean(-1, keepdim=True)

agg = collections.defaultdict(list); flags = []
for seed in range(10):
    net, ck = M26.load("reward", seed, t, dev, M26.R23 / "runs")
    lt = logits_all(net)
    devl = lt - lt.mean(1, keepdim=True)
    G = torch.zeros(t.max_prefix + 1, 7, 3, 4, dtype=torch.float64, device=dev)
    for l in range(t.max_prefix + 1):
        for s in range(7):
            m = fitm & (L == l) & (stepbin == s)
            if m.any(): G[l, s] = devl[m].mean(0)
    Gt = G[L, stepbin]                                    # [m, 3, 4]
    H = lt.mean(1)
    add_a, nat_a = (H[:, None] + Gt).argmax(-1), lt.argmax(-1)
    okf = lambda act: torch.gather(opt, 2, act[..., None]).squeeze(-1)
    nat_ok, add_ok = okf(nat_a), okf(add_a)
    delta = R42.margin(astar, Gt); uns = delta <= 0
    te = ~fitm
    D = {}
    for name, m in [("step 0", te & (st == 0)), ("later", te & (st > 0))]:
        dd = dep & m[:, None]; du, ds = dd & uns[:, None], dd & ~uns[:, None]
        D[name] = dict(dep_cells=int(dd.sum()), uns_share=float(du.sum() / dd.sum()), uns_cells=int(du.sum()),
                       nat_uns=float(nat_ok[du].double().mean()), nat_solv=float(nat_ok[ds].double().mean()),
                       add_uns=float(add_ok[du].double().mean()), add_solv=float(add_ok[ds].double().mean()))
    m = te & (st > 0)
    union = opt.any(1)
    prefa = torch.where(union[:, None, :], Gt, torch.full_like(Gt, -1e9)).argmax(-1)
    against = ((prefa != astar) & dep).sum(-1).double()
    for nm, f in [("entropy", ent), ("pmax", pmax), ("min Q-gap", qgap), ("goals against own bias", against)]:
        D[nm] = (round(f[m & uns].mean().item(), 3), round(f[m & ~uns].mean().item(), 3))
    fail = (m & uns)[:, None] & dep & ~nat_ok
    D["failing: goal share"] = [round((fail[:, g].sum() / fail.sum().clamp(min=1)).item(), 2) for g in range(3)]
    flags.append(uns.cpu())
    for k, v in D.items(): agg[k].append(v)
    print(seed, ck, {k: ({kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in D.items()}, flush=True)

U = torch.stack(flags, 1)                                  # [m, 10] unsolvable under each model's G
core = U.float().mean(1) >= 0.5
print("cases unsolvable in >= half the models:", int(core.sum()), "decisions,", len(torch.unique(key[core.to(dev)], dim=0)), "distinct posterior triples;",
      "of them later decisions:", int((core & (st.cpu() > 0)).sum()))
by = collections.Counter(st.cpu()[core].tolist()); print("by step:", dict(sorted(by.items())))
pat = collections.Counter("".join("UDLR"[x] for x in r) for r in astar.cpu()[core].tolist()); print("top a* patterns (G1G2G3):", pat.most_common(8))
torch.save(dict(tok=b.tok[ep].cpu(), pos=pos.cpu(), prefix=L.cpu(), step=st.cpu(), true_goal=b.goal[ep].cpu(), nodes=nd.cpu(),
                astar=astar.cpu(), opt=opt.cpu(), dep=dep.cpu(), unsolvable_by_model=U, core=core, fit_half=fitm.cpu()), OUT)
print("saved", OUT)
