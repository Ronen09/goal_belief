from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, itertools, collections, torch
sys.path.insert(0, __ROOT__)
from pathlib import Path
import importlib.util
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R42 = load("r42", __ROOT__ + "/studies/5_goal_belief_mechanism/additive_code/run.py")
M26, R30 = R42.M26, R42.R30
torch.set_grad_enabled(False)
dev = "cuda"
t = M26.T18.tables(dev); ctx = M26.Ctx(t)
nL = t.max_prefix + 1
fit, test = torch.nonzero(ctx.fit).squeeze(1), torch.nonzero(ctx.test & (ctx.L >= 1)).squeeze(1)
opt = ctx.opt[test]; Q = ctx.Q[test].double(); astar = Q.argmax(-1)
dep = torch.zeros(len(test), 3, dtype=torch.bool, device=dev)
for g in range(3):
    for g2 in range(3):
        if g != g2: dep[:, g] |= ~(opt[:, g] & opt[:, g2]).any(-1)
bel = t.belief[ctx.node[test]].double()          # [n, 14]
cells = t.cells; L = ctx.L[test]
A = "UDLR"
col = torch.tensor([c for r, c in cells], device=dev, dtype=torch.double)
row_ = torch.tensor([r for r, c in cells], device=dev, dtype=torch.double)
left = (col <= 4).double(); right = (col >= 6).double()
ent = -(bel.clamp_min(1e-12).log() * bel).sum(-1)
pmax = bel.max(-1).values
pl, pr = bel @ left, bel @ right
split = torch.minimum(pl, pr)                     # mass on the minority arm
nd = len(set(map(tuple, astar.tolist())))
pattern = [ "".join(A[a] for a in r) for r in astar.tolist()]
gap = (Q.sort(-1, descending=True).values[..., 0] - Q.sort(-1, descending=True).values[..., 1])  # [n,3] Q margin
ndist = torch.tensor([len(set(r)) for r in astar.tolist()], device=dev)

def violated_cycles(astar, G):
    n = len(astar); W = torch.full((n, 4, 4), -torch.inf, dtype=torch.float64, device=dev); rows = torch.arange(n, device=dev)
    who = torch.full((n, 4, 4), -1, dtype=torch.long, device=dev)
    for g in range(3):
        a = astar[:, g]; w = G[:, g] - G[rows, g, a][:, None]
        for b in range(4):
            m = (a != b) & (w[:, b] > W[rows, a, b])
            W[rows[m], a[m], b] = w[m, b]; who[rows[m], a[m], b] = g
    best = torch.full((n,), torch.inf, dtype=torch.float64, device=dev); arg = [None]*n
    bestc = torch.full((n,), -1, device=dev)
    for ci, c in enumerate(R42.CYCLES):
        tot = sum(W[:, c[i], c[(i+1) % len(c)]] for i in range(len(c)))
        v = torch.where(torch.isfinite(tot), -tot/len(c), torch.full_like(tot, torch.inf))
        bestc = torch.where(v < best, ci, bestc); best = torch.minimum(best, v)
    return best, bestc

agg = collections.defaultdict(list)
for s in range(10):
    net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
    lf = R42.logits(net, ctx, fit); dev_f = lf - lf.mean(1, keepdim=True)
    G = torch.zeros(3, nL, 4, dtype=torch.float64, device=dev)
    for l in range(nL):
        m = ctx.L[fit] == l
        if m.any(): G[:, l] = dev_f[m].mean(0)
    lt = R42.logits(net, ctx, test); Gt = G[:, L].permute(1, 0, 2)
    delta, bc = violated_cycles(astar, Gt)
    uns = delta <= 0
    nat_ok = torch.gather(opt, 2, lt.argmax(-1)[..., None]).squeeze(-1)
    if s == 0:
        print("G (seed0, per length), actions U D L R:")
        for l in range(1, nL): print(" L", l, [[round(x, 1) for x in G[g, l].tolist()] for g in range(3)])
    # cycle length of the binding cycle
    clen = torch.tensor([len(R42.CYCLES[i]) for i in bc.tolist()], device=dev)
    # which goal fails in unsolvable natural
    D = dict(uns=uns.double().mean().item())
    for name, f in [("entropy", ent), ("pmax", pmax), ("minority-arm mass", split), ("L", L.double()), ("#distinct a*", ndist.double()),
                    ("min Q-gap over goals", gap.min(-1).values)]:
        D[name] = (f[uns].mean().item(), f[~uns].mean().item())
    D["2-cycle binding"] = (clen[uns] == 2).double().mean().item()
    # natural accuracy per goal on dependent cells, unsolv vs solv
    for g in range(3):
        D[f"G{g+1} nat ok"] = (nat_ok[uns & dep[:, g], g].double().mean().item(), nat_ok[~uns & dep[:, g], g].double().mean().item())
    # 'goal against its own bias': a*_g is not G_g's preferred action among the union of optimal actions
    union = opt.any(1)
    pref = torch.where(union[:, None, :], Gt, torch.full_like(Gt, -1e9)).argmax(-1)  # each goal's favourite among candidates
    against = (pref != astar) & dep
    D["goals going against own bias (#)"] = (against.sum(-1)[uns].double().mean().item(), against.sum(-1)[~uns].double().mean().item())
    # which goal's bias is wrong where natural fails in unsolvable
    fail = uns[:, None] & dep & ~nat_ok
    D["failing cells: goal share"] = [round((fail[:, g].sum() / fail.sum()).item(), 2) for g in range(3)]
    D["failing cells: chose own-bias favourite"] = (lt.argmax(-1) == pref)[fail].double().mean().item()
    cnt = collections.Counter(p for p, u in zip(pattern, uns.tolist()) if u)
    tot = collections.Counter(pattern)
    D["top unsolvable patterns (G1G2G3: n_uns/n_all)"] = [(p, c, tot[p]) for p, c in cnt.most_common(6)]
    for k, v in D.items(): agg[k].append(v)
    print(f"seed{s}", {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else v) for k, v in D.items()}, flush=True)
