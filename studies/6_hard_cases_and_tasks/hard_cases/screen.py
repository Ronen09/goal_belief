"""Model-free screen of candidate maze tasks: how badly does the best additive policy H(history) + G(goal), fitted to the
solver's own Q*, do? Solver with eps-random moves generates histories; every decision is replayed under every goal."""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, itertools, json, time, numpy as np, torch
sys.path.insert(0, __ROOT__)
from goalgeo import mazebelief as MB, mazegraph as MG, mazeppo as P, mazemodel as MM
dev = "cuda"; torch.set_grad_enabled(False)

def make(rows, goals_rc, start_rc, H=12, eps=0.4, gamma=0.9):
    cells, sym = [], []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == ".": continue
            cells.append((r, c)); sym.append({"a": 0, "b": 1, "L": 2}[ch])
    ix = {c: i for i, c in enumerate(cells)}
    return MB.Maze(cells, np.array(sym), tuple(ix[g] for g in goals_rc), eps, gamma, H, tuple(ix[s] for s in start_rc))

def margin(astar, G):
    """Best additive margin for K goals (difference constraints; cycles over the 4 actions)."""
    n, K = astar.shape
    W = torch.full((n, 4, 4), -torch.inf, dtype=torch.float64, device=dev); rows = torch.arange(n, device=dev)
    for g in range(K):
        a = astar[:, g]; w = G[:, g] - G[rows, g, a][:, None]
        for b in range(4):
            m = a != b
            W[rows[m], a[m], b] = torch.maximum(W[rows[m], a[m], b], w[m, b])
    best = torch.full((n,), torch.inf, dtype=torch.float64, device=dev)
    for c in [c for k in (2, 3, 4) for c in itertools.permutations(range(4), k) if c[0] == min(c)]:
        tot = sum(W[:, c[i], c[(i + 1) % len(c)]] for i in range(len(c)))
        best = torch.minimum(best, torch.where(torch.isfinite(tot), -tot / len(c), torch.full_like(tot, torch.inf)))
    return best

def score(maze, max_prefix=4, N=20000, seed=0):
    g = MG.build(maze, max_prefix=max_prefix)
    t = P.Tables(g, dev, max_prefix=max_prefix); K = len(t.goal_cell)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    b = P.rollout(None, t, N, gen, behaviour=P.solver_behaviour(t, 0.2, gen))
    n = torch.arange(N, device=dev); pre = b.pre_node[n, b.prefix]
    cur = torch.stack([t.reveal[pre, k] for k in range(K)], 1); nodes = torch.full((N, t.H, K), -1, dtype=torch.long, device=dev)
    for s in range(t.H):
        nodes[:, s] = cur
        o = b.tok[n, (b.pos[:, s] + 1).clamp(max=t.L - 1), MM.F_SYM] - 1; a = b.act[:, s]
        ok = (cur >= 0) & (o >= 0)[:, None]
        cur = torch.where(ok, t.node_nxt[cur.clamp(min=0), a[:, None].expand(-1, K), o.clamp(min=0)[:, None].expand(-1, K)], torch.full_like(cur, -1))
    keep = b.alive & (nodes >= 0).all(-1)
    ep, st = torch.nonzero(keep, as_tuple=True); nd = nodes[ep, st]
    Q = t.Q[nd].double(); V = Q.max(-1).values; opt = Q >= V[..., None] - 1e-6
    dep = torch.zeros_like(opt[..., 0])
    for k in range(K):
        for k2 in range(K):
            if k != k2: dep[:, k] |= ~(opt[:, k] & opt[:, k2]).any(-1)
    Qc = Q - Q.mean(-1, keepdim=True); within = Qc - Qc.mean(1, keepdim=True)
    L, sb = b.prefix[ep], st.clamp(max=6)
    G = torch.zeros(max_prefix + 1, 7, K, 4, dtype=torch.float64, device=dev)
    for l in range(max_prefix + 1):
        for s in range(7):
            m = (L == l) & (sb == s)
            if m.any(): G[l, s] = within[m].mean(0)
    Gt = G[L, sb]; H = Qc.mean(1)
    add_a = (H[:, None] + Gt).argmax(-1)
    aok = torch.gather(opt, 2, add_a[..., None]).squeeze(-1)
    loss = V - torch.gather(Q, 2, add_a[..., None]).squeeze(-1)
    uns = margin(Q.argmax(-1), Gt) <= 0
    # how much the belief matters: acting on the most likely cell (fully observed Q at the MAP cell)
    vstar = t.V[t.reveal[pre, b.goal]]
    return dict(goals=K, nodes=len(g.belief), dep_share=float(dep.double().mean()),
                additive_share=float(1 - ((within - Gt) ** 2).sum() / (within ** 2).sum()),
                add_fail_dep=float((~aok)[dep].double().mean()), add_loss_per_decision=float(loss.mean()),
                add_loss_where_fails=float(loss[~aok].mean()) if (~aok).any() else 0.0,
                unsolvable_dep=float(uns[:, None].expand_as(dep)[dep].double().mean()),
                v_star=float(vstar.mean()), solver_success=float((b.rew.sum(1) > 0).double().mean()))

X = ["..L...b....", ".aaaaabbbbb", "..a...b...."]                      # the maze-belief experiment's layout (goal cells keep their symbols)
X_START = [(1, 1), (1, 2), (1, 8), (1, 9)]
LAYOUTS = {
  "cross (r18)": (X, X_START),
  "mirror": (["..L.....b..", ".aaaaabbbbb", "..b.....a.."], X_START),            # stubs carry the other arm's symbol
  "twin-stubs": (["..a.....b..", ".aaaaLbbbbb", "..b.....a.."], X_START),        # landmark in the centre, aliased stubs
}
if __name__ == "__main__":
    mode = sys.argv[1]
    out = {}
    if mode == "baseline":
        out["cross (r18), 3 goals"] = score(make(X, [(1, 5), (2, 6), (1, 10)], X_START))
        cells = [(r, c) for r, row in enumerate(X) for c, ch in enumerate(row) if ch != "." and (r, c) not in X_START]
        out["cross, goal = any non-start cell"] = score(make(X, cells, X_START))
        out["cross (r18), 3 goals, 8 moves"] = score(make(X, [(1, 5), (2, 6), (1, 10)], X_START, H=8))
    elif mode == "search":
        for name, (rows, starts) in LAYOUTS.items():
            cells = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != "." and (r, c) not in starts]
            for trip in itertools.combinations(cells, 3):
                try:
                    out[f"{name} | {trip}"] = score(make(rows, list(trip), starts, H=8), max_prefix=3, N=8000)
                except Exception as e:
                    out[f"{name} | {trip}"] = dict(error=str(e))
            print(name, "done", flush=True)
    json.dump(out, open(f__HERE__ + "screen_{mode}.json", "w"), indent=1)
    for k, v in out.items():
        if mode == "baseline": print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
