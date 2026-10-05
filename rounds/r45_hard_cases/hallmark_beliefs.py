from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[2])
import sys, torch, importlib.util
sys.path.insert(0, __ROOT__)
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R42 = load("r42", __ROOT__ + "/rounds/r42_additive_code/run.py"); M26 = R42.M26
torch.set_grad_enabled(False)
t = M26.T18.tables("cuda"); ctx = M26.Ctx(t)
test = torch.nonzero(ctx.test & (ctx.L >= 1)).squeeze(1)
Q = ctx.Q[test].double(); astar = Q.argmax(-1); bel = t.belief[ctx.node[test]].double()
lay = ["..L...b....", ".aaaa1bbbb3", "..a...2...."]
def show(mask, name):
    b = bel[mask].mean(0)
    print(f"{name}: {int(mask.sum())} histories, {len(torch.unique(ctx.node[test][mask]))} distinct posteriors; Q per goal (U D L R) mean:")
    for g in range(3): print("   G%d" % (g+1), [round(x, 3) for x in Q[mask, g].mean(0).tolist()])
    grid = [["  .  "] * 11 for _ in range(3)]
    for i, (r, c) in enumerate(t.cells): grid[r][c] = f"{lay[r][c]}{b[i]:.2f}"[:5].rjust(5)
    print("\n".join(" ".join(r) for r in grid))
for pat in ["URR", "DLD", "URU", "RLR"]:
    A = "UDLR"; code = torch.tensor([A.index(ch) for ch in pat], device="cuda")
    show((astar == code).all(-1), pat)
