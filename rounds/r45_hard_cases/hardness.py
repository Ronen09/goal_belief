"""Model-free hardness of a task: how far is the solver's own Q* from additive (H(history) + G(goal))?"""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[2])
import torch
SP = __HERE__ + ""
d = torch.load(SP + "cache/later_cases.pt")
import sys; sys.path.insert(0, __ROOT__)
import importlib.util
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R42 = load("r42", __ROOT__ + "/rounds/r42_additive_code/run.py"); M26 = R42.M26
t = M26.T18.tables("cpu")
Q = t.Q[d["nodes"]].double()                          # [m, 3, 4]
Qc = Q - Q.mean(-1, keepdim=True)
opt, dep = d["opt"], d["dep"]
L, sb = d["prefix"], d["step"].clamp(max=6)
within = Qc - Qc.mean(1, keepdim=True)                # goal-dependent part
G = torch.zeros(5, 7, 3, 4, dtype=torch.float64)
for l in range(5):
    for s in range(7):
        m = (L == l) & (sb == s)
        if m.any(): G[l, s] = within[m].mean(0)
Gt = G[L, sb]
share = 1 - ((within - Gt) ** 2).sum() / (within ** 2).sum()
H = Qc.mean(1)
add_a = (H[:, None] + Gt).argmax(-1)
ok = torch.gather(opt, 2, add_a[..., None]).squeeze(-1)
uns = R42.margin(Q.argmax(-1), Gt) <= 0
gap = (Q.sort(-1, descending=True).values[..., 0] - Q.sort(-1, descending=True).values[..., 1])
cost = (Q.max(-1).values - torch.gather(Q, 2, add_a[..., None]).squeeze(-1))
print(f"additive share of Q*'s goal dependence: {share:.3f}")
print(f"additive argmax optimal on goal-dependent cells: {ok[dep].double().mean():.3f}")
print(f"additively unsolvable (solver-level G): {uns[:, None].expand_as(dep)[dep].double().mean():.3f} of goal-dependent cells")
print(f"mean value lost by the additive move, goal-dependent cells: {cost[dep].mean():.4f}; where it errs: {cost[dep & ~ok].mean():.4f}; Q-gap there {gap[dep & ~ok].mean():.4f}")
