from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, json, torch, statistics as st, itertools
from pathlib import Path
SP = __HERE__ + ""
exec(open(SP + "finetune.py").read().split("res = {}")[0])
from goalgeo import mazeaux as AX
def net_of(arm, s):
    if arm == "r23": return M26.load("reward", s, t, dev, M26.R23 / "runs")[0]
    ck = sorted((Path(SP) / "runs" / arm / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
    n = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix); n.load_state_dict(torch.load(ck, map_location=dev)); return n.to(dev).eval()
ARMS = ["r23", "targeted", "control"]; SEEDS = range(10)
idx = torch.nonzero(te).squeeze(1)                               # held-out posteriors
A, P_ = {}, {}
for arm in ARMS:
    for s in SEEDS:
        l = all_logits(net_of(arm, s), idx); A[arm, s] = l.argmax(-1); P_[arm, s] = torch.softmax(l, -1)
opt = d["opt"][idx]; dep = d["dep"][idx]; core = d["core"][idx][:, None].expand_as(dep)
astar_ok = lambda a: torch.gather(opt, 2, a[..., None]).squeeze(-1)
cats = {"all": torch.ones_like(dep), "hard (goal-dep)": dep & core, "other goal-dep": dep & ~core, "goal irrelevant": ~dep}
def agree(x, y, m): return float((x == y)[m].double().mean())
def tv(x, y, m): return float((0.5 * (x - y).abs().sum(-1))[m].mean())
# 'same optimal-set decision': both pick an optimal move (possibly different ones among ties)
rows = {}
for c, m in cats.items():
    r = {}
    for a1, a2 in [("targeted", "control"), ("r23", "targeted"), ("r23", "control")]:
        r[f"{a1} vs {a2} (same seed)"] = st.median([agree(A[a1, s], A[a2, s], m) for s in SEEDS])
    for arm in ARMS:
        r[f"{arm} vs {arm} (other seed)"] = st.median([agree(A[arm, s], A[arm, s2], m) for s, s2 in itertools.combinations(SEEDS, 2)])
    r["TV targeted vs control"] = st.median([tv(P_["targeted", s], P_["control", s], m) for s in SEEDS])
    r["TV targeted vs targeted"] = st.median([tv(P_["targeted", s], P_["targeted", s2], m) for s, s2 in itertools.combinations(SEEDS, 2)])
    r["TV control vs control"] = st.median([tv(P_["control", s], P_["control", s2], m) for s, s2 in itertools.combinations(SEEDS, 2)])
    rows[c] = r
for c, r in rows.items():
    print(f"\n== {c} ({int(cats[c].sum())} cells)"); [print(f"  {k:32s} {v:.3f}") for k, v in r.items()]
# where targeted and control disagree: who is right?
dis = {k: [] for k in ("targeted right", "control right", "both right (ties)", "neither")}
for s in SEEDS:
    m = (A["targeted", s] != A["control", s])
    tr_, co_ = astar_ok(A["targeted", s]), astar_ok(A["control", s])
    for k, v in [("targeted right", tr_ & ~co_), ("control right", co_ & ~tr_), ("both right (ties)", tr_ & co_), ("neither", ~tr_ & ~co_)]:
        dis[k].append(float(v[m].double().mean()))
print("\nwhen targeted and control pick different moves:", {k: round(st.median(v), 3) for k, v in dis.items()})
# by goal and by step, hard cells
for g in range(3):
    print(f"G{g+1} hard-cell optimal: " + " | ".join(f"{arm} {st.median([float(astar_ok(A[arm, s])[:, g][(dep & core)[:, g]].double().mean()) for s in SEEDS]):.3f}" for arm in ARMS))
stp = d["step"][idx]
for lo, hi in [(0, 0), (1, 3), (4, 7), (8, 11)]:
    mm = (dep & core) & ((stp >= lo) & (stp <= hi))[:, None]
    print(f"steps {lo}-{hi} hard-cell optimal: " + " | ".join(f"{arm} {st.median([float(astar_ok(A[arm, s])[mm].double().mean()) for s in SEEDS]):.3f}" for arm in ARMS))
# on-policy behaviour
print()
for arm in ARMS:
    regs = {k: [] for k in ("regret", "regret_G1", "regret_G2", "regret_G3", "length", "success")}
    for s in SEEDS:
        gen = torch.Generator(device=dev); gen.manual_seed(99)
        r = P.summarize(P.rollout(net_of(arm, s), t, 16384, gen, greedy=True), t, 1)[0]
        for k in regs: regs[k].append(r[k])
    print(arm, {k: round(st.median(v), 4) for k, v in regs.items()})
