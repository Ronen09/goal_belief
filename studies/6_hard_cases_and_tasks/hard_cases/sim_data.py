"""Matched episodes for the visual simulation: the model (greedy) and the Bayes-optimal solver from the same prefix,
goal and noise. Records beliefs, Q*, the model's policy, its additive decomposition H + G and the goal-swap view."""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[3])
import sys, json, torch, importlib.util
from pathlib import Path
sys.path.insert(0, __ROOT__)
from goalgeo import mazeppo as P, mazemodel as MM, mazeaux as AX
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
R42 = load("r42", __ROOT__ + "/studies/5_goal_belief_mechanism/additive_code/run.py"); M26 = R42.M26
torch.set_grad_enabled(False)
SP = __HERE__ + ""
dev = "cuda"; t = M26.T18.tables(dev)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "r23"          # r23 | targeted | control
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if MODEL == "r23":
    net, _ = M26.load("reward", SEED, t, dev, M26.R23 / "runs")
else:
    ck = sorted((Path(SP) / "runs" / MODEL / f"seed{SEED}" / "ckpt").glob("u*.pt"))[-1]
    net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()

# goal bias G for this model, per prefix length and step bin, from the fit half of the generated cases
d = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in torch.load(SP + "cache/later_cases.pt").items()}
fi = torch.nonzero(d["fit_half"]).squeeze(1)
def logits3(tok, pos, L):
    out = []
    for g in range(3):
        tk = tok.clone(); tk[torch.arange(len(tk), device=dev), 1 + L, MM.F_GOAL] = g + 1
        lg = []
        for s in range(0, len(tk), 8192):
            lg.append(net(tk[s:s + 8192])[0][torch.arange(len(tk[s:s + 8192]), device=dev), pos[s:s + 8192]])
        out.append(torch.cat(lg))
    l = torch.stack(out, 1).double(); return l - l.mean(-1, keepdim=True)
lf = logits3(d["tok"][fi], d["pos"][fi], d["prefix"][fi]); devl = lf - lf.mean(1, keepdim=True)
G = torch.zeros(t.max_prefix + 1, 7, 3, 4, dtype=torch.float64, device=dev)
Lf, sf = d["prefix"][fi], d["step"][fi].clamp(max=6)
for l in range(t.max_prefix + 1):
    for s in range(7):
        m = (Lf == l) & (sf == s)
        if m.any(): G[l, s] = devl[m].mean(0)

class Env(P.Env):
    """mazeppo.Env with the prefix cells recorded."""
    def __init__(self, t, N, gen):
        self.t, self.N, self.gen = t, N, gen
        r = lambda hi: torch.randint(hi, (N,), device=dev, generator=gen)
        self.cell = t.starts[r(len(t.starts))]
        self.prefix = r(t.max_prefix + 1); self.goal = r(len(t.goal_cell))
        self.tok = torch.zeros(N, t.L, MM.NF, dtype=torch.long, device=dev)
        o = self.sample_symbol()
        self.tok[:, 0, MM.F_TYPE], self.tok[:, 0, MM.F_SYM] = MM.OBS, o + 1
        node = t.start[o]
        self.pre_node = torch.full((N, t.max_prefix + 1), -1, device=dev); self.pre_node[:, 0] = node
        self.pre_cell = torch.full((N, t.max_prefix + 1), -1, device=dev); self.pre_cell[:, 0] = self.cell
        for i in range(t.max_prefix):
            on = i < self.prefix; a = r(4)
            self.cell = torch.where(on, t.nxt_cell[self.cell, a], self.cell)
            o = self.sample_symbol()
            node = torch.where(on, t.node_nxt[node, a, o], node)
            self.tok[on, 1 + i] = torch.stack([torch.full_like(a, MM.EVT), o + 1, a + 1, torch.zeros_like(a), torch.zeros_like(a)], -1)[on]
            self.pre_node[on, i + 1] = node[on]; self.pre_cell[on, i + 1] = self.cell[on]
        n = torch.arange(N, device=dev)
        self.idx = 1 + self.prefix
        self.tok[n, self.idx, MM.F_TYPE], self.tok[n, self.idx, MM.F_GOAL] = MM.GOAL, self.goal + 1
        self.node = t.reveal[node, self.goal]
        self.cf = torch.stack([t.reveal[node, g] for g in range(3)], 1)    # nodes under each goal (same history)
        self.k = torch.full((N,), t.H, device=dev); self.done = torch.zeros(N, dtype=torch.bool, device=dev)

N = 8192
def run(agent):
    gen = torch.Generator(device=dev); gen.manual_seed(2024)
    e = Env(t, N, gen); n = torch.arange(N, device=dev)
    rec = []
    for s in range(t.H):
        live = ~e.done
        q = t.Q[e.node].double()
        top = int(e.idx.max()) + 1
        l3 = logits3(e.tok[:, :top], e.idx, e.prefix)                 # model logits under each goal, same history
        lg = l3[n, e.goal]
        H = l3.mean(1); Gt = G[e.prefix, torch.full_like(e.prefix, min(s, 6))]
        cfv = (e.cf >= 0)
        cq = t.Q[e.cf.clamp(min=0)].double()
        astar = cq.argmax(-1)
        uns = R42.margin(astar, Gt) <= 0
        if agent == "model":
            a = lg.argmax(-1)
        else:
            a = q.argmax(-1)                                                # first optimal action (as the additive-code experiment's a*)
        rec.append(dict(live=live.clone(), cell=e.cell.clone(), node=e.node.clone(), q=q, a=a, probs=torch.softmax(lg, -1), H=H, G=Gt[n, e.goal],
                        cf_arg=l3.argmax(-1), cf_q=cq, cf_valid=cfv.all(-1), uns=uns))
        rew = e.step(a)
        o = e.tok[n, e.idx.clamp(max=t.L - 1), MM.F_SYM] - 1
        rec[-1].update(rew=rew.clone(), sym=torch.where(~e.done, o, torch.full_like(o, -1)))
        ok = (e.cf >= 0) & ~e.done[:, None]
        e.cf = torch.where(ok, t.node_nxt[e.cf.clamp(min=0), a[:, None].expand(-1, 3), o.clamp(min=0)[:, None].expand(-1, 3)], torch.full_like(e.cf, -1))
    return e, rec

em, rm = run("model"); es, rs = run("solver")
assert torch.equal(em.pre_cell, es.pre_cell) and torch.equal(em.goal, es.goal)
gam = t.gamma
ret = lambda rec: sum((gam ** s) * r["rew"].double() for s, r in enumerate(rec))
Rm, Rs = ret(rm), ret(rs)
def first_diff():
    fd = torch.full((N,), -1, device=dev)
    for s in range(t.H):
        m = (fd < 0) & rm[s]["live"] & (rm[s]["a"] != rs[s]["a"])
        fd[m] = s
    return fd
fd = first_diff()
# the model's decision at the divergence: was it an additively unsolvable case where the model's action is suboptimal?
def at(rec, s, key, i): return rec[s][key][i]
bad_hard, ok_hard, same, other = [], [], [], []
for i in range(N):
    s = int(fd[i])
    if s < 0:
        if Rm[i] > 0 and int(em.prefix[i]) >= 2: same.append(i)
        continue
    q = rm[s]["q"][i]; opt = q >= q.max() - 1e-6
    sub = not bool(opt[rm[s]["a"][i]])
    hard = bool(rm[s]["uns"][i])
    if sub and hard and Rs[i] - Rm[i] > 0.05: bad_hard.append(i)
    elif sub and not hard and Rs[i] - Rm[i] > 0.05: other.append(i)
# any episode where the model meets a hard case and still acts optimally
for i in range(N):
    for s in range(t.H):
        if not rm[s]["live"][i]: break
        q = rm[s]["q"][i]
        if rm[s]["uns"][i] and q[rm[s]["a"][i]] >= q.max() - 1e-6 and len(set(rm[s]["cf_q"][i].argmax(-1).tolist())) >= 2:
            ok_hard.append(i); break
print("pool sizes: model fails a hard case", len(bad_hard), "| model handles a hard case", len(ok_hard), "| identical", len(same), "| fails elsewhere", len(other))

def pick(pool, k, key=lambda i: 0):
    seen, out = set(), []
    for i in sorted(pool, key=key):
        sig = (int(em.goal[i]), int(fd[i]) if fd[i] >= 0 else -1)
        if sig in seen: continue
        seen.add(sig); out.append(i)
        if len(out) == k: break
    return out
gap = lambda i: -float(Rs[i] - Rm[i])
chosen = [("hard", i) for i in pick(bad_hard, 6, gap)] + [("hard-ok", i) for i in pick(ok_hard, 2)] + \
         [("other", i) for i in pick(other, 2, gap)] + [("same", i) for i in pick(same, 2)]
r3 = lambda x: [round(float(v), 3) for v in x]
def agent_frames(rec, i):
    fr = []
    for s in range(t.H):
        r = rec[s]
        if not r["live"][i]: break
        q = r["q"][i]
        fr.append(dict(cell=int(r["cell"][i]), belief=r3(t.belief[r["node"][i]]), q=r3(q), opt=[bool(x) for x in (q >= q.max() - 1e-6)],
                       act=int(r["a"][i]), probs=r3(r["probs"][i]), H=r3(r["H"][i]), G=r3(r["G"][i]),
                       cf_arg=[int(x) for x in r["cf_arg"][i]] if r["cf_valid"][i] else None,
                       cf_opt=[[bool(x) for x in (r["cf_q"][i][g] >= r["cf_q"][i][g].max() - 1e-6)] for g in range(3)] if r["cf_valid"][i] else None,
                       hard=bool(r["uns"][i]), sym=int(r["sym"][i]), reward=float(r["rew"][i])))
    return fr
eps = []
for kind, i in chosen:
    L = int(em.prefix[i])
    pre = [dict(cell=int(em.pre_cell[i, j]), belief=r3(t.belief[em.pre_node[i, j]]),
                move=int(em.tok[i, j, MM.F_ACT]) - 1 if j > 0 else None, sym=int(em.tok[i, j, MM.F_SYM]) - 1) for j in range(L + 1)]
    eps.append(dict(kind=kind, goal=int(em.goal[i]), prefix=pre, diverge=int(fd[i]), ret_model=round(float(Rm[i]), 3), ret_bayes=round(float(Rs[i]), 3),
                    model=agent_frames(rm, i), bayes=agent_frames(rs, i)))
summary = dict(n=N, ret_model=round(float(Rm.mean()), 4), ret_bayes=round(float(Rs.mean()), 4),
               success_model=round(float((Rm > 0).double().mean()), 4), success_bayes=round(float((Rs > 0).double().mean()), 4))
maze = dict(cells=[list(c) for c in t.cells], sym=M26.T18.MB.cross_maze().sym.tolist() if hasattr(M26.T18, "MB") else None,
            goals=[int(x) for x in t.goal_cell], starts=[int(x) for x in t.starts], eps=0.4, gamma=gam, H=t.H)
json.dump(dict(model=MODEL, seed=SEED, maze=maze, summary=summary, episodes=eps), open(SP + f"sim_{MODEL}_s{SEED}.json", "w"))
print(summary, [(k, e["goal"], e["diverge"], e["ret_model"], e["ret_bayes"]) for (k, _), e in zip(chosen, eps)])
