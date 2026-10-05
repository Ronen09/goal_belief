"""Screen 4-goal placements for the multi-goal collection task."""
from pathlib import Path as _P
__HERE__ = str(_P(__file__).resolve().parent) + "/"
__ROOT__ = str(_P(__file__).resolve().parents[2])
import sys, itertools, json, numpy as np, torch
SP = __HERE__ + ""
sys.path.insert(0, SP)
from goalgeo import multigoal as MGL
from screen import make, margin, X, X_START
dev = "cuda"; torch.set_grad_enabled(False)
VALS = MGL.value_settings(4).to(dev)                                # [256, 4]

def full_obs_Q(maze, vals):
    """Fully observed Q [H+1, n, 16, nv, 4] by DP over (moves left, cell, collected mask)."""
    n, K, H, gm = maze.n, len(maze.goals), maze.H, maze.gamma
    nv = len(vals); goals = list(maze.goals)
    nxt = torch.as_tensor(maze.nxt, device=dev)                     # [n, 4]
    V = torch.zeros(H + 1, n, 2 ** K, nv, device=dev); Q = torch.zeros(H + 1, n, 2 ** K, nv, 4, device=dev)
    gidx = {g: j for j, g in enumerate(goals)}
    for k in range(1, H + 1):
        for C in range(2 ** K):
            for a in range(4):
                s2 = nxt[:, a]
                q = gm * V[k - 1, s2, C]                            # [n, nv] default: no pickup
                for s in range(n):
                    c2 = int(s2[s]); j = gidx.get(c2)
                    if j is not None and not (C >> j) & 1:
                        q[s] = vals[:, j] + gm * V[k - 1, c2, C | (1 << j)]
                Q[k, :, C, :, a] = q
            V[k, :, C] = Q[k, :, C].max(-1).values
    return Q

def simulate(g, t_nxt, t_pobs, t_pc, t_nxc, maze, N, Q, gen):
    """Solver with 20 % random moves, random value setting per episode; returns decision records."""
    n_sym, K = maze.n_sym, len(maze.goals)
    E = torch.as_tensor(maze.E, device=dev, dtype=torch.float32); nxtc = torch.as_tensor(maze.nxt, device=dev)
    starts = torch.as_tensor(maze.starts, device=dev); goals = torch.as_tensor(maze.goals, device=dev)
    r = lambda hi: torch.randint(hi, (N,), device=dev, generator=gen)
    cell = starts[r(len(starts))]; L = r(5); v = r(len(VALS))
    o = torch.multinomial(E[cell], 1, generator=gen).squeeze(1)
    node = torch.as_tensor(g.start, device=dev)[o]
    for i in range(4):
        on = i < L; a = r(4)
        cell = torch.where(on, nxtc[cell, a], cell); o = torch.multinomial(E[cell], 1, generator=gen).squeeze(1)
        node = torch.where(on, t_nxt[node, a, o], node)
    node = torch.as_tensor(g.reveal, device=dev)[node]
    C = torch.zeros(N, dtype=torch.long, device=dev); alive = torch.ones(N, dtype=torch.bool, device=dev)
    rec = []; ret = torch.zeros(N, device=dev)
    for s in range(maze.H):
        q = Q[node, v]
        a = torch.where(torch.rand(N, device=dev, generator=gen) < 0.2, r(4), q.argmax(-1))
        rec.append(dict(node=node.clone(), v=v.clone(), cell=cell.clone(), C=C.clone(), L=L.clone(), step=torch.full_like(L, s), alive=alive.clone()))
        cell = nxtc[cell, a]
        isg = (cell[:, None] == goals[None]); j = isg.float().argmax(1); pick = isg.any(1) & ~((C >> j) & 1).bool()
        ret += torch.where(pick & alive, VALS[v, j] * maze.gamma ** s, torch.zeros_like(ret))
        o = torch.multinomial(E[cell], 1, generator=gen).squeeze(1)
        nn_ = torch.where(pick, t_nxc[node, a, j], t_nxt[node, a, o])
        C = torch.where(pick, C | (1 << j), C)
        alive = alive & (nn_ >= 0); node = nn_.clamp(min=0)
    cat = {k: torch.cat([x[k] for x in rec]) for k in rec[0]}
    m = cat.pop("alive")
    return {k: x[m] for k, x in cat.items()}, ret

def score(goals, N=20000, seed=0):
    maze = make(X, goals, X_START)
    g = MGL.build(maze, 4); Q = MGL.solve(g, VALS)
    T = lambda x: torch.as_tensor(x, device=dev)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    d, ret = simulate(g, T(g.nxt), T(g.p_obs), T(g.pc), T(g.nxt_c), maze, N, Q, gen)
    q = Q[d["node"]].double()                                        # [m, 256, 4]
    V = q.max(-1).values; opt = q >= V[..., None] - 1e-6
    # belief matters: QMDP and most-likely-cell policies, scored by the exact Q, at the setting actually played
    Fq = full_obs_Q(maze, VALS)
    bel = T(g.belief[d["node"].cpu().numpy()]).float()
    k = (maze.H - d["step"])
    fq = Fq[k[:, None].expand(-1, maze.n), torch.arange(maze.n, device=dev)[None].expand(len(k), -1), d["C"][:, None].expand(-1, maze.n), d["v"][:, None].expand(-1, maze.n)]  # [m, n, 4]
    qmdp = (bel[..., None] * fq).sum(1).argmax(-1); mapa = fq[torch.arange(len(k), device=dev), bel.argmax(1)].argmax(-1)
    qv = q[torch.arange(len(k), device=dev), d["v"]]
    loss = lambda a: (qv.max(-1).values - qv.gather(1, a[:, None]).squeeze(1))
    uncertain = bel.max(1).values < 0.999
    # additive code over value settings: H(history) + G(setting), G per (prefix length, step bin)
    Qc = q - q.mean(-1, keepdim=True); within = Qc - Qc.mean(1, keepdim=True)
    sb = d["step"].clamp(max=6); G = torch.zeros(5, 7, len(VALS), 4, dtype=torch.float64, device=dev)
    for l in range(5):
        for s in range(7):
            m = (d["L"] == l) & (sb == s)
            if m.any(): G[l, s] = within[m].mean(0)
    Gt = G[d["L"], sb]; Hh = Qc.mean(1)
    add_a = (Hh[:, None] + Gt).argmax(-1); aok = torch.gather(opt, 2, add_a[..., None]).squeeze(-1)
    dep = ~opt.all(1).any(-1)                                        # no action optimal under every setting
    aloss = V - torch.gather(q, 2, add_a[..., None]).squeeze(-1)
    return dict(goals=goals, nodes=len(g.k), solver_return=float(ret.mean()), decisions=len(k), share_uncertain=float(uncertain.double().mean()),
                qmdp_loss=float(loss(qmdp).mean()), qmdp_loss_uncertain=float(loss(qmdp)[uncertain].mean()),
                map_loss=float(loss(mapa).mean()), map_loss_uncertain=float(loss(mapa)[uncertain].mean()),
                value_dependent=float(dep.double().mean()), additive_share=float(1 - ((within - Gt) ** 2).sum() / (within ** 2).sum()),
                add_fail=float((~aok).double().mean()), add_loss=float(aloss.mean()), add_loss_uncertain=float(aloss[uncertain].mean()),
                add_fail_uncertain=float((~aok)[uncertain].double().mean()))

if __name__ == "__main__":
    cells = [(r, c) for r, row in enumerate(X) for c, ch in enumerate(row) if ch != "." and (r, c) not in X_START]
    out = []
    for gs in itertools.combinations(cells, 4):
        out.append(score(list(gs), N=8000))
    json.dump(out, open(SP + "screen_multi.json", "w"), indent=1)
    print(len(out), "placements")
