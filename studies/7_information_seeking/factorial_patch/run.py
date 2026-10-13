"""Factorial patching: two histories x two goals, every cell patched into every other, component by component.
Registered (PLAN.md). Nothing is fitted. For each quartet (h1, h2) x (A, B) of same-step decisions the four natural
runs are the references; a quantity (the decision token's residual at a site, the history tokens' or the goal token's
residual, a sublayer output or a head's output at the decision token) is taken from a source cell and written into a
recipient cell, all 4 x 3 directions. Where the patched decision goes (source / the source's history under the
recipient's goal / the recipient's history under the source's goal / kept), the per-component causal additivity in
history and goal, the symmetry across recipients.

    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/factorial_patch/run.py                       # results.json
    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/factorial_patch/run.py --untrained --seeds 0 --out _smoke
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
HS, M10 = HERE.parent / "H_simplex", HERE.parent / "maze10"
sys.path.insert(0, str(HS)); sys.path.insert(0, str(M10))
import run as R, measure as MS, train as TR          # noqa: E402
from equiv import qvals                                # noqa: E402

N_DEC, N_PAIRS, CHUNK = 8192, 4096, 2048
EQ, DIFF, N_TEST_EQ, CANDS, MINQ, MAX_EQ = 0.5, 1.0, 8192, 256, 0.4, 6000      # equiv.py's selection, reproduced
ROLES = ("h1A", "h1B", "h2A", "h2B")                   # role r: history bit r >> 1 (0 = h1), goal bit r & 1 (0 = A)
TYPES = ("same_goal", "same_hist", "both")
RANDOM_AT = [("dec", 2)] + [("attn", l) for l in range(4)] + [("mlp", l) for l in range(4)]


def quantities():
    return ([("dec", l) for l in (1, 2, 3, 4)] + [("hist", l) for l in (1, 2, 3)] + [("goaltok", l) for l in (1, 2, 3)]
            + [("attn", l) for l in range(4)] + [("mlp", l) for l in range(4)] + [("head", l, k) for l in range(4) for k in range(4)])


def qname(q):
    return f"{q[0]}@resid{q[1]}" if q[0] in ("dec", "hist", "goaltok") else (f"{q[0]} {q[1]}" if q[0] != "head" else f"head {q[1]}.{q[2]}")


def ptype(r, s):
    return "same_goal" if (r & 1) == (s & 1) else ("same_hist" if (r >> 1) == (s >> 1) else "both")


# ------------------------------------------------------------------ natural runs

@torch.no_grad()
def records(net, tok, p, K):
    """Natural runs of every decision under every goal: the whole sequence's residual entering blocks 1-3
    [N, K, L, d]; at the decision token the residual at sites 1-4, every sublayer's output and every head's output;
    the centred logits [N, K, 4]."""
    N, L = tok.shape[:2]; dev = tok.device; d = net.d
    E = lambda *s: torch.empty(N, K, *s, device=dev)
    out = dict(resid_full={l: E(L, d) for l in (1, 2, 3)}, resid={l: E(d) for l in (1, 2, 3, 4)},
               attn={l: E(d) for l in range(net.nl)}, mlp={l: E(d) for l in range(net.nl)}, heads={l: E(net.nh, d) for l in range(net.nl)},
               logits=torch.empty(N, K, 4, device=dev, dtype=torch.float64))
    for i in range(0, N, CHUNK):
        n_ = min(CHUNK, N - i); tk = tok[i:i + n_].repeat(K, 1, 1); pp = p[i:i + n_].repeat(K)
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
        lg, _, rec = net(tk, record=True); ar = torch.arange(K * n_, device=dev)
        V = lambda x: x.view(K, n_, *x.shape[1:]).transpose(0, 1)
        for l in (1, 2, 3): out["resid_full"][l][i:i + n_] = V(rec["resid"][l])
        for l in (1, 2, 3, 4): out["resid"][l][i:i + n_] = V(rec["resid"][l][ar, pp])
        for l in range(net.nl):
            out["attn"][l][i:i + n_] = V(rec["attn"][l][ar, pp]); out["mlp"][l][i:i + n_] = V(rec["mlp"][l][ar, pp])
            out["heads"][l][i:i + n_] = V(rec["heads"][l][ar, :, pp])
        z = V(lg[ar, pp]).double(); out["logits"][i:i + n_] = z - z.mean(-1, keepdim=True)
    return out


def source_values(rec, q, si, sg):
    """The quantity q of the source cells (record index si, goal sg): [Q, ...]."""
    if q[0] == "dec": return rec["resid"][q[1]][si, sg]
    if q[0] in ("hist", "goaltok"): return rec["resid_full"][q[1]][si, sg]
    if q[0] == "head": return rec["heads"][q[1]][si, sg, q[2]]
    return rec[q[0]][q[1]][si, sg]


@torch.no_grad()
def run_patched(net, tok, p, goal, make_patch):
    """Centred logits at p [n, 4] for the tokens under `goal` [n], with the patch functions of make_patch(slice, ar, pp)."""
    out = []
    for i in range(0, len(tok), CHUNK):
        sl = slice(i, i + CHUNK); tk = tok[sl].clone(); tk[:, 1, MM.F_GOAL] = goal[sl] + 1
        pp = p[sl]; ar = torch.arange(len(tk), device=tok.device)
        lg, _ = net(tk, patch=make_patch(sl, ar, pp))
        z = lg[ar, pp].double(); out.append(z - z.mean(-1, keepdim=True))
    return torch.cat(out)


def patch_for(q, val, rnd=None):
    """Patch functions writing `val` [Q, ...] (the source's quantity) into the recipient's run; with rnd [Q, d] the
    random vector is added at the decision token instead."""
    def make(sl, ar, pp):
        v = val[sl]
        if rnd is not None:
            key = ("resid", q[1]) if q[0] == "dec" else (q[0], q[1])
            def fn(x, r_=rnd[sl]):
                x = x.clone(); x[ar, pp] = x[ar, pp] + r_.to(x.dtype); return x
            return {key: fn}
        if q[0] == "dec":
            def fn(x):
                x = x.clone(); x[ar, pp] = v.to(x.dtype); return x
            return {("resid", q[1]): fn}
        if q[0] == "hist":
            def fn(x):
                x = x.clone(); pos = torch.arange(x.shape[1], device=x.device)[None]
                m = (pos <= pp[:, None]) & (pos != 1) & (pos != pp[:, None])
                x[m] = v[:, : x.shape[1]][m].to(x.dtype); return x
            return {("resid", q[1]): fn}
        if q[0] == "goaltok":
            def fn(x):
                x = x.clone(); x[:, 1] = v[:, 1].to(x.dtype); return x
            return {("resid", q[1]): fn}
        if q[0] == "head":
            def fn(h):
                h = h.clone(); h[ar, pp] = v.to(h.dtype); return h
            return {("head", q[1], q[2]): fn}
        def fn(y):
            y = y.clone(); y[ar, pp] = v.to(y.dtype); return y
        return {(q[0], q[1]): fn}
    return make


# ------------------------------------------------------------------ quartets

def random_pairs(dt, dev):
    """block1.py's draw: a subset of decisions after the first step and same-step pairs with posteriors >= 0.5 apart."""
    g = torch.Generator(device=dev); g.manual_seed(5)
    ok = torch.nonzero(dt["st"] >= 1).squeeze(1); ti = ok[torch.randperm(len(ok), device=dev, generator=g)[:N_DEC]]
    sub_st, sub_b = dt["st"][ti], dt["b"][ti]; by_step = {}
    for i in torch.randint(len(ti), (N_PAIRS * 4,), device=dev, generator=g).tolist(): by_step.setdefault(int(sub_st[i]), []).append(i)
    pr = []
    for i in torch.randint(len(ti), (N_PAIRS * 4,), device=dev, generator=g).tolist():
        cands = by_step.get(int(sub_st[i]), [])
        if not cands: continue
        j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=g))]
        if j != i and float((sub_b[i] - sub_b[j]).abs().sum()) >= 0.5: pr.append((i, j))
        if len(pr) >= N_PAIRS: break
    pr = torch.tensor(pr, device=dev)
    return ti[pr[:, 0]], ti[pr[:, 1]]


def equiv_pairs(t, dt, n_fit_decisions, K, dev):
    """equiv.py's selection, the same draw: same step, A-equivalent (same optimal set, QMDP profiles within EQ
    effective step, A worth >= MINQ), B-different (disjoint optimal sets, >= DIFF apart), one pair per recipient, at
    most MAX_EQ. Returns decision indices i1, i2 and the ordered goals A, B; pairs at the first step are dropped
    afterwards."""
    gsub = torch.Generator(device=dev); gsub.manual_seed(1)
    torch.randperm(n_fit_decisions, device=dev, generator=gsub)                         # equiv.py draws the fit subset first
    ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:N_TEST_EQ]
    Qt = qvals(t, dt, ti)
    cQ = lambda Q: (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Q.clamp(min=1e-9)) / np.log(t.gamma))
    sett = Qt >= Qt.max(-1, keepdim=True).values - 1e-6
    gp = torch.Generator(device=dev); gp.manual_seed(7)
    m = len(ti); cand = torch.randint(m, (m, CANDS), device=dev, generator=gp)
    same_step = dt["st"][ti][:, None] == dt["st"][ti][cand]
    cqt = cQ(Qt); dq = (cqt[:, None] - cqt[cand]).abs().amax(-1)
    same_set = (sett[:, None] == sett[cand]).all(-1); disjoint = ~(sett[:, None] & sett[cand]).any(-1)
    reach = Qt.amax(-1) >= MINQ
    eq = same_set & (dq < EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > DIFF)
    pairs = []
    for A in range(K):
        for B in range(K):
            if A == B: continue
            ok = same_step & eq[..., A] & dif[..., B] & (cand != torch.arange(m, device=dev)[:, None])
            ii, cc = torch.nonzero(ok, as_tuple=True)
            first = torch.ones(m, dtype=torch.bool, device=dev); keep = []
            for a_, c_ in zip(ii.tolist(), cc.tolist()):
                if first[a_]: first[a_] = False; keep.append((a_, int(cand[a_, c_]), A, B))
            pairs += keep
    if len(pairs) > MAX_EQ:
        sel = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0))[:MAX_EQ].tolist(); pairs = [pairs[i] for i in sel]
    P = torch.tensor(pairs, device=dev)
    return ti[P[:, 0]], ti[P[:, 1]], P[:, 2], P[:, 3]


# ------------------------------------------------------------------ statistics

def gmean(v, m, gid, ng):
    """Mean of v over the quartets with m true, per group: list of ng floats (nan where empty)."""
    v = v.double() * m; s = torch.zeros(ng, device=v.device, dtype=torch.float64).index_add_(0, gid, v)
    c = torch.zeros(ng, device=v.device, dtype=torch.float64).index_add_(0, gid, m.double())
    return (s / c).tolist()


def gcount(m, gid, ng):
    return torch.zeros(ng, device=m.device, dtype=torch.float64).index_add_(0, gid, m.double()).tolist()


def score(Lhat, r, s, Lc, dc, gid, ng):
    """Where the patched decision goes, for recipient role r and source role s."""
    dh = Lhat.argmax(-1); Lr, Ls, dr, ds = Lc[r], Lc[s], dc[r], dc[s]
    qh, qg = ((s >> 1) << 1) | (r & 1), ((r >> 1) << 1) | (s & 1)
    inf = ds != dr
    out = dict(n_inf=gcount(inf, gid, ng), to_source=gmean(dh == ds, inf, gid, ng), kept=gmean(dh == dr, inf, gid, ng),
               elsewhere=gmean((dh != ds) & (dh != dr), inf, gid, ng), changed=gmean(dh != dr, torch.ones_like(inf), gid, ng))
    for name, q in (("qh", qh), ("qg", qg)):
        if q != s and q != r:
            out["to_" + name] = gmean(dh == dc[q], dc[q] != dr, gid, ng)
    d4 = (dc[0] != dc[1]) & (dc[0] != dc[2]) & (dc[0] != dc[3]) & (dc[1] != dc[2]) & (dc[1] != dc[3]) & (dc[2] != dc[3])
    out["n4"] = gcount(d4, gid, ng); out["u_source"] = gmean(dh == ds, d4, gid, ng); out["u_kept"] = gmean(dh == dr, d4, gid, ng)
    if qh != s and qh != r: out["u_qh"] = gmean(dh == dc[qh], d4, gid, ng); out["u_qg"] = gmean(dh == dc[qg], d4, gid, ng)
    diff = Ls - Lr; nn = (diff ** 2).sum(1); ok = nn > 1e-9
    out["transfer"] = gmean(((Lhat - Lr) * diff).sum(1) / nn.clamp(min=1e-9), ok, gid, ng)
    return out


def additivity(Db, Dh, Dg, gid, ng):
    """The causal interaction: the residual of the two-factor change against the sum of the one-factor changes,
    its squared norm over the two-factor change's (per group), and the mean cosine between residual and change."""
    I = Db - Dh - Dg
    s_i = torch.zeros(ng, device=I.device, dtype=torch.float64).index_add_(0, gid, (I ** 2).sum(1))
    s_b = torch.zeros(ng, device=I.device, dtype=torch.float64).index_add_(0, gid, (Db ** 2).sum(1))
    ok = Db.norm(dim=1) > 1e-9
    cos = (I * Db).sum(1) / (I.norm(dim=1) * Db.norm(dim=1)).clamp(min=1e-12)
    return dict(iota=(s_i / s_b).tolist(), cos=gmean(cos, ok, gid, ng), rel_norm=gmean(I.norm(dim=1) / Db.norm(dim=1).clamp(min=1e-12), ok, gid, ng))


def output_interaction(Y, gid, ng):
    """Y: {role: [Q, ...]} a quantity's natural value in the four cells; the variance share of its goal x history
    interaction (the double difference) in the four cells' variation, per group."""
    flat = {r: y.reshape(len(y), -1).double() for r, y in Y.items()}
    I = flat[3] - flat[2] - flat[1] + flat[0]
    mean = sum(flat.values()) / 4
    tot = sum(((y - mean) ** 2).sum(1) for y in flat.values())
    s_i = torch.zeros(ng, device=I.device, dtype=torch.float64).index_add_(0, gid, (I ** 2).sum(1) / 4)
    s_t = torch.zeros(ng, device=I.device, dtype=torch.float64).index_add_(0, gid, tot)
    return (s_i / s_t).tolist()


@torch.no_grad()
def run_set(net, rec, tok_of, p_of, i1, i2, gA, gB, gid, ng, label, log):
    """All directed patches of every quantity on one quartet set. i1, i2: record indices; gA, gB: goals [Q]."""
    dev = i1.device; Q = len(i1)
    cells = {0: (i1, gA), 1: (i1, gB), 2: (i2, gA), 3: (i2, gB)}
    Lc = {r: rec["logits"][idx, g] for r, (idx, g) in cells.items()}; dc = {r: L.argmax(-1) for r, L in Lc.items()}
    res = dict(n=Q, n_by_group=gcount(torch.ones(Q, dtype=torch.bool, device=dev), gid, ng),
               hist_dependent_A=gmean(dc[0] != dc[2], torch.ones(Q, dtype=torch.bool, device=dev), gid, ng),
               hist_dependent_B=gmean(dc[1] != dc[3], torch.ones(Q, dtype=torch.bool, device=dev), gid, ng),
               goal_dependent_h1=gmean(dc[0] != dc[1], torch.ones(Q, dtype=torch.bool, device=dev), gid, ng),
               goal_dependent_h2=gmean(dc[2] != dc[3], torch.ones(Q, dtype=torch.bool, device=dev), gid, ng),
               natural=dict(iota={}, patches={}, output_interaction={}))
    d4 = (dc[0] != dc[1]) & (dc[0] != dc[2]) & (dc[0] != dc[3]) & (dc[1] != dc[2]) & (dc[1] != dc[3]) & (dc[2] != dc[3])
    res["four_distinct"] = gmean(d4, torch.ones(Q, dtype=torch.bool, device=dev), gid, ng)
    for r in range(4):
        sh, sg, sb = r ^ 2, r ^ 1, r ^ 3
        res["natural"]["iota"][ROLES[r]] = additivity(Lc[sb] - Lc[r], Lc[sh] - Lc[r], Lc[sg] - Lc[r], gid, ng)
    res["natural"]["output_interaction"]["logits"] = output_interaction(Lc, gid, ng)
    gr = torch.Generator(device=dev); gr.manual_seed(11)
    t0 = time.time()
    for q in quantities() + [("random",) + q for q in RANDOM_AT]:
        is_rnd = q[0] == "random"; qq = q[1:] if is_rnd else q
        name = ("random " if is_rnd else "") + qname(qq)
        row = dict(patches={}, additivity={})
        if not is_rnd and qq[0] not in ("hist", "goaltok"):
            res["natural"]["output_interaction"][name] = output_interaction({r: source_values(rec, qq, idx, g) for r, (idx, g) in cells.items()}, gid, ng)
        for r in range(4):
            ridx, rg = cells[r]; tok, p = tok_of[ridx], p_of[ridx]
            D = {}
            for s in range(4):
                if s == r: continue
                sidx, sg = cells[s]
                val = source_values(rec, qq, sidx, sg)
                rnd = None
                if is_rnd:
                    vr = source_values(rec, qq, ridx, rg); rnd = torch.randn(vr.shape, device=dev, generator=gr)
                    rnd = rnd / rnd.norm(dim=1, keepdim=True) * (val - vr).norm(dim=1, keepdim=True)
                Lhat = run_patched(net, tok, p, rg, patch_for(qq, val, rnd))
                if not is_rnd and qq == ("dec", 4):
                    assert float((Lhat - Lc[s]).abs().max()) < 1e-2, float((Lhat - Lc[s]).abs().max())
                row["patches"][f"{ROLES[r]}<-{ROLES[s]}"] = dict(type=ptype(r, s), **score(Lhat, r, s, Lc, dc, gid, ng))
                D[s] = Lhat - Lc[r]
            row["additivity"][ROLES[r]] = additivity(D[r ^ 3], D[r ^ 2], D[r ^ 1], gid, ng)
        res["natural"]["patches"][name] = row
        pooled = lambda ty, k: float(np.nanmean([np.nanmean(v[k]) for v in row["patches"].values() if v["type"] == ty]))
        log(f"  {label} {name:18s} to source: same_goal {pooled('same_goal', 'to_source'):.2f} same_hist {pooled('same_hist', 'to_source'):.2f} both {pooled('both', 'to_source'):.2f}"
            f" (qh {pooled('both', 'to_qh'):.2f} qg {pooled('both', 'to_qg'):.2f}) | iota {np.nanmean([np.nanmean(v['iota']) for v in row['additivity'].values()]):.2f} | {time.time() - t0:.0f}s")
    res["quantities"] = res["natural"].pop("patches")
    return res


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--untrained", action="store_true"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--runs", default=str(M10 / "runs" / "ppo")); ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    out_dir = Path(a.out) if a.out else HERE; out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "run.log", "a")
    def log(s):
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    runs = Path(a.runs); cfg = json.load(open(runs / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K = len(t.goal_cell)
    n_ep = 512 if (a.untrained or a.quick) else R.N_EP
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, n_ep, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, n_ep, gen); fgoal, fcell = e.goal, e.cell          # equiv.py's fit episodes: only their count is needed
    res = dict(task=cfg["task"], n_episodes=n_ep, roles=ROLES, runs={})
    for s in a.seeds:
        d = runs / f"seed{s}" / "ckpt"; ck = d / "u000000.pt" if a.untrained else sorted(d.glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, cfg["args"].get("d", 128), cfg["args"].get("layers", 4))
        net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        dt = R.decisions(t, net, r, K); n_fit = int(rf["alive"].sum())
        i1r, i2r = random_pairs(dt, dev)
        e1, e2, eA, eB = equiv_pairs(t, dt, n_fit, K, dev)
        keep = (dt["st"][e1] >= 1); n_eq_step0 = int((~keep).sum()); e1, e2, eA, eB = e1[keep], e2[keep], eA[keep], eB[keep]
        uidx = torch.unique(torch.cat([i1r, i2r, e1, e2])); pos_of = torch.full((len(dt["st"]),), -1, device=dev, dtype=torch.long); pos_of[uidx] = torch.arange(len(uidx), device=dev)
        tok_u, p_u = r["tok"][dt["ep"][uidx]], dt["st"][uidx] + 1
        rec = records(net, tok_u, p_u, K)
        dlog = float((rec["logits"] - dt["lc"][uidx]).abs().max()); scale = float(dt["lc"][uidx].abs().max())
        assert dlog < 1e-2 * scale, (dlog, scale)                                      # full-length against truncated sequences: float noise only
        row = dict(checkpoint=ck.name, decisions=len(dt["st"]), decisions_used=len(uidx), logit_check=dict(max_abs_diff=dlog, logit_scale=scale), sets={})
        # random quartets: every pair x every unordered goal pair
        gp = [(A, B) for A in range(K) for B in range(A + 1, K)]
        Qp = len(i1r); i1 = pos_of[i1r].repeat(len(gp)); i2 = pos_of[i2r].repeat(len(gp))
        gA = torch.tensor([A for A, _ in gp], device=dev).repeat_interleave(Qp); gB = torch.tensor([B for _, B in gp], device=dev).repeat_interleave(Qp)
        gid = torch.arange(len(gp), device=dev).repeat_interleave(Qp)
        log(f"seed{s} {ck.name}: {len(dt['st'])} decisions, {Qp} random pairs x {len(gp)} goal pairs, {len(e1)} A-equivalent pairs ({n_eq_step0} at step 0 dropped)")
        row["sets"]["random"] = dict(groups=[f"{A}{B}" for A, B in gp], **run_set(net, rec, tok_u, p_u, i1, i2, gA, gB, gid, len(gp), "random", log))
        # A-equivalent quartets: ordered goal pairs
        go = [(A, B) for A in range(K) for B in range(K) if A != B]; gmap = {ab: i for i, ab in enumerate(go)}
        gid = torch.tensor([gmap[(int(A), int(B))] for A, B in zip(eA.tolist(), eB.tolist())], device=dev)
        row["sets"]["equiv"] = dict(groups=[f"{A}->{B}" for A, B in go], **run_set(net, rec, tok_u, p_u, pos_of[e1], pos_of[e2], eA, eB, gid, len(go), "equiv", log))
        res["runs"][f"seed{s}"] = row
        log(f"seed{s} done in {time.time() - t0:.0f}s")
        (out_dir / ("smoke.json" if a.untrained else "results.json")).write_text(json.dumps(res))
        del rec; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
