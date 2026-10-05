"""The maze-belief experiment: fixed histories, decoders of the exact location posterior, and interventions.

Bank. Histories from an epsilon-greedy solver (never from a model under study), replayed unchanged at every
checkpoint. Decoders are fitted on histories whose prefix evidence (the tokens before the goal) hashes to the fit
side, and tested on the others, so no test decision shares its prefix with a fit decision.

Pairs, all at the reveal (the goal token), where the exact posterior does not depend on the goal:
    recipient  prefix A, goal g
    hybrid     prefix B, goal g       the recipient's goal with other evidence: the reference for a belief patch
    donor      prefix B, goal g'      other evidence and another goal (g' = g for same-goal pairs)
    swapped    prefix A, goal g'      the reference for a goal patch
A and B have the same length, so every token position matches.

Patches. At the goal token: a head's output, a block's attention or MLP output, the residual stream. At prefix
tokens: the residual stream entering block l, at every prefix token or at the last one only. Prefix tokens
cannot see the goal, so whatever is read from them is goal-free by construction.
"""

from __future__ import annotations

import torch

from goalgeo import mazemodel as MM, mazeppo as P
from goalgeo.navprobe import Affine, site_names, site_stack


# ------------------------------------------------------------------ bank

def make_bank(t: P.Tables, seed, n=40000):
    gen = torch.Generator(device=t.dev); gen.manual_seed(seed)
    parts = [P.rollout(None, t, n // 4, gen, behaviour=P.solver_behaviour(t, e, gen)) for e in (0.1, 0.3, 0.6, 1.0)]
    b = P.Batch(**{k: torch.cat([getattr(p, k) for p in parts]) for k in P.Batch.__dataclass_fields__})
    # hash of the prefix evidence
    L = b.tok.shape[1]
    before = torch.arange(L, device=t.dev)[None] <= b.prefix[:, None]
    code = (b.tok[:, :, MM.F_SYM] * 5 + b.tok[:, :, MM.F_ACT]) * before
    w = torch.tensor([(7919 * (i + 1)) % 1000003 for i in range(L)], device=t.dev)
    h = ((code * w).sum(1) * 2654435761) % 1000
    fit = torch.where(b.prefix == 0, torch.rand(len(h), device=t.dev, generator=gen) < 0.7, h < 700)
    return b, fit


def decisions(t: P.Tables, b: P.Batch, fit):
    hi, s = torch.nonzero(b.alive, as_tuple=True)
    node = b.node[hi, s]
    q = t.Q[node]
    bel = t.belief[node]
    left = torch.tensor([c <= 4 for _, c in _cells(t)], device=t.dev, dtype=torch.float32)
    return dict(hist=hi, step=s, pos=b.pos[hi, s], node=node, goal=b.goal[hi], prefix=b.prefix[hi], belief=bel, q_star=q,
                optimal=q >= q.max(1, keepdim=True).values - 1e-6, fit=fit[hi], p_left=bel @ left, cell=b.cell[hi, s],
                entropy=-(bel * bel.clamp(min=1e-12).log()).sum(1))


def _cells(t):
    return t.cells


def prefix_rows(t: P.Tables, b: P.Batch, fit):
    """One row per prefix token (the first symbol included): the goal-free posterior after that token."""
    hi, i = torch.nonzero(b.pre_node >= 0, as_tuple=True)
    return dict(hist=hi, pos=i, node=b.pre_node[hi, i], belief=t.belief[b.pre_node[hi, i]], fit=fit[hi], last=i == b.prefix[hi])


# ------------------------------------------------------------------ activations and decoders

@torch.no_grad()
def collect(net, tok, rows, chunk=8192):
    """Activations at (history, position) rows: [sites, N, d], and logits [N, 4]."""
    S, N = 2 * net.nl + 1, len(rows["hist"])
    acts = torch.zeros(S, N, net.d, device=tok.device); logits = torch.zeros(N, 4, device=tok.device)
    for s in range(0, tok.shape[0], chunk):
        sel = (rows["hist"] >= s) & (rows["hist"] < s + chunk)
        if sel.any():
            lg, _, rec = net(tok[s:s + chunk], record=True)
            hi, p = rows["hist"][sel] - s, rows["pos"][sel]
            for i, a in enumerate(site_stack(rec, net.nl)):
                acts[i, sel] = a[hi, p]
            logits[sel] = lg[hi, p]
    return acts, logits


def belief_scores(pred, Y):
    """Decoding of a probability vector: share of the total variance explained, mean L1 error, validity."""
    Y = Y.double()
    err = ((pred - Y) ** 2).sum(0)
    var = ((Y - Y.mean(0)) ** 2).sum(0)
    return dict(r2=float(1 - err.sum() / var.sum()), l1=float((pred - Y).abs().sum(1).mean()),
                invalid=float(((pred < 0) | (pred > 1)).any(1).double().mean()), sum_error=float((pred.sum(1) - 1).abs().mean()))


def scalar_r2(pred, y):
    y = y.double()
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# ------------------------------------------------------------------ pairs

def build_pairs(t: P.Tables, b: P.Batch, fit, n, seed, min_prefix=2, min_l1=0.5, test_only=True):
    """Recipients and donors among histories with the same prefix length; the donor's posterior at the reveal is at
    least min_l1 (L1) from the recipient's. Half the pairs have the same goal."""
    gen = torch.Generator(device=t.dev); gen.manual_seed(seed)
    dev = t.dev
    ok = (b.prefix >= min_prefix) & (~fit if test_only else torch.ones_like(fit))
    pool = torch.nonzero(ok).squeeze(1)
    r = pool[torch.randint(len(pool), (4 * n,), device=dev, generator=gen)]          # with replacement: a recipient meets several donors
    d = pool[torch.randint(len(pool), (len(r),), device=dev, generator=gen)]
    nr, nd = b.pre_node[r, b.prefix[r]], b.pre_node[d, b.prefix[d]]
    good = (b.prefix[r] == b.prefix[d]) & ((t.belief[nr] - t.belief[nd]).abs().sum(1) >= min_l1)
    r, d, nr, nd = r[good][:n], d[good][:n], nr[good][:n], nd[good][:n]
    m = len(r)
    same = torch.rand(m, device=dev, generator=gen) < 0.5
    g = b.goal[r]
    g2 = torch.where(same, g, (g + 1 + torch.randint(2, (m,), device=dev, generator=gen)) % 3)
    L = b.prefix[r]
    pos = 1 + L
    rows = torch.arange(m, device=dev)

    def seq(src, goal):
        tok = torch.zeros(m, t.L, MM.NF, dtype=torch.long, device=dev)
        keep = torch.arange(t.L, device=dev)[None] <= L[:, None]
        tok[keep] = b.tok[src][keep]
        tok[rows, pos, MM.F_TYPE], tok[rows, pos, MM.F_GOAL] = MM.GOAL, goal + 1
        return tok
    q = lambda node, goal: t.Q[t.reveal[node, goal]]
    opt = lambda node, goal: q(node, goal) >= q(node, goal).max(1, keepdim=True).values - 1e-6
    return dict(n=m, pos=pos, prefix=L, same_goal=same, goal=g, goal2=g2,
                tok=seq(r, g), tok_hybrid=seq(d, g), tok_donor=seq(d, g2), tok_swapped=seq(r, g2),
                belief=t.belief[nr], belief_cf=t.belief[nd],
                opt=opt(nr, g), opt_hybrid=opt(nd, g), opt_donor=opt(nd, g2), opt_swapped=opt(nr, g2))


# ------------------------------------------------------------------ interventions

def components(net):
    out = []
    for l in range(net.nl):
        out += [("head", l, h) for h in range(net.nh)] + [("attn", l), ("mlp", l), ("resid", l)]
    return out + [("resid", net.nl)]


def name(c):
    return f"L{c[1]}.H{c[2]}" if c[0] == "head" else f"L{c[1]}.{c[0]}" if c[0] != "resid" else f"resid{c[1]}"


@torch.no_grad()
def run(net, tok, pos, patch=None):
    n = torch.arange(len(tok), device=tok.device)
    lg, _, rec = net(tok, record=True, patch=patch)
    return lg[n, pos].softmax(-1), rec


def _at(rec, c, n, pos):
    if c[0] == "head":
        return rec["heads"][c[1]][n, c[2], pos]
    return rec[{"attn": "attn", "mlp": "mlp", "resid": "resid"}[c[0]]][c[1]][n, pos]


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def moved(pp, pb, target, sel):
    """How far the patch moves the distribution to the target, 1 = all the way, on pairs `sel`."""
    if sel.sum() < 20:
        return None
    return float(1 - tv(pp, target)[sel].mean() / tv(pb, target)[sel].mean())


@torch.no_grad()
def interventions(net, pairs, probe, gap=0.3, groups=None):
    """Every patch, for same-goal and cross-goal pairs.
    Same goal: the donor is the hybrid. 'to_hybrid' is the effect of the evidence.
    Cross goal: 'to_hybrid' = behaviour for (donor's evidence, recipient's goal), 'to_donor' = the donor's own behaviour,
    'to_swapped' = behaviour for (recipient's evidence, donor's goal).
    groups: {name: [components]} patched together.
    Also: the decoded posterior's displacement along the exact one ('belief'), and the gain in probability of the
    solver's optimal set for the target ('solver'), both for the hybrid."""
    n = torch.arange(pairs["n"], device=pairs["tok"].device)
    pos = pairs["pos"]
    pb, rb = run(net, pairs["tok"], pos)
    ph, _ = run(net, pairs["tok_hybrid"], pos)
    pd, rd = run(net, pairs["tok_donor"], pos)
    ps, _ = run(net, pairs["tok_swapped"], pos)
    same, cross = pairs["same_goal"], ~pairs["same_goal"]
    far = lambda a, b: tv(a, b) > gap
    sel_same = same & far(pb, ph)
    sel_cross = cross & far(pb, ph) & far(pb, pd) & far(ph, pd) & far(pb, ps) & far(ps, ph)
    target = (pairs["belief_cf"] - pairs["belief"]).double()
    dec0 = probe(rb["resid"][net.nl][n, pos])
    excl = (pairs["opt_hybrid"] & ~pairs["opt"]).float()
    out = dict(n_same=int(sel_same.sum()), n_cross=int(sel_cross.sum()),
               model=dict(tv_evidence=float(tv(pb, ph).mean()), tv_goal=float(tv(pb, ps).mean()),
                          use=float((((pb * pairs["opt"]).sum(1) - (pb * pairs["opt_hybrid"]).sum(1))[~(pairs["opt"] & pairs["opt_hybrid"]).any(1)]).mean())))
    L = pairs["tok"].shape[1]
    prefix_mask = torch.arange(L, device=pos.device)[None] < pos[:, None]
    last_mask = torch.arange(L, device=pos.device)[None] == (pos - 1)[:, None]

    def measure(patch):
        pp, rp = run(net, pairs["tok"], pos, patch)
        d = probe(rp["resid"][net.nl][n, pos]) - dec0
        has = excl.sum(1) > 0
        return dict(same_to_hybrid=moved(pp, pb, ph, sel_same), cross_to_hybrid=moved(pp, pb, ph, sel_cross),
                    cross_to_donor=moved(pp, pb, pd, sel_cross), cross_to_swapped=moved(pp, pb, ps, sel_cross),
                    belief=float((d * target).sum() / (target ** 2).sum()),
                    solver=float(((pp - pb) * excl).sum(1)[has].mean() / ((ph - pb) * excl).sum(1)[has].mean()))

    for c in components(net):
        val = _at(rd, c, n, pos)
        def fn(x, val=val):
            x = x.clone(); x[n, pos] = val
            return x
        out[name(c)] = measure({c: fn})
    for g, comps in (groups or {}).items():
        patch = {}
        for c in comps:
            val = _at(rd, c, n, pos)
            def fn(x, val=val):
                x = x.clone(); x[n, pos] = val
                return x
            patch[c] = fn
        out[g] = measure(patch) if patch else None
    for l in range(net.nl + 1):
        for label, mask in (("prefix", prefix_mask), ("last", last_mask)):
            src = rd["resid"][l]
            def fn(x, src=src, mask=mask):
                x = x.clone(); x[mask] = src[mask]
                return x
            out[f"{label}@resid{l}"] = measure({("resid", l): fn})
    return out


@torch.no_grad()
def goal_swap(net, pairs, probe):
    """The same history with another goal: displacement of the decoded posterior (L1) against the displacement the
    evidence swap produces, and the change of the action distribution."""
    n = torch.arange(pairs["n"], device=pairs["tok"].device)
    pos, sel = pairs["pos"], ~pairs["same_goal"]
    out = {}
    pb, rb = run(net, pairs["tok"], pos); ps, rs = run(net, pairs["tok_swapped"], pos); ph, rh = run(net, pairs["tok_hybrid"], pos)
    for l in range(net.nl + 1):
        f = lambda r: probe[l](r["resid"][l][n, pos])
        out[f"resid{l}"] = dict(goal=float((f(rs) - f(rb)).abs().sum(1)[sel].mean()), evidence=float((f(rh) - f(rb)).abs().sum(1)[sel].mean()))
    out["tv_goal"], out["tv_evidence"] = float(tv(pb, ps)[sel].mean()), float(tv(pb, ph)[sel].mean())
    return out


def head_groups(net, res, thresh=0.3):
    """From a discovery run: heads whose patch carries the goal (it moves a cross-goal recipient at least `thresh` of
    the way to the behaviour for the donor's goal), and the other heads, per layer."""
    out = {}
    for l in range(net.nl):
        heads = [("head", l, h) for h in range(net.nh)]
        goal = [c for c in heads if (res[name(c)]["cross_to_swapped"] or 0) >= thresh]
        out[f"L{l}.goal_heads"] = goal
        out[f"L{l}.other_heads"] = [c for c in heads if c not in goal]
    return out
