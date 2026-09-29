"""Round 17: matched pairs from the dense bank and interventions on components at the decision token.

A pair is a recipient decision and a donor decision at the same token position (same remaining horizon).
Donor kinds:
    twin     the recipient's own history replayed with one seen report flipped: every public field is
             identical, only the evidence (and so the belief) differs
    same     another history that reaches the same public state with the same belief by a different
             token sequence (another route or clue order): the patch should change nothing
    cross    the twin's history up to its last query, followed by a different walk that ends, at the same
             time, on a different cell with the same legal actions: the patch carries the twin's belief
             together with a different position and, often, a different best action. (The legal sets must
             match: logits of actions that are illegal where they were computed were never trained.)
             'strict' cross pairs have an action that is optimal only for the recipient's position with
             the counterfactual belief, and another that is optimal only for the donor's position.
The reference for every patch is the model's own behaviour on the twin (the recipient's position with the
counterfactual evidence).

Components patched at the decision token only: a head's output, a block's whole attention output, its MLP
output, and (the broad patch) the residual stream entering the block or leaving the last one.
"""

from __future__ import annotations

import torch

from goalgeo import navbank as B, navmodel as NM, navppo as P, navprobe as PR


def _pick(keys_sorted, order, want, gen, tries=6, ok=None):
    """For each wanted key, a random decision index with that key (or -1), subject to ok(candidate, row)."""
    lo = torch.searchsorted(keys_sorted, want)
    hi = torch.searchsorted(keys_sorted, want, right=True)
    out = torch.full_like(want, -1)
    for _ in range(tries):
        r = torch.rand(len(want), device=want.device, generator=gen)
        j = (lo + (r * (hi - lo)).long()).clamp(max=len(order) - 1)
        cand = order[j]
        good = (hi > lo) & (out < 0)
        if ok is not None:
            good = good & ok(cand)
        out = torch.where(good, cand, out)
    return out


def build_pairs(tab, dense, dec, layouts, n, seed):
    """Recipients: decisions with at least one clue, in the given layouts. Returns a dict of aligned tensors:
    recipient / twin token sequences, the decision index of the same-belief and cross-position donors (-1 if
    none), positions, exact beliefs and optimal sets."""
    dev = tab.dev
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    pool = torch.nonzero(((dec["sh"] > 0) | (dec["sv"] > 0)) & torch.isin(dec["layout"], layouts)).squeeze(1)
    r = pool[torch.randperm(len(pool), device=dev, generator=gen)[:n]]
    sh, sv, hist = dec["sh"][r], dec["sv"][r], dec["hist"][r]
    both = (sh > 0) & (sv > 0)
    flip_h = (sh > 0) & (~both | (torch.rand(len(r), device=dev, generator=gen) < 0.5))
    sh2 = torch.where(flip_h, 3 - sh, sh); sv2 = torch.where(~flip_h, 3 - sv, sv)
    # the twin: replay with the counterfactual reports (a report not yet seen at the decision is irrelevant to it)
    rep = dense.tok[hist][:, :, NM.F_REP]
    yh0, yv0 = (rep == NM.REP_R).any(1).long(), (rep == NM.REP_B).any(1).long()
    yh, yv = torch.where(flip_h, 1 - yh0, yh0), torch.where(~flip_h, 1 - yv0, yv0)
    tw = B.replay(tab, dense.cfg[hist], dense.state[hist, 0, 1], dense.act[hist], dense.alive[hist], yh, yv)
    t = dec["t"][r]
    n_ = torch.arange(len(r), device=dev)
    assert torch.equal(tw.state[n_, t], torch.stack([dec["k"][r], dec["cell"][r], sh2, sv2], -1))
    # donors from the bank
    flags = (dec["sh"] > 0) * 2 + (dec["sv"] > 0)
    key_same = dec["public"] * 9 + dec["sh"] * 3 + dec["sv"]
    o1 = torch.argsort(key_same)
    L = dense.tok.shape[1]
    prefix = (torch.arange(L, device=dev)[None] <= dec["pos"][r][:, None])
    def differs(c):
        return ((dense.tok[dec["hist"][c]] != dense.tok[hist]).any(-1) & prefix).any(1)
    same = _pick(key_same[o1], o1, key_same[r], gen, ok=differs)
    cfg, k, cell = dec["cfg"][r], dec["k"][r], dec["cell"][r]
    q_cf = tab.Q[cfg, k, cell, sh2, sv2]
    opt_cf = q_cf >= q_cf.max(-1, keepdim=True).values - 1e-6
    cross = cross_donors(tab, dense, hist, t, yh, yv, dec["legal"][r], cell, dec["optimal"][r], opt_cf, sh2, sv2, gen)
    f = lambda q, s: torch.where(s == 0, torch.full_like(q, 0.5), torch.where(s == 2, q, 1 - q))
    marg = lambda a, b_: torch.stack([f(dec["qh"][r], a), f(dec["qv"][r], b_)], -1)
    return dict(dec=r, pos=dec["pos"][r], tok=dense.tok[hist], q=dense.q[hist], tok_twin=tw.tok, legal=dec["legal"][r],
                same=same, flip_h=flip_h, cross_tok=cross["tok"], cross_ok=cross["ok"], cross_strict=cross["strict"],
                cross_opt=cross["opt"], marg=marg(sh, sv), marg_cf=marg(sh2, sv2),
                opt=dec["optimal"][r], opt_cf=opt_cf,
                q_star=dec["q_star"][r], q_star_cf=q_cf, cell=cell)


@torch.no_grad()
def cross_donors(tab, dense, hist, t, yh, yv, legal, cell, opt, opt_cf, sh2, sv2, gen, tries=12):
    """For each recipient, a history with the twin's reports whose actions after the last query before step t
    are a random walk. Accepted if at step t it stands on another cell with the same legal actions; strict
    candidates are preferred."""
    dev, N, H = tab.dev, len(hist), tab.H
    act0, steps = dense.act[hist], torch.arange(H, device=dev)[None]
    isq = (act0 == 4) & (steps < t[:, None])
    tq = torch.where(isq, steps, -1).max(1).values                          # >= 0: every recipient has a clue
    n = torch.arange(N, device=dev)
    best = dict(tok=dense.tok[hist].clone(), ok=torch.zeros(N, dtype=torch.bool, device=dev),
                strict=torch.zeros(N, dtype=torch.bool, device=dev), opt=torch.zeros(N, 6, dtype=torch.bool, device=dev))
    for _ in range(tries):
        env = P.Env(tab, dense.cfg[hist], dense.state[hist, 0, 1])
        at = dict()
        for s in range(H):
            m = env.legal()
            walk = torch.multinomial(m[:, :4].float() + 1e-9, 1, generator=gen).squeeze(1)
            a = torch.where((s > tq) & (s < t), walk, act0[:, s])
            a = torch.where(m.gather(1, a[:, None]).squeeze(1), a, walk)       # past step t the original may be illegal here
            here = s == t
            if here.any():
                qs = env.qstar()
                at.setdefault("cell", torch.zeros_like(cell)); at.setdefault("legal", torch.zeros_like(legal))
                at.setdefault("opt", torch.zeros_like(legal)); at.setdefault("state", torch.zeros(N, 2, dtype=torch.long, device=dev))
                at["cell"][here], at["legal"][here] = env.cell[here], m[here]
                at["opt"][here] = (qs >= qs.max(-1, keepdim=True).values - 1e-6)[here]
                at["state"][here] = torch.stack([env.sh, env.sv], -1)[here]
            env.done = env.done & (s >= t)                                     # keep the history going up to the decision
            at_h = env.cell == tab.hcell[env.cfg]
            env.step(a, y=torch.where(at_h, yh, yv))
        assert torch.equal(at["state"], torch.stack([sh2, sv2], -1))
        ok = (at["cell"] != cell) & (at["legal"] == legal).all(1)
        strict = ok & (opt_cf & ~opt & ~at["opt"]).any(1) & (at["opt"] & ~opt & ~opt_cf).any(1)
        take = (strict & ~best["strict"]) | (ok & ~best["ok"])
        best["tok"][take], best["opt"][take] = env.tok[take], at["opt"][take]
        best["strict"] = best["strict"] | strict; best["ok"] = best["ok"] | ok
    return best


def components(net, broad=True):
    out = []
    for l in range(net.nl):
        out += [("head", l, h) for h in range(net.nh)] + [("attn", l), ("mlp", l)]
        if broad:
            out.append(("resid", l))
    return out + ([("resid", net.nl)] if broad else [])


def name(c):
    return f"L{c[1]}.H{c[2]}" if c[0] == "head" else f"L{c[1]}.{c[0]}" if c[0] != "resid" else f"resid{c[1]}"


@torch.no_grad()
def record_at(net, tok, q, pos):
    """Every component's output at the decision position, the final residual stream there and the logits."""
    n = torch.arange(len(tok), device=tok.device)
    lg, _, rec = net(tok, q, record=True)
    out = {("resid", l): rec["resid"][l][n, pos] for l in range(net.nl + 1)}
    for l in range(net.nl):
        out[("attn", l)], out[("mlp", l)] = rec["attn"][l][n, pos], rec["mlp"][l][n, pos]
        for h in range(net.nh):
            out[("head", l, h)] = rec["heads"][l][n, h, pos]
    return out, lg[n, pos]


@torch.no_grad()
def patched(net, tok, q, pos, comp, value):
    """Logits and final residual at pos when component `comp` at pos is set to `value`."""
    n = torch.arange(len(tok), device=tok.device)
    def fn(x):
        x = x.clone(); x[n, pos] = value
        return x
    lg, _, rec = net(tok, q, record=True, patch={comp: fn})
    return lg[n, pos], rec["resid"][net.nl][n, pos]


def effects(net, probe, pairs, donor_tok, donor_q, donor_pos, keep, random=False, gen=None, comps=None):
    """Patch each component of the recipients `keep` from the donors. Returns {component: metrics}.
      belief   decoded (P(right), P(bottom)) displacement along the exact counterfactual displacement, as a share of it
      off      decoded displacement of the marginal the counterfactual leaves alone (absolute)
      action   movement of the action distribution towards the model's own distribution on the twin:
               1 - TV(patched, twin) / TV(base, twin), as a ratio of means
      cf_mass  on pairs whose exact optimal sets are disjoint: gain in probability of the counterfactual optimal
               set, as a share of the twin's gain
    With random=True the component is moved by a random vector of the same norm as (donor - base)."""
    tok, q, pos = pairs["tok"][keep], pairs["q"][keep], pairs["pos"][keep]
    legal = pairs["legal"][keep]
    base, lg_b = record_at(net, tok, q, pos)
    don, _ = record_at(net, donor_tok, donor_q, donor_pos)
    _, lg_t = record_at(net, pairs["tok_twin"][keep], q, pos)
    sm = lambda lg: lg.masked_fill(~legal, -1e9).softmax(-1)
    pb, pt = sm(lg_b), sm(lg_t)
    m_b = probe(base[("resid", net.nl)])[:, 4:6]
    target = (pairs["marg_cf"][keep] - pairs["marg"][keep]).double()
    on = target.abs() > 1e-9
    disjoint = ~(pairs["opt"][keep] & pairs["opt_cf"][keep]).any(1)
    cf = pairs["opt_cf"][keep].float()
    tv = lambda a, b_: 0.5 * (a - b_).abs().sum(-1)
    out = {}
    for c in (comps or components(net)):
        v = don[c]
        if random:
            r = torch.randn(v.shape, device=v.device, generator=gen)
            v = base[c] + r / r.norm(dim=-1, keepdim=True) * (don[c] - base[c]).norm(dim=-1, keepdim=True)
        lg_p, fin = patched(net, tok, q, pos, c, v)
        pp = sm(lg_p)
        d = probe(fin)[:, 4:6] - m_b
        gain = ((pp - pb) * cf).sum(-1)[disjoint]; full = ((pt - pb) * cf).sum(-1)[disjoint]
        out[name(c)] = dict(belief=((d * target).sum() / (target ** 2).sum()).item(),
                            off=d[~on].abs().mean().item(),
                            action=1 - (tv(pp, pt).mean() / tv(pb, pt).mean()).item(),
                            cf_mass=(gain.mean() / full.mean()).item(), cf_gain=gain.mean().item(), cf_full=full.mean().item(),
                            tv_base=tv(pb, pt).mean().item(), n=len(tok), n_disjoint=int(disjoint.sum()))
    return out


def run_pairs(net, probe, dense, dec, pairs, seed=0, comps=None):
    """All four interventions for one set of pairs."""
    dev = pairs["tok"].device
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    every = torch.arange(len(pairs["dec"]), device=dev)
    out = dict(twin=effects(net, probe, pairs, pairs["tok_twin"], pairs["q"], pairs["pos"], every, comps=comps),
               random=effects(net, probe, pairs, pairs["tok_twin"], pairs["q"], pairs["pos"], every, random=True, gen=gen, comps=comps))
    keep = torch.nonzero(pairs["same"] >= 0).squeeze(1)
    h = dec["hist"][pairs["same"][keep]]
    out["same"] = effects(net, probe, pairs, dense.tok[h], dense.q[h], pairs["pos"][keep], keep, comps=comps)
    keep = torch.nonzero(pairs["cross_ok"]).squeeze(1)
    out["cross"] = effects(net, probe, pairs, pairs["cross_tok"][keep], pairs["q"][keep], pairs["pos"][keep], keep, comps=comps)
    return out


@torch.no_grad()
def source_contributions(net, decoders, tok, q, pos):
    """For each layer and head, what each source token sends to the decision token, read through that block's
    shared decoder: [layers, N, h, L, 2] in (P(right), P(bottom)) units. decoders[l] is a PR.Affine."""
    _, _, rec = net(tok, q, record=True)
    out = []
    for l, b in enumerate(net.blocks):
        terms = b.source_terms(rec["pattern"][l], rec["value"][l], pos)             # [N, h, L, d]
        out.append(decoders[l].delta(terms)[..., 4:6])
    return torch.stack(out), rec
