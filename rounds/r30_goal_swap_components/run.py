"""Round 30: a goal swap with the history held fixed. Which components after the two-route interface carry the switch
from one goal's action to another's?

    .venv/bin/python rounds/r30_goal_swap_components/run.py            # writes results.json

Recipient: history h under goal g. Donor: the same history under goal g', where the two goals' optimal action sets for
h's posterior are disjoint. The prefix states are identical in the two runs (prefix tokens never see the goal), so every
difference arises at the goal token, and every patch below is at the goal token only.

Components: each attention head's output and each MLP's output in blocks 1-3 (after the interface: the residual stream
entering block 1); block 0's heads and MLP for reference.

    patch       the donor's output in the recipient's run (sufficiency): how far toward the donor's decision
    restore     the recipient's output in the donor's run (necessity): how far the donor's decision is undone
    dla         direct logit attribution of the component's change, through the final layer norm at the recipient's scale

Discovery on fit-side histories; the set of components whose patch moves the decision at least 0.3 of the way is then
patched together, and its complement, on held-out histories.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazemodel as MM

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R27 = load("r27run", ROUNDS / "r27_belief_encoding_edit" / "run.py")
M26 = R27.M26
N_HIST, CHUNK, THRESH, LENGTHS = 12000, 8192, 0.3, (1, 2, 3, 4)
GOAL_PAIRS = [(g, h) for g in range(3) for h in range(3) if g != h]


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def components(net, first=1):
    out = []
    for l in range(first, net.nl):
        out += [("head", l, h) for h in range(net.nh)] + [("mlp", l)]
    return out


def name(c):
    return f"L{c[1]}.H{c[2]}" if c[0] == "head" else f"L{c[1]}.mlp"


class Cells:
    """(history, recipient goal, donor goal) with disjoint optimal sets, for one side of the bank."""

    def __init__(self, ctx, side, seed):
        dev = ctx.t.dev
        g = torch.Generator(device=dev); g.manual_seed(seed)
        pool = torch.nonzero(side & (ctx.L >= LENGTHS[0])).squeeze(1)
        h = pool[torch.randperm(len(pool), device=dev, generator=g)[:N_HIST]]
        rows = []
        for gr, gd in GOAL_PAIRS:
            dj = ~(ctx.opt[h, gr] & ctx.opt[h, gd]).any(-1)
            rows.append(torch.stack([h[dj], torch.full_like(h[dj], gr), torch.full_like(h[dj], gd)], 1))
        r = torch.cat(rows)
        self.hist, self.gr, self.gd = r[:, 0], r[:, 1], r[:, 2]
        self.n = len(r)


@torch.no_grad()
def seq(ctx, hist, goal):
    t = ctx.t
    L = ctx.L[hist]
    tok = torch.zeros(len(hist), t.L, MM.NF, dtype=torch.long, device=t.dev)
    keep = torch.arange(t.L, device=t.dev)[None] <= L[:, None]
    tok[keep] = ctx.bank.tok[hist][keep]
    rows = torch.arange(len(hist), device=t.dev)
    tok[rows, 1 + L, MM.F_TYPE], tok[rows, 1 + L, MM.F_GOAL] = MM.GOAL, goal + 1
    return tok


def value(rec, c, n, pos):
    if c[0] == "head":
        return rec["heads"][c[1]][n, c[2], pos]
    return rec["mlp"][c[1]][n, pos]


def make_patch(comps, rec_src, n, pos):
    patch = {}
    for c in comps:
        val = value(rec_src, c, n, pos)
        def fn(x, val=val):
            x = x.clone(); x[n, pos] = val
            return x
        patch[c] = fn
    return patch


@torch.no_grad()
def measure(net, ctx, cells, groups):
    """Per component and per group: patch (toward the donor), restore (back toward the recipient), donor-optimal rate,
    and DLA share; per goal pair for the groups. Means over cells."""
    t = ctx.t
    comps = components(net, 0)
    acc = {}
    def add(key, sub, x):
        acc.setdefault(key, {}).setdefault(sub, []).append(x)
    for s in range(0, cells.n, CHUNK):
        h, gr, gd = cells.hist[s:s + CHUNK], cells.gr[s:s + CHUNK], cells.gd[s:s + CHUNK]
        n = torch.arange(len(h), device=t.dev)
        pos = 1 + ctx.L[h]
        tR, tD = seq(ctx, h, gr), seq(ctx, h, gd)
        lR, _, rR = net(tR, record=True); lD, _, rD = net(tD, record=True)
        lR, lD = lR[n, pos], lD[n, pos]
        pR, pD = lR.softmax(-1), lD.softmax(-1)
        den = tv(pR, pD)
        optD = ctx.opt[h, gd]
        cR = torch.log_softmax(lR, -1); cR = cR - cR.mean(-1, keepdim=True)
        cD = torch.log_softmax(lD, -1); cD = cD - cD.mean(-1, keepdim=True)
        dl = cD - cR
        # direct logit attribution: change of each component's output, through the final layer norm at the recipient's scale
        xR = rR["resid"][net.nl][n, pos]
        sd = xR.var(-1, unbiased=False, keepdim=True).add(net.ln.eps).sqrt()
        def dla(u):
            z = (u - u.mean(-1, keepdim=True)) / sd * net.ln.weight
            lg = z @ net.pi.weight.T
            lg = lg - lg.mean(-1, keepdim=True)
            return (lg * dl).sum(-1) / (dl ** 2).sum(-1).clamp(min=1e-9)
        emb = net.embed.f[MM.F_GOAL].weight
        add("goal_embedding", "dla", dla(emb[gd + 1] - emb[gr + 1]))
        for c in comps:
            add(name(c), "dla", dla(value(rD, c, n, pos) - value(rR, c, n, pos)))
        items = [(name(c), [c]) for c in comps] + list(groups.items())
        for key, cs in items:
            if not cs:
                continue
            pp = net(tR, patch=make_patch(cs, rD, n, pos))[0][n, pos].softmax(-1)
            pq = net(tD, patch=make_patch(cs, rR, n, pos))[0][n, pos].softmax(-1)
            add(key, "patch", 1 - tv(pp, pD) / den.clamp(min=1e-6))
            add(key, "restore", 1 - tv(pq, pR) / den.clamp(min=1e-6))
            add(key, "donor_optimal", optD[n, pp.argmax(-1)].float())
            add(key, "gpair", gr * 3 + gd)
        add("_base", "donor_optimal_unpatched", optD[n, pR.argmax(-1)].float())
        add("_base", "donor_optimal_donor", optD[n, pD.argmax(-1)].float())
        add("_base", "tv", den)
        add("_base", "gpair", gr * 3 + gd)
    out = {}
    for key, d in acc.items():
        cat = {k: torch.cat(v) for k, v in d.items()}
        out[key] = {k: float(v.float().mean()) for k, v in cat.items() if k != "gpair"}
        if "gpair" in cat and key in groups:
            out[key]["by_goal_pair"] = {f"G{a + 1}->G{b + 1}": float(cat["patch"][cat["gpair"] == a * 3 + b].mean()) for a, b in GOAL_PAIRS if (cat["gpair"] == a * 3 + b).any()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    disc, test = Cells(ctx, ctx.fit, 30), Cells(ctx, ctx.test, 31)
    res = dict(cells=dict(discovery=disc.n, test=test.n), runs={})
    for s in a.seeds:
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        net.eval()
        after = components(net, 1)
        all_after = {"all_blocks_1_3": after}
        d = measure(net, ctx, disc, all_after)
        S = [c for c in after if d[name(c)]["patch"] >= THRESH]
        groups = dict(all_blocks_1_3=after, selected=S, complement=[c for c in after if c not in S],
                      attention_1_3=[c for c in after if c[0] == "head"], mlp_1_3=[c for c in after if c[0] == "mlp"])
        e = measure(net, ctx, test, groups)
        dg = measure(net, ctx, disc, {k: v for k, v in groups.items() if k != "all_blocks_1_3"}) if S else {}
        d.update({k: v for k, v in dg.items() if k in groups})
        res["runs"][f"seed{s}"] = dict(checkpoint=ck, selected=[name(c) for c in S], discovery=d, test=e)
        print(f"seed{s} selected {[name(c) for c in S]} | discovery patch " + " ".join(f"{name(c)} {d[name(c)]['patch']:.2f}" for c in after) +
              f" | test: selected {e['selected']['patch'] if S else float('nan'):.2f} complement {e['complement']['patch']:.2f} all {e['all_blocks_1_3']['patch']:.2f} | {time.time() - t0:.0f}s", flush=True)
        (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
