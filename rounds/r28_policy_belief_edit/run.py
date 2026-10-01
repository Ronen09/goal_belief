"""Round 28: one belief-encoding edit at the pre-goal interface of the reward-trained models, read by two controllers:
the original policy (at the goal token) and round 26's frozen head (at the last prefix token).

    .venv/bin/python rounds/r28_policy_belief_edit/run.py            # writes results.json

Interface: the residual stream entering block 1 at every prefix token (rounds 20-22). Prefix tokens never see the goal,
so one edit serves all three goals. Encoding h ~ c_position + E b, fitted on prefix tokens of the fit side (b: the
exact goal-free posterior at that token). Per prefix position i of a held-out pair (recipient A, donor B):

    none        h_A,i
    whole       h_B,i                                   every prefix state replaced
    encoding    h_A,i + E (b_B,i - b_A,i)               the donor's posteriors only
    rotated     h_A,i + Q E (b_B,i - b_A,i)             Q a random rotation (5 draws)
    random      h_A,i + P_R (h_B,i - h_A,i)             a random rank-13 projection (5 draws)
    pca         h_A,i + P_PCA (h_B,i - h_A,i)           the top 13 principal components (secondary)

Readers: the policy's action distribution at the goal token, with natural access (primary) or with its first attention
layer's output at the goal token replaced by its mean (direct route removed, round 22; secondary); and round 26's
head (100 000 examples, three seeds) on the propagated state after the last block at the last prefix token.
References: the policy on the donor's own prefix (hybrid; with the same ablation in the secondary condition), and the
head on the donor's own state (equal to `whole` by construction).
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazeedit as ED, mazemeasure as MS, mazepred as PR
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R27 = load("r27run", ROUNDS / "r27_belief_encoding_edit" / "run.py")
M26, R22 = R27.M26, R27.M26.R22
L_IF, RANK, DRAWS, HEAD_N = 1, 13, 5, 100000
KINDS = ("none", "whole", "encoding", "rotated", "random", "pca")
CHUNK = 8192


class Interface:
    """Encoding and subspaces at the interface of one model, fitted on the fit side's prefix tokens."""

    def __init__(self, net, ctx, gen):
        t, b = ctx.t, ctx.bank
        rows = MS.prefix_rows(t, b, ctx.fit)
        X = torch.zeros(len(rows["hist"]), net.d, device=t.dev)
        for s in range(0, len(b.tok), CHUNK):
            sel = (rows["hist"] >= s) & (rows["hist"] < s + CHUNK)
            st = ED.states(net, b.tok[s:s + CHUNK], L_IF)
            X[sel] = st[rows["hist"][sel] - s, rows["pos"][sel]]
        fit, test = rows["fit"], ~rows["fit"]
        pos1 = torch.nn.functional.one_hot(rows["pos"], t.max_prefix + 1).double()
        B = rows["belief"].double()
        Xd = X.double()
        enc = Affine(torch.cat([B, pos1], 1)[fit], Xd[fit])
        self.E = enc.W[: B.shape[1]]                                                         # [14, d]
        pred = enc(torch.cat([B, pos1], 1)[test])
        res = ((Xd[test] - pred) ** 2).sum()
        pm = torch.stack([Xd[fit][rows["pos"][fit] == p].mean(0) for p in range(t.max_prefix + 1)])
        self.r2 = float(1 - res / ((Xd[test] - Xd[test].mean(0)) ** 2).sum())
        self.r2_within_position = float(1 - res / ((Xd[test] - pm[rows["pos"][test]]) ** 2).sum())
        Xc = Xd[fit] - pm[rows["pos"][fit]]
        self.P_PCA = torch.linalg.eigh(Xc.T @ Xc / len(Xc))[1].flip(1)[:, :RANK]
        d = net.d
        self.P_R = [R27.orth(torch.randn(d, RANK, dtype=torch.float64, device=t.dev, generator=gen)) for _ in range(DRAWS)]
        self.Q = [R27.orth(torch.randn(d, d, dtype=torch.float64, device=t.dev, generator=gen)) for _ in range(DRAWS)]

    def edits(self, xA, xB, bA, bB):
        """New interface states for every kind: {kind: [M, d] or a list of draws}. xA, xB [M, d]; bA, bB [M, 14]."""
        xA, xB = xA.double(), xB.double()
        e = (bB - bA).double() @ self.E
        diff = xB - xA
        proj = lambda P: (xA + (diff @ P) @ P.T).float()
        return dict(none=xA.float(), whole=xB.float(), encoding=(xA + e).float(), rotated=[(xA + e @ Q).float() for Q in self.Q],
                    random=[proj(P) for P in self.P_R], pca=proj(self.P_PCA))

    def stats(self, xA, xB, bA, bB):
        e = (bB - bA).double() @ self.E
        diff = (xB - xA).double()
        tot = (diff ** 2).sum(1).mean()
        return dict(edit_norm_over_diff_norm=float((e.norm(dim=1) / diff.norm(dim=1).clamp(min=1e-9)).median()),
                    cosine=float(torch.nn.functional.cosine_similarity(e, diff, dim=1).mean()),
                    residual_share=float(((diff - e) ** 2).sum(1).mean() / tot))


@torch.no_grad()
def readers(net, ctx, itf, hd, rows, a, b, means):
    """Action distributions of both readers for every kind. Policy: [1, n, goals, 4] per access condition; head:
    [heads, n, goals, 4]."""
    data, t = ctx.data, ctx.t
    L = ctx.L[a]
    gpos = 1 + L
    nn_ = torch.arange(len(a), device=t.dev)
    pol = {c: {k: [] for k in KINDS + ("hybrid",)} for c in ("natural", "removed")}
    head = {k: [] for k in KINDS}
    stats = None
    for s in range(0, len(a), CHUNK):
        sl = slice(s, s + CHUNK)
        aa, bb, LL, gp = a[sl], b[sl], L[sl], gpos[sl]
        n = torch.arange(len(aa), device=t.dev)
        out_p = {c: {k: [None] * 3 for k in KINDS + ("hybrid",)} for c in pol}
        out_h = {}
        for g in range(3):
            tA, tH = data.seq(aa, LL, g), data.seq(bb, LL, g)
            mask = ED.prefix_mask(tA, gp)
            hi, pi = torch.nonzero(mask, as_tuple=True)
            xA, xB = ED.states(net, tA, L_IF)[mask], ED.states(net, tH, L_IF)[mask]
            bA, bB = t.belief[ctx.bank.pre_node[aa[hi], pi]], t.belief[ctx.bank.pre_node[bb[hi], pi]]
            new = itf.edits(xA, xB, bA, bB)
            if g == 0 and s == 0:
                stats = itf.stats(xA, xB, bA, bB)
            for c in pol:
                abl = None if c == "natural" else means[g, LL]
                out_p[c]["hybrid"][g] = ED.run(net, tH, gp, ablate=abl, goal_pos=gp)[0]
                for k, v in new.items():
                    runs = [ED.run(net, tA, gp, l=L_IF, new=x, mask=mask, ablate=abl, goal_pos=gp) for x in (v if isinstance(v, list) else [v])]
                    out_p[c][k][g] = torch.stack([r[0] for r in runs]).mean(0)
                    if c == "natural" and g == 0:                                              # the head reads the last prefix token: the same for every goal
                        out_h[k] = torch.stack([R27.head_probs(hd, r[2]["resid"][net.nl][n, LL], rows) for r in runs]).mean(0)
        for c in pol:
            for k in out_p[c]:
                pol[c][k].append(torch.stack(out_p[c][k], 1))                                  # [n, goals, 4]
        for k in KINDS:
            head[k].append(out_h[k])
    pol = {c: {k: torch.cat(v)[None] for k, v in d.items()} for c, d in pol.items()}
    head = {k: torch.cat(v, 1) for k, v in head.items()}
    return pol, head, stats


def summarise(P, ref, ctx, pairs, a, b):
    dj, sm = pairs.disjoint(a, b), pairs.same(a, b)
    optA, optB = ctx.opt[a], ctx.opt[b]
    sc = lambda p, sel: R27.score(p, P["none"], ref, optB, optA, ctx.V[b], ctx.Q[b], sel)
    out = {}
    for k in P:
        out[k] = dict(change=sc(P[k], dj), preserve=sc(P[k], sm))
    base, top = out["none"]["change"]["donor_optimal"], (out["hybrid"] if "hybrid" in out else out["whole"])["change"]["donor_optimal"]
    for k in out:
        out[k]["transfer"] = (out[k]["change"]["donor_optimal"] - base) / (top - base) if top - base > 0.02 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    hk = torch.load(ROUNDS / "r27_belief_encoding_edit" / "heads" / f"site4_n{HEAD_N}.pt")
    hd = PR.Heads.__new__(PR.Heads); hd.hidden = hk["hidden"]; hd.params = [p.to(dev) for p in hk["params"]]
    res = dict(counts=pairs.counts(), runs={})
    for s in a.seeds:
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        gen = torch.Generator(device=dev); gen.manual_seed(2800 + s)
        itf = Interface(net, ctx, gen)
        means = R22.route_means(net, ctx.data)
        rows = torch.nonzero(hk["xmap"] == hk["keys"].index(("reward", s))).squeeze(1).to(dev)
        row = dict(checkpoint=ck, encoding_r2=itf.r2, encoding_r2_within_position=itf.r2_within_position)
        for name, pr in (("main", pairs.main), ("onestep", pairs.onestep)):
            ra, rb = pr["r"], pr["d"]
            pol, head, stats = readers(net, ctx, itf, hd, rows, ra, rb, means)
            row[name] = dict(stats=stats, head=summarise(head, head["whole"], ctx, pairs, ra, rb),
                             **{f"policy_{c}": summarise(pol[c], pol[c]["hybrid"], ctx, pairs, ra, rb) for c in pol})
            # the policy's interface effect: the encoding's share of the whole interface patch, per reader
            for reader in ("head", "policy_natural", "policy_removed"):
                r = row[name][reader]
                base, whole = r["none"]["change"]["donor_optimal"], r["whole"]["change"]["donor_optimal"]
                for k in KINDS:
                    r[k]["share_of_interface"] = (r[k]["change"]["donor_optimal"] - base) / (whole - base) if whole - base > 0.02 else None
        res["runs"][f"seed{s}"] = row
        m = row["main"]
        print(f"seed{s} enc R² {itf.r2:.3f} | head T: " + " ".join(f"{k} {m['head'][k]['transfer']:.2f}" for k in KINDS) +
              " | policy natural T: " + " ".join(f"{k} {m['policy_natural'][k]['transfer']:.2f}" for k in KINDS + ('hybrid',)) +
              " | removed T: " + " ".join(f"{k} {m['policy_removed'][k]['transfer']:.2f}" for k in ('whole', 'encoding', 'rotated')) + f" | {time.time() - t0:.0f}s", flush=True)
        (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
