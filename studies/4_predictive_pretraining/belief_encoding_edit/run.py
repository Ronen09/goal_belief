"""The belief-encoding-edit experiment: belief-encoding edits of the frozen representation, read by the predictive-transfer experiment's frozen goal-conditioned heads.

    .venv/bin/python studies/4_predictive_pretraining/belief_encoding_edit/run.py --counts     # pair counts only (solver tables)
    .venv/bin/python studies/4_predictive_pretraining/belief_encoding_edit/run.py              # writes results.json

Per backbone, at the head's site (the residual stream at the last prefix token): an encoding model h ~ c + E b is fitted
on the fit side of the pair-types experiment's bank. On held-out pairs (recipient A, donor B) the recipient's state is edited and passed
through layer norm and the frozen head, under each goal:

    none            h_A
    whole           h_B
    encoding        h_A + E (b_B - b_A)                 uses the donor's posterior only (the candidate)
    rotated         h_A + Q E (b_B - b_A)               Q a random rotation: the same edit vectors in a generic subspace
    onestep         h_A + F (p_B - p_A)                 an encoding of the exact one-step prediction p instead of b
    enc_patch       h_A + P_E (h_B - h_A)               the donor's state within the encoding subspace (rank 13)
    decoder_patch   h_A + P_D (h_B - h_A)               the same in the posterior decoder's subspace (the belief-edit experiment's)
    pca_patch       h_A + P_PCA (h_B - h_A)             the top 13 principal components
    random_patch    h_A + P_R (h_B - h_A)               a random 13-dimensional subspace
    wrong           h_A + E (b_C - b_A)                 a third posterior C: the edit should give C's action
rotated and random_patch are averaged over five draws.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazemodel as MM, mazepred as PR
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
R26 = HERE.parent.parent / "4_predictive_pretraining" / "predictive_transfer"
spec = importlib.util.spec_from_file_location("r26measure", R26 / "measure.py")
M26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M26)

PAIR_SEED, RANK, DRAWS, MIN_L1 = 27, 13, 5, 0.25
N_MAIN, N_ONESTEP = 30000, 20000
SITES = dict(predict1=4, predict2=4, reward=4, random=1)              # the predictive-transfer experiment's selected sites
NS = (100000, 1000)                                                   # primary: the heads trained on 100 000 examples
KINDS = ("none", "whole", "encoding", "rotated", "onestep", "enc_patch", "decoder_patch", "pca_patch", "random_patch", "wrong")


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


# ------------------------------------------------------------------ pairs (solver tables only)

class Pairs:
    def __init__(self, ctx):
        t, dev = ctx.t, ctx.t.dev
        self.ctx = ctx
        g = torch.Generator(device=dev); g.manual_seed(PAIR_SEED)
        L, node = ctx.L, ctx.node
        pool = ctx.test_hist[ctx.L[ctx.test_hist] >= 2]
        bel = t.belief[node]
        p1 = PR.predictive(t, bel.double(), 1).flatten(1)
        _, cls1 = torch.unique(torch.round(p1 * 1e9), dim=0, return_inverse=True)
        self.cls1 = cls1                                                               # one-step class of each history
        # main: random pairs of the same length, posteriors at least MIN_L1 apart
        r = pool[torch.randint(len(pool), (8 * N_MAIN,), device=dev, generator=g)]
        d = pool[torch.randint(len(pool), (8 * N_MAIN,), device=dev, generator=g)]
        ok = (L[r] == L[d]) & ((bel[r] - bel[d]).abs().sum(1) >= MIN_L1)
        r, d = r[ok][:N_MAIN], d[ok][:N_MAIN]
        c = pool[torch.randint(len(pool), (len(r) * 8,), device=dev, generator=g)].view(len(r), 8)          # a third posterior for 'wrong'
        okc = (L[c] == L[r][:, None]) & ((bel[c] - bel[r][:, None]).abs().sum(-1) >= MIN_L1) & ((bel[c] - bel[d][:, None]).abs().sum(-1) >= MIN_L1)
        first = torch.where(okc.any(1), okc.float().argmax(1), torch.zeros_like(r))
        cc = c[torch.arange(len(r), device=dev), first]
        self.main = dict(r=r, d=d, c=torch.where(okc.any(1), cc, -1))
        # one-step agreeing: same one-step class, different posterior, disjoint optimal sets under at least one goal
        r = pool[torch.randint(len(pool), (400 * N_ONESTEP,), device=dev, generator=g)]
        order = torch.argsort(cls1[r] * 8 + L[r], stable=True)                         # pair consecutive histories of the same class and length
        r = r[order]
        a, b = r[:-1], r[1:]
        ok = (cls1[a] == cls1[b]) & (L[a] == L[b]) & (node[a] != node[b]) & (self.disjoint(a, b).any(1))
        idx = torch.nonzero(ok).squeeze(1)
        idx = idx[torch.randperm(len(idx), device=dev, generator=g)][:N_ONESTEP]
        self.onestep = dict(r=a[idx], d=b[idx])
        # shared belief: the pair-types experiment's type A pairs (same posterior, different tokens) as two recipients, one donor
        A = ctx.data.types["A"]
        r1, r2 = A["r"], A["d"]
        dd = pool[torch.randint(len(pool), (len(r1) * 8,), device=dev, generator=g)].view(len(r1), 8)
        okd = (L[dd] == L[r1][:, None]) & ((bel[dd] - bel[r1][:, None]).abs().sum(-1) >= MIN_L1)
        keep = okd.any(1)
        dsel = dd[torch.arange(len(r1), device=dev), okd.float().argmax(1)]
        self.shared = dict(r=r1[keep], r2=r2[keep], d=dsel[keep])

    def opt(self, h):
        return self.ctx.opt[h]                                                           # [n, goals, 4]

    def disjoint(self, a, b):
        return ~(self.opt(a) & self.opt(b)).any(-1)

    def same(self, a, b):
        return (self.opt(a) == self.opt(b)).all(-1)

    def counts(self):
        m, o, s = self.main, self.onestep, self.shared
        dj, sm = self.disjoint(m["r"], m["d"]), self.same(m["r"], m["d"])
        both = dj.any(1) & sm.any(1)
        hasc = m["c"] >= 0
        wc = hasc[:, None] & self.disjoint(m["r"], m["c"].clamp(min=0)) & self.disjoint(m["d"], m["c"].clamp(min=0))
        return dict(main=dict(pairs=len(m["r"]), change_cells=int(dj.sum()), preserve_cells=int(sm.sum()), overlap_cells=int((~dj & ~sm).sum()),
                              by_goal_change=dj.sum(0).tolist(), by_goal_preserve=sm.sum(0).tolist(), selectivity_pairs=int(both.sum()), wrong_cells=int(wc.sum())),
                    onestep=dict(pairs=len(o["r"]), change_cells=int(self.disjoint(o["r"], o["d"]).sum()), preserve_cells=int(self.same(o["r"], o["d"]).sum()),
                                 by_goal_change=self.disjoint(o["r"], o["d"]).sum(0).tolist(), classes=int(len(torch.unique(self.cls1[o["r"]])))),
                    shared=dict(triples=len(s["r"]), change_cells=int(self.disjoint(s["r"], s["d"]).sum()), preserve_cells=int(self.same(s["r"], s["d"]).sum())))


# ------------------------------------------------------------------ edits

def orth(V):
    return torch.linalg.qr(V)[0]


class Editor:
    """Encoding model and subspaces for one backbone at one site, fitted on the fit side."""

    def __init__(self, H, ctx, gen):
        t = ctx.t
        fit = ctx.fit
        bel = t.belief[ctx.node].double()
        self.bel = bel
        self.p1 = PR.predictive(t, bel, 1).flatten(1)
        Hd = H.double()
        enc = Affine(bel[fit], Hd[fit])                                                   # h ~ c + E b
        self.E = enc.W                                                                    # [14, d]
        test = ctx.test
        pred = enc(bel[test])
        self.encoding_r2 = float(1 - ((Hd[test] - pred) ** 2).sum() / ((Hd[test] - Hd[test].mean(0)) ** 2).sum())
        one = Affine(self.p1[fit], Hd[fit])                                                # h ~ c + F p
        self.F = one.W
        self.onestep_r2 = float(1 - ((Hd[test] - one(self.p1[test])) ** 2).sum() / ((Hd[test] - Hd[test].mean(0)) ** 2).sum())
        n = bel.shape[1]
        Z = torch.eye(n, dtype=torch.float64, device=H.device) - 1.0 / n                  # sum-zero belief directions
        U, S, _ = torch.linalg.svd((Z @ self.E).T, full_matrices=False)
        self.P_E = U[:, :RANK]                                                            # [d, 13]: the encoding subspace
        dec = Affine(Hd[fit], bel[fit])
        U, S, _ = torch.linalg.svd(dec.W @ Z, full_matrices=False)
        self.P_D = U[:, :RANK]                                                            # the decoder's subspace
        Xc = Hd[fit] - Hd[fit].mean(0)
        self.P_PCA = torch.linalg.eigh(Xc.T @ Xc / len(Xc))[1].flip(1)[:, :RANK]
        d = H.shape[1]
        self.P_R = [orth(torch.randn(d, RANK, dtype=torch.float64, device=H.device, generator=gen)) for _ in range(DRAWS)]
        self.Q = [orth(torch.randn(d, d, dtype=torch.float64, device=H.device, generator=gen)) for _ in range(DRAWS)]
        self.share = None

    def states(self, H, a, b, c=None):
        """Edited states for recipients a and donors b (history ids): {kind: [n, d] or list of draws}."""
        hA, hB = H[a].double(), H[b].double()
        dB = self.bel[b] - self.bel[a]
        e = dB @ self.E
        diff = hB - hA
        proj = lambda P: hA + (diff @ P) @ P.T
        out = dict(none=hA, whole=hB, encoding=hA + e, rotated=[hA + e @ Q for Q in self.Q], onestep=hA + (self.p1[b] - self.p1[a]) @ self.F,
                   enc_patch=proj(self.P_E), decoder_patch=proj(self.P_D), pca_patch=proj(self.P_PCA), random_patch=[proj(P) for P in self.P_R])
        if c is not None:
            out["wrong"] = hA + (self.bel[c] - self.bel[a]) @ self.E
        return out

    def stats(self, H, a, b):
        """How much of the actual state difference each edit accounts for (squared norm, share)."""
        hA, hB = H[a].double(), H[b].double()
        diff = hB - hA
        e = (self.bel[b] - self.bel[a]) @ self.E
        tot = (diff ** 2).sum(1).mean()
        sh = lambda P: float(((diff @ P) ** 2).sum(1).mean() / tot)
        return dict(edit_norm_over_diff_norm=float((e.norm(dim=1) / diff.norm(dim=1)).median()),
                    cosine_edit_diff=float(torch.nn.functional.cosine_similarity(e, diff, dim=1).mean()),
                    residual_share=float(((diff - e) ** 2).sum(1).mean() / tot),
                    share_enc=sh(self.P_E), share_dec=sh(self.P_D), share_pca=sh(self.P_PCA), share_random=float(np.mean([sh(P) for P in self.P_R])))


@torch.no_grad()
def head_probs(hd, X, rows):
    """Action distributions of the heads `rows` (the three head seeds of one backbone) on states X [n, d]:
    [heads, n, goals, 4]."""
    Z = PR.layer_norm(X.float())
    K = len(rows)
    eye = torch.eye(3, device=X.device)
    n = len(Z)
    xg = Z[None, :, None].expand(K, -1, 3, -1).reshape(K, n * 3, -1)
    gg = eye.repeat(n, 1)[None].expand(K, -1, -1)
    sub = PR.Heads.__new__(PR.Heads); sub.hidden = hd.hidden
    sub.params = [p[rows] for p in hd.params]
    return sub(xg, gg).view(K, n, 3, 4).softmax(-1)


def score(p, base, whole, optB, optA, VB, QB, sel):
    """Means over heads of the cells `sel` [n, goals]."""
    w = sel.float()
    cnt = w.sum().clamp(min=1)
    g = p.argmax(-1)                                                                   # [heads, n, goals]
    take = lambda m: float(((m.float() * w).sum((1, 2)) / cnt).mean())
    pick = lambda T: torch.gather(T[None].expand(len(g), -1, -1, -1), 3, g[..., None]).squeeze(-1)
    inB, inA = pick(optB), pick(optA)
    reg = VB[None] - pick(QB)
    den = tv(base, whole)
    mv = 1 - tv(p, whole) / den.clamp(min=1e-6)
    wm = w[None] * (den > 0.1).float()
    return dict(n=int(sel.sum()), donor_optimal=take(inB), recipient_optimal=take(inA), unchanged=take(g == base.argmax(-1)), as_whole=take(g == whole.argmax(-1)),
                regret_donor=take(reg), moved=float(((mv * wm).sum((1, 2)) / wm.sum((1, 2)).clamp(min=1)).mean()))


@torch.no_grad()
def analyse(H, ed, hd, rows, pairs, ctx):
    t = ctx.t
    out = {}
    probs = lambda states: {k: (torch.stack([head_probs(hd, s, rows) for s in v]).mean(0) if isinstance(v, list) else head_probs(hd, v, rows)) for k, v in states.items()}
    # main pairs
    m = pairs.main
    has_c = m["c"] >= 0
    st = ed.states(H, m["r"], m["d"], m["c"].clamp(min=0))
    P = probs(st)
    a, b, c = m["r"], m["d"], m["c"].clamp(min=0)
    optA, optB, optC = ctx.opt[a], ctx.opt[b], ctx.opt[c]
    VB, QB = ctx.V[b], ctx.Q[b]
    dj, sm = pairs.disjoint(a, b), pairs.same(a, b)
    res = {}
    for k in KINDS:
        if k == "wrong":
            sel = has_c[:, None] & pairs.disjoint(a, c) & pairs.disjoint(b, c)
            res[k] = dict(to_C=score(P[k], P["none"], P["none"], optC, optA, ctx.V[c], ctx.Q[c], sel), to_B=score(P[k], P["none"], P["whole"], optB, optA, VB, QB, sel))
            continue
        res[k] = dict(change=score(P[k], P["none"], P["whole"], optB, optA, VB, QB, dj), preserve=score(P[k], P["none"], P["whole"], optB, optA, VB, QB, sm),
                      all=score(P[k], P["none"], P["whole"], optB, optA, VB, QB, torch.ones_like(dj)))
        # selectivity: pairs with a goal where the action must change and a goal where it must not; the same edit
        both = dj.any(1) & sm.any(1)
        g = P[k].argmax(-1)
        inB = torch.gather(optB[None].expand(len(g), -1, -1, -1), 3, g[..., None]).squeeze(-1)                 # [heads, n, goals]
        ok_change = ((inB | ~dj[None]).all(-1))                                                                  # right on every change goal
        ok_keep = ((inB | ~sm[None]).all(-1))
        res[k]["selective"] = float((ok_change & ok_keep)[:, both].float().mean())
    out["main"] = res
    out["edit_stats"] = ed.stats(H, a, b)
    # one-step agreeing pairs
    o = pairs.onestep
    a, b = o["r"], o["d"]
    P = probs(ed.states(H, a, b))
    dj, sm = pairs.disjoint(a, b), pairs.same(a, b)
    out["onestep"] = {k: dict(change=score(P[k], P["none"], P["whole"], ctx.opt[b], ctx.opt[a], ctx.V[b], ctx.Q[b], dj)) for k in P}
    out["onestep_stats"] = ed.stats(H, a, b)
    # shared belief: the same edit on two histories with the same posterior
    s = pairs.shared
    P1, P2 = probs(ed.states(H, s["r"], s["d"])), probs(ed.states(H, s["r2"], s["d"]))
    dj = pairs.disjoint(s["r"], s["d"])
    w = dj.float()
    cnt = w.sum().clamp(min=1)
    sh = {}
    for k in P1:
        g1, g2 = P1[k].argmax(-1), P2[k].argmax(-1)
        optB = ctx.opt[s["d"]]
        in1 = torch.gather(optB[None].expand(len(g1), -1, -1, -1), 3, g1[..., None]).squeeze(-1)
        in2 = torch.gather(optB[None].expand(len(g2), -1, -1, -1), 3, g2[..., None]).squeeze(-1)
        m_ = lambda x: float(((x.float() * w).sum((1, 2)) / cnt).mean())
        sh[k] = dict(both_donor_optimal=m_(in1 & in2), agree=m_(g1 == g2), tv=m_(tv(P1[k], P2[k])))
    out["shared"] = sh
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = Pairs(ctx)
    res = dict(counts=pairs.counts(), runs={})
    print(json.dumps(res["counts"], indent=1), flush=True)
    if a.counts:
        (HERE / "counts.json").write_text(json.dumps(res["counts"], indent=1))
        return
    heads = {(l, n): torch.load(HERE / "heads" / f"site{l}_n{n}.pt") for l in set(SITES.values()) for n in NS}
    for cond, l in SITES.items():
        res["runs"][cond] = {}
        for s in range(10):
            net, _ = M26.load(cond, s, t, dev, R26 / "runs")
            H = PR.features(net, ctx.bank.tok, ctx.bank.prefix)[l]
            gen = torch.Generator(device=dev); gen.manual_seed(2700 + s)
            ed = Editor(H, ctx, gen)
            row = dict(encoding_r2=ed.encoding_r2, onestep_encoding_r2=ed.onestep_r2)
            for n in NS:
                hk = heads[(l, n)]
                hd = PR.Heads.__new__(PR.Heads); hd.hidden = hk["hidden"]; hd.params = [p.to(dev) for p in hk["params"]]
                b = hk["keys"].index((cond, s))
                rows = torch.nonzero(hk["xmap"] == b).squeeze(1).to(dev)                    # the three head seeds of this backbone
                row[str(n)] = analyse(H, ed, hd, rows, pairs, ctx)
            res["runs"][cond][f"seed{s}"] = row
            r = row[str(NS[0])]["main"]
            print(f"{cond} seed{s} enc R² {ed.encoding_r2:.3f} | change donor-optimal: " + " ".join(f"{k} {r[k]['change']['donor_optimal']:.2f}" for k in KINDS if k != "wrong") +
                  f" | preserve: none {r['none']['preserve']['donor_optimal']:.2f} enc {r['encoding']['preserve']['donor_optimal']:.2f} | {time.time() - t0:.0f}s", flush=True)
            (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
