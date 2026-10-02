"""Round 32: a shared belief encoding at the goal token's state entering block 1 (the direct route), fitted without
action labels, and edits of that state with the donor's posterior. Round 23's reward models, frozen.

    .venv/bin/python rounds/r32_direct_belief_edit/run.py            # writes results.json
    .venv/bin/python rounds/r32_direct_belief_edit/run.py --untrained --seeds 0     # smoke test (initial checkpoint)

Site: z(h, g), the residual stream entering block 1 at the goal token of history h under goal g. It is computed by
block 0 from the raw tokens and the goal (round 29's direct route). Encoding, fitted on the fit side's histories under
all three goals (b: the exact goal-free posterior at the last prefix token):

    shared (primary)    z ~ c_{g,L} + E b           one E for every goal; offsets per goal and prefix length
    brief_offset        z ~ c_g + E b               the brief's literal form (offset per goal only)
    goal_specific       z ~ c_{g,L} + E_g b         a different map per goal (secondary comparison)

Per held-out pair (recipient A, donor B, same prefix length) and goal g, only the goal token's state entering block 1
is replaced; the recipient's prefix states are kept and the network runs on:

    none           z(A,g)
    whole          z(B,g)                              the whole direct-route state, same goal
    encoding       z(A,g) + E (b_B - b_A)              the donor's posterior only; the same vector under every goal
    enc_brief      z(A,g) + E' (b_B - b_A)             E' from the brief's offset form
    enc_goal       z(A,g) + E_g (b_B - b_A)            goal-specific map
    rotated        z(A,g) + Q E (b_B - b_A)            Q a random rotation: same norms, generic directions (5 draws)
    pca            z(A,g) + P P' (z(B,g) - z(A,g))     the donor's state in the top-13 principal components
    random         the same in a random 13-dimensional subspace (5 draws)

References: the hybrid (the model run on B under g, natural) and the solver's optimal set for B's posterior under g.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R27 = load("r27run", ROUNDS / "r27_belief_encoding_edit" / "run.py")
R30 = load("r30run", ROUNDS / "r30_goal_swap_components" / "run.py")
M26 = R27.M26
L_IF, RANK, DRAWS, CHUNK, N_EQ, EQ_SEED = 1, 13, 5, 8192, 4, 32
KINDS = ("none", "whole", "encoding", "enc_brief", "enc_goal", "rotated", "pca", "random")


@torch.no_grad()
def goal_state(net, ctx, hist, g):
    """z(h, g): the goal token's state entering block 1, [n, d]."""
    out = []
    for s in range(0, len(hist), CHUNK):
        h = hist[s:s + CHUNK]
        n = torch.arange(len(h), device=h.device)
        out.append(net(R30.seq(ctx, h, torch.full_like(h, g)), record=True)[2]["resid"][L_IF][n, 1 + ctx.L[h]])
    return torch.cat(out)


def r2(Y, P, groups=None):
    """Held-out R² of predictions P for Y; against the overall mean, or within groups (their means on Y)."""
    res = ((Y - P) ** 2).sum()
    if groups is None:
        return float(1 - res / ((Y - Y.mean(0)) ** 2).sum())
    u, inv = torch.unique(groups, return_inverse=True)
    m = torch.zeros(len(u), Y.shape[1], dtype=Y.dtype, device=Y.device).index_add_(0, inv, Y)
    m = m / torch.bincount(inv, minlength=len(u))[:, None]
    return float(1 - res / ((Y - m[inv]) ** 2).sum())


class Encoding:
    """The three encodings and the generic subspaces at the goal token, fitted on fit-side histories under all goals."""

    def __init__(self, net, ctx, gen):
        t, dev = ctx.t, ctx.t.dev
        nh = len(ctx.L)
        all_h = torch.arange(nh, device=dev)
        Z = torch.stack([goal_state(net, ctx, all_h, g) for g in range(3)], 1).double()        # [hist, goals, d]
        self.Z = Z
        bel = t.belief[ctx.node].double()
        self.bel = bel
        G = torch.nn.functional.one_hot(torch.arange(3, device=dev), 3).double()
        GL = torch.nn.functional.one_hot(torch.arange(3, device=dev)[None] * (t.max_prefix + 1) + ctx.L[:, None], 3 * (t.max_prefix + 1)).double()
        def design(rows, offset):
            n = len(rows)
            b = bel[rows][:, None].expand(n, 3, -1)
            off = G[None].expand(n, 3, 3) if offset == "g" else GL[rows]
            return torch.cat([b, off], -1).reshape(n * 3, -1)
        fit, test = all_h[ctx.fit], all_h[ctx.test]
        Yf, Yt = Z[fit].reshape(-1, Z.shape[-1]), Z[test].reshape(-1, Z.shape[-1])
        grp = (torch.arange(3, device=dev)[None] * (t.max_prefix + 1) + ctx.L[test][:, None]).reshape(-1)
        nb = bel.shape[1]
        self.fit_stats = {}
        # shared map, offsets per goal and length (primary) and per goal only (the brief's literal form)
        for name, off in (("shared", "gl"), ("brief_offset", "g")):
            enc = Affine(design(fit, off), Yf)
            P = enc(design(test, off))
            setattr(self, "E" if name == "shared" else "E_brief", enc.W[:nb])
            self.fit_stats[name] = dict(r2=r2(Yt, P), r2_within_goal_length=r2(Yt, P, grp))
        # goal-specific maps, offsets per length
        self.E_goal, Pg = [], torch.zeros_like(Z[test])
        L1 = torch.nn.functional.one_hot(ctx.L, t.max_prefix + 1).double()
        for g in range(3):
            X = torch.cat([bel, L1], 1)
            enc = Affine(X[fit], Z[fit, g])
            self.E_goal.append(enc.W[:nb])
            Pg[:, g] = enc(X[test])
        self.fit_stats["goal_specific"] = dict(r2=r2(Yt, Pg.reshape(-1, Z.shape[-1])), r2_within_goal_length=r2(Yt, Pg.reshape(-1, Z.shape[-1]), grp))
        # offsets only: how much of the state the goal and length explain without the posterior
        self.fit_stats["offsets_only"] = dict(r2=r2(Yt, Affine(GL[fit].reshape(len(fit) * 3, -1), Yf)(GL[test].reshape(len(test) * 3, -1))))
        # how different the goal-specific maps are from the shared one (sum-zero belief directions)
        C = torch.eye(nb, dtype=torch.float64, device=dev) - 1.0 / nb
        es = C @ self.E
        self.fit_stats["goal_map_distance"] = [float(((C @ Eg - es) ** 2).sum() / (es ** 2).sum()) for Eg in self.E_goal]
        # generic subspaces of the goal-token state, centred within goal and length
        Zf = Z[fit]
        gm = torch.zeros(3, t.max_prefix + 1, Z.shape[-1], dtype=torch.float64, device=dev)
        for g in range(3):
            for l in range(t.max_prefix + 1):
                m = ctx.L[fit] == l
                if m.any():
                    gm[g, l] = Zf[m, g].mean(0)
        Xc = (Zf - gm[:, ctx.L[fit]].permute(1, 0, 2)).reshape(-1, Z.shape[-1])
        self.P_PCA = torch.linalg.eigh(Xc.T @ Xc / len(Xc))[1].flip(1)[:, :RANK]
        d = Z.shape[-1]
        self.P_R = [R27.orth(torch.randn(d, RANK, dtype=torch.float64, device=dev, generator=gen)) for _ in range(DRAWS)]
        self.Q = [R27.orth(torch.randn(d, d, dtype=torch.float64, device=dev, generator=gen)) for _ in range(DRAWS)]

    def edits(self, a, b, g):
        """New goal-token states for recipients a, donors b under goal g: {kind: [n, d] or a list of draws}."""
        zA, zB = self.Z[a, g], self.Z[b, g]
        db = self.bel[b] - self.bel[a]
        e = db @ self.E
        diff = zB - zA
        proj = lambda P: zA + (diff @ P) @ P.T
        return dict(none=zA, whole=zB, encoding=zA + e, enc_brief=zA + db @ self.E_brief, enc_goal=zA + db @ self.E_goal[g],
                    rotated=[zA + e @ Q for Q in self.Q], pca=proj(self.P_PCA), random=[proj(P) for P in self.P_R])

    def stats(self, a, b):
        """The edit against the actual direct-route difference, all goals pooled."""
        out = {}
        for k, E in (("encoding", lambda g: self.E), ("enc_goal", lambda g: self.E_goal[g])):
            diff = torch.cat([self.Z[b, g] - self.Z[a, g] for g in range(3)])
            e = torch.cat([(self.bel[b] - self.bel[a]) @ E(g) for g in range(3)])
            out[k] = dict(norm_ratio=float((e.norm(dim=1) / diff.norm(dim=1).clamp(min=1e-9)).median()),
                          cosine=float(torch.nn.functional.cosine_similarity(e, diff, dim=1).mean()),
                          residual_share=float(((diff - e) ** 2).sum(1).mean() / (diff ** 2).sum(1).mean()))
        diff = torch.cat([self.Z[b, g] - self.Z[a, g] for g in range(3)])
        out["pca_share"] = float(((diff @ self.P_PCA) ** 2).sum(1).mean() / (diff ** 2).sum(1).mean())
        out["random_share"] = float(np.mean([float(((diff @ P) ** 2).sum(1).mean() / (diff ** 2).sum(1).mean()) for P in self.P_R]))
        return out


def centred(lg):
    lp = torch.log_softmax(lg, -1)
    return lp - lp.mean(-1, keepdim=True)


@torch.no_grad()
def outputs(net, ctx, enc, a, b):
    """Per recipient-donor pair and goal: action distribution [n, goals, 4] for every kind and the hybrid, and the
    projection of each kind's logit change onto the hybrid's (numerator and the shared denominator)."""
    t = ctx.t
    n_all = len(a)
    P = {k: torch.zeros(n_all, 3, 4, device=t.dev) for k in KINDS + ("hybrid",)}
    num = {k: torch.zeros(n_all, 3, dtype=torch.float64, device=t.dev) for k in KINDS}
    den = torch.zeros(n_all, 3, dtype=torch.float64, device=t.dev)
    check = 0.0                                                                            # 'none' patch against the natural run
    for s in range(0, n_all, CHUNK):
        sl = slice(s, s + CHUNK)
        aa, bb = a[sl], b[sl]
        n = torch.arange(len(aa), device=t.dev)
        gp = 1 + ctx.L[aa]
        for g in range(3):
            gg = torch.full_like(aa, g)
            tA, tB = R30.seq(ctx, aa, gg), R30.seq(ctx, bb, gg)
            l0, lh = net(tA)[0][n, gp], net(tB)[0][n, gp]
            c0 = centred(l0); d = (centred(lh) - c0).double()
            den[sl, g] = (d ** 2).sum(-1)
            P["hybrid"][sl, g] = lh.softmax(-1)
            for k, v in enc.edits(aa, bb, g).items():
                ps, nm = [], []
                for z in (v if isinstance(v, list) else [v]):
                    def f(x, z=z):
                        x = x.clone(); x[n, gp] = z.float()
                        return x
                    lg = net(tA, patch={("resid", L_IF): f})[0][n, gp]
                    ps.append(lg.softmax(-1)); nm.append(((centred(lg) - c0).double() * d).sum(-1))
                P[k][sl, g] = torch.stack(ps).mean(0)
                if k == "none":
                    check = max(check, float((ps[0] - l0.softmax(-1)).abs().max()))
                num[k][sl, g] = torch.stack(nm).mean(0)
    outputs.check = max(getattr(outputs, "check", 0.0), check)
    return P, num, den


def pick(T, act):
    """T [n, goals, 4] boolean or float, act [n, goals]: T at the chosen action."""
    return torch.gather(T, 2, act[..., None]).squeeze(-1)


def summarise(P, num, den, ctx, a, b, sel):
    """Absolute rates over the cells `sel` [n, goals] for every kind; pooled transfer against hybrid and whole."""
    optA, optB = ctx.opt[a], ctx.opt[b]
    hyb = P["hybrid"].argmax(-1)
    base = P["none"].argmax(-1)
    w = sel
    cnt = max(int(w.sum()), 1)
    m = lambda x: float((x.double() * w).sum() / cnt)
    out = dict(n=int(w.sum()))
    for k in KINDS + ("hybrid",):
        g = P[k].argmax(-1)
        out[k] = dict(donor_optimal=m(pick(optB, g)), recipient_optimal=m(pick(optA, g)), as_hybrid=m(g == hyb), unchanged=m(g == base),
                      regret_donor=m(ctx.V[b] - pick(ctx.Q[b], g)))
        if k in num:
            out[k]["projection"] = float((num[k] * w).sum() / (den * w).sum().clamp(min=1e-12))   # pooled: sum of numerators / sum of denominators
    d0 = out["none"]["donor_optimal"]
    for k in KINDS:
        dk = out[k]["donor_optimal"]
        out[k]["transfer_hybrid"] = (dk - d0) / (out["hybrid"]["donor_optimal"] - d0) if out["hybrid"]["donor_optimal"] - d0 > 0.02 else None
        out[k]["share_of_whole"] = (dk - d0) / (out["whole"]["donor_optimal"] - d0) if out["whole"]["donor_optimal"] - d0 > 0.02 else None
    return out


def preserve(P, ctx, a, b, sel):
    """Cells where the two posteriors' optimal sets are identical: share still optimal, and harm against no edit."""
    optA = ctx.opt[a]
    w = sel
    cnt = max(int(w.sum()), 1)
    m = lambda x: float((x.double() * w).sum() / cnt)
    out = dict(n=int(w.sum()))
    for k in KINDS + ("hybrid",):
        out[k] = dict(optimal=m(pick(optA, P[k].argmax(-1))))
    for k in KINDS:
        out[k]["harm"] = out["none"]["optimal"] - out[k]["optimal"]
    return out


def goal_dependent(P, ctx, a, b):
    """Pairs where the belief change matters under two goals whose donor optimal sets are disjoint: the same edit must
    give different, appropriate actions. Per kind: share of such pairs right under every change goal, and right with
    different actions under the two goals."""
    optA, optB = ctx.opt[a], ctx.opt[b]
    dj = ~(optA & optB).any(-1)
    gp = []
    for g1 in range(3):
        for g2 in range(g1 + 1, 3):
            gp.append(dj[:, g1] & dj[:, g2] & ~(optB[:, g1] & optB[:, g2]).any(-1))
    gp = torch.stack(gp, 1)                                                                # [n, 3 goal pairs]
    pairs = gp.any(1)
    out = dict(n=int(pairs.sum()))
    for k in KINDS + ("hybrid",):
        g = P[k].argmax(-1)
        ok = pick(optB, g)
        right_all = (ok | ~dj).all(1)
        ix = [(0, 1), (0, 2), (1, 2)]
        both = torch.stack([ok[:, i] & ok[:, j] & (g[:, i] != g[:, j]) for i, j in ix], 1)
        out[k] = dict(right_every_change_goal=float(right_all[pairs].float().mean()),
                      right_distinct_actions=float((both & gp).any(1)[pairs].float().mean()))
    return out


def equivalents(ctx, rec):
    """For each recipient, N_EQ - 1 further held-out histories with the same posterior node and prefix length and
    different token sequences (all N_EQ distinct). Returns [n, N_EQ] (column 0 the recipient) and a validity mask."""
    dev = ctx.t.dev
    g = torch.Generator(device=dev); g.manual_seed(EQ_SEED)
    th = ctx.test_hist
    _, seqid = torch.unique(R30.seq(ctx, th, torch.zeros_like(th)).flatten(1), dim=0, return_inverse=True)
    seq_of = torch.full((len(ctx.L),), -1, dtype=torch.long, device=dev); seq_of[th] = seqid
    # one random representative history per distinct token sequence, sorted by (node, length) in random order
    perm = torch.randperm(len(th), device=dev, generator=g)
    rep = torch.full((int(seqid.max()) + 1,), -1, dtype=torch.long, device=dev)
    rep[seqid[perm]] = th[perm]
    rep = rep[torch.randperm(len(rep), device=dev, generator=g)]
    key = lambda h: ctx.node[h] * 8 + ctx.L[h]
    order = rep[torch.argsort(key(rep), stable=True)]
    k = key(order)
    rk = key(rec)
    s0, sz = torch.searchsorted(k, rk), torch.searchsorted(k, rk, right=True) - torch.searchsorted(k, rk)
    off = torch.randint(1 << 30, (len(rec),), device=dev, generator=g)
    # N_EQ consecutive members of the group are distinct sequences; at most one is the recipient's own
    cand = torch.stack([order[(s0 + (off + j) % sz.clamp(min=1)).clamp(max=len(order) - 1)] for j in range(N_EQ)], 1)
    own = seq_of[cand] == seq_of[rec][:, None]
    rank = torch.argsort(own.long(), dim=1, stable=True)                                 # others first
    out = torch.cat([rec[:, None], torch.gather(cand, 1, rank)[:, :N_EQ - 1]], 1)
    return out, sz >= N_EQ


def equivalence(Ps, ctx, b, sel):
    """Ps: a list of N_EQ outputs, one per equivalent recipient (same posterior). On change cells `sel`: share where every
    recipient reaches the donor's optimal set, mean pairwise agreement of the recipients' greedy actions, and the share of
    the variance of donor-optimality that is between cells (belief change) rather than across recipients."""
    optB = ctx.opt[b]
    out = {}
    for k in KINDS + ("hybrid",):
        G = torch.stack([P[k].argmax(-1) for P in Ps], 0)                                  # [N_EQ, n, goals]
        ok = torch.stack([pick(optB, g) for g in G]).float()
        agree = torch.stack([(G[i] == G[j]).float() for i in range(len(G)) for j in range(i + 1, len(G))]).mean(0)
        mean_c = ok.mean(0)
        tot = ok[:, sel].var(unbiased=False)
        within = ((ok - mean_c) ** 2).mean(0)[sel].mean()
        out[k] = dict(all_donor_optimal=float(ok.prod(0)[sel].mean()), donor_optimal=float(ok.mean(0)[sel].mean()), agreement=float(agree[sel].mean()),
                      between_share=float(1 - within / tot) if tot > 1e-9 else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    m, o = pairs.main, pairs.onestep
    eq, eq_ok = equivalents(ctx, m["r"])
    res = dict(counts=dict(main_pairs=len(m["r"]), onestep_pairs=len(o["r"]), equivalent_pairs=int(eq_ok.sum())), runs={})
    out = HERE / ("smoke.json" if a.untrained else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        gen = torch.Generator(device=dev); gen.manual_seed(3200 + s)
        enc = Encoding(net, ctx, gen)
        row = dict(checkpoint=ck, fit=enc.fit_stats, edit_stats=enc.stats(m["r"], m["d"]))
        ra, rb = m["r"], m["d"]
        P, num, den = outputs(net, ctx, enc, ra, rb)
        dj, sm = pairs.disjoint(ra, rb), pairs.same(ra, rb)
        row["main"] = dict(change=summarise(P, num, den, ctx, ra, rb, dj), preserve=preserve(P, ctx, ra, rb, sm),
                           goal_dependent=goal_dependent(P, ctx, ra, rb),
                           by_goal={f"G{g + 1}": summarise(P, num, den, ctx, ra, rb, dj & (torch.arange(3, device=dev) == g)[None]) for g in range(3)})
        Po, numo, deno = outputs(net, ctx, enc, o["r"], o["d"])
        row["onestep"] = dict(change=summarise(Po, numo, deno, ctx, o["r"], o["d"], pairs.disjoint(o["r"], o["d"])))
        # equivalent recipients: the same belief change applied to N_EQ histories with the recipient's posterior
        ie = torch.nonzero(eq_ok).squeeze(1)
        Ps = [P if j == 0 else outputs(net, ctx, enc, eq[ie, j], rb[ie])[0] for j in range(N_EQ)]
        Ps[0] = {k: v[ie] for k, v in P.items()}
        row["equivalent"] = dict(n_pairs=len(ie), change=equivalence(Ps, ctx, rb[ie], dj[ie]))
        row["check_none_equals_natural"] = outputs.check; outputs.check = 0.0
        res["runs"][f"seed{s}"] = row
        c = row["main"]["change"]
        print(f"seed{s} {ck} R² shared {enc.fit_stats['shared']['r2']:.3f} (within {enc.fit_stats['shared']['r2_within_goal_length']:.3f}) goal-specific {enc.fit_stats['goal_specific']['r2']:.3f} | "
              f"change donor-optimal: " + " ".join(f"{k} {c[k]['donor_optimal']:.3f}" for k in KINDS + ('hybrid',)) +
              f" | preserve harm enc {row['main']['preserve']['encoding']['harm']:.3f} | goal-dep enc {row['main']['goal_dependent']['encoding']['right_every_change_goal']:.2f} "
              f"whole {row['main']['goal_dependent']['whole']['right_every_change_goal']:.2f} | {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
