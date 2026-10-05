"""The pattern-specificity experiment: is the Sigma-W edit of the prefix states specific? Replication on new histories, with controls.

    .venv/bin/python studies/3_reward_trained_agents/pattern_specificity/run.py            # writes subspaces/ and results.json

Order: every subspace is fitted on the maze-belief experiment's fitting histories and saved for all models; only then is the new bank
made and the evaluation pairs drawn.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazeedit as ED, mazemeasure as MS, mazemodel as MM
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
R18 = HERE.parent.parent / "3_reward_trained_agents" / "maze_belief"
R20 = HERE.parent.parent / "3_reward_trained_agents" / "belief_edit"
sys.path.insert(0, str(R18))
import measure as M18, train as T18                                  # noqa: E402

MODELS = ("ppo/seed1", "ppo/seed3", "ppo/seed4", "supervised/seed0", "ppo/seed0", "ppo/seed2")
BANK_SEED, PAIR_SEED = 210021, 21
L_IF = 1                                                             # the interface: entering block 1
SWEEP = (1, 2, 4, 7)
DRAWS = 5
BINS = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0 + 1e-6)
MATCHED = ("whole", "pattern", "pattern_shift", "pattern_complement", "pca", "random_share_0", "belief")
CROSS = ("whole", "pattern", "pattern_shift", "pattern_complement", "pca", "random_share_0")


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def opt_set(q):
    return q >= q.max(1, keepdim=True).values - 1e-6


# ------------------------------------------------------------------ subspaces (fitting histories only)

@torch.no_grad()
def fit_subspaces(net, t, bank, fit, share20, seed=21):
    pr = MS.prefix_rows(t, bank, fit)
    acts, _ = MS.collect(net, bank.tok, pr)
    X, Y = acts[2 * L_IF][pr["fit"]], pr["belief"][pr["fit"]]
    dec = Affine(X, Y)
    W = dec.W
    Xc = X.double() - X.double().mean(0)
    Sig = Xc.T @ Xc / len(Xc)
    A = Sig @ W
    Ua, Sa, _ = torch.linalg.svd(A, full_matrices=False)
    k = int((Sa > Sa.max() * 1e-8).sum())
    Ua = Ua[:, :k]
    ev, V = torch.linalg.eigh(Sig)
    V, ev = V.flip(1), ev.flip(0)
    d = X.shape[1]
    I = torch.eye(d, dtype=torch.float64, device=X.device)
    P = lambda U: U @ U.T
    g = torch.Generator().manual_seed(seed)
    rnd = lambda r: torch.linalg.qr(torch.randn(d, r, generator=g, dtype=torch.float64))[0].to(X.device)
    r_share = int(round(d * share20))
    subs = dict(pattern=P(Ua), pattern_shift=(A @ torch.linalg.pinv(W.T @ A, hermitian=True) @ W.T).T, belief=ED.projector(W),
                pca=P(V[:, :k]), pca_out_pattern=ED.projector((I - P(Ua)) @ V[:, :k]), pattern_out_pca=ED.projector((I - P(V[:, :k])) @ Ua))
    rank = dict(pattern=k, pattern_shift=k, belief=int(round(torch.trace(subs["belief"]).item())), pca=k,
                pca_out_pattern=int(round(torch.trace(subs["pca_out_pattern"]).item())), pattern_out_pca=int(round(torch.trace(subs["pattern_out_pca"]).item())))
    for i in range(DRAWS):
        subs[f"random_rank_{i}"], rank[f"random_rank_{i}"] = P(rnd(k)), k
        subs[f"random_share_{i}"], rank[f"random_share_{i}"] = P(rnd(r_share)), r_share
    for r in SWEEP:
        if r >= k:
            continue
        subs[f"pattern_r{r}"], subs[f"pca_r{r}"] = P(Ua[:, :r]), P(V[:, :r])
        rank[f"pattern_r{r}"] = rank[f"pca_r{r}"] = r
        for i in range(DRAWS):
            subs[f"random_r{r}_{i}"], rank[f"random_r{r}_{i}"] = P(rnd(r)), r
    test = ~pr["fit"]
    info = dict(k=k, random_share_rank=r_share, decoder_r2=MS.belief_scores(dec(acts[2 * L_IF][test]), pr["belief"][test])["r2"],
                pca_variance_share=float(ev[:k].sum() / ev.sum()),
                pattern_in_pca=float((subs["pattern"] * subs["pca"]).sum() / k),          # 1: the same span; k/d: unrelated
                variance_in_pattern=float((Sig * subs["pattern"]).sum() / ev.sum()))
    return subs, rank, info


KINDS_EXTRA = ("whole", "pattern_complement") + tuple(f"random_norm_{i}" for i in range(DRAWS))


def make_edit(kind, xA, xB, subs, gen):
    d = (xB - xA).double()
    if kind == "whole":
        return xB
    if kind == "pattern_complement":
        return (xB.double() - d @ subs["pattern"]).float()
    if kind.startswith("random_norm"):
        r = torch.randn(d.shape, device=d.device, generator=gen, dtype=torch.float64)
        return (xA.double() + r / r.norm(dim=1, keepdim=True) * (d @ subs["pattern"]).norm(dim=1, keepdim=True)).float()
    return (xA.double() + d @ subs[kind]).float()


@torch.no_grad()
def edited(net, tokA, tokH, gpos, subs, kinds, abl, gen, chunk=16384):
    """Action distributions at the reveal: base, hybrid and one per edit; and each edit's magnitude and captured share."""
    pb, ph, pp = [], [], {k: [] for k in kinds}
    sq = {k: [0.0, 0.0] for k in kinds}; tot = 0.0
    for s in range(0, len(tokA), chunk):
        sl = slice(s, s + chunk)
        tA, tH, gp = tokA[sl], tokH[sl], gpos[sl]
        ab = None if abl is None else abl[sl]
        mask = ED.prefix_mask(tA, gp)
        mA, mB = ED.states(net, tA, L_IF)[mask], ED.states(net, tH, L_IF)[mask]
        pb.append(ED.run(net, tA, gp, ablate=ab, goal_pos=gp)[0]); ph.append(ED.run(net, tH, gp, ablate=ab, goal_pos=gp)[0])
        tot += float(((mB - mA).double() ** 2).sum())
        for k in kinds:
            new = make_edit(k, mA, mB, subs, gen)
            sq[k][0] += float(((new - mA).double() ** 2).sum()); sq[k][1] += float(((mB - new).double() ** 2).sum())
            pp[k].append(ED.run(net, tA, gp, l=L_IF, new=new, mask=mask, ablate=ab, goal_pos=gp)[0])
    meta = {k: dict(magnitude=(v[0] / tot) ** 0.5, share=1 - v[1] / tot) for k, v in sq.items()}
    return torch.cat(pb), torch.cat(ph), {k: torch.cat(v) for k, v in pp.items()}, meta


@torch.no_grad()
def raw_route_mean(net, tok, gpos, goal):
    """The decision token's first attention output, replaced by its mean for that goal."""
    rows = torch.arange(len(tok), device=tok.device)
    a0 = torch.cat([net(tok[s:s + 16384], record=True)[2]["attn"][0][rows[:len(tok[s:s + 16384])], gpos[s:s + 16384]] for s in range(0, len(tok), 16384)])
    mean = torch.stack([a0[goal == g].mean(0) if (goal == g).any() else a0.mean(0) for g in range(3)])
    return mean[goal]


# ------------------------------------------------------------------ evaluation pairs (the new bank)

class Data:
    def __init__(self, t, quick=False):
        self.t, self.dev = t, t.dev
        dev = t.dev
        self.bank, fit = MS.make_bank(t, BANK_SEED, 4000 if quick else 80000)
        b = self.bank
        maze = MB.cross_maze(H=5) if quick else MB.cross_maze()
        self.qmdp = torch.tensor(np.stack([maze.mdp_values(g)[maze.H] for g in maze.goals]), device=dev, dtype=torch.float32)
        gen = torch.Generator(device=dev); gen.manual_seed(PAIR_SEED)
        self.pool = torch.nonzero(~fit & (b.prefix >= (1 if quick else 2))).squeeze(1)
        self.main = self.pairs(*self.draw(8000 if quick else 120000, gen))
        m = self.main
        mapA = self.qmdp[m["goal"], m["bA"].argmax(1)].argmax(1); mapB = self.qmdp[m["goal"], m["bB"].argmax(1)].argmax(1)
        disjoint = ~(m["optA"] & m["optB"]).any(1)
        self.sets = dict(mode=disjoint & (mapA != mapB) & (m["bA"].argmax(1) != m["bB"].argmax(1)), uncertainty=disjoint & (mapA == mapB),
                         similar=(m["l1"] < 0.1) & m["differs"], evidence=m["l1"] >= 0.5)
        # posterior-matched: pairs binned by the distance between the posteriors, different token histories
        r, d = self.draw(200000 if quick else 4000000, gen)
        nA, nB = b.pre_node[r, b.prefix[r]], b.pre_node[d, b.prefix[d]]
        l1 = (t.belief[nA] - t.belief[nB]).abs().sum(1)
        keep = torch.arange(t.L, device=dev)[None] <= b.prefix[r][:, None]
        differs = ((b.tok[r] != b.tok[d]).any(-1) & keep).any(1)
        rs, ds, bins = [], [], []
        for i in range(len(BINS) - 1):
            idx = torch.nonzero(differs & (l1 >= BINS[i]) & (l1 < BINS[i + 1])).squeeze(1)[:3000]
            rs.append(r[idx]); ds.append(d[idx]); bins.append(torch.full_like(idx, i))
        self.matched = self.pairs(torch.cat(rs), torch.cat(ds))
        self.matched["bin"] = torch.cat(bins)

    def draw(self, n, gen):
        b = self.bank
        r = self.pool[torch.randint(len(self.pool), (n,), device=self.dev, generator=gen)]
        d = self.pool[torch.randint(len(self.pool), (n,), device=self.dev, generator=gen)]
        ok = (b.prefix[r] == b.prefix[d]) & (r != d)
        return r[ok], d[ok]

    def seq(self, src, L, goal):
        t = self.t
        tok = torch.zeros(len(src), t.L, MM.NF, dtype=torch.long, device=self.dev)
        keep = torch.arange(t.L, device=self.dev)[None] <= L[:, None]
        tok[keep] = self.bank.tok[src][keep]
        rows = torch.arange(len(src), device=self.dev)
        tok[rows, 1 + L, MM.F_TYPE], tok[rows, 1 + L, MM.F_GOAL] = MM.GOAL, goal + 1
        return tok

    def pairs(self, r, d):
        t, b = self.t, self.bank
        L, goal = b.prefix[r], b.goal[r]
        nA, nB = b.pre_node[r, L], b.pre_node[d, L]
        bA, bB = t.belief[nA], t.belief[nB]
        keep = torch.arange(t.L, device=self.dev)[None] <= L[:, None]
        qA, qB = t.Q[t.reveal[nA, goal]], t.Q[t.reveal[nB, goal]]
        return dict(r=r, d=d, L=L, goal=goal, gpos=1 + L, nA=nA, nB=nB, bA=bA, bB=bB, l1=(bA - bB).abs().sum(1),
                    differs=((b.tok[r] != b.tok[d]).any(-1) & keep).any(1), qA=qA, qB=qB, optA=opt_set(qA), optB=opt_set(qB),
                    tokA=self.seq(r, L, goal), tokH=self.seq(d, L, goal))


def scores(pp, pb, ph, optA, optB, sel, responsive=0.3):
    """The belief-edit experiment's measures on pairs `sel`; moved on those where the model responds to the evidence."""
    if sel.sum() < 30:
        return dict(n=int(sel.sum()))
    resp = sel & (tv(pb, ph) > responsive)
    only_b, neither = (optB & ~optA).float(), (~optA & ~optB).float()
    gain = ((pp - pb) * only_b).sum(1)[sel].mean(); full = ((ph - pb) * only_b).sum(1)[sel].mean()
    return dict(n=int(sel.sum()), n_responsive=int(resp.sum()),
                moved=float(1 - tv(pp, ph)[resp].mean() / tv(pb, ph)[resp].mean()) if resp.sum() >= 30 else None,
                solver_gain=float(gain / full) if abs(float(full)) > 1e-3 else None,
                arbitrary=float(((pp - pb) * neither).sum(1)[sel].mean()))


def binned(x_unc, x_mode, size_unc, size_mode, edges, gen, boot=500):
    """x = (tv(edited, hybrid), tv(base, hybrid)) per pair. moved per bin of `size` for each set, the unmatched
    difference (uncertainty - mode), and the difference within bins weighted by the pooled bin counts, with a
    bootstrap interval over pairs."""
    nb = len(edges) + 1
    bu, bm = torch.bucketize(size_unc, edges), torch.bucketize(size_mode, edges)

    def stat(iu, im):
        def per(x, bins, idx):
            num = torch.zeros(nb, device=bins.device, dtype=torch.float64).index_add_(0, bins[idx], x[0][idx].double())
            den = torch.zeros(nb, device=bins.device, dtype=torch.float64).index_add_(0, bins[idx], x[1][idx].double())
            return num, den, torch.bincount(bins[idx], minlength=nb)
        nu, du, cu = per(x_unc, bu, iu); nm, dm, cm = per(x_mode, bm, im)
        mu, mm = 1 - nu / du.clamp(min=1e-9), 1 - nm / dm.clamp(min=1e-9)
        ok = (cu >= 30) & (cm >= 30)
        w = (cu + cm).double() * ok
        matched = ((mu - mm) * w)[ok].sum() / w.sum().clamp(min=1e-9)
        return mu, mm, cu, cm, float((1 - nu.sum() / du.sum()) - (1 - nm.sum() / dm.sum())), float(matched)
    au, am = torch.arange(len(size_unc), device=size_unc.device), torch.arange(len(size_mode), device=size_mode.device)
    mu, mm, cu, cm, raw, matched = stat(au, am)
    bs = []
    for _ in range(boot):
        bs.append(stat(torch.randint(len(au), (len(au),), device=au.device, generator=gen), torch.randint(len(am), (len(am),), device=am.device, generator=gen))[4:])
    bs = np.array(bs)
    return dict(edges=[float(e) for e in edges], uncertainty=[float(x) if c >= 30 else None for x, c in zip(mu, cu)], mode=[float(x) if c >= 30 else None for x, c in zip(mm, cm)],
                n_uncertainty=cu.tolist(), n_mode=cm.tolist(), unmatched_difference=raw, matched_difference=matched,
                unmatched_ci=[float(np.quantile(bs[:, 0], q)) for q in (0.025, 0.975)], matched_ci=[float(np.quantile(bs[:, 1], q)) for q in (0.025, 0.975)])


@torch.no_grad()
def analyse(net, data, subs, rank):
    t, dev = data.t, data.dev
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    kinds = tuple(subs) + KINDS_EXTRA
    out = {}
    # ---- replication and controls, on mode and uncertainty pairs
    m = data.main
    keep = {}
    for cname in ("intact", "raw_route_removed"):
        abl = raw_route_mean(net, m["tokA"], m["gpos"], m["goal"]) if cname != "intact" else None
        pb, ph, pp, meta = edited(net, m["tokA"], m["tokH"], m["gpos"], subs, kinds, abl, gen)
        keep[cname] = (pb, ph, pp)
        res = dict(tv_base_hybrid={k: float(tv(pb, ph)[v].mean()) for k, v in data.sets.items()})
        for k in kinds:
            res[k] = dict(rank=rank.get(k), **meta[k], **{s: scores(pp[k], pb, ph, m["optA"], m["optB"], data.sets[s]) for s in ("mode", "uncertainty")},
                          similar_tv_from_base=float(tv(pp[k], pb)[data.sets["similar"]].mean()) if data.sets["similar"].any() else None)
        out[cname] = res
    # ---- uncertainty against mode, at matched size of the change (raw route intact)
    pb, ph, pp = keep["intact"]
    aA = m["qA"].argmax(1, keepdim=True)
    s_solver = (m["qB"].max(1).values - m["qB"].gather(1, aA).squeeze(1))
    s_model = tv(pb, ph)
    um = {}
    for size_name, size, base in (("s_solver", s_solver, tv(pb, ph) > 0.3), ("s_model", s_model, torch.ones_like(s_model, dtype=torch.bool))):
        u, mo = data.sets["uncertainty"] & base, data.sets["mode"] & base
        if u.sum() < 30 or mo.sum() < 30:
            continue
        edges = torch.quantile(size[u | mo].double(), torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float64, device=dev)).to(size.dtype)
        edges = torch.unique(edges)
        um[size_name] = dict(mean_size=dict(uncertainty=float(size[u].mean()), mode=float(size[mo].mean())),
                             **{k: binned((tv(pp[k], ph)[u], tv(pb, ph)[u]), (tv(pp[k], ph)[mo], tv(pb, ph)[mo]), size[u], size[mo], edges, gen) for k in ("whole", "pattern")})
    out["uncertainty_vs_mode"] = um
    # ---- posterior-matched histories
    q = data.matched
    pm = {}
    for cname in ("intact", "raw_route_removed"):
        abl = raw_route_mean(net, q["tokA"], q["gpos"], q["goal"]) if cname != "intact" else None
        pb, ph, pp, _ = edited(net, q["tokA"], q["tokH"], q["gpos"], subs, MATCHED, abl, gen)
        rows = []
        for i in range(len(BINS) - 1):
            sel = q["bin"] == i
            if sel.sum() < 30:
                rows.append(dict(lo=BINS[i], hi=BINS[i + 1], n=int(sel.sum()))); continue
            w = tv(pp["whole"], pb)[sel].mean()
            rows.append(dict(lo=BINS[i], hi=BINS[i + 1], n=int(sel.sum()), l1=float(q["l1"][sel].mean()), tv_base_hybrid=float(tv(pb, ph)[sel].mean()),
                             **{k: dict(tv_from_base=float(tv(v, pb)[sel].mean()), share_of_whole=float(tv(v, pb)[sel].mean() / w)) for k, v in pp.items()}))
        pm[cname] = rows
    out["posterior_matched"] = pm
    # ---- the same edit across goals, scored against the solver
    sel = data.sets["evidence"]
    cg = {}
    for cname in ("intact", "raw_route_removed"):
        P_, oA, oB = [], [], []
        for g in range(3):
            gg = torch.full_like(m["goal"], g)
            tA, tH = data.seq(m["r"], m["L"], gg), data.seq(m["d"], m["L"], gg)
            abl = raw_route_mean(net, tA, m["gpos"], gg) if cname != "intact" else None
            pb, ph, pp, _ = edited(net, tA, tH, m["gpos"], subs, CROSS, abl, gen)
            P_.append(dict(base=pb, hybrid=ph, **pp))
            oA.append(opt_set(t.Q[t.reveal[m["nA"], gg]])); oB.append(opt_set(t.Q[t.reveal[m["nB"], gg]]))
        res = dict(per_goal={}, both={})
        for g in range(3):
            c = sel & ~(oA[g] & oB[g]).any(1)
            only_b = (oB[g] & ~oA[g]).float()
            full = ((P_[g]["hybrid"] - P_[g]["base"]) * only_b).sum(1)[c].mean()
            res["per_goal"][f"G{g + 1}"] = dict(n=int(c.sum()), hybrid_gain=float(full),
                                                **{k: dict(gain=float(((P_[g][k] - P_[g]["base"]) * only_b).sum(1)[c].mean() / full) if abs(float(full)) > 1e-3 else None,
                                                           changed=float(tv(P_[g][k], P_[g]["base"])[c].mean())) for k in CROSS})
        for g1, g2 in ((0, 1), (0, 2), (1, 2)):
            c = sel & ~(oB[g1] & oB[g2]).any(1) & ~(oA[g1] & oB[g1]).any(1) & ~(oA[g2] & oB[g2]).any(1)
            row = dict(n=int(c.sum()))
            if c.sum() >= 200:
                rows = torch.arange(len(c), device=dev)
                for k in ("base", "hybrid") + CROSS:
                    a1, a2 = P_[g1][k].argmax(1), P_[g2][k].argmax(1)
                    in1, in2 = oB[g1][rows, a1], oB[g2][rows, a2]
                    ch = (a1 != P_[g1]["base"].argmax(1)) & (a2 != P_[g2]["base"].argmax(1))
                    row[k] = dict(appropriate_both=float((in1 & in2)[c].float().mean()), changed_not_appropriate=float((ch & ~(in1 & in2))[c].float().mean()))
            res["both"][f"G{g1 + 1},G{g2 + 1}"] = row
        cg[cname] = res
    out["cross_goal"] = cg
    return out


def load(name, t, device, quick):
    net = MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(device).eval()
    if not quick:                                                    # the smoke run uses a small maze: untrained networks
        net.load_state_dict(torch.load(sorted((R18 / "runs" / name / "ckpt").glob("u*.pt"))[-1]))
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); (out / "subspaces").mkdir(parents=True, exist_ok=True)
    t = T18.tables(a.device, a.quick)
    models = MODELS[:2] if a.quick else MODELS
    # 1. construction, on the maze-belief experiment's fitting histories; saved before any evaluation pair exists
    bank, fit = MS.make_bank(t, M18.BANK_SEED, 4000 if a.quick else 40000)
    r20 = json.load(open(R20 / "results.json"))["runs"]
    info = {}
    for name in models:
        net = load(name, t, a.device, a.quick)
        subs, rank, info[name] = fit_subspaces(net, t, bank, fit, r20[name]["resid1"]["norm_share"]["pattern"])
        torch.save(dict(subs={k: v.cpu() for k, v in subs.items()}, rank=rank, info=info[name]), out / "subspaces" / (name.replace("/", "_") + ".pt"))
        print("fitted", name, info[name], flush=True)
    del bank
    # 2. evaluation
    data = Data(t, a.quick)
    res = dict(pairs=len(data.main["r"]), sets={k: int(v.sum()) for k, v in data.sets.items()}, matched_pairs=len(data.matched["r"]), runs={})
    print(res, flush=True)
    for name in models:
        net = load(name, t, a.device, a.quick)
        f = torch.load(out / "subspaces" / (name.replace("/", "_") + ".pt"))
        r = analyse(net, data, {k: v.to(a.device) for k, v in f["subs"].items()}, f["rank"])
        r["subspaces"] = info[name]
        res["runs"][name] = r
        x = r["raw_route_removed"]
        print(name, {k: (x[k]["mode"].get("moved"), round(x[k]["share"], 3)) for k in ("pattern", "pattern_shift", "pattern_complement", "pca", "random_rank_0", "random_share_0")}, flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
