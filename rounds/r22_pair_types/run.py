"""Round 22: three types of donor-recipient pair, whole and rank-13 PCA patches, every pair under all three goals.

    .venv/bin/python rounds/r22_pair_types/run.py            # writes results.json

The PCA subspaces are the ones round 21 saved (rounds/r21_pattern_specificity/subspaces/); run r21 first.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from goalgeo import mazeedit as ED, mazemeasure as MS, mazemodel as MM

HERE = Path(__file__).resolve().parent
R18 = HERE.parent / "r18_maze_belief"
R21 = HERE.parent / "r21_pattern_specificity"
sys.path.insert(0, str(R18))
import train as T18                                                  # noqa: E402

MODELS = ("ppo/seed1", "ppo/seed3", "ppo/seed4", "supervised/seed0", "ppo/seed0", "ppo/seed2")
BANK_SEED, PAIR_SEED = 220022, 22
L_IF = 1
KINDS = ("hybrid", "whole", "pca", "pca_complement")
CLASSES = ("same", "disjoint", "overlapping")
MIN_L1, CAP = 0.25, 50


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def opt_set(q):
    return q >= q.max(1, keepdim=True).values - 1e-6


# ------------------------------------------------------------------ pairs (solver tables only)

class Data:
    def __init__(self, t, quick=False):
        self.t, self.dev = t, t.dev
        dev = t.dev
        self.bank, fit = MS.make_bank(t, BANK_SEED, 8000 if quick else 200000)
        b = self.bank
        gen = torch.Generator(device=dev); gen.manual_seed(PAIR_SEED)
        self.gen = gen
        lengths = (1, 2) if quick else (2, 3, 4)
        self.pool = pool = torch.nonzero(~fit & (b.prefix >= lengths[0])).squeeze(1)
        n = 400000 if quick else 20000000
        r = pool[torch.randint(len(pool), (n,), device=dev, generator=gen)]
        d = pool[torch.randint(len(pool), (n,), device=dev, generator=gen)]
        ok = (b.prefix[r] == b.prefix[d]) & (r != d)
        r, d = r[ok], d[ok]
        L = b.prefix[r]
        nA, nB = b.pre_node[r, L], b.pre_node[d, L]
        differs = torch.cat([((b.tok[r[s:s + 10 ** 6]] != b.tok[d[s:s + 10 ** 6]]).any(-1) & (torch.arange(t.L, device=dev)[None] <= L[s:s + 10 ** 6, None])).any(1)
                             for s in range(0, len(r), 10 ** 6)])
        l1 = (t.belief[nA] - t.belief[nB]).abs().sum(1)
        eq, dj = self.classes(nA, nB)
        far = differs & (l1 >= MIN_L1)
        per = 200 if quick else 4000
        take = lambda m, k: torch.nonzero(m).squeeze(1)[:k]
        # A: the same node of the belief graph, different tokens; at most CAP pairs per node
        same = torch.nonzero(differs & (nA == nB)).squeeze(1)
        order = same[torch.argsort(nA[same], stable=True)]
        nodes = nA[order]
        start = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), nodes[1:] != nodes[:-1]])
        first = torch.cummax(torch.where(start, torch.arange(len(order), device=dev), torch.zeros_like(order)), 0).values
        capped = torch.zeros(len(r), dtype=torch.bool, device=dev)
        capped[order[torch.arange(len(order), device=dev) - first < CAP]] = True
        iA = torch.cat([take(capped & (L == l), per) for l in lengths])
        # B: same optimal set under one goal, disjoint under another; equal numbers per (g_same, g_disjoint, length)
        cell = per * 3 // 18 + (1 if quick else 34)                                    # 700 in the full run
        iB, pick = [], []
        for ge in range(3):
            for gd in range(3):
                if ge != gd:
                    for l in lengths:
                        idx = take(far & eq[:, ge] & dj[:, gd] & (L == l), cell)
                        iB.append(idx); pick.append(torch.tensor([[ge, gd]], device=dev).expand(len(idx), 2))
        iB, pick = torch.cat(iB), torch.cat(pick)
        # C: disjoint under all three goals
        iC = torch.cat([take(far & dj.all(1) & (L == l), per) for l in lengths])
        self.types = {}
        for name, idx in (("A", iA), ("B", iB), ("C", iC)):
            e, j = eq[idx], dj[idx]
            self.types[name] = dict(r=r[idx], d=d[idx], L=L[idx], nA=nA[idx], nB=nB[idx], l1=l1[idx],
                                    cls=torch.where(e, 0, torch.where(j, 1, 2)))            # [N, goal]: same / disjoint / overlapping
        self.types["B"]["pick"] = pick
        self.info = {k: dict(n=len(v["r"]), by_length={int(l): int((v["L"] == l).sum()) for l in lengths}, mean_l1=float(v["l1"].mean()),
                             distinct_nodes=len(torch.unique(v["nA"])), cells={c: int((v["cls"] == i).sum()) for i, c in enumerate(CLASSES)}) for k, v in self.types.items()}
        self.lengths = lengths
        self.ref = pool[torch.randint(len(pool), (2000 if quick else 20000,), device=dev, generator=gen)]

    def classes(self, nA, nB):
        t = self.t
        eq, dj = [], []
        for g in range(3):
            gg = torch.full_like(nA, g)
            oA, oB = opt_set(t.Q[t.reveal[nA, gg]]), opt_set(t.Q[t.reveal[nB, gg]])
            eq.append((oA == oB).all(1)); dj.append(~(oA & oB).any(1))
        return torch.stack(eq, 1), torch.stack(dj, 1)

    def seq(self, src, L, g):
        t = self.t
        tok = torch.zeros(len(src), t.L, MM.NF, dtype=torch.long, device=self.dev)
        keep = torch.arange(t.L, device=self.dev)[None] <= L[:, None]
        tok[keep] = self.bank.tok[src][keep]
        rows = torch.arange(len(src), device=self.dev)
        tok[rows, 1 + L, MM.F_TYPE], tok[rows, 1 + L, MM.F_GOAL] = MM.GOAL, g + 1
        return tok


# ------------------------------------------------------------------ patches

@torch.no_grad()
def route_means(net, data):
    """The goal token's first attention output, averaged over the reference histories: [goal, prefix length, d]."""
    L = data.bank.prefix[data.ref]
    rows = torch.arange(len(L), device=data.dev)
    out = torch.zeros(3, data.t.max_prefix + 1, net.d, device=data.dev)
    for g in range(3):
        a0 = net(data.seq(data.ref, L, g), record=True)[2]["attn"][0][rows, 1 + L]
        for l in data.lengths:
            out[g, l] = a0[L == l].mean(0)
    return out


@torch.no_grad()
def patched(net, tokA, tokH, gpos, P, abl, chunk=16384):
    """Action distributions at the reveal: base and one per kind."""
    out = {k: [] for k in ("base",) + KINDS}
    for s in range(0, len(tokA), chunk):
        sl = slice(s, s + chunk)
        tA, tH, gp = tokA[sl], tokH[sl], gpos[sl]
        ab = None if abl is None else abl[sl]
        mask = ED.prefix_mask(tA, gp)
        mA, mB = ED.states(net, tA, L_IF)[mask], ED.states(net, tH, L_IF)[mask]
        d = (mB - mA).double()
        new = dict(whole=mB, pca=(mA.double() + d @ P).float(), pca_complement=(mB.double() - d @ P).float())
        out["base"].append(ED.run(net, tA, gp, ablate=ab, goal_pos=gp)[0])
        out["hybrid"].append(ED.run(net, tH, gp, ablate=ab, goal_pos=gp)[0])
        for k, v in new.items():
            out[k].append(ED.run(net, tA, gp, l=L_IF, new=v, mask=mask, ablate=ab, goal_pos=gp)[0])
    return {k: torch.cat(v) for k, v in out.items()}


def summarise(p, optB, sel):
    """Measures on the cells `sel` (rows of one goal, or rows of several goals concatenated)."""
    n = int(sel.sum())
    if n < 30:
        return dict(n=n)
    rows = torch.arange(len(sel), device=sel.device)
    pb, ph = p["base"], p["hybrid"]
    on = lambda q: (q * optB).sum(1)
    res = dict(n=n, base_on_optimal=float(on(pb)[sel].mean()), base_greedy_optimal=float(optB[rows, pb.argmax(1)][sel].float().mean()))
    den = tv(pb, ph)[sel].mean()
    full = (on(ph) - on(pb))[sel].mean()
    for k in KINDS:
        q = p[k]
        res[k] = dict(tv_from_base=float(tv(q, pb)[sel].mean()), d_optimal=float((on(q) - on(pb))[sel].mean()),
                      d_optimal_share=float((on(q) - on(pb))[sel].mean() / full) if abs(float(full)) > 0.01 else None,
                      moved=float(1 - tv(q, ph)[sel].mean() / den) if float(den) > 0.01 else None,
                      greedy_changed=float((q.argmax(1) != pb.argmax(1))[sel].float().mean()), greedy_optimal=float(optB[rows, q.argmax(1)][sel].float().mean()),
                      # added after the first look (pairs are drawn symmetrically, so the signed change cancels): the
                      # probability that crosses the boundary of the optimal set, and greedy changes by where they start and end
                      crossing=float((on(q) - on(pb)).abs()[sel].mean()),
                      greedy_within=float(((q.argmax(1) != pb.argmax(1)) & optB[rows, q.argmax(1)] & optB[rows, pb.argmax(1)])[sel].float().mean()),
                      greedy_lost=float((~optB[rows, q.argmax(1)] & optB[rows, pb.argmax(1)])[sel].float().mean()),
                      greedy_gained=float((optB[rows, q.argmax(1)] & ~optB[rows, pb.argmax(1)])[sel].float().mean()))
    return res


@torch.no_grad()
def analyse(net, data, P):
    t = data.t
    means = route_means(net, data)
    out = {}
    for cname in ("natural", "direct_route_removed"):
        res = {}
        for name, v in data.types.items():
            if len(v["r"]) == 0:
                continue
            gpos = 1 + v["L"]
            ps, opts = [], []
            for g in range(3):
                abl = means[g, v["L"]] if cname != "natural" else None
                ps.append(patched(net, data.seq(v["r"], v["L"], g), data.seq(v["d"], v["L"], g), gpos, P, abl))
                opts.append(opt_set(t.Q[t.reveal[v["nB"], torch.full_like(v["nB"], g)]]))
            allp = {k: torch.cat([p[k] for p in ps]) for k in ps[0]}
            allo = torch.cat(opts)
            cls = v["cls"].T.reshape(-1)                                           # goal-major, as the concatenation
            r = dict(all_goals={c: summarise(allp, allo, cls == i) for i, c in enumerate(CLASSES)},
                     per_goal={f"G{g + 1}": {c: summarise(ps[g], opts[g], v["cls"][:, g] == i) for i, c in enumerate(CLASSES)} for g in range(3)})
            Ls = v["L"].repeat(3)
            r["by_length"] = {int(l): {c: summarise(allp, allo, (cls == i) & (Ls == l)) for i, c in enumerate(CLASSES)} for l in data.lengths}
            if name == "B":                                                        # the cells each pair was chosen for
                n = len(v["r"]); rows = torch.arange(n, device=data.dev)
                for j, c in enumerate(("same", "disjoint")):
                    sel = torch.zeros(3 * n, dtype=torch.bool, device=data.dev)
                    sel[v["pick"][:, j] * n + rows] = True
                    r[f"chosen_{c}"] = summarise(allp, allo, sel)
            res[name] = r
        out[cname] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = T18.tables(a.device, a.quick)
    data = Data(t, a.quick)
    res = dict(types=data.info, runs={})
    print(res, flush=True)
    for name in MODELS[:2] if a.quick else MODELS:
        net = MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(a.device).eval()
        if a.quick:                                                  # the smoke run uses a small maze: untrained networks, a random subspace
            P = torch.linalg.qr(torch.randn(net.d, 13, dtype=torch.float64, device=a.device))[0]; P = P @ P.T
        else:
            net.load_state_dict(torch.load(sorted((R18 / "runs" / name / "ckpt").glob("u*.pt"))[-1]))
            P = torch.load(R21 / "subspaces" / (name.replace("/", "_") + ".pt"))["subs"]["pca"].to(a.device)
        res["runs"][name] = r = analyse(net, data, P)
        for c in r:
            print(name, c, {ty: {cl: (x.get("hybrid", {}).get("tv_from_base"), x.get("pca", {}).get("tv_from_base")) for cl, x in r[c][ty]["all_goals"].items() if x["n"] >= 30} for ty in r[c]}, flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
