"""The belief-edit experiment: edits of the decoded belief at the prefix interface of the maze-belief models.

    .venv/bin/python studies/3_reward_trained_agents/belief_edit/run.py            # writes results.json
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazeedit as ED, mazemeasure as MS, mazemodel as MM, mazeppo as P
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
R18 = HERE.parent.parent / "3_reward_trained_agents" / "maze_belief"
sys.path.insert(0, str(R18))
import measure as M18, train as T18                                  # noqa: E402

STEPS = (0, 1, 2, 4)
FIRST = ("ppo/seed1", "ppo/seed3", "ppo/seed4", "supervised/seed0")
LATER = ("ppo/seed0", "ppo/seed2")


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


class Data:
    def __init__(self, device, quick=False):
        self.t = t = T18.tables(device, quick)
        self.dev = device
        self.bank, self.fit = MS.make_bank(t, M18.BANK_SEED, 4000 if quick else 40000)
        b = self.bank
        self.prows = MS.prefix_rows(t, b, self.fit)
        maze = MB.cross_maze(H=5) if quick else MB.cross_maze()
        self.qmdp = torch.tensor(np.stack([maze.mdp_values(g)[maze.H] for g in maze.goals]), device=device, dtype=torch.float32)   # [goal, cell, 4]
        gen = torch.Generator(device=device); gen.manual_seed(20)
        n = 4000 if quick else 60000
        pool = torch.nonzero(~self.fit & (b.prefix >= (1 if quick else 2))).squeeze(1)
        r = pool[torch.randint(len(pool), (n,), device=device, generator=gen)]
        d = pool[torch.randint(len(pool), (n,), device=device, generator=gen)]
        ok = (b.prefix[r] == b.prefix[d]) & (r != d)
        self.r, self.d = r[ok], d[ok]
        r, d = self.r, self.d
        self.L = b.prefix[r]
        self.goal = b.goal[r]
        self.gpos = 1 + self.L
        nA, nB = b.pre_node[r, self.L], b.pre_node[d, self.L]
        self.bA, self.bB = t.belief[nA], t.belief[nB]
        self.preA, self.preB = nA, nB
        l1 = (self.bA - self.bB).abs().sum(1)
        keep = torch.arange(t.L, device=device)[None] <= self.L[:, None]
        differs = ((b.tok[r] != b.tok[d]).any(-1) & keep).any(1)
        qA, qB = t.Q[t.reveal[nA, self.goal]], t.Q[t.reveal[nB, self.goal]]
        self.optA = qA >= qA.max(1, keepdim=True).values - 1e-6
        self.optB = qB >= qB.max(1, keepdim=True).values - 1e-6
        disjoint = ~(self.optA & self.optB).any(1)
        mapA = self.qmdp[self.goal, self.bA.argmax(1)].argmax(1)
        mapB = self.qmdp[self.goal, self.bB.argmax(1)].argmax(1)
        self.sets = dict(mode=disjoint & (mapA != mapB) & (self.bA.argmax(1) != self.bB.argmax(1)),
                         uncertainty=disjoint & (mapA == mapB),
                         similar=(l1 < 0.1) & differs,
                         evidence=l1 >= 0.5)
        # token sequences: the recipient's whole history; the hybrid has the donor's prefix and the recipient's continuation
        self.tokA = b.tok[r].clone()
        self.tokH = b.tok[r].clone()
        self.tokH[keep] = b.tok[d][keep]
        self.mask = ED.prefix_mask(self.tokA, self.gpos)
        # exact nodes of the hybrid along the recipient's continuation (-1 where the donor's belief rules it out)
        node = t.reveal[nB, self.goal]
        rows = torch.arange(len(r), device=device)
        self.alive = b.alive[r]
        self.nodeA = b.node[r]
        self.nodeH = torch.full_like(self.nodeA, -1)
        self.nodeH[:, 0] = node
        for s in range(1, t.H):
            ev = self.tokA[rows, (self.gpos + s).clamp(max=t.L - 1)]
            a, o = ev[:, MM.F_ACT] - 1, ev[:, MM.F_SYM] - 1
            live = self.alive[:, s] & (node >= 0)
            nxt = t.node_nxt[node.clamp(min=0), a.clamp(min=0), o.clamp(min=0)]
            node = torch.where(live, nxt, torch.full_like(node, -1))
            self.nodeH[:, s] = node

    def with_goal(self, tok, g):
        """The sequence up to the reveal, with goal g."""
        out = torch.zeros_like(tok)
        keep = torch.arange(tok.shape[1], device=self.dev)[None] <= self.L[:, None]
        out[keep] = tok[keep]
        rows = torch.arange(len(tok), device=self.dev)
        out[rows, self.gpos, MM.F_TYPE], out[rows, self.gpos, MM.F_GOAL] = MM.GOAL, g + 1
        return out


def scores(pp, pb, ph, optA, optB, sel, responsive=0.3):
    """On pairs `sel`. moved: on those where the model responds to the evidence."""
    if sel.sum() < 30:
        return dict(n=int(sel.sum()))
    resp = sel & (tv(pb, ph) > responsive)
    only_b, neither = (optB & ~optA).float(), (~optA & ~optB).float()
    gain = ((pp - pb) * only_b).sum(1)[sel].mean(); full = ((ph - pb) * only_b).sum(1)[sel].mean()
    return dict(n=int(sel.sum()), n_responsive=int(resp.sum()),
                moved=float(1 - tv(pp, ph)[resp].mean() / tv(pb, ph)[resp].mean()) if resp.sum() >= 30 else None,
                solver_gain=float(gain / full) if abs(float(full)) > 1e-3 else None, hybrid_gain=float(full),
                arbitrary=float(((pp - pb) * neither).sum(1)[sel].mean()),
                greedy_match=float((pp.argmax(1) == ph.argmax(1))[resp & (pb.argmax(1) != ph.argmax(1))].float().mean()) if resp.sum() >= 30 else None)


@torch.no_grad()
def analyse(net, data, quick=False):
    t, b = data.t, data.bank
    dev = data.dev
    gen = torch.Generator(device=dev); gen.manual_seed(7)
    out = dict(sets={k: int(v.sum()) for k, v in data.sets.items()})
    # decoders: at the interface (prefix rows of the fitting histories) and at the goal token's output
    pr = data.prows
    fitp = pr["fit"]
    acts_p, _ = MS.collect(net, b.tok, pr)                                # [9 sites, rows, d]; resid l is site 2l
    d0 = MS.decisions(t, b, data.fit)
    at0 = (d0["step"] == 0) & d0["fit"]
    rows0 = {k: v[at0] for k, v in d0.items()}
    acts0, _ = MS.collect(net, b.tok, rows0)
    out_probe = Affine(acts0[-1], rows0["belief"])
    rows = torch.arange(len(data.r), device=dev)
    for l in (1, 2):
        dec, subs, k = ED.decoder_subspaces(acts_p[2 * l][fitp], pr["belief"][fitp])
        test_r2 = MS.belief_scores(dec(acts_p[2 * l][~fitp]), pr["belief"][~fitp])["r2"]
        xA, xB = ED.states(net, data.tokA, l), ED.states(net, data.tokH, l)
        mA, mB = xA[data.mask], xB[data.mask]
        diff = (mB - mA).double()
        share = {kk: float(((diff @ P_) ** 2).sum() / (diff ** 2).sum()) for kk, P_ in subs.items()}
        res = dict(decoder_r2=test_r2, rank=k, norm_share=share, steps={})
        news = {kind: ED.make_edit(kind, mA, mB, subs, gen) for kind in ED.EDITS + ED.POST_HOC}
        # validity at the interface: decoded posterior after the edit, as a share of the way to the donor's decoded one
        dA, dB = dec(mA), dec(mB)
        res["interface_decoded_to_donor"] = {kind: float(((dec(v) - dA) * (dB - dA)).sum() / ((dB - dA) ** 2).sum()) for kind, v in news.items()}
        mr, mp = torch.nonzero(data.mask, as_tuple=True)                  # the donor's exact posterior at each edited position
        res["interface_decoded_donor_r2"] = MS.belief_scores(dB, t.belief[b.pre_node[data.d][mr, mp]])["r2"]
        for s in STEPS:
            at = (data.gpos + s).clamp(max=t.L - 1)
            live = data.alive[:, s] & (data.nodeH[:, s] >= 0)
            qA, qH = t.Q[data.nodeA[:, s]], t.Q[data.nodeH[:, s].clamp(min=0)]
            optA = qA >= qA.max(1, keepdim=True).values - 1e-6
            optB = qH >= qH.max(1, keepdim=True).values - 1e-6
            if s == 0:
                sets = {kk: v & live for kk, v in data.sets.items()}
            else:
                dis = ~(optA & optB).any(1)
                sets = dict(mode=data.sets["mode"] & live & dis, uncertainty=data.sets["uncertainty"] & live & dis, similar=data.sets["similar"] & live)
            conds = [("intact", None)]
            if s == 0:
                a0 = net(data.tokA, record=True)[2]["attn"][0][rows, data.gpos]
                mean = torch.stack([a0[data.goal == g].mean(0) for g in range(3)])[data.goal]
                conds.append(("raw_route_ablated", mean))
            step = {}
            for cname, abl in conds:
                pb, fb, _ = ED.run(net, data.tokA, at, ablate=abl, goal_pos=data.gpos)
                ph, fh, _ = ED.run(net, data.tokH, at, ablate=abl, goal_pos=data.gpos)
                ob, oh = out_probe(fb), out_probe(fh)
                cond = dict(model=dict(tv_base_hybrid={kk: float(tv(pb, ph)[v].mean()) if v.sum() else None for kk, v in sets.items()},
                                       hybrid_on_solver_set={kk: float((ph * optB)[v].sum(1).mean()) if v.sum() else None for kk, v in sets.items()}))
                for kind, new in news.items():
                    pp, fp, _ = ED.run(net, data.tokA, at, l=l, new=new, mask=data.mask, ablate=abl, goal_pos=data.gpos)
                    op = out_probe(fp)
                    row = {}
                    for kk, sel in sets.items():
                        if kk == "evidence":
                            continue
                        if kk == "similar":
                            row[kk] = dict(n=int(sel.sum()), tv_from_base=float(tv(pp, pb)[sel].mean()) if sel.sum() else None)
                        else:
                            sc = scores(pp, pb, ph, optA, optB, sel)
                            if sel.sum() >= 30 and s == 0:
                                num = ((op - ob) * (oh - ob))[sel].sum(); den = ((oh - ob) ** 2)[sel].sum()
                                sc["output_decoded_to_hybrid"] = float(num / den)
                            row[kk] = sc
                    cond[kind] = row
                step[cname] = cond
            res["steps"][s] = step
        out[f"resid{l}"] = res
    # cross-goal reuse: the same edited prefix states under each goal (l = 1)
    l = 1
    dec, subs, _ = ED.decoder_subspaces(acts_p[2 * l][fitp], pr["belief"][fitp])
    sel = data.sets["evidence"]
    kinds = ("whole", "belief", "complement", "random_subspace", "pattern", "pattern_shift", "pattern_complement")
    for cname in ("intact", "raw_route_ablated"):                       # the ablated condition was added post hoc
        base, hyb, ed = [], [], {k: [] for k in kinds}
        for g in range(3):
            gg = torch.full_like(data.goal, g)
            tA, tH = data.with_goal(data.tokA, gg), data.with_goal(data.tokH, gg)
            abl = None
            if cname != "intact":
                abl = net(tA, record=True)[2]["attn"][0][rows, data.gpos].mean(0, keepdim=True).expand(len(rows), -1)
            xA, xB = ED.states(net, tA, l), ED.states(net, tH, l)
            mA, mB = xA[data.mask], xB[data.mask]
            base.append(ED.run(net, tA, data.gpos, ablate=abl, goal_pos=data.gpos)[0]); hyb.append(ED.run(net, tH, data.gpos, ablate=abl, goal_pos=data.gpos)[0])
            for k in ed:
                ed[k].append(ED.run(net, tA, data.gpos, l=l, new=ED.make_edit(k, mA, mB, subs, gen), mask=data.mask, ablate=abl, goal_pos=data.gpos)[0])
        cg = dict(per_goal={}, both={})
        for g in range(3):
            resp = sel & (tv(base[g], hyb[g]) > 0.3)
            cg["per_goal"][f"G{g + 1}"] = dict(n=int(resp.sum()), **{k: float(1 - tv(v[g], hyb[g])[resp].mean() / tv(base[g], hyb[g])[resp].mean()) for k, v in ed.items()})
        for g1, g2 in ((0, 1), (0, 2), (1, 2)):
            h1, h2, b1, b2 = hyb[g1].argmax(1), hyb[g2].argmax(1), base[g1].argmax(1), base[g2].argmax(1)
            c = sel & (h1 != h2) & (h1 != b1) & (h2 != b2)
            cg["both"][f"G{g1 + 1},G{g2 + 1}"] = dict(n=int(c.sum()), **{k: float(((v[g1].argmax(1) == h1) & (v[g2].argmax(1) == h2))[c].float().mean()) if c.sum() >= 30 else None
                                                                       for k, v in ed.items()},
                                                     same_action_both={k: float((v[g1].argmax(1) == v[g2].argmax(1))[c].float().mean()) if c.sum() >= 30 else None
                                                                       for k, v in ed.items()})
        out["cross_goal" if cname == "intact" else "cross_goal_ablated"] = cg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Data(a.device, a.quick)
    t = data.t
    runs = [(Path(r).parent.name + "/" + Path(r).name, Path(r)) for r in a.runs] if a.runs is not None else [(k, R18 / "runs" / k) for k in FIRST + LATER]
    res = dict(pairs=len(data.r), sets={k: int(v.sum()) for k, v in data.sets.items()}, runs={})
    for name, run in runs:
        net = MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(a.device).eval()
        net.load_state_dict(torch.load(sorted((run / "ckpt").glob("u*.pt"))[-1]))
        r = analyse(net, data, a.quick)
        res["runs"][name] = r
        s0 = r["resid1"]["steps"][0]["intact"]
        print(name, {k: (s0[k]["mode"].get("moved"), s0[k]["uncertainty"].get("moved")) for k in ED.EDITS}, flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
