"""The maze-occupancy experiment: occupancy against the posterior and the action values, on the maze-belief models.

    .venv/bin/python studies/3_reward_trained_agents/maze_occupancy/run.py            # every maze-belief run; writes results.json
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazebelief as MB, mazegraph as MG, mazemeasure as MS, mazemodel as MM, mazeocc as MO, mazeppo as P
from goalgeo.navprobe import Affine, site_names, site_stack

HERE = Path(__file__).resolve().parent
R18 = HERE.parent.parent / "3_reward_trained_agents" / "maze_belief"
sys.path.insert(0, str(R18))
import measure as M18, train as T18                                  # noqa: E402


def r2(pred, Y):
    Y = Y.double()
    return float(1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


class Data:
    def __init__(self, device, quick=False):
        self.t = t = T18.tables(device, quick)
        g = MG.build(MB.cross_maze(H=5), max_prefix=2) if quick else MG.cached(MB.cross_maze(), R18 / "cache" / "graph.npz")
        D, _ = MO.solver_occupancy(g)
        self.D = torch.tensor(D, device=device)
        bank, _ = MS.make_bank(t, M18.BANK_SEED, 4000 if quick else 40000)
        gen = torch.Generator(device=device); gen.manual_seed(19)
        pool = torch.nonzero(bank.prefix >= 1).squeeze(1)
        h = pool[torch.randperm(len(pool), device=device, generator=gen)[: 1200 if quick else 12000]]
        n = len(h)
        self.prefix = bank.prefix[h].repeat(3)
        self.goal = torch.arange(3, device=device).repeat_interleave(n)
        self.ev = torch.arange(n, device=device).repeat(3)             # index of the evidence (history)
        pre = bank.pre_node[h, bank.prefix[h]].repeat(3)
        self.node = t.reveal[pre, self.goal]
        L = t.L
        tok = torch.zeros(3 * n, L, MM.NF, dtype=torch.long, device=device)
        keep = torch.arange(L, device=device)[None] <= self.prefix[:, None]
        tok[keep] = bank.tok[h].repeat(3, 1, 1)[keep]
        rows = torch.arange(3 * n, device=device)
        self.pos = 1 + self.prefix
        tok[rows, self.pos, MM.F_TYPE], tok[rows, self.pos, MM.F_GOAL] = MM.GOAL, self.goal + 1
        self.tok = tok
        # evidence hash -> split
        code = ((bank.tok[h][:, :, MM.F_SYM] * 5 + bank.tok[h][:, :, MM.F_ACT]) * keep[:n])
        w = torch.tensor([(7919 * (i + 1)) % 1000003 for i in range(L)], device=device)
        hh = ((code * w).sum(1) * 2654435761) % 3000
        held_ev = (hh % 10 >= 7).repeat(3)
        held_goal = ((hh // 10) % 3).repeat(3)
        self.test_ev = held_ev
        self.test_combo = ~held_ev & (self.goal == held_goal)
        self.fit = ~held_ev & ~self.test_combo
        self.b = t.belief[self.node]
        self.q = t.Q[self.node]
        self.d = self.D[self.node]
        self.opt = self.q >= self.q.max(1, keepdim=True).values - 1e-6
        self.onehot = torch.nn.functional.one_hot(self.goal, 3).float()
        # exact-feature predictors of the solver's occupancy
        self.exact = self.predictors(self.d)

    def predictors(self, d, extra=None):
        """extra: further exact features for a fifth predictor P4, affine in [b, Q*, extra] per goal."""
        fit = self.fit
        out = dict(P0=torch.zeros_like(d, dtype=torch.float64), P1=None, P2=torch.zeros_like(d, dtype=torch.float64),
                   P3=torch.zeros_like(d, dtype=torch.float64))
        X1 = torch.cat([self.b, self.onehot], 1)
        out["P1"] = Affine(X1[fit], d[fit], lam=1e-6)(X1)
        for g in range(3):
            s = self.goal == g
            out["P0"][s] = d[fit & s].double().mean(0)
            out["P2"][s] = Affine(self.b[fit & s], d[fit & s], lam=1e-6)(self.b[s])
            X3 = torch.cat([self.b, self.q], 1)
            out["P3"][s] = Affine(X3[fit & s], d[fit & s], lam=1e-6)(X3[s])
            if extra is not None:
                X4 = torch.cat([self.b, self.q, extra], 1)
                out.setdefault("P4", torch.zeros_like(d, dtype=torch.float64))[s] = Affine(X4[fit & s], d[fit & s], lam=1e-6)(X4[s])
        return out


@torch.no_grad()
def activations(net, data, at_prefix=False, chunk=8192):
    S = 2 * net.nl + 1
    N = len(data.tok)
    acts = torch.zeros(S, N, net.d, device=data.tok.device); logits = torch.zeros(N, 4, device=data.tok.device)
    pos = data.pos - 1 if at_prefix else data.pos
    for s in range(0, N, chunk):
        lg, _, rec = net(data.tok[s:s + chunk], record=True)
        n = torch.arange(lg.shape[0], device=lg.device)
        for i, a in enumerate(site_stack(rec, net.nl)):
            acts[i, s:s + chunk] = a[n, pos[s:s + chunk]]
        logits[s:s + chunk] = lg[n, pos[s:s + chunk]]
    return acts, logits


def pair_sets(data, greedy, n=20000, seed=5, min_l1=0.5):
    """(A) the same evidence, two goals; (B) the same goal, two pieces of evidence. Both: the same solver-optimal
    action set, the same greedy action of the model, occupancies at least min_l1 apart. Test rows only."""
    dev = data.tok.device
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    test = data.test_ev | data.test_combo
    ne = len(data.tok) // 3
    out = {}
    e = torch.randint(ne, (n,), device=dev, generator=gen)
    g1 = torch.randint(3, (n,), device=dev, generator=gen); g2 = (g1 + 1 + torch.randint(2, (n,), device=dev, generator=gen)) % 3
    out["same_evidence"] = (g1 * ne + e, g2 * ne + e)
    pool = torch.nonzero(test).squeeze(1)
    i = pool[torch.randint(len(pool), (4 * n,), device=dev, generator=gen)]
    j = pool[torch.randint(len(pool), (4 * n,), device=dev, generator=gen)]
    ok = (data.goal[i] == data.goal[j]) & (data.prefix[i] == data.prefix[j])
    out["same_goal"] = (i[ok], j[ok])
    res = {}
    for k, (i, j) in out.items():
        ok = (test[i] | test[j]) & (data.opt[i] == data.opt[j]).all(1) & (greedy[i] == greedy[j]) & ((data.d[i] - data.d[j]).abs().sum(1) >= min_l1)
        res[k] = (i[ok], j[ok])
    return res


def slope(pred, true, i, j):
    dp, dt = (pred[i] - pred[j]).double(), (true[i] - true[j]).double()
    return float((dp * dt).sum() / (dt * dt).sum()) if len(i) else None


def analyse(net, data, K, quick=False):
    names = site_names(net.nl)
    acts, lg = activations(net, data)
    pacts, _ = activations(net, data, at_prefix=True)
    greedy = lg.argmax(1)
    dm, ret, pi0 = MO.model_occupancy(net, data.t, data.tok, data.prefix, data.goal, data.node, K=K)
    fit = data.fit
    tests = dict(held_evidence=data.test_ev, held_combination=data.test_combo)
    pm = data.predictors(dm, extra=pi0)                                 # P4: with the model's own action distribution at the reveal
    targets = dict(posterior=data.b, action_values=data.q, advantage=data.q - data.q.max(1, keepdim=True).values,
                   occupancy_solver=data.d, occupancy_model=dm,
                   residual_solver=(data.d.double() - data.exact["P3"]).float(), residual_model=(dm.double() - pm["P3"]).float(),
                   residual_solver_P2=(data.d.double() - data.exact["P2"]).float(),
                   residual_model_policy=(dm.double() - pm["P4"]).float())
    out = dict(model=dict(regret=float((pi0 * (data.q.max(1, keepdim=True).values - data.q)).sum(1).mean()),
                          **{f"regret_G{g + 1}": float((pi0 * (data.q.max(1, keepdim=True).values - data.q)).sum(1)[data.goal == g].mean()) for g in range(3)},
                          occupancy_gap_l1=float((dm - data.d).abs().sum(1).mean()),
                          **{f"occupancy_gap_G{g + 1}": float((dm - data.d).abs().sum(1)[data.goal == g].mean()) for g in range(3)}),
               exact={}, decode={}, prefix_decode={}, transfer={}, pairs={})
    for nm, d, pr in (("solver", data.d, data.exact), ("model", dm, pm)):
        var = ((d - d.mean(0)) ** 2).sum()
        out["exact"][nm] = {k: {tn: r2(v[ts], d[ts]) for tn, ts in tests.items()} for k, v in pr.items()}
        out["exact"][nm]["residual_share"] = float(((d.double() - pr["P3"]) ** 2).sum() / var)
        out["exact"][nm]["residual_share_P2"] = float(((d.double() - pr["P2"]) ** 2).sum() / var)
        if "P4" in pr:
            out["exact"][nm]["residual_share_P4"] = float(((d.double() - pr["P4"]) ** 2).sum() / var)
    probes = {}
    for tn_, Y in targets.items():
        out["decode"][tn_] = {}
        for i, n in enumerate(names):
            pr = Affine(acts[i][fit], Y[fit])
            if tn_ in ("occupancy_solver", "occupancy_model", "action_values", "posterior") and i == len(names) - 1:
                probes[tn_] = pr(acts[i])
            out["decode"][tn_][n] = {k: r2(pr(acts[i][ts]), Y[ts]) for k, ts in tests.items()}
    # the last prefix token: the posterior, and each goal's occupancy and action values (functions of b alone there)
    for tn_, Y in (("posterior", data.b), ("occupancy_solver", data.d), ("action_values", data.q)):
        out["prefix_decode"][tn_] = {}
        for i, n in enumerate(names):
            row = []
            for g in range(3):
                s = data.goal == g
                pr = Affine(pacts[i][fit & s], Y[fit & s])
                row.append(r2(pr(pacts[i][data.test_ev & s]), Y[data.test_ev & s]))
            out["prefix_decode"][tn_][n] = float(np.mean(row))
    # a decoder fitted without one goal (goal means removed), tested on it
    gm = torch.stack([acts[:, data.goal == g].mean(1) for g in range(3)], 1)
    for tn_ in ("posterior", "occupancy_solver", "action_values"):
        Y = targets[tn_]
        ym = torch.stack([Y[data.goal == g].mean(0) for g in range(3)])
        out["transfer"][tn_] = {}
        for i, n in enumerate(names):
            cen = acts[i] - gm[i][data.goal]
            yc = Y - ym[data.goal] if tn_ != "posterior" else Y
            row, own = [], []
            for g in range(3):
                f_, t_ = ~data.test_ev & (data.goal != g), data.test_ev & (data.goal == g)
                o_ = ~data.test_ev & (data.goal == g)
                row.append(r2(Affine(cen[f_], yc[f_])(cen[t_]), yc[t_])); own.append(r2(Affine(cen[o_], yc[o_])(cen[t_]), yc[t_]))
            out["transfer"][tn_][n] = dict(held_out_goal=float(np.mean(row)), own_goal=float(np.mean(own)))
    ps = pair_sets(data, greedy, 2000 if quick else 20000)
    for k, (i, j) in ps.items():
        out["pairs"][k] = dict(n=int(len(i)),
                               occupancy=dict(activations=slope(probes["occupancy_solver"], data.d, i, j),
                                              **{p: slope(data.exact[p], data.d, i, j) for p in ("P0", "P1", "P2", "P3")}),
                               occupancy_model=dict(activations=slope(probes["occupancy_model"], dm, i, j), **{p: slope(pm[p], dm, i, j) for p in ("P0", "P1", "P2", "P4")}),
                               action_values=dict(activations=slope(probes["action_values"], data.q, i, j)),
                               posterior=dict(activations=slope(probes["posterior"], data.b, i, j)) if k == "same_goal" else None,
                               true_l1=float((data.d[i] - data.d[j]).abs().sum(1).mean()) if len(i) else None,
                               model_tv=float(0.5 * (pi0[i] - pi0[j]).abs().sum(1).mean()) if len(i) else None)
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
    runs = [Path(r) for r in a.runs] if a.runs is not None else sorted((R18 / "runs").glob("*/seed*"))
    res = dict(rows=len(data.tok), fit=int(data.fit.sum()), held_evidence=int(data.test_ev.sum()), held_combination=int(data.test_combo.sum()),
               sites=site_names(4), runs={})
    for run in runs:
        cks = sorted((run / "ckpt").glob("u*.pt"))
        net = MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(a.device).eval()
        for label, ck in (("init", cks[0]), ("final", cks[-1])):
            net.load_state_dict(torch.load(ck))
            r = analyse(net, data, 4 if a.quick else 32, a.quick)
            res["runs"][f"{run.parent.name}/{run.name}/{label}"] = r
            print(run.parent.name, run.name, label, "regret", round(r["model"]["regret"], 4), "occ solver", round(max(v["held_combination"] for v in r["decode"]["occupancy_solver"].values()), 3),
                  "residual", round(max(v["held_combination"] for v in r["decode"]["residual_solver"].values()), 3), flush=True)
        (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
