"""The cross-history-MLP experiment: what do the goal token's MLPs carry? A cross-history patch of MLP outputs, the observation-prediction experiment's reward models, frozen.

    .venv/bin/python studies/5_goal_belief_mechanism/cross_history_mlp/run.py            # writes results.json

Cross-goal cells (the brief's test). Donor: history A under goal g', where the model chooses a_A' (optimal). Recipient:
history B under goal g, where it chooses a_B (optimal); under g' it would choose a_B' (optimal). a_A', a_B, a_B' are
three different actions. The donor's MLP output at the goal token replaces the recipient's. The patched choice is
    a_B'   B's own action under the donor's goal      the output carries a goal instruction
    a_A'   the donor's action                         it carries the donor's action preference (or A's evidence with g')
    a_B    the recipient's action                     it is overridden by the rest of the recipient's computation
Same-goal cells (control): donor A under the recipient's goal g, choosing a_A != a_B. A goal instruction changes
nothing (retain a_B); an action preference, or A's evidence combined with g, gives a_A.
Representation (no patch): each MLP's output at the goal token, regressed on goal x chosen action, on goal x posterior,
and on both; held-out R².
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazemodel as MM
from goalgeo.navprobe import Affine

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


R30 = load("r30run", ROUNDS / "5_goal_belief_mechanism" / "goal_swap_components" / "run.py")
M26 = R30.M26
CHUNK, N_CELLS, LENGTHS, SEED = 8192, 20000, (1, 2, 3, 4), 31
R30_SELECTED = json.load(open(ROUNDS / "5_goal_belief_mechanism" / "goal_swap_components" / "results.json"))["runs"]


@torch.no_grad()
def run_all(net, ctx, hist, goal):
    """Greedy action, logits and record at the goal token for histories under goals (vectors)."""
    out_lg, mlps = [], [[] for _ in range(net.nl)]
    for s in range(0, len(hist), CHUNK):
        h, g = hist[s:s + CHUNK], goal[s:s + CHUNK]
        n = torch.arange(len(h), device=h.device); pos = 1 + ctx.L[h]
        lg, _, rec = net(R30.seq(ctx, h, g), record=True)
        out_lg.append(lg[n, pos])
        for l in range(net.nl):
            mlps[l].append(rec["mlp"][l][n, pos])
    return torch.cat(out_lg), [torch.cat(m) for m in mlps]


def greedy_ok(ctx, hist, goal, lg):
    a = lg.argmax(-1)
    return a, ctx.opt[hist, goal][torch.arange(len(hist), device=hist.device), a]


class Cells:
    """Model-specific cells from held-out histories of the same prefix length (the goal token at the same position)."""

    def __init__(self, net, ctx):
        dev = ctx.t.dev
        g = torch.Generator(device=dev); g.manual_seed(SEED)
        pool = torch.nonzero(ctx.test & (ctx.L >= LENGTHS[0])).squeeze(1)
        M = 600000
        B = pool[torch.randint(len(pool), (M,), device=dev, generator=g)]
        A = pool[torch.randint(len(pool), (M,), device=dev, generator=g)]
        gr = torch.randint(3, (M,), device=dev, generator=g)
        gd = (gr + 1 + torch.randint(2, (M,), device=dev, generator=g)) % 3
        ok = (ctx.L[A] == ctx.L[B]) & (A != B)
        A, B, gr, gd = A[ok], B[ok], gr[ok], gd[ok]
        aB, okB = greedy_ok(ctx, B, gr, run_all(net, ctx, B, gr)[0])
        aB2, okB2 = greedy_ok(ctx, B, gd, run_all(net, ctx, B, gd)[0])
        aA2, okA2 = greedy_ok(ctx, A, gd, run_all(net, ctx, A, gd)[0])
        aA, okA = greedy_ok(ctx, A, gr, run_all(net, ctx, A, gr)[0])
        cross = okB & okB2 & okA2 & (aB != aB2) & (aA2 != aB) & (aA2 != aB2)
        same = okB & okA & (aA != aB)
        i = torch.nonzero(cross).squeeze(1)[:N_CELLS]
        j = torch.nonzero(same).squeeze(1)[:N_CELLS]
        self.cross = dict(A=A[i], B=B[i], gr=gr[i], gd=gd[i], aB=aB[i], aB2=aB2[i], aA2=aA2[i])
        self.same = dict(A=A[j], B=B[j], gr=gr[j], aB=aB[j], aA=aA[j])
        self.candidates = int(ok.sum())


@torch.no_grad()
def patched(net, ctx, B, gr, A, gd, layers):
    """The recipient (B, gr) with the MLP outputs of `layers` at the goal token taken from the donor (A, gd)."""
    out = []
    for s in range(0, len(B), CHUNK):
        b, a, g1, g2 = B[s:s + CHUNK], A[s:s + CHUNK], gr[s:s + CHUNK], gd[s:s + CHUNK]
        n = torch.arange(len(b), device=b.device); pos = 1 + ctx.L[b]
        rec = net(R30.seq(ctx, a, g2), record=True)[2]
        patch = R30.make_patch([("mlp", l) for l in layers], rec, n, pos)
        out.append(net(R30.seq(ctx, b, g1), patch=patch)[0][n, pos])
    return torch.cat(out)


def outcome(lg, named):
    """Greedy shares and probability on each named action."""
    a, p = lg.argmax(-1), lg.softmax(-1)
    rows = torch.arange(len(a), device=a.device)
    out = {f"greedy_{k}": float((a == v).float().mean()) for k, v in named.items()}
    hit = torch.zeros_like(a, dtype=torch.bool)
    for v in named.values():
        hit |= a == v
    out["greedy_other"] = float((~hit).float().mean())
    out.update({f"prob_{k}": float(p[rows, v].mean()) for k, v in named.items()})
    return out


@torch.no_grad()
def representation(net, ctx):
    """Held-out R² of each MLP's goal-token output from goal x greedy action, goal x posterior, and both."""
    nh = len(ctx.L)
    hist = torch.arange(nh, device=ctx.t.dev).repeat(3)
    goal = torch.arange(3, device=ctx.t.dev).repeat_interleave(nh)
    lg, mlps = run_all(net, ctx, hist, goal)
    act = lg.argmax(-1)
    G = torch.nn.functional.one_hot(goal, 3).double()
    GA = (G[:, :, None] * torch.nn.functional.one_hot(act, 4).double()[:, None]).flatten(1)
    GB = (G[:, :, None] * ctx.t.belief[ctx.node[hist]].double()[:, None]).flatten(1)
    feats = dict(goal=G, goal_x_action=GA, goal_x_posterior=GB, both=torch.cat([GA, GB], 1))
    fit, test = ctx.fit[hist], ctx.test[hist]
    out = {}
    for l, Y in enumerate(mlps):
        Y = Y.double()
        var = ((Y[test] - Y[test].mean(0)) ** 2).sum()
        out[f"L{l}.mlp"] = {k: float(1 - ((Y[test] - Affine(F[fit], Y[fit])(F[test])) ** 2).sum() / var) for k, F in feats.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    res = dict(runs={})
    for s in a.seeds:
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs"); net.eval()
        cells = Cells(net, ctx)
        c, q = cells.cross, cells.same
        sel = [int(x[1]) for x in R30_SELECTED[f"seed{s}"]["selected"]]
        sets = {"L0.mlp": [0], "L1.mlp": [1], "L2.mlp": [2], "L3.mlp": [3], "selected_r30": sel, "mlps_1_3": [1, 2, 3]}
        row = dict(checkpoint=ck, n_cross=len(c["B"]), n_same=len(q["B"]), candidates=cells.candidates, selected_r30=sel, cross={}, same={}, within={})
        row["cross"]["none"] = outcome(run_all(net, ctx, c["B"], c["gr"])[0], dict(donor_action=c["aA2"], own_under_donor_goal=c["aB2"], recipient=c["aB"]))
        for k, layers in sets.items():
            row["cross"][k] = outcome(patched(net, ctx, c["B"], c["gr"], c["A"], c["gd"], layers), dict(donor_action=c["aA2"], own_under_donor_goal=c["aB2"], recipient=c["aB"]))
            row["within"][k] = outcome(patched(net, ctx, c["B"], c["gr"], c["B"], c["gd"], layers), dict(donor_action=c["aA2"], own_under_donor_goal=c["aB2"], recipient=c["aB"]))
            row["same"][k] = outcome(patched(net, ctx, q["B"], q["gr"], q["A"], q["gr"], layers), dict(donor_action=q["aA"], recipient=q["aB"]))
        row["representation"] = representation(net, ctx)
        res["runs"][f"seed{s}"] = row
        x = row["cross"]
        print(f"seed{s} cells {row['n_cross']}/{row['n_same']} | cross: " + " | ".join(f"{k} donor {x[k]['greedy_donor_action']:.2f} own {x[k]['greedy_own_under_donor_goal']:.2f} keep {x[k]['greedy_recipient']:.2f}" for k in sets) +
              f" | same-goal mlps_1_3 donor {row['same']['mlps_1_3']['greedy_donor_action']:.2f} | {time.time() - t0:.0f}s", flush=True)
        (HERE / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
