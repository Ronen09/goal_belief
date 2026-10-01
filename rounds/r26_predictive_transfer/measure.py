"""Round 26: representations of four backbones on the same held-out histories, and the transfer of each frozen
backbone to a small goal-conditioned head trained on optimal actions.

    .venv/bin/python rounds/r26_predictive_transfer/measure.py            # writes results.json

Backbones (seeds 0-9 each; a seed is the same initialisation in every condition):
    predict1  next symbol after each supplied move, goal-free random walks (this round)
    predict2  the two symbols after each supplied pair of moves (this round; secondary)
    reward    round 23's reward-only PPO models, final checkpoint
    random    the same initialisations, untrained
Histories: round 22's bank (fit side for anything fitted, held-out side for every number) and round 22's pairs.
The representation is read at the last prefix token (goal-free in every backbone), entering each block and after
the last. The decision is the first one after the reveal, under each of the three goals (12 moves left).
"""

from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazeaux as AX, mazemeasure as MS, mazemodel as MM, mazepred as PR
from goalgeo.navprobe import Affine

torch.backends.cuda.matmul.allow_tf32 = False
HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent
R18, R23 = ROUNDS / "r18_maze_belief", ROUNDS / "r23_obs_prediction"
sys.path.insert(0, str(R18))
import train as T18                                                  # noqa: E402

spec = importlib.util.spec_from_file_location("r22run", ROUNDS / "r22_pair_types" / "run.py")
R22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R22)

BANK_SEED, N_BANK = 220022, 200000                                   # round 22's bank (R22.Data builds the same one)
CONDS = ("predict1", "predict2", "reward", "random")
NS = (30, 100, 300, 1000, 3000, 10000, 30000, 100000)
HEAD_SEEDS = (0, 1, 2)
HIDDEN = (16, 0)                                                     # primary: one hidden layer of 16; secondary: linear
STEPS, BATCH, LR = 4000, 512, 1e-3
SPLIT_SEED, EXAMPLE_SEED = 2626, 26


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def load(cond, seed, t, dev, runs):
    if cond in ("predict1", "predict2"):
        net = PR.NetPred(t.n_sym, len(t.goal_cell), t.H, t.max_prefix, k=int(cond[-1]))
        ck = sorted((runs / cond / f"seed{seed}" / "ckpt").glob("u*.pt"))[-1]
    else:
        net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix)
        d = R23 / "runs" / "reward" / f"seed{seed}" / "ckpt"
        ck = d / "u000000.pt" if cond == "random" else sorted(d.glob("u*.pt"))[-1]
    net.load_state_dict(torch.load(ck, map_location=dev))
    return net.to(dev).eval(), ck.name


def check_same_init(t, dev, seeds):
    """A seed's untrained backbone is the same in every condition (round 23's initial checkpoint = NetPred's init)."""
    for s in seeds:
        torch.manual_seed(s)
        fresh = PR.NetPred(t.n_sym, len(t.goal_cell), t.H, t.max_prefix, k=1).state_dict()
        init = torch.load(R23 / "runs" / "reward" / f"seed{s}" / "ckpt" / "u000000.pt", map_location="cpu")
        bad = [k for k in init if not k.startswith(("obs.",)) and not torch.equal(init[k], fresh[k].cpu())]
        assert not bad, f"seed {s}: initialisation differs in {bad[:3]}"


class Ctx:
    def __init__(self, t, quick=False):
        self.t = t
        dev = t.dev
        self.data = R22.Data(t, quick)
        self.bank, fit = MS.make_bank(t, BANK_SEED, 8000 if quick else N_BANK)
        assert torch.equal(self.bank.tok, self.data.bank.tok)
        b = self.bank
        nh = len(b.prefix)
        self.L = b.prefix
        self.node = b.pre_node[torch.arange(nh, device=dev), b.prefix]                    # posterior at the last prefix token
        self.fit, self.test = fit, ~fit
        self.test_hist = torch.nonzero(self.test).squeeze(1)
        self.inv = torch.full((nh,), -1, dtype=torch.long, device=dev); self.inv[self.test_hist] = torch.arange(len(self.test_hist), device=dev)
        g = torch.Generator(device=dev); g.manual_seed(SPLIT_SEED)
        self.select = torch.rand(len(self.test_hist), device=dev, generator=g) < 0.5        # selection half / report half
        # decisions at the reveal for every history and goal
        gg = torch.arange(3, device=dev)
        self.Q = t.Q[t.reveal[self.node[:, None], gg[None]]]                               # [histories, goals, 4]
        self.V = self.Q.max(-1).values
        self.opt = self.Q >= self.V[..., None] - 1e-6
        det = json.load(open(HERE / "checks.json"))["determined_1step"] if not quick else None
        self.det1 = (torch.stack([torch.tensor(det[f"G{j + 1}"], device=dev)[self.node] for j in range(3)], 1) if det else
                     torch.ones(nh, 3, dtype=torch.bool, device=dev))                       # optimal set fixed by the one-step class
        # labelled examples: fit-side histories in a fixed random order, each with a random goal; nested in N
        fit_hist = torch.nonzero(fit).squeeze(1)
        self.ex_hist, self.ex_goal = [], []
        for j in HEAD_SEEDS:
            g.manual_seed(EXAMPLE_SEED + j)
            self.ex_hist.append(fit_hist[torch.randperm(len(fit_hist), device=dev, generator=g)])
            self.ex_goal.append(torch.randint(3, (len(fit_hist),), device=dev, generator=g))
        self.ns = [n for n in NS if n <= len(fit_hist)] if not quick else [30, 300, 3000]
        # exact references at the last prefix token
        bel = t.belief[self.node].double()
        self.ref_features = dict(goal_only=torch.zeros(nh, 0, device=dev), posterior=bel.float(), predict1_exact=PR.predictive(t, bel, 1).flatten(1).float(),
                                 predict2_exact=PR.predictive(t, bel, 2).flatten(1).float(), raw_history=self.raw(b))

    def raw(self, b):
        """One-hot of the prefix tokens (symbol and move at positions 0..4) and of the prefix length."""
        t = self.t
        mp = t.max_prefix
        keep = (torch.arange(mp + 1, device=t.dev)[None] <= b.prefix[:, None])
        s = torch.nn.functional.one_hot(b.tok[:, : mp + 1, MM.F_SYM] * keep, t.n_sym + 1)
        a = torch.nn.functional.one_hot(b.tok[:, : mp + 1, MM.F_ACT] * keep, 5)
        return torch.cat([s.flatten(1), a.flatten(1), torch.nn.functional.one_hot(b.prefix, mp + 1)], 1).float()


# ------------------------------------------------------------------ stage 1: representations

@torch.no_grad()
def representation(net, cond, ctx, F):
    """F [sites, histories, d]: the residual stream at the last prefix token."""
    t, b, data = ctx.t, ctx.bank, ctx.data
    bel = t.belief[ctx.node].double()
    targets = dict(posterior=bel, predict1=PR.predictive(t, bel, 1).flatten(1), predict2=PR.predictive(t, bel, 2).flatten(1))
    fit, test = ctx.fit, ctx.test
    A, C = data.types["A"], data.types["C"]
    out = {}
    for l in range(len(F)):
        X = F[l]
        row = {}
        for name, Y in targets.items():
            pr = Affine(X[fit], Y[fit])
            row[f"r2_{name}"] = MS.belief_scores(pr(X[test]), Y[test])["r2"]
            if name == "posterior":
                dec = pr(X)
        Z = PR.layer_norm(X)
        d2 = lambda v: ((Z[v["r"]] - Z[v["d"]]) ** 2).sum(-1).mean()
        row["consistency_ratio"] = float(d2(A) / d2(C))                  # identical posteriors against different ones; 0 = identical states
        l1 = lambda v: (dec[v["r"]] - dec[v["d"]]).abs().sum(-1).mean()
        row["decoded_consistency_ratio"] = float(l1(A) / l1(C))
        out[l] = row
    res = dict(sites=out)
    if cond.startswith("predict"):                                       # the predictor's own head, at the last prefix token
        k = int(cond[-1])
        p = torch.cat([net(b.tok[s:s + 8192], pred=True)[torch.arange(len(b.tok[s:s + 8192]), device=t.dev), b.prefix[s:s + 8192]].softmax(-1)
                       for s in range(0, len(b.tok), 8192)])                 # [histories, seq, out]
        ex = PR.predictive(t, bel, k).float()
        p1 = p if k == 1 else p.view(-1, 4, 4, t.n_sym, t.n_sym).sum(-1).mean(2)
        ex1 = PR.predictive(t, bel, 1).float()
        res["head"] = dict(error=float(tv(p, ex)[test].mean()), error_1step=float(tv(p1, ex1)[test].mean()),
                           d_same=float(tv(p1[A["r"]], p1[A["d"]]).mean()), d_diff=float(tv(p1[C["r"]], p1[C["d"]]).mean()),
                           error_by_length={int(L): float(tv(p1, ex1)[test & (ctx.L == L)].mean()) for L in torch.unique(ctx.L)})
    if cond == "reward":                                                 # the model's own policy at the reveal, same decisions
        h = ctx.test_hist
        reg = []
        for g in range(3):
            lg = torch.cat([net(data.seq(h[s:s + 8192], ctx.L[h[s:s + 8192]], g))[0][torch.arange(len(h[s:s + 8192]), device=t.dev), 1 + ctx.L[h[s:s + 8192]]]
                            for s in range(0, len(h), 8192)])
            reg.append(ctx.V[h, g] - ctx.Q[h, g].gather(1, lg.argmax(1, keepdim=True)).squeeze(1))
        reg = torch.stack(reg, 1)
        res["own_policy"] = dict(regret_report=float(reg[~ctx.select].mean()), regret_select=float(reg[ctx.select].mean()),
                                 by_goal=[float(reg[~ctx.select, g].mean()) for g in range(3)])
    return res


# ------------------------------------------------------------------ stage 2: heads on frozen features

@torch.no_grad()
def evaluate(lg, ctx):
    """lg [K, test histories, goals, 4] -> per head: regret measures on the report half (and the selection half)."""
    h = ctx.test_hist
    Q, V, opt = ctx.Q[h], ctx.V[h], ctx.opt[h]
    a = lg.argmax(-1)                                                    # greedy
    reg = V[None] - Q[None].expand(len(lg), -1, -1, -1).gather(3, a[..., None]).squeeze(3)          # [K, n, goals]
    p = lg.softmax(-1)
    ereg = (p * (V[..., None] - Q)[None]).sum(-1)
    rep, sel = ~ctx.select, ctx.select
    L = ctx.L[h]
    det = ctx.det1[h]
    m = lambda x, w: ((x * w).sum((1, 2)) / w.sum().clamp(min=1)).tolist()
    W = lambda mask: mask[:, None].float().expand(-1, 3)
    out = dict(regret=m(reg, W(rep)), regret_select=m(reg, W(sel)), expected_regret=m(ereg, W(rep)),
               optimal=m(opt[None].float().expand(len(lg), -1, -1, -1).gather(3, a[..., None]).squeeze(3), W(rep)),
               by_goal=[[float(x) for x in (reg[:, rep, g]).mean(1)] for g in range(3)],
               by_length={int(l): m(reg, W(rep & (L == l))) for l in torch.unique(L)},
               determined=m(reg, W(rep) * det), undetermined=m(reg, W(rep) * ~det))
    # identical-posterior consistency of the head's action distribution (type A) against different posteriors (type C)
    for name in ("A", "C"):
        v = ctx.data.types[name]
        i, j = ctx.inv[v["r"]], ctx.inv[v["d"]]
        out[f"tv_{name}"] = tv(p[:, i], p[:, j]).mean((1, 2)).tolist()
    return out


def run_heads(X, xmap, ctx, hidden, n):
    K = len(xmap)
    nb = K // len(HEAD_SEEDS)
    hs = torch.tensor(HEAD_SEEDS, device=X.device).repeat_interleave(nb)                         # head j: backbone xmap[j], head seed hs[j]
    ex_hist = torch.stack(ctx.ex_hist)                                                             # [head seeds, E]
    ex_goal = torch.stack(ctx.ex_goal)
    E = ex_hist.shape[1]
    off = (torch.arange(len(HEAD_SEEDS), device=X.device) * E)
    idx = (off[hs][:, None] + torch.arange(n, device=X.device)[None])                              # the first n examples of each head seed
    flat_h, flat_g = ex_hist.flatten(), ex_goal.flatten()
    hd = PR.train_heads(X, xmap, flat_h, flat_g, ctx.opt[flat_h, flat_g], idx, hidden, [100 * int(s) + hidden for s in hs.tolist()], STEPS, BATCH, LR)
    return evaluate(PR.apply_heads(hd, X, xmap, ctx.test_hist), ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    global STEPS
    if a.quick:
        a.seeds, STEPS = a.seeds[:2], 300
    dev, runs, t0 = a.device, Path(a.runs), time.time()
    t = T18.tables(dev)
    check_same_init(t, dev, a.seeds)
    ctx = Ctx(t, a.quick)
    res = dict(types=ctx.data.info, ns=ctx.ns, head_seeds=list(HEAD_SEEDS), hidden=list(HIDDEN), steps=STEPS, batch=BATCH, lr=LR,
               n_test=int(len(ctx.test_hist)), n_select=int(ctx.select.sum()), backbones={}, heads={}, references={})
    feats, keys = [], []
    for cond in CONDS:
        if cond.startswith("predict") and not (runs / cond).exists():
            continue
        for s in a.seeds:
            net, ck = load(cond, s, t, dev, runs)
            F = PR.features(net, ctx.bank.tok, ctx.bank.prefix)                                    # [sites, histories, d]
            res["backbones"].setdefault(cond, {})[f"seed{s}"] = dict(checkpoint=ck, **representation(net, cond, ctx, F))
            feats.append(F.half()); keys.append((cond, s))
            r = res["backbones"][cond][f"seed{s}"]["sites"]
            print(f"{cond} seed{s} posterior R² by site " + " ".join(f"{r[l]['r2_posterior']:.3f}" for l in r) +
                  " | consistency " + " ".join(f"{r[l]['consistency_ratio']:.3f}" for l in r) + f" | {time.time() - t0:.0f}s", flush=True)
        (Path(a.out) / "results.json").write_text(json.dumps(res))
    nb = len(keys)
    xmap = torch.arange(nb, device=dev).repeat(len(HEAD_SEEDS))
    for hidden in HIDDEN:
        # references: exact features of the history and the raw tokens, through the same head
        for name, Fr in ctx.ref_features.items():
            Xr = (PR.layer_norm(Fr) if Fr.shape[1] > 1 else Fr)[None]
            for n in ctx.ns:
                ev = run_heads(Xr, torch.zeros(len(HEAD_SEEDS), dtype=torch.long, device=dev), ctx, hidden, n)
                res["references"].setdefault(f"h{hidden}", {}).setdefault(name, {})[n] = ev
            print(f"reference {name} hidden {hidden}: regret by N " + " ".join(f"{np.mean(res['references'][f'h{hidden}'][name][n]['regret']):.4f}" for n in ctx.ns), flush=True)
        for l in range(feats[0].shape[0]):
            X = torch.stack([PR.layer_norm(f[l].float()) for f in feats])                         # [backbones, histories, d]
            for n in ctx.ns:
                ev = run_heads(X, xmap, ctx, hidden, n)
                for i, (cond, s) in enumerate(keys):
                    pick = lambda v: [v[i + j * nb] for j in range(len(HEAD_SEEDS))] if isinstance(v, list) and len(v) == len(xmap) else v
                    row = {k: (pick(v) if not isinstance(v, dict) else {kk: pick(vv) for kk, vv in v.items()}) for k, v in ev.items() if k != "by_goal"}
                    row["by_goal"] = [[g[i + j * nb] for j in range(len(HEAD_SEEDS))] for g in ev["by_goal"]]
                    res["heads"].setdefault(f"h{hidden}", {}).setdefault(cond, {}).setdefault(f"seed{s}", {}).setdefault(l, {})[n] = row
            del X
            torch.cuda.empty_cache()
            med = {c: np.median([np.mean([np.mean(res['heads'][f'h{hidden}'][c][f'seed{s}'][l][n]['regret']) for n in ctx.ns]) for s in a.seeds])
                   for c in res["heads"][f"h{hidden}"]}
            print(f"hidden {hidden} site {l}: median AUC " + " ".join(f"{c} {v:.4f}" for c, v in med.items()) + f" | {time.time() - t0:.0f}s", flush=True)
            (Path(a.out) / "results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
