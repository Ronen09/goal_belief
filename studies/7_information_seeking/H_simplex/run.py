"""H on the belief simplex: fits of the history profile H as functions of the exact posterior, the per-cell profiles of
the affine (QMDP-form) fit, and the fitted forms run in the additive code offline and online. Plan: PLAN.md.

    .venv/bin/python studies/7_information_seeking/H_simplex/run.py                       # writes results.json, fits.pt
    .venv/bin/python studies/7_information_seeking/H_simplex/run.py --untrained --seeds 0 --out DIR     # smoke test
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
M10 = HERE.parent / "maze10"
sys.path.insert(0, str(M10))
import measure as MS                                                  # noqa: E402  (the maze10 experiment's episodes, logits, G)
import train as TR                                                    # noqa: E402

N_EP, SB, STEP_BINS = 4096, MS.SB, 8


# ------------------------------------------------------------------ per-cell candidates (shortest paths only)

def cell_candidates(t):
    """popt [n, 4]: the share of goals for which the move is a shortest-path move from the cell; reach [n, 4]: the mean
    over goals of gamma^d(next(cell, a), goal)."""
    dn = t.dist[:, t.nxt_cell]                                                                 # [K, n, 4] distance after the move
    popt = (dn <= dn.min(-1, keepdim=True).values + 1e-6).float().mean(0)
    reach = (t.gamma ** dn).mean(0)
    return popt.double(), reach.double()


# ------------------------------------------------------------------ decisions of one set of episodes

@torch.no_grad()
def decisions(t, net, r, K):
    """Per live decision: the posterior, step, entropy, prefix features, and the centred logits under every goal."""
    N, H = r["act"].shape
    dev = t.dev
    L = []
    for st in range(H):
        tok = r["tok"][:, : st + 2].repeat(K, 1, 1)
        tok[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(N) + 1
        lg = net(tok)[0][:, st + 1].view(K, N, 4).transpose(0, 1).double()
        L.append(lg - lg.mean(-1, keepdim=True))
    lc = torch.stack(L, 1)                                                                     # [N, H, K, 4]
    ep, st = torch.nonzero(r["alive"], as_tuple=True)
    # prefix features: counts of each symbol and each move so far, last move, last symbol (one-hot)
    tok = r["tok"]
    sym = torch.nn.functional.one_hot(tok[..., MM.F_SYM], t.n_sym + 1)[..., 1:].double()       # [N, L, n_sym]
    mov = torch.nn.functional.one_hot(tok[..., MM.F_ACT], 5)[..., 1:].double()                 # [N, L, 4]
    csym, cmov = sym.cumsum(1), mov.cumsum(1)
    pos = st + 1                                                                               # token index of the decision
    feats = torch.cat([csym[ep, pos], cmov[ep, pos], mov[ep, pos], sym[ep, pos], (st[:, None].double() / H)], 1)
    return dict(ep=ep, st=st, b=r["bel"][ep, st].double(), ent=r["en"][ep, st].double(), cell=r["cell"][ep, st],
                goal=r["goal"][ep], lc=lc[ep, st], H=lc[ep, st].mean(1), prefix=feats)


# ------------------------------------------------------------------ fits

def step_onehot(st, H):
    return torch.nn.functional.one_hot((st * STEP_BINS // H).clamp(max=STEP_BINS - 1), STEP_BINS).double()


class Ridge:
    """Y ≈ X W + c on the fit rows, penalty chosen on a tenth of them (relative to the mean feature variance)."""

    def __init__(self, X, Y, va_mask, lams=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
        """va_mask [rows] bool: the validation rows (whole episodes), used only to choose the penalty."""
        X, Y = X.double(), Y.double()
        self.mu = X.mean(0); self.my = Y.mean(0)
        Xc, Yc = X - self.mu, Y - self.my
        n = len(X); idx = torch.arange(n, device=X.device)
        va, tr = idx[va_mask], idx[~va_mask]
        eye = torch.eye(X.shape[1], dtype=torch.float64, device=X.device)
        scale = (Xc ** 2).mean()

        def solve(rows, lam):
            A = Xc[rows].T @ Xc[rows]
            return torch.linalg.solve(A + lam * scale * len(rows) * eye, Xc[rows].T @ Yc[rows])

        self.lam = min(lams, key=lambda lam: float(((Xc[va] @ solve(tr, lam) - Yc[va]) ** 2).sum()))
        self.W = solve(idx, self.lam)

    def __call__(self, X):
        return (X.double() - self.mu) @ self.W + self.my


def r2(P, Y, Yfit_mean):
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Yfit_mean) ** 2).sum())


class MLP(torch.nn.Module):
    def __init__(self, d_in, d=64, d_out=4):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(d_in, d), torch.nn.GELU(), torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, d_out))

    def forward(self, x):
        return self.f(x)


def fit_mlp(X, Y, va_mask, seed=0, steps=3000, d=64, std_floor=1e-6):
    """A small MLP fitted by Adam with weight decay; early stopping on the validation episodes."""
    torch.manual_seed(seed)
    X, Y = X.float(), Y.float()
    n = len(X); idx = torch.arange(n, device=X.device); va, tr = idx[va_mask], idx[~va_mask]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=std_floor)                           # std_floor: for one-hot inputs with rare columns
    net = MLP(X.shape[1], d, Y.shape[1]).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_state = float("inf"), None
    gen = torch.Generator(device=X.device); gen.manual_seed(seed)
    with torch.enable_grad():
        for i in range(steps):
            rows = tr[torch.randint(len(tr), (4096,), device=X.device, generator=gen)]
            loss = ((net((X[rows] - mu) / sd) - Y[rows]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if i % 100 == 0 or i == steps - 1:
                with torch.no_grad():
                    v = float(((net((X[va] - mu) / sd) - Y[va]) ** 2).mean())
                if v < best:
                    best, best_state = v, {k: w.clone() for k, w in net.state_dict().items()}
    net.load_state_dict(best_state); net.eval()
    return lambda Z: net((Z.float() - mu) / sd).double()


def fit_all(t, d_fit, d_test, popt, reach, H_len):
    """Every model of H in the plan: held-out R² on d_test, and the predictors (functions of a decision dict)."""
    n = t.n
    S = lambda d: step_onehot(d["st"], H_len)
    B = lambda d: d["b"]
    sharp = lambda d: (d["b"] ** 2) / (d["b"] ** 2).sum(1, keepdim=True)
    P = lambda d: d["b"] @ popt
    R = lambda d: d["b"] @ reach
    mls = lambda d: torch.nn.functional.one_hot(d["b"].argmax(1), n).double()
    E = lambda d: d["ent"][:, None]
    forms = {
        "step": lambda d: S(d),
        "affine": lambda d: torch.cat([S(d), B(d)], 1),
        "popt": lambda d: torch.cat([S(d), P(d)], 1),
        "reach": lambda d: torch.cat([S(d), R(d)], 1),
        "mix": lambda d: torch.cat([S(d), P(d), R(d)], 1),
        "mls": lambda d: torch.cat([S(d), mls(d)], 1),
        "affine+sharp": lambda d: torch.cat([S(d), B(d), sharp(d)], 1),
        "affine+ent": lambda d: torch.cat([S(d), B(d), E(d), E(d) * B(d)], 1),
    }
    Y, Yt = d_fit["H"], d_test["H"]
    ym = Y.mean(0)
    va = d_fit["ep"] % 10 == 0                                                                 # validation: every tenth fit episode
    out, pred = {}, {}
    for name, f in forms.items():
        m = Ridge(f(d_fit), Y, va)
        pred[name] = (lambda f, m: lambda d: m(f(d)))(f, m)
        out[name] = r2(pred[name](d_test), Yt, ym)
        if name == "affine":
            out["affine_lambda"] = m.lam
            h_cells = m.W[STEP_BINS:]                                                          # [n, 4] per-cell profiles (centred rows)
    # the three-number mix: a single slope per term (alpha popt + beta reach), with step and action intercepts
    def mix3_features(d):                                                                      # rows (decision, action)
        m = len(d["st"]); act = torch.eye(4, dtype=torch.float64, device=d["b"].device).repeat(m, 1)
        return torch.cat([S(d).repeat_interleave(4, 0), act, P(d).reshape(-1, 1), R(d).reshape(-1, 1)], 1)
    m3 = Ridge(mix3_features(d_fit), Y.reshape(-1, 1), va.repeat_interleave(4))
    pred["mix3"] = lambda d: m3(mix3_features(d)).reshape(-1, 4)
    out["mix3"] = r2(pred["mix3"](d_test), Yt, ym)
    out["mix3_slopes"] = dict(popt=float(m3.W[-2, 0]), reach=float(m3.W[-1, 0]))
    # nonlinear ceilings
    f_mlp = lambda d: torch.cat([S(d), B(d)], 1)
    g = fit_mlp(f_mlp(d_fit), Y, va)
    pred["mlp"] = lambda d: g(f_mlp(d))
    out["mlp"] = r2(pred["mlp"](d_test), Yt, ym)
    f_hist = lambda d: torch.cat([S(d), B(d), d["prefix"]], 1)
    g2 = fit_mlp(f_hist(d_fit), Y, va, seed=1)
    pred["history"] = lambda d: g2(f_hist(d))
    out["history"] = r2(pred["history"](d_test), Yt, ym)
    f_hl = lambda d: torch.cat([S(d), B(d), d["prefix"]], 1)
    out["history_linear"] = r2(Ridge(f_hl(d_fit), Y, va)(f_hl(d_test)), Yt, ym)
    out["gain_nonlinear_in_b"] = out["mlp"] - out["affine"]
    out["gain_beyond_b"] = out["history"] - out["mlp"]
    return out, pred, h_cells


# ------------------------------------------------------------------ the per-cell profiles

def profiles(t, h_cells, popt, reach):
    hc = h_cells - h_cells.mean(1, keepdim=True)
    pc, rc = popt - popt.mean(1, keepdim=True), reach - reach.mean(1, keepdim=True)
    any_opt = (popt > 0)                                                                        # a shortest-path move for some goal
    top_ok = any_opt.gather(1, hc.argmax(1, keepdim=True)).squeeze(1)
    corr = lambda a, b: float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
    X = torch.stack([pc.flatten(), rc.flatten(), torch.ones(pc.numel(), dtype=torch.float64, device=pc.device)], 1)
    beta = torch.linalg.lstsq(X, hc.flatten()[:, None]).solution.squeeze(1)
    return dict(top_action_optimal_for_some_goal=float(top_ok.double().mean()), corr_popt=corr(hc, pc), corr_reach=corr(hc, rc),
                corr_mix=corr(hc, (X @ beta)), mix_beta=dict(popt=float(beta[0]), reach=float(beta[1])),
                h_cells=hc.cpu().tolist(), popt=pc.cpu().tolist(), reach=rc.cpu().tolist())


# ------------------------------------------------------------------ decisions with the fitted forms

def goal_bias(d_fit, K, H_len):
    """G [K, SB, 4]: the mean goal deviation per step bin (as in maze10), on the fit decisions."""
    dv = d_fit["lc"] - d_fit["H"][:, None]                                                     # [m, K, 4]
    sb = d_fit["st"].clamp(max=SB - 1)
    G = torch.zeros(K, SB, 4, dtype=torch.float64, device=dv.device)
    cnt = torch.zeros(SB, dtype=torch.float64, device=dv.device)
    G.index_add_(1, sb, dv.transpose(0, 1)); cnt.index_add_(0, sb, torch.ones_like(sb, dtype=torch.float64))
    return G / cnt.clamp(min=1)[None, :, None]


def offline(d, pred, G, names):
    """On decisions where the model's move depends on the goal: agreement of argmax(Ĥ + G) with the model's move."""
    lc, st = d["lc"], d["st"].clamp(max=SB - 1)
    nat = lc.argmax(-1)                                                                        # [m, K]
    matters = (nat != nat[:, :1]).any(1)
    Gt = G[:, st].permute(1, 0, 2)                                                             # [m, K, 4]
    own = nat[torch.arange(len(nat), device=nat.device), d["goal"]]
    out = dict(goal_matters_share=float(matters.double().mean()))
    cand = {"H": lambda dd: dd["H"], "goal_blind": None}
    for name in ["H", "goal_blind"] + names:
        if name == "goal_blind":
            a = d["H"][:, None].expand(-1, G.shape[0], -1).argmax(-1)
        else:
            Hh = d["H"] if name == "H" else pred[name](d)
            a = (Hh[:, None] + Gt).argmax(-1)
        out[name] = float((a == nat)[matters].double().mean())
    return out


def fitted_policy(pred_fn, G, t, H_len):
    def f(env, s):
        d = dict(st=torch.full((env.N,), s, device=t.dev), b=env.belief.double(), ent=BM.entropy(env.belief).double())
        Hh = pred_fn(d)
        return (Hh + G[env.goal, min(s, SB - 1)]).argmax(-1)
    return f


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--runs", default=str(M10 / "runs" / "ppo"))
    ap.add_argument("--out", default=None, help="output directory (default: here)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=N_EP)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    out_dir = Path(a.out) if a.out else HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = Path(a.runs)
    cfg = json.load(open(runs / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev)
    K, H_len = len(t.goal_cell), t.H
    n_ep = 512 if a.untrained else a.n
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BM.Env(t, n_ep, gen); goal, cell = e.goal, e.cell                                       # the maze10 evaluation spawns (first n_ep)
    gen.manual_seed(178)
    e = BM.Env(t, n_ep, gen); fgoal, fcell = e.goal, e.cell
    popt, reach = cell_candidates(t)
    res = dict(task=cfg["task"], cells=t.n, n_episodes=n_ep, runs={})
    fits = {}
    for s in a.seeds:
        d = runs / f"seed{s}" / "ckpt"
        ck = d / "u000000.pt" if a.untrained else sorted(d.glob("u*.pt"))[-1]
        args = cfg["args"]
        net = TR.build(t.n_sym, K, t.H, args.get("d", 128), args.get("layers", 4))
        net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        nat_fit, r_fit = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        nat, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        d_fit, d_test = decisions(t, net, r_fit, K), decisions(t, net, r, K)
        row = dict(checkpoint=ck.name, natural=nat, decisions_fit=len(d_fit["st"]), decisions_test=len(d_test["st"]))
        row["r2"], pred, h_cells = fit_all(t, d_fit, d_test, popt, reach, H_len)
        row["profiles"] = profiles(t, h_cells, popt, reach)
        G = goal_bias(d_fit, K, H_len)
        names = ["affine", "mix", "mix3", "mlp", "affine+ent", "mls"]
        row["offline"] = offline(d_test, pred, G, names)
        H_policy = lambda env, s_: (MS.logits_goals(net, env, s_).mean(1) + G[env.goal, min(s_, SB - 1)]).argmax(-1)
        row["online"] = dict(additive=MS.episodes(t, H_policy, goal, cell, 79)[0],
                             goal_blind=MS.episodes(t, lambda env, s_: MS.logits_goals(net, env, s_).mean(1).argmax(-1), goal, cell, 79)[0],
                             history_blind=MS.episodes(t, lambda env, s_: G[env.goal, min(s_, SB - 1)].argmax(-1), goal, cell, 79)[0])
        for name in ["affine", "mix", "mix3", "mlp"]:
            row["online"][name] = MS.episodes(t, fitted_policy(pred[name], G, t, H_len), goal, cell, 79)[0]
        rec = lambda x: (x - row["online"]["goal_blind"]["ret"]) / (nat["ret"] - row["online"]["goal_blind"]["ret"])
        row["recovery"] = {k: rec(v["ret"]) for k, v in row["online"].items()}
        res["runs"][f"seed{s}"] = row
        fits[f"seed{s}"] = dict(h_cells=h_cells.cpu(), G=G.cpu())
        q = row["r2"]
        print(f"seed{s} {ck.name} natural {nat['ret']:.3f} | R2 step {q['step']:.3f} affine {q['affine']:.3f} popt {q['popt']:.3f} reach {q['reach']:.3f} "
              f"mix {q['mix']:.3f} mix3 {q['mix3']:.3f} mls {q['mls']:.3f} +sharp {q['affine+sharp']:.3f} +ent {q['affine+ent']:.3f} mlp {q['mlp']:.3f} "
              f"history {q['history']:.3f} (linear {q['history_linear']:.3f}) | profiles top-ok {row['profiles']['top_action_optimal_for_some_goal']:.2f} "
              f"corr mix {row['profiles']['corr_mix']:.2f} | offline H {row['offline']['H']:.3f} affine {row['offline']['affine']:.3f} mix {row['offline']['mix']:.3f} "
              f"mlp {row['offline']['mlp']:.3f} blind {row['offline']['goal_blind']:.3f} | recovery additive {row['recovery']['additive']:.2f} affine {row['recovery']['affine']:.2f} "
              f"mix {row['recovery']['mix']:.2f} mix3 {row['recovery']['mix3']:.2f} mlp {row['recovery']['mlp']:.2f} | {time.time() - t0:.0f}s", flush=True)
        (out_dir / ("smoke.json" if a.untrained else "results.json")).write_text(json.dumps(res))
        torch.save(fits, out_dir / ("smoke_fits.pt" if a.untrained else "fits.pt"))


if __name__ == "__main__":
    main()
