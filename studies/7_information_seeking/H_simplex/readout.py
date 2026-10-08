"""Is the encoded posterior reusable for new decisions? New decision tasks: the QMDP-optimal move under the exact
posterior for 6 new goal cells (the interior-goals experiment's junction goals) and for a new rule (toward the nearest
landmark). Readouts (linear, MLP) from the frozen decision-token state (goal-averaged; sites 2 and 4) to each task;
baselines from the exact posterior, from H, from the 16 current logits, and the majority move. The matched-preference
subset: test decisions paired with another of the same step, the same own-goal move and nearly the same H whose
new optimal moves are disjoint. Causal: on causal.py's matched pairs, the posterior-part / residual-part / random swap
at site 2 propagated to site 4, read by the new readouts: do the new decisions transfer to the donor's?

    .venv/bin/python studies/7_information_seeking/H_simplex/readout.py        # writes readout.md / readout.json
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C          # noqa: E402

SITES = (2, 4)
N_FIT, N_TEST = 24576, 8192


def new_tasks(t):
    """Per-cell fully observed values Qcell [T, n, 4] of the new tasks, and their names."""
    maze = t.maze
    junc = BM.make(**{**json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))["task"], "junctions": 6})
    goals = [g for g in junc.goals if g not in maze.goals]
    names, Q = [], []
    for g in goals:
        d = torch.tensor(BM.distances(maze.nxt, g), device=t.dev).float()
        Q.append(t.gamma ** d[t.nxt_cell]); names.append(f"goal@{maze.cells[g]}")
    dl = torch.stack([torch.tensor(BM.distances(maze.nxt, l), device=t.dev).float() for l in maze.landmarks])
    Q.append((t.gamma ** dl[:, t.nxt_cell]).max(0).values); names.append("nearest landmark")
    return torch.stack(Q).double(), names


def opt_sets(b, Qcell):
    """[m, T, 4] optimal-move sets under the belief for every task (QMDP: belief-weighted value)."""
    v = torch.einsum("ms,tsa->mta", b, Qcell)
    return v >= v.max(-1, keepdim=True).values - 1e-6


def fit_readout(X, sets, va, kind, seed=0):
    """A readout to the optimal-move set (loss: -log of the set's softmax mass). kind 'linear' or 'mlp'. Returns f(X) -> logits."""
    torch.manual_seed(seed)
    X = X.float(); n = len(X); idx = torch.arange(n, device=X.device); tr, vv = idx[~va], idx[va]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-2)
    net = (torch.nn.Linear(X.shape[1], 4) if kind == "linear" else R.MLP(X.shape[1], 64, 4)).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = 2000; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=X.device); gen.manual_seed(seed)
    best, state = float("inf"), None
    loss_of = lambda rows: -(torch.logsumexp(torch.log_softmax(net((X[rows] - mu) / sd), -1).masked_fill(~sets[rows], -1e9), -1)).mean()
    with torch.enable_grad():
        for i in range(steps):
            rows = tr[torch.randint(len(tr), (4096,), device=X.device, generator=gen)]
            loss = loss_of(rows); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if i % 100 == 0 or i == steps - 1:
                with torch.no_grad():
                    v = float(loss_of(vv))
                if v < best:
                    best, state = v, {k: w.clone() for k, w in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    return lambda Z: net((Z.float() - mu) / sd)


def hit(logits, sets):
    return sets.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)


@torch.no_grad()
def run_edit_state(net, tok, p, K, site, new_state, read_site):
    """As causal.run_edit, but returns the goal-averaged residual at read_site [N, d] after the edit at site."""
    N, dev = len(tok), tok.device
    out = []
    for i in range(0, N, C.CHUNK):
        tk = tok[i:i + C.CHUNK].repeat(K, 1, 1); pp = p[i:i + C.CHUNK].repeat(K); n_ = len(tok[i:i + C.CHUNK])
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
        ns = new_state[i:i + C.CHUNK].transpose(0, 1).reshape(K * n_, -1)
        ar = torch.arange(len(tk), device=dev)

        def patch(x):
            x = x.clone(); x[ar, pp] = ns.to(x.dtype); return x
        _, _, rec = net(tk, record=True, patch={("resid", site): patch})
        out.append(rec["resid"][read_site][ar, pp].view(K, n_, -1).mean(0))
    return torch.cat(out).double()


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len, n = len(t.goal_cell), t.H, t.n
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    Qcell, names = new_tasks(t); T = len(names)
    res = dict(tasks=names, runs={})
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        gsub = torch.Generator(device=dev); gsub.manual_seed(1)
        fi = torch.randperm(len(df["st"]), device=dev, generator=gsub)[:N_FIT]
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:N_TEST]
        Xf, _ = C.states(net, rf["tok"][df["ep"][fi]], df["st"][fi] + 1, K, SITES)
        Xt, _ = C.states(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K, SITES)
        S = lambda d, idx: R.step_onehot(d["st"][idx], H_len)
        va = df["ep"][fi] % 10 == 0
        sets_f, sets_t = opt_sets(df["b"][fi], Qcell), opt_sets(dt["b"][ti], Qcell)
        feats = {"state_site2": (Xf[2].mean(1), Xt[2].mean(1)), "state_site4": (Xf[4].mean(1), Xt[4].mean(1)),
                 "posterior": (df["b"][fi], dt["b"][ti]), "H": (df["H"][fi], dt["H"][ti]),
                 "logits_all_goals": (df["lc"][fi].flatten(1), dt["lc"][ti].flatten(1))}
        feats = {k: (torch.cat([S(df, fi), a.double()], 1), torch.cat([S(dt, ti), b_.double()], 1)) for k, (a, b_) in feats.items()}
        # the matched-preference subset among test decisions: same step, same own-goal move, nearly the same H, disjoint new sets
        own = dt["lc"][ti].argmax(-1)[torch.arange(len(ti), device=dev), dt["goal"][ti]]
        Ht = dt["H"][ti]; stt = dt["st"][ti]
        gp = torch.Generator(device=dev); gp.manual_seed(3)
        perm = torch.randperm(len(ti), device=dev, generator=gp)
        dist = (Ht - Ht[perm]).norm(dim=1); thr = torch.quantile(dist, 0.25)
        # a greedy pass: for each decision, a partner among 64 random candidates
        cand = torch.randint(len(ti), (len(ti), 64), device=dev, generator=gp)
        dH = (Ht[:, None] - Ht[cand]).norm(dim=-1)
        ok = (stt[:, None] == stt[cand]) & (own[:, None] == own[cand]) & (dH < thr) & (cand != torch.arange(len(ti), device=dev)[:, None])
        matched = {}
        for j in range(T):
            disjoint = ~(sets_t[:, j][:, None] & sets_t[cand, j]).any(-1)
            m = (ok & disjoint).any(1)
            matched[j] = m
        row = dict(checkpoint=ck.name, majority={}, readout={}, matched_share={names[j]: float(matched[j].double().mean()) for j in range(T)})
        readers = {}
        for fk, (Xa, Xb) in feats.items():
            for kind in ("linear", "mlp"):
                if fk in ("H", "logits_all_goals", "posterior") and kind == "linear" and fk != "posterior":
                    pass
                accs, accs_m = [], []
                for j in range(T):
                    f = fit_readout(Xa, sets_f[:, j], va, kind, seed=j)
                    h = hit(f(Xb), sets_t[:, j])
                    accs.append(float(h.double().mean())); accs_m.append(float(h[matched[j]].double().mean()) if matched[j].any() else None)
                    if fk == "state_site4" and kind == "mlp":
                        readers[j] = f
                row["readout"][f"{fk}/{kind}"] = dict(all=float(np.mean(accs)), matched=float(np.mean([a for a in accs_m if a is not None])), by_task=accs)
        for j in range(T):
            maj = sets_t[:, j].double().mean(0).argmax()
            row["majority"][names[j]] = float(sets_t[:, j, maj].double().mean())
        row["majority_mean"] = float(np.mean(list(row["majority"].values())))
        row["chance_matched_mean"] = float(np.mean([sets_t[matched[j], j].double().mean(0).max().item() for j in range(T) if matched[j].any()]))
        # causal: the swaps at site 2, read at site 4 by the site-4 MLP readouts
        gp2 = torch.Generator(device=dev); gp2.manual_seed(5)
        A = torch.randperm(len(dt["st"]), device=dev, generator=gp2)[:C.N_PAIRS * 3]
        Bc = torch.randperm(len(dt["st"]), device=dev, generator=gp2)[:C.N_PAIRS * 3]
        by_step = {}
        for i in Bc.tolist():
            by_step.setdefault(int(dt["st"][i]), []).append(i)
        pairs = []
        for i in A.tolist():
            cands = by_step.get(int(dt["st"][i]), [])
            for _ in range(8):
                if not cands: break
                j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp2))]
                if j != i and float((dt["b"][i] - dt["b"][j]).abs().sum()) >= C.MIN_DIFF:
                    pairs.append((i, j)); break
            if len(pairs) >= C.N_PAIRS: break
        ia, ib = torch.tensor([p_[0] for p_ in pairs], device=dev), torch.tensor([p_[1] for p_ in pairs], device=dev)
        ta, tb = r["tok"][dt["ep"][ia]], r["tok"][dt["ep"][ib]]; pa = dt["st"][ia] + 1
        XA, _ = C.states(net, ta, pa, K, SITES); XB, _ = C.states(net, tb, pa, K, SITES)
        Fb = lambda d, idx: torch.cat([S(d, idx), d["b"][idx], (d["b"][idx] + 1e-3).log()], 1)
        f_enc = R.fit_mlp(Fb(df, fi), Xf[2].mean(1).double(), va, steps=4000, d=256)
        fa, fb = f_enc(Fb(dt, ia)), f_enc(Fb(dt, ib))
        ra, rb = XA[2].mean(1).double() - fa, XB[2].mean(1).double() - fb
        gr = torch.Generator(device=dev); gr.manual_seed(11)
        d_post = (fb - fa).float(); rnd = torch.randn(d_post.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * d_post.norm(dim=1, keepdim=True)
        edits = {"posterior_part": XA[2] + d_post[:, None], "residual_part": XA[2] + (rb - ra).float()[:, None], "random": XA[2] + rnd[:, None], "whole_swap": XB[2], "none": XA[2]}
        setsA, setsB = opt_sets(dt["b"][ia], Qcell), opt_sets(dt["b"][ib], Qcell)
        Sa = S(dt, ia)
        row["causal"] = {}
        for ek, ns in edits.items():
            x4 = run_edit_state(net, ta, pa, K, 2, ns, 4)
            X4 = torch.cat([Sa, x4], 1)
            to_donor, kept = [], []
            for j in range(T):
                disjoint = ~(setsA[:, j] & setsB[:, j]).any(1)
                a = readers[j](X4).argmax(1)
                to_donor.append(float(setsB[:, j].gather(1, a[:, None]).squeeze(1)[disjoint].double().mean()))
                kept.append(float(setsA[:, j].gather(1, a[:, None]).squeeze(1)[disjoint].double().mean()))
            row["causal"][ek] = dict(to_donor=float(np.mean(to_donor)), kept=float(np.mean(kept)), by_task=to_donor)
        res["runs"][f"seed{s}"] = row
        q = row["readout"]
        print(f"seed{s}: majority {row['majority_mean']:.2f} | all: " + " ".join(f"{k} {v['all']:.2f}" for k, v in q.items()) + f" | matched (chance {row['chance_matched_mean']:.2f}): " + " ".join(f"{k} {v['matched']:.2f}" for k, v in q.items())
              + " | causal (new decisions to donor / kept): " + " ".join(f"{k} {v['to_donor']:.2f}/{v['kept']:.2f}" for k, v in row["causal"].items()), flush=True)
        (HERE / "readout.json").write_text(json.dumps(res))
    res = json.loads((HERE / "readout.json").read_text()); runs = list(res["runs"].values())
    mm = lambda f, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([f(r_) for r_ in runs])
    L = ["# Is the encoded posterior reusable for new decisions?", "",
         f"Generated by `readout.py` (exploratory). New decision tasks: the QMDP-optimal move under the exact posterior for {', '.join(res['tasks'])}. Readouts fitted on {N_FIT} fit decisions, hit rate "
         f"(the readout's move is in the task's optimal set) on {N_TEST} test decisions, mean over the {len(res['tasks'])} tasks; medians (min–max) over six models. "
         "Matched-preference subset: test decisions with a partner of the same step, the same own-goal move and H within the 25th percentile of distances, whose optimal sets for the task are disjoint.", "",
         f"Majority move: {mm(lambda r_: r_['majority_mean'])} of decisions; chance on the matched subset {mm(lambda r_: r_['chance_matched_mean'])}; matched share of decisions "
         + ", ".join(f"{k} {mm(lambda r_, k=k: r_['matched_share'][k])}" for k in res["tasks"]) + ".", "",
         "| readout from | all decisions | matched-preference subset |", "|---|---|---|"]
    for k in runs[0]["readout"]:
        L.append(f"| {k} | {mm(lambda r_: r_['readout'][k]['all'])} | {mm(lambda r_: r_['readout'][k]['matched'])} |")
    L += ["", "## Causal: swaps at the input of block 2, propagated to the final residual, read by the site-4 MLP readouts", "",
          "On matched pairs (same step, posteriors differing by ≥ 0.5 of mass) whose optimal sets for a task are disjoint: the share of the readout's moves in the donor's set / in the recipient's set, mean over tasks.", "",
          "| edit | new decision to donor | kept |", "|---|---|---|"]
    for k in runs[0]["causal"]:
        L.append(f"| {k} | {mm(lambda r_: r_['causal'][k]['to_donor'])} | {mm(lambda r_: r_['causal'][k]['kept'])} |")
    (HERE / "readout.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
