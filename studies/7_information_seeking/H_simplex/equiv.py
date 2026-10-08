"""Beliefs equivalent for one goal, different for another: what changes across goals, the information kept or its
representation? Exploratory, maze10 models. Pairs of test decisions at the same step with nearly identical exact QMDP
action-value profiles under goal A (same optimal set, centred effective steps log_gamma Q within EQ) and different profiles under goal B
(disjoint optimal sets, effective steps apart by >= DIFF), pooled over ordered goal pairs (A, B); values are horizon-free (gamma^d) and goal A must be worth >= MINQ to both beliefs.
  1. representation: ||x1(g) - x2(g)|| under g = A and g = B at the residual sites, each over the median same-step
     distance under that goal (and the same for the logits);
  2. decoding: an MLP decoder from the state under A to B's optimal move, on all test decisions and on the pairs,
     against the decoder from the state under B and chance;
  3. patching at the input of block 2, logits read under the recipient goal: the distinction as represented under A,
     x2(A) - x1(A), applied to the recipient under B (and B's under A); the goal-free posterior part f(b2) - f(b1)
     under B and under A; outcome: B's decision to b2's, A's decision changed.

    .venv/bin/python studies/7_information_seeking/H_simplex/equiv.py        # writes equiv.md / equiv.json
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

SITES = (1, 2, 4)
EQ, DIFF, N_FIT, N_TEST, CANDS, MINQ = 0.5, 1.0, 16384, 8192, 256, 0.4      # EQ, DIFF in effective steps (log_gamma of the value)


def qvals(t, d, idx):
    """Horizon-free QMDP action values under every goal [m, K, 4]: sum_s b(s) gamma^d(next(s, a), goal). (With the
    horizon, late decisions make a goal out of reach worthless under every action, a trivial equivalence.)"""
    b = d["b"][idx].float()
    Qcell = t.gamma ** t.dist[:, t.nxt_cell]                                                   # [K, n, 4]
    return torch.einsum("ms,gsa->mga", b, Qcell).double()


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len, n = len(t.goal_cell), t.H, t.n
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        gsub = torch.Generator(device=dev); gsub.manual_seed(1)
        fi = torch.randperm(len(df["st"]), device=dev, generator=gsub)[:N_FIT]
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:N_TEST]
        Xf, Lf = C.states(net, rf["tok"][df["ep"][fi]], df["st"][fi] + 1, K, SITES)
        Xt, Lt = C.states(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K, SITES)
        Qf, Qt = qvals(t, df, fi), qvals(t, dt, ti)
        cQ = lambda Q: (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Q.clamp(min=1e-9)) / np.log(t.gamma))      # centred effective steps
        setf, sett = Qf >= Qf.max(-1, keepdim=True).values - 1e-6, Qt >= Qt.max(-1, keepdim=True).values - 1e-6      # [m, K, 4]
        S = lambda d, idx: R.step_onehot(d["st"][idx], H_len)
        va = df["ep"][fi] % 10 == 0
        # ---- pairs: same step; A-equivalent, B-different
        gp = torch.Generator(device=dev); gp.manual_seed(7)
        m = len(ti); cand = torch.randint(m, (m, CANDS), device=dev, generator=gp)
        same_step = dt["st"][ti][:, None] == dt["st"][ti][cand]
        cqt = cQ(Qt)
        dq = (cqt[:, None] - cqt[cand]).abs().amax(-1)                                                   # [m, CANDS, K]
        same_set = (sett[:, None] == sett[cand]).all(-1)                                                  # [m, CANDS, K]
        disjoint = ~(sett[:, None] & sett[cand]).any(-1)
        reach = (Qt.amax(-1) >= MINQ)                                                            # the goal is near enough to matter
        eq = same_set & (dq < EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > DIFF)
        pairs = []
        for A in range(K):
            for B in range(K):
                if A == B: continue
                ok = same_step & eq[..., A] & dif[..., B] & (cand != torch.arange(m, device=dev)[:, None])
                ii, cc = torch.nonzero(ok, as_tuple=True)
                # one pair per recipient at most
                first = torch.ones(m, dtype=torch.bool, device=dev); keep = []
                for a_, c_ in zip(ii.tolist(), cc.tolist()):
                    if first[a_]: first[a_] = False; keep.append((a_, int(cand[a_, c_]), A, B))
                pairs += keep
        if len(pairs) > 6000:
            sel = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0))[:6000].tolist(); pairs = [pairs[i] for i in sel]
        P = torch.tensor(pairs, device=dev); i1, i2, gA, gB = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
        N = len(P); ar = torch.arange(N, device=dev)
        row = dict(pairs=N, pairs_by_goal_pair={f"{A}->{B}": int(((gA == A) & (gB == B)).sum()) for A in range(K) for B in range(K) if A != B},
                   qdiff_A=float((cqt[i1, gA] - cqt[i2, gA]).abs().amax(-1).mean()), qdiff_B=float((cqt[i1, gB] - cqt[i2, gB]).abs().amax(-1).mean()),
                   belief_l1=float((dt["b"][ti][i1] - dt["b"][ti][i2]).abs().sum(1).mean()))
        # ---- 1. representation distances under A and under B, normalised by same-step random pairs under that goal
        rp = torch.randint(m, (m,), device=dev, generator=gp)
        row["distance"] = {}
        for l in SITES:
            X = Xt[l]
            base = torch.stack([(X[:, g] - X[rp, g]).norm(dim=-1).median() for g in range(K)])                   # typical distance under g
            dA = (X[i1, gA] - X[i2, gA]).norm(dim=-1) / base[gA]; dB = (X[i1, gB] - X[i2, gB]).norm(dim=-1) / base[gB]
            row["distance"][f"resid{l}"] = dict(under_A=float(dA.median()), under_B=float(dB.median()), ratio_B_over_A=float((dB / dA).median()))
        baseL = torch.stack([(Lt[:, g] - Lt[rp, g]).norm(dim=-1).median() for g in range(K)])
        row["distance"]["logits"] = dict(under_A=float(((Lt[i1, gA] - Lt[i2, gA]).norm(dim=-1) / baseL[gA]).median()), under_B=float(((Lt[i1, gB] - Lt[i2, gB]).norm(dim=-1) / baseL[gB]).median()))
        decA_same = float((Lt[i1, gA].argmax(-1) == Lt[i2, gA].argmax(-1)).double().mean()); decB_same = float((Lt[i1, gB].argmax(-1) == Lt[i2, gB].argmax(-1)).double().mean())
        row["model_decision_same"] = dict(under_A=decA_same, under_B=decB_same)
        # ---- 2. decoding B's move from the state under A (site 2 and 4), against from the state under B
        row["decode"] = {}
        for l in (2, 4):
            out = {"from_A": [], "from_B": [], "from_A_pairs": [], "from_B_pairs": [], "chance_pairs": []}
            for A in range(K):
                for B in range(K):
                    if A == B: continue
                    sel = (gA == A) & (gB == B)
                    for src, key in ((A, "from_A"), (B, "from_B")):
                        f = R.fit_readout_like if False else None
                        # readout to B's optimal set from the state under src
                        Xa = torch.cat([S(df, fi), Xf[l][:, src].double()], 1); Xb = torch.cat([S(dt, ti), Xt[l][:, src].double()], 1)
                        f = fit_set_readout(Xa, setf[:, B], va, seed=A * 4 + B)
                        h = sett[:, B].gather(1, f(Xb).argmax(1, keepdim=True)).squeeze(1)
                        out[key].append(float(h.double().mean()))
                        if sel.any():
                            hp = torch.cat([h[i1[sel]], h[i2[sel]]]); out[key + "_pairs"].append(float(hp.double().mean()))
                    if sel.any():
                        ss = torch.cat([sett[i1[sel], B], sett[i2[sel], B]]).double(); out["chance_pairs"].append(float(ss.mean(0).max()))
            row["decode"][f"resid{l}"] = {k: float(np.mean(v)) for k, v in out.items()}
        # ---- 3. patching at site 2, read under the recipient goal
        tok1 = r["tok"][dt["ep"][ti][i1]]; p1 = dt["st"][ti][i1] + 1
        X1, X2 = Xt[2][i1], Xt[2][i2]                                                                       # [N, K, d]
        Fb = lambda d, idx: torch.cat([S(d, idx), d["b"][idx], (d["b"][idx] + 1e-3).log()], 1)
        f2 = R.fit_mlp(Fb(df, fi), Xf[2].mean(1).double(), va, steps=4000, d=256)
        d_post = (f2(Fb(dt, ti[i2])) - f2(Fb(dt, ti[i1]))).float()
        dA_vec, dB_vec = (X2[ar, gA] - X1[ar, gA]).float(), (X2[ar, gB] - X1[ar, gB]).float()
        gr = torch.Generator(device=dev); gr.manual_seed(3)
        rnd = torch.randn(d_post.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * dB_vec.norm(dim=1, keepdim=True)
        dec1B, dec2B, dec1A, dec2A = Lt[i1, gB].argmax(-1), Lt[i2, gB].argmax(-1), Lt[i1, gA].argmax(-1), Lt[i2, gA].argmax(-1)
        differB = dec1B != dec2B
        def apply(vec, under):                                                                              # edit only the copy under `under`, read under it
            ns = X1.clone(); ns[ar, under] = ns[ar, under] + vec.to(ns.dtype)
            Le = C.run_edit(net, tok1, p1, K, 2, ns)
            return Le[ar, under].argmax(-1)
        row["patch"] = {}
        for name, vec in (("distinction_as_under_A", dA_vec), ("distinction_as_under_B", dB_vec), ("posterior_part", d_post), ("random_B_norm", rnd)):
            eB, eA = apply(vec, gB), apply(vec, gA)
            row["patch"][name] = dict(B_to_b2=float((eB == dec2B)[differB].double().mean()), B_kept=float((eB == dec1B)[differB].double().mean()),
                                      A_changed=float((eA != dec1A).double().mean()), A_to_b2=float((eA == dec2A).double().mean()))
        res[f"seed{s}"] = row
        print(f"seed{s}: pairs {N} (qdiff A {row['qdiff_A']:.3f} B {row['qdiff_B']:.3f}, belief L1 {row['belief_l1']:.2f}) | dist " + " ".join(f"{k} A {v['under_A']:.2f} B {v['under_B']:.2f}" for k, v in row["distance"].items())
              + f" | model same decision A {decA_same:.2f} B {decB_same:.2f} | decode " + " ".join(f"{k}: fromA {v['from_A']:.2f}/{v['from_A_pairs']:.2f} fromB {v['from_B']:.2f}/{v['from_B_pairs']:.2f} chance {v['chance_pairs']:.2f}" for k, v in row["decode"].items())
              + " | patch " + " ".join(f"{k}: B->b2 {v['B_to_b2']:.2f} A chg {v['A_changed']:.2f}" for k, v in row["patch"].items()), flush=True)
        (HERE / "equiv.json").write_text(json.dumps(res))
    res = json.loads((HERE / "equiv.json").read_text()); runs = list(res.values())
    mm = lambda f, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([f(r_) for r_ in runs])
    L = ["# Beliefs equivalent for one goal, different for another", "",
         f"Generated by `equiv.py` (exploratory). Pairs of test decisions at the same step with the same optimal set and horizon-free QMDP values (Σ_s b(s) γ^d), in centred effective steps (log_γ), within {EQ} step under goal A, goal A worth ≥ {MINQ} to both beliefs, and disjoint optimal sets at least {DIFF} step apart under goal B, "
         f"pooled over ordered goal pairs: {mm(lambda r_: r_['pairs'], 0)} pairs per model (largest difference of effective steps under A {mm(lambda r_: r_['qdiff_A'])}, under B {mm(lambda r_: r_['qdiff_B'])}; posteriors {mm(lambda r_: r_['belief_l1'])} apart in L1). "
         f"The model's own decision is the same for the two beliefs under A in {mm(lambda r_: r_['model_decision_same']['under_A'])} and under B in {mm(lambda r_: r_['model_decision_same']['under_B'])}. Medians (min–max) over six models.", "",
         "## 1. Does the representation distinguish them more under B than under A?", "", "Distance between the two states under each goal, over the median distance of random same-step pairs under that goal.", "",
         "| site | under A | under B | ratio B / A |", "|---|---|---|---|"]
    for k in runs[0]["distance"]:
        d_ = lambda f_: mm(lambda r_: f_(r_["distance"][k]))
        L.append(f"| {k} | {d_(lambda v: v['under_A'])} | {d_(lambda v: v['under_B'])} | {d_(lambda v: v.get('ratio_B_over_A', float('nan')))} |")
    L += ["", "## 2. Can B's distinction be decoded from the state under A?", "", "MLP decoder to B's optimal move; hit rate on all test decisions / on the pairs' decisions.", "",
          "| site | from the state under A | from the state under B | chance on the pairs |", "|---|---|---|---|"]
    for k in runs[0]["decode"]:
        d_ = lambda f_: mm(lambda r_: f_(r_["decode"][k]))
        L.append(f"| {k} | {d_(lambda v: v['from_A'])} / {d_(lambda v: v['from_A_pairs'])} | {d_(lambda v: v['from_B'])} / {d_(lambda v: v['from_B_pairs'])} | {d_(lambda v: v['chance_pairs'])} |")
    L += ["", "## 3. Patching the distinction at the input of block 2", "",
          "Recipient b1; the vector added to its state under the recipient goal only, logits read under that goal. B → b2: among pairs whose decisions under B differ, the share now taking b2's; A changed: the share of A's decisions that change (they should not: the beliefs are A-equivalent).", "",
          "| vector | under B: to b2's decision | under B: kept b1's | under A: decision changed |", "|---|---|---|---|"]
    for k in runs[0]["patch"]:
        d_ = lambda f_: mm(lambda r_: f_(r_["patch"][k]))
        L.append(f"| {k} | {d_(lambda v: v['B_to_b2'])} | {d_(lambda v: v['B_kept'])} | {d_(lambda v: v['A_changed'])} |")
    (HERE / "equiv.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def fit_set_readout(X, sets, va, seed=0, steps=1500):
    """An MLP readout to an optimal-move set (loss: −log of the set's softmax mass); early stopping on the validation episodes."""
    torch.manual_seed(seed)
    X = X.float(); idx = torch.arange(len(X), device=X.device); tr, vv = idx[~va], idx[va]
    mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-2)
    net = R.MLP(X.shape[1], 64, 4).to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator(device=X.device); gen.manual_seed(seed)
    loss_of = lambda rows: -(torch.logsumexp(torch.log_softmax(net((X[rows] - mu) / sd), -1).masked_fill(~sets[rows], -1e9), -1)).mean()
    best, state = float("inf"), None
    with torch.enable_grad():
        for i in range(steps):
            rows = tr[torch.randint(len(tr), (4096,), device=X.device, generator=gen)]
            loss = loss_of(rows); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if i % 100 == 0 or i == steps - 1:
                with torch.no_grad():
                    v = float(loss_of(vv))
                if v < best: best, state = v, {k: w.clone() for k, w in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    return lambda Z: net((Z.float() - mu) / sd)


if __name__ == "__main__":
    main()
