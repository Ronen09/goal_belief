"""Does the goal change what block 0 retrieves from the history, or how the retrieved history is transformed?
Exploratory. The goal is one token (position 1). For every decision after the first the decision token's embedding
is goal-free, so at block 0 its query is goal-free and the goal reaches it through two routes only: the goal token's
KEY, which reshapes the attention pattern over the history (retrieval), and the goal token's VALUE (content). History
fixed, goal A -> B, each route swapped alone (pattern of B with the values of A; pattern of A with the value of B),
then the block-0 MLP applied to the route's state:
  (1) how far each route moves the attention output / the state after block 0 toward goal B's (projection on the
      natural change, R²), for the whole change and for its belief-dependent (interaction) part;
  (2) against the change of coordinates: a ridge map M(A->B) at the input of block 1 fitted on same-step pairs of
      fit decisions; on held-out pairs, the route-swapped difference x2_route - x1_route against the B-written
      difference (R²), beside the untransformed A-written difference and M(A->B) Delta_A;
  (3) what the MLP adds: the MLP's natural change against its first-order (Jacobian) response to the attention change,
      the growth of each route's change through the MLP, and the interaction shares before and after it;
  (4) the decision: the route's block-0 state patched at the decision token under goal A, logits under A: the share
      of decisions that differ between A and B now taking B's, and the logit transfer; ceiling: the whole state under B.
Structure checks: at block 0 the pattern route is a per-head rescaling of the history (q is goal-free), reported as the
attention mass on the goal token per goal and the identity of the history weights' ratios.

    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/H_simplex/block0.py        # writes block0.md / block0.json
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

N_DEC, N_PAIRS, N_FIT_PAIRS, CHUNK = 8192, 4096, 12288, 1024


def r2(P, T):
    return float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


def proj(dx, dnat):
    """Mean projection coefficient of dx on dnat per row, and the R² of dx against dnat as vectors."""
    c = (dx * dnat).sum(1) / (dnat ** 2).sum(1).clamp(min=1e-12)
    return float(c.mean()), r2(dx, dnat)


@torch.no_grad()
def block0_routes(net, tok, p, K):
    """At the decision token p under every goal: x_in [N, d] (goal-free), and per goal the pattern row [N, K, h, L],
    attention output, mid, mlp output and resid after block 0 [N, K, d]; per ordered goal pair the two routes'
    attention output and resid-1 state {(A, B): {"pat": (a, mid, mlp, x1), "val": (...), "both": (...)}}."""
    N, dev = len(tok), tok.device
    b0 = net.blocks[0]; d, h = net.d, net.nh; dh = d // h; W = b0.o.weight.view(d, h, dh).double()
    out = {k: [] for k in ("x_in", "pat", "a", "mid", "mlp", "x1", "w_goal")}
    routes = {(A, B): {rt: {k: [] for k in ("a", "mid", "mlp", "x1")} for rt in ("pat", "val", "both")} for A in range(K) for B in range(K) if A != B}
    for i in range(0, N, CHUNK):
        tk = tok[i:i + CHUNK].repeat(K, 1, 1); pp = p[i:i + CHUNK].repeat(K); n_ = len(tok[i:i + CHUNK])
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
        _, _, rec = net(tk, record=True)
        ar = torch.arange(len(tk), device=dev)
        A_ = rec["pattern"][0][ar, :, pp].view(K, n_, h, -1).transpose(0, 1).double()      # [n, K, h, L]
        V_ = rec["value"][0].view(K, n_, h, -1, dh).transpose(0, 1).double()               # [n, K, h, L, dh]
        x_in = rec["resid"][0][ar, pp].view(K, n_, -1)[0].double()
        out["x_in"].append(x_in); out["pat"].append(A_); out["w_goal"].append(A_[:, :, :, 1])
        for k in ("attn", "mid", "mlp"):
            out["a" if k == "attn" else k].append(rec[k][0][ar, pp].view(K, n_, -1).transpose(0, 1).double())
        out["x1"].append(rec["resid"][1][ar, pp].view(K, n_, -1).transpose(0, 1).double())
        att = lambda Arow, Vals: torch.einsum("nhl,nhld,ohd->no", Arow, Vals, W) + b0.o.bias.double()
        for (A, B), rr in routes.items():
            VA, VB = V_[:, A], V_[:, B]
            Vmix = VA.clone(); Vmix[:, :, 1] = VB[:, :, 1]                                   # values of A, the goal token's value from B
            for rt, (Arow, Vals) in {"pat": (A_[:, B], VA), "val": (A_[:, A], Vmix), "both": (A_[:, B], Vmix)}.items():
                a = att(Arow, Vals); mid = x_in + a
                m = b0.mlp(b0.ln2(mid.float())).double(); x1 = mid + m
                for k, v in zip(("a", "mid", "mlp", "x1"), (a, mid, m, x1)): rr[rt][k].append(v)
        # check: "both" reproduces the natural state under B
        if i == 0:
            nat = out["x1"][-1][:, 1]; got = routes[(0, 1)]["both"]["x1"][-1]
            assert float((nat - got).abs().max()) < 1e-3, float((nat - got).abs().max())
    out = {k: torch.cat(v) for k, v in out.items()}
    routes = {ab: {rt: {k: torch.cat(v) for k, v in d_.items()} for rt, d_ in rr.items()} for ab, rr in routes.items()}
    return out, routes


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K = len(t.goal_cell)
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        g = torch.Generator(device=dev); g.manual_seed(5)
        # ---- test decisions after the first step (the decision token is an event token, goal-free)
        ok = torch.nonzero(dt["st"] >= 1).squeeze(1); ti = ok[torch.randperm(len(ok), device=dev, generator=g)[:N_DEC]]
        tok, p = r["tok"][dt["ep"][ti]], dt["st"][ti] + 1
        nat, routes = block0_routes(net, tok, p, K)
        Lnat = dt["lc"][ti]                                                                   # [N, K, 4] centred logits under every goal
        # ---- structure: attention mass on the goal token; the history weights' ratios across goals
        pat = nat["pat"]                                                                       # [N, K, h, L]
        hist_mask = torch.ones(pat.shape[-1], dtype=torch.bool, device=dev); hist_mask[1] = False
        ratio = pat[:, 1][..., hist_mask] / pat[:, 0][..., hist_mask].clamp(min=1e-12)       # [N, h, L-1], over tokens <= p only meaningful
        valid = (pat[:, 0][..., hist_mask] > 1e-9)
        vv = valid.double(); cnt = vv.sum(-1).clamp(min=1)
        mu = (ratio * vv).sum(-1) / cnt; sd = (((ratio - mu[..., None]) ** 2) * vv).sum(-1).div(cnt).sqrt()
        rvar = (sd / mu.clamp(min=1e-12))[cnt > 1]
        rvar = [float(rvar.mean())]
        struct = dict(goal_mass=[float(nat["w_goal"][:, k].mean()) for k in range(K)],
                      goal_mass_per_head=[float(nat["w_goal"][:, :, hh].mean()) for hh in range(net.nh)],
                      history_ratio_cv=float(np.mean(rvar)))
        # ---- decomposition x = hist + M + I under the K goals (for the interaction part)
        def parts(X):                                                                          # X [N, K, d] -> hist [N, d], M [K, d], I [N, K, d]
            hist = X.mean(1); M = (X - hist[:, None]).mean(0); return hist, M, X - hist[:, None] - M[None]
        I_a = parts(nat["a"])[2]; I_x1 = parts(nat["x1"])[2]; I_mlp = parts(nat["mlp"])[2]
        share = lambda X: (lambda h_, M_, I_: float((I_ ** 2).sum() / ((X - X.mean((0, 1))) ** 2).sum()))(*parts(X))
        # ---- the site-1 change of coordinates, fitted on fit-episode pairs (as equiv2.py at site 2)
        okf = torch.nonzero(df["st"] >= 1).squeeze(1)
        gp = torch.Generator(device=dev); gp.manual_seed(21)
        by_step = {}
        for i in okf[torch.randint(len(okf), (N_FIT_PAIRS * 4,), device=dev, generator=gp)].tolist():
            by_step.setdefault(int(df["st"][i]), []).append(i)
        tp = []
        for i in okf[torch.randint(len(okf), (N_FIT_PAIRS * 4,), device=dev, generator=gp)].tolist():
            cands = by_step.get(int(df["st"][i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
            if j != i and float((df["b"][i] - df["b"][j]).abs().sum()) >= 0.5: tp.append((i, j))
            if len(tp) >= N_FIT_PAIRS: break
        tp = torch.tensor(tp, device=dev); f1, f2 = tp[:, 0], tp[:, 1]
        XF1, _ = C.states(net, rf["tok"][df["ep"][f1]], df["st"][f1] + 1, K, (1,)); XF2, _ = C.states(net, rf["tok"][df["ep"][f2]], df["st"][f2] + 1, K, (1,))
        XF1, XF2 = XF1[1].double(), XF2[1].double(); va = df["ep"][f1] % 10 == 0
        Mmap = {(A, B): R.Ridge(XF2[:, A] - XF1[:, A], XF2[:, B] - XF1[:, B], va) for A in range(K) for B in range(K) if A != B}
        # ---- held-out pairs among the test decisions
        sub = {k: dt[k][ti] for k in ("st", "b")}
        by_step = {}
        for i in torch.randint(len(ti), (N_PAIRS * 4,), device=dev, generator=g).tolist():
            by_step.setdefault(int(sub["st"][i]), []).append(i)
        pr = []
        for i in torch.randint(len(ti), (N_PAIRS * 4,), device=dev, generator=g).tolist():
            cands = by_step.get(int(sub["st"][i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=g))]
            if j != i and float((sub["b"][i] - sub["b"][j]).abs().sum()) >= 0.5: pr.append((i, j))
            if len(pr) >= N_PAIRS: break
        pr = torch.tensor(pr, device=dev); q1, q2 = pr[:, 0], pr[:, 1]
        # ---- MLP: first-order response to the attention change (finite difference in double)
        import copy
        b0d = copy.deepcopy(net.blocks[0]).double()
        mlp_d = lambda X: b0d.mlp(b0d.ln2(X))
        agg = {}
        add = lambda k, v: agg.setdefault(k, []).append(v)
        rows = {}
        for (A, B), rr in routes.items():
            aA, aB, xA, xB, mA, mB = nat["a"][:, A], nat["a"][:, B], nat["x1"][:, A], nat["x1"][:, B], nat["mlp"][:, A], nat["mlp"][:, B]
            d_nat_a, d_nat_x = aB - aA, xB - xA
            d_nat_I_a, d_nat_I_x = I_a[:, B] - I_a[:, A], I_x1[:, B] - I_x1[:, A]
            row = {}
            for rt in ("pat", "val", "both"):
                da, dx = rr[rt]["a"] - aA, rr[rt]["x1"] - xA
                row[f"{rt}_attn_proj"], row[f"{rt}_attn_r2"] = proj(da, d_nat_a)
                row[f"{rt}_x1_proj"], row[f"{rt}_x1_r2"] = proj(dx, d_nat_x)
                # interaction part of the route's change: remove its mean over decisions (the offset part)
                row[f"{rt}_attn_I_r2"] = r2(da - da.mean(0), d_nat_I_a); row[f"{rt}_x1_I_r2"] = r2(dx - dx.mean(0), d_nat_I_x)
                # (3) growth through the MLP: |d mlp| / |d attn| for the route against the natural change
                dm = rr[rt]["mlp"] - mA
                row[f"{rt}_mlp_gain"] = float((dm.norm(dim=1) / da.norm(dim=1).clamp(min=1e-12)).mean())
                row[f"{rt}_mlp_r2_vs_nat"] = r2(dm, mB - mA)
                # (2) against the change of coordinates on held-out pairs
                dr = rr[rt]["x1"][q2] - rr[rt]["x1"][q1]; dB = xB[q2] - xB[q1]; dA = xA[q2] - xA[q1]
                row[f"{rt}_pair_r2_vs_B"] = r2(dr, dB)
                row[f"{rt}_pair_I_r2_vs_B"] = r2(dr - dA, dB - dA)                            # the route's change of the difference against the natural change of the difference
            row["nat_mlp_gain"] = float(((mB - mA).norm(dim=1) / d_nat_a.norm(dim=1).clamp(min=1e-12)).mean())
            # first-order MLP response to the natural attention change
            eps = 1e-6; midA, midB = nat["mid"][:, A], nat["mid"][:, B]; dm = mB - mA
            lin = (mlp_d(midA + eps * d_nat_a) - mlp_d(midA)) / eps
            row["mlp_first_order_r2"] = r2(lin, dm); row["mlp_first_order_I_r2"] = r2(lin - lin.mean(0), dm - dm.mean(0))
            linm = (mlp_d((midA + midB) / 2 + eps * d_nat_a) - mlp_d((midA + midB) / 2)) / eps
            row["mlp_midpoint_r2"] = r2(linm, dm)
            row["step_rel"] = float((d_nat_a.norm(dim=1) / midA.norm(dim=1)).mean())
            row["mlp_linear_fit_r2"] = r2(R.Ridge(d_nat_a[:N_DEC // 2], (mB - mA)[:N_DEC // 2], torch.arange(N_DEC // 2, device=dev) % 10 == 0)(d_nat_a[N_DEC // 2:]), (mB - mA)[N_DEC // 2:])
            dA_, dB_ = xA[q2] - xA[q1], xB[q2] - xB[q1]
            row["pair_r2_identity"] = r2(dA_, dB_); row["pair_r2_map"] = r2(Mmap[(A, B)](dA_), dB_)
            # (4) decisions: patch the route's x1 at the decision token under goal A (site 1), logits under A
            decA, decB = Lnat[:, A].argmax(-1), Lnat[:, B].argmax(-1); differ = decA != decB
            LA, LB = Lnat[:, A], Lnat[:, B]
            def patched(x1_new):
                ns = nat["x1"].clone().float(); ns[:, A] = x1_new.float()
                Le = C.run_edit(net, tok, p, K, 1, ns)[:, A]
                sw = float((Le.argmax(-1) == decB)[differ].double().mean())
                tr = float((((Le - LA) * (LB - LA)).sum(1) / ((LB - LA) ** 2).sum(1).clamp(min=1e-12)).mean())
                return sw, tr
            for rt in ("pat", "val", "both"): row[f"{rt}_dec_switch"], row[f"{rt}_dec_transfer"] = patched(rr[rt]["x1"])
            row["ceiling_dec_switch"], row["ceiling_dec_transfer"] = patched(xB)
            gr = torch.Generator(device=dev); gr.manual_seed(3)
            rnd = torch.randn(xA.shape, device=dev, generator=gr, dtype=torch.float64); rnd = rnd / rnd.norm(dim=1, keepdim=True) * d_nat_x.norm(dim=1, keepdim=True)
            row["random_dec_switch"], row["random_dec_transfer"] = patched(xA + rnd)
            row["differ"] = float(differ.double().mean())
            for k, v in row.items(): add(k, v)
            rows[f"{A}->{B}"] = row
        mean = {k: float(np.mean(v)) for k, v in agg.items()}
        mean.update(share_attn=share(nat["a"]), share_mlp=share(nat["mlp"]), share_x1=share(nat["x1"]))
        res[f"seed{s}"] = dict(mean=mean, struct=struct, by_pair=rows)
        print(f"seed{s}: " + " ".join(f"{k} {v:.3f}" for k, v in mean.items()), flush=True)
        print(f"  struct: {struct}", flush=True)
        (HERE / "block0.json").write_text(json.dumps(res))
    report()


def report():
    res = json.loads((HERE / "block0.json").read_text()); runs = list(res.values())
    mm = lambda k, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([r_["mean"][k] for r_ in runs])
    gm = [np.mean(r_["struct"]["goal_mass"]) for r_ in runs]; gmh = np.array([r_["struct"]["goal_mass_per_head"] for r_ in runs])
    cv = [r_["struct"]["history_ratio_cv"] for r_ in runs]
    L = ["# Block 0: does the goal change what is retrieved, or how it is transformed?", "",
         f"Generated by `block0.py` (exploratory). The maze10 models' decision tokens after the first step ({N_DEC} test decisions; the decision token's embedding is goal-free, so block 0's query is goal-free and the goal reaches the token through the goal token's key, which reshapes the attention pattern, and the goal token's value). History fixed, goal A → B over the 12 ordered goal pairs; each route swapped alone and the block-0 MLP applied to the result. Means over goal pairs, medians (min–max) over six models.", "",
         "## Structure", "",
         f"* Attention mass of the decision token on the goal token in block 0: {np.median(gm):.2f} ({min(gm):.2f}–{max(gm):.2f}) averaged over heads; per head " + ", ".join(f"{x:.2f}" for x in np.median(gmh, 0)) + ".",
         f"* With a goal-free query, changing the goal rescales every history weight by one factor per head: the coefficient of variation of the history weights' ratios across goals is {np.median(cv):.1e} (an identity, checked). The pattern route therefore changes *how much* of the history aggregate enters, per head, and nothing about *which* history tokens.", "",
         "## 1. How far each route moves the state toward goal B's (history fixed)", "",
         "| route swapped A → B | attention output: projection on the natural change / R² | its interaction part, R² | state after block 0: projection / R² | its interaction part, R² |", "|---|---|---|---|---|",
         f"| pattern (goal token's key) | {mm('pat_attn_proj')} / {mm('pat_attn_r2')} | {mm('pat_attn_I_r2')} | {mm('pat_x1_proj')} / {mm('pat_x1_r2')} | {mm('pat_x1_I_r2')} |",
         f"| value (goal token's value) | {mm('val_attn_proj')} / {mm('val_attn_r2')} | {mm('val_attn_I_r2')} | {mm('val_x1_proj')} / {mm('val_x1_r2')} | {mm('val_x1_I_r2')} |",
         f"| both (= the natural change) | {mm('both_attn_proj')} / {mm('both_attn_r2')} | {mm('both_attn_I_r2')} | {mm('both_x1_proj')} / {mm('both_x1_r2')} | {mm('both_x1_I_r2')} |", "",
         "The interaction part of a route's change is the change minus its mean over decisions, against the natural change of the goal × history interaction I(h, g).", "",
         "## 2. Against the change of coordinates at the input of block 1", "",
         f"On {N_PAIRS} held-out same-step pairs (posteriors ≥ 0.5 apart): the difference between the two decisions' route-swapped states against the B-written difference Δ_B.", "",
         "| difference | R² for Δ_B | R² of its change from Δ_A against Δ_B − Δ_A |", "|---|---|---|",
         f"| untransformed Δ_A | {mm('pair_r2_identity')} | 0 |",
         f"| M(A→B) Δ_A (ridge at site 1) | {mm('pair_r2_map')} | |",
         f"| pattern-route states | {mm('pat_pair_r2_vs_B')} | {mm('pat_pair_I_r2_vs_B')} |",
         f"| value-route states | {mm('val_pair_r2_vs_B')} | {mm('val_pair_I_r2_vs_B')} |",
         f"| both routes (= Δ_B) | {mm('both_pair_r2_vs_B')} | {mm('both_pair_I_r2_vs_B')} |", "",
         "## 3. What the block-0 MLP adds", "",
         f"* Interaction (goal × history) share of the variance: attention output {mm('share_attn')}, MLP output {mm('share_mlp')}, state after block 0 {mm('share_x1')}.",
         f"* The goal change at the MLP's input is a large step: ‖Δattn‖ / ‖mid‖ = {mm('step_rel')}. The MLP's natural change (goal A → B) against its first-order response at A to the attention change: R² {mm('mlp_first_order_r2')} (its interaction part {mm('mlp_first_order_I_r2')}); at the midpoint of the step {mm('mlp_midpoint_r2')}; against the best single linear map of the attention change over decisions: {mm('mlp_linear_fit_r2')}.",
         f"* Growth through the MLP, ‖Δmlp‖ / ‖Δattn‖: natural {mm('nat_mlp_gain')}, pattern route {mm('pat_mlp_gain')}, value route {mm('val_mlp_gain')}; the route's MLP change against the natural MLP change (R²): pattern {mm('pat_mlp_r2_vs_nat')}, value {mm('val_mlp_r2_vs_nat')}.", "",
         "## 4. The decision", "",
         f"The route's block-0 state patched at the decision token (input of block 1) under goal A, everything else under A; decisions where the model's moves under A and B differ ({mm('differ')} of decisions).", "",
         "| patched at the input of block 1 | A's decision switches to B's | logit transfer toward B's |", "|---|---|---|",
         f"| whole state under B (ceiling for the token) | {mm('ceiling_dec_switch')} | {mm('ceiling_dec_transfer')} |",
         f"| pattern route | {mm('pat_dec_switch')} | {mm('pat_dec_transfer')} |",
         f"| value route | {mm('val_dec_switch')} | {mm('val_dec_transfer')} |",
         f"| both routes | {mm('both_dec_switch')} | {mm('both_dec_transfer')} |",
         f"| random, the natural change's norm | {mm('random_dec_switch')} | {mm('random_dec_transfer')} |"]
    (HERE / "block0.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
