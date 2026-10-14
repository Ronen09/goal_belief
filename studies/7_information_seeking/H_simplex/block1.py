"""Does block 1 introduce goal-dependent selection of the history, or continue transforming what block 0 combined?
Exploratory. At the input of block 1 every token but the first carries the goal (block 0 attended to the goal token):
the decision token's own state x1_p (section 12), the history tokens' states and the goal token's. Block 1's attention
at the decision token p is recomputed with each source of the goal switched A -> B alone, the rest held at A:
  query       q_p from x1_p(B)                         does the goal change how the token selects history?
  hist_kv     keys / values of the history tokens      has the history already been rewritten goal-dependently?
  self_kv     key / value of the decision token itself (its own block-0 goal content, read back by attention)
  goal_kv     key / value of the goal token            another injection of the goal's content
  resid       x1_p(B) in the residual only (no attention change): the block-0 content carried forward
  token = query + self_kv + resid (everything the token already holds from block 0); sources = hist_kv + goal_kv.
For the query route the attention is also measured renormalised over the history tokens only (total variation of the
per-head distribution against A's; the goal token's mass separately), beside the causal measures: projection of the
route's change of block 1's attention output and of the state after block 1 on the natural change, the interaction
part's R², the held-out pair difference against the B-written difference at the input of block 2 (equiv2's site), and
the decision with the route's state patched at the input of block 2 under A. Then what block 1's MLP adds.

    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/H_simplex/block1.py        # writes block1.md / block1.json
"""

from __future__ import annotations

import copy, json, math, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C          # noqa: E402
from block0 import r2, proj                                        # noqa: E402

N_DEC, N_PAIRS, N_FIT_PAIRS, CHUNK = 8192, 4096, 12288, 1024
ROUTES = {"query": {"q"}, "hist_kv": {"hist"}, "hist_k": {"hist_k"}, "hist_v": {"hist_v"}, "self_kv": {"self"}, "goal_kv": {"goal"}, "goal_k": {"goal_k"}, "goal_v": {"goal_v"}, "resid": {"res"},
          "token": {"q", "self", "res"}, "sources": {"hist", "goal"}, "all": {"q", "hist", "self", "goal", "res"}}


@torch.no_grad()
def inputs(net, tok, p, K):
    """Block-1 inputs of the whole sequence under every goal [N, K, L, d] (float32) and the natural block-1 quantities at
    the decision token: attn output, mid, mlp, resid-2 [N, K, d] (double), plus the natural pattern rows [N, K, h, L]."""
    N, dev = len(tok), tok.device
    X, A_, a, mid, m, x2 = [], [], [], [], [], []
    for i in range(0, N, CHUNK):
        tk = tok[i:i + CHUNK].repeat(K, 1, 1); pp = p[i:i + CHUNK].repeat(K); n_ = len(tok[i:i + CHUNK])
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
        _, _, rec = net(tk, record=True); ar = torch.arange(len(tk), device=dev)
        X.append(rec["resid"][1].view(K, n_, *rec["resid"][1].shape[1:]).transpose(0, 1))
        A_.append(rec["pattern"][1][ar, :, pp].view(K, n_, net.nh, -1).transpose(0, 1).double())
        for lst, k in ((a, "attn"), (mid, "mid"), (m, "mlp")):
            lst.append(rec[k][1][ar, pp].view(K, n_, -1).transpose(0, 1).double())
        x2.append(rec["resid"][2][ar, pp].view(K, n_, -1).transpose(0, 1).double())
    return torch.cat(X), torch.cat(A_), torch.cat(a), torch.cat(mid), torch.cat(m), torch.cat(x2)


@torch.no_grad()
def route_attn(b1, XA, XB, p, parts):
    """Block 1's attention at p with the goal sources in `parts` taken from B's block-1 inputs XB [n, L, d] and the rest
    from XA; returns the attention output [n, d], the pattern [n, h, L] and the residual input x1_p used."""
    n, L, d = XA.shape; h = b1.h; dh = d // h; dev = XA.device
    ar = torch.arange(n, device=dev)
    pos = torch.arange(L, device=dev)[None].expand(n, L)
    is_goal = pos == 1; is_self = pos == p[:, None]; is_hist = ~is_goal & ~is_self & (pos <= p[:, None])
    useK = torch.zeros(n, L, dtype=torch.bool, device=dev); useV = useK.clone()
    for name, mask in (("hist", is_hist), ("self", is_self), ("goal", is_goal)):
        if name in parts: useK |= mask; useV |= mask
        if name + "_k" in parts: useK |= mask
        if name + "_v" in parts: useV |= mask
    qA, qB = b1.qkv(b1.ln1(XA)).view(n, L, 3, h, dh), b1.qkv(b1.ln1(XB)).view(n, L, 3, h, dh)
    k = torch.where(useK[:, :, None, None], qB[:, :, 1], qA[:, :, 1]).transpose(1, 2)   # [n, h, L, dh]
    v = torch.where(useV[:, :, None, None], qB[:, :, 2], qA[:, :, 2]).transpose(1, 2)
    xq = XB[ar, p] if "q" in parts else XA[ar, p]
    q = b1.qkv(b1.ln1(xq)).view(n, 3, h, dh)[:, 0]                                      # [n, h, dh]
    s = torch.einsum("nhd,nhld->nhl", q, k) / math.sqrt(dh)
    s = s.masked_fill(pos[:, None] > p[:, None, None], -torch.inf)
    A = s.softmax(-1)
    z = torch.einsum("nhl,nhld->nhd", A, v).reshape(n, d)
    xres = XB[ar, p] if "res" in parts else XA[ar, p]
    return b1.o(z).double(), A.double(), xres.double()


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
        b1 = net.blocks[1]; b1d = copy.deepcopy(b1).double(); mlp_d = lambda X: b1d.mlp(b1d.ln2(X))
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        g = torch.Generator(device=dev); g.manual_seed(5)
        ok = torch.nonzero(dt["st"] >= 1).squeeze(1); ti = ok[torch.randperm(len(ok), device=dev, generator=g)[:N_DEC]]
        tok, p = r["tok"][dt["ep"][ti]], dt["st"][ti] + 1
        X, Pnat, a_nat, mid_nat, m_nat, x2_nat = inputs(net, tok, p, K)
        Lnat = dt["lc"][ti]
        parts_fn = lambda Y: (lambda hist, M: (hist, M, Y - hist[:, None] - M[None]))(Y.mean(1), (Y - Y.mean(1)[:, None]).mean(0))
        I_a, I_x2, I_m = parts_fn(a_nat)[2], parts_fn(x2_nat)[2], parts_fn(m_nat)[2]
        share = lambda Y: float((parts_fn(Y)[2] ** 2).sum() / ((Y - Y.mean((0, 1))) ** 2).sum())
        # site-2 maps on fit pairs (equiv2's recipe)
        okf = torch.nonzero(df["st"] >= 1).squeeze(1); gp = torch.Generator(device=dev); gp.manual_seed(21); by_step = {}
        for i in okf[torch.randint(len(okf), (N_FIT_PAIRS * 4,), device=dev, generator=gp)].tolist(): by_step.setdefault(int(df["st"][i]), []).append(i)
        tp = []
        for i in okf[torch.randint(len(okf), (N_FIT_PAIRS * 4,), device=dev, generator=gp)].tolist():
            cands = by_step.get(int(df["st"][i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
            if j != i and float((df["b"][i] - df["b"][j]).abs().sum()) >= 0.5: tp.append((i, j))
            if len(tp) >= N_FIT_PAIRS: break
        tp = torch.tensor(tp, device=dev); f1, f2 = tp[:, 0], tp[:, 1]
        XF1, _ = C.states(net, rf["tok"][df["ep"][f1]], df["st"][f1] + 1, K, (2,)); XF2, _ = C.states(net, rf["tok"][df["ep"][f2]], df["st"][f2] + 1, K, (2,))
        XF1, XF2 = XF1[2].double(), XF2[2].double(); va = df["ep"][f1] % 10 == 0
        Mmap = {(A, B): R.Ridge(XF2[:, A] - XF1[:, A], XF2[:, B] - XF1[:, B], va) for A in range(K) for B in range(K) if A != B}
        # held-out pairs among the test decisions
        sub_st, sub_b = dt["st"][ti], dt["b"][ti]; by_step = {}
        for i in torch.randint(len(ti), (N_PAIRS * 4,), device=dev, generator=g).tolist(): by_step.setdefault(int(sub_st[i]), []).append(i)
        pr = []
        for i in torch.randint(len(ti), (N_PAIRS * 4,), device=dev, generator=g).tolist():
            cands = by_step.get(int(sub_st[i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=g))]
            if j != i and float((sub_b[i] - sub_b[j]).abs().sum()) >= 0.5: pr.append((i, j))
            if len(pr) >= N_PAIRS: break
        pr = torch.tensor(pr, device=dev); q1, q2 = pr[:, 0], pr[:, 1]
        pos = torch.arange(X.shape[2], device=dev)[None].expand(len(p), -1); is_hist = (pos != 1) & (pos <= p[:, None])
        agg = {}; add = lambda k, v: agg.setdefault(k, []).append(v); rows = {}
        for A in range(K):
            for B in range(K):
                if A == B: continue
                XA, XB = X[:, A], X[:, B]
                aA, aB, xA, xB, mA, mB = a_nat[:, A], a_nat[:, B], x2_nat[:, A], x2_nat[:, B], m_nat[:, A], m_nat[:, B]
                d_a, d_x = aB - aA, xB - xA; dI_a, dI_x = I_a[:, B] - I_a[:, A], I_x2[:, B] - I_x2[:, A]
                PA = Pnat[:, A]
                renorm = lambda P: (P * is_hist[:, None]) / (P * is_hist[:, None]).sum(-1, keepdim=True).clamp(min=1e-12)
                row = {}
                decA, decB = Lnat[:, A].argmax(-1), Lnat[:, B].argmax(-1); differ = decA != decB; LA, LB = Lnat[:, A], Lnat[:, B]
                def patched(x2_new):
                    ns = x2_nat.clone().float(); ns[:, A] = x2_new.float()
                    Le = C.run_edit(net, tok, p, K, 2, ns)[:, A]
                    return float((Le.argmax(-1) == decB)[differ].double().mean()), float((((Le - LA) * (LB - LA)).sum(1) / ((LB - LA) ** 2).sum(1).clamp(min=1e-12)).mean())
                for name, parts in ROUTES.items():
                    a_r, P_r, xres = [], [], []
                    for i in range(0, len(p), CHUNK):
                        o = route_attn(b1, XA[i:i + CHUNK], XB[i:i + CHUNK], p[i:i + CHUNK], parts); a_r.append(o[0]); P_r.append(o[1]); xres.append(o[2])
                    a_r, P_r, xres = torch.cat(a_r), torch.cat(P_r), torch.cat(xres)
                    if name == "all": assert float((a_r - aB).abs().max()) < 1e-2 and float((P_r - Pnat[:, B]).abs().max()) < 1e-3, (float((a_r - aB).abs().max()), float((P_r - Pnat[:, B]).abs().max()))
                    mid_r = xres + a_r; m_r = mlp_d(mid_r); x_r = mid_r + m_r
                    da, dx = a_r - aA, x_r - xA
                    row[f"{name}_attn_proj"], _ = proj(da, d_a); row[f"{name}_attn_I_r2"] = r2(da - da.mean(0), dI_a)
                    row[f"{name}_x2_proj"], _ = proj(dx, d_x); row[f"{name}_x2_I_r2"] = r2(dx - dx.mean(0), dI_x)
                    # attention: redistribution over the history tokens (TV per head, mean) and the goal token's mass
                    row[f"{name}_tv_hist"] = float((0.5 * (renorm(P_r) - renorm(PA)).abs().sum(-1)).mean())
                    row[f"{name}_goal_mass_change"] = float((P_r[:, :, 1] - PA[:, :, 1]).mean())
                    # pairs against the site-2 change of coordinates
                    dr, dB_, dA_ = x_r[q2] - x_r[q1], xB[q2] - xB[q1], xA[q2] - xA[q1]
                    row[f"{name}_pair_r2_vs_B"] = r2(dr, dB_); row[f"{name}_pair_I_r2_vs_B"] = r2(dr - dA_, dB_ - dA_)
                    row[f"{name}_dec_switch"], row[f"{name}_dec_transfer"] = patched(x_r)
                    row[f"{name}_mlp_r2_vs_nat"] = r2(m_r - mA, mB - mA)
                row["tv_hist_natural"] = row["all_tv_hist"]; row["goal_mass_A"] = float(PA[:, :, 1].mean())
                dA_, dB_ = xA[q2] - xA[q1], xB[q2] - xB[q1]
                row["pair_r2_identity"] = r2(dA_, dB_); row["pair_r2_map"] = r2(Mmap[(A, B)](dA_), dB_)
                gr = torch.Generator(device=dev); gr.manual_seed(3)
                rnd = torch.randn(xA.shape, device=dev, generator=gr, dtype=torch.float64); rnd = rnd / rnd.norm(dim=1, keepdim=True) * d_x.norm(dim=1, keepdim=True)
                row["random_dec_switch"], row["random_dec_transfer"] = patched(xA + rnd)
                row["differ"] = float(differ.double().mean())
                # the MLP of block 1
                midA, midB = mid_nat[:, A], mid_nat[:, B]; dmid = midB - midA; dm = mB - mA; eps = 1e-6
                lin = (mlp_d(midA + eps * dmid) - mlp_d(midA)) / eps; linm = (mlp_d((midA + midB) / 2 + eps * dmid) - mlp_d((midA + midB) / 2)) / eps
                row["mlp_first_order_r2"] = r2(lin, dm); row["mlp_midpoint_r2"] = r2(linm, dm)
                row["mlp_linear_fit_r2"] = r2(R.Ridge(dmid[:N_DEC // 2], dm[:N_DEC // 2], torch.arange(N_DEC // 2, device=dev) % 10 == 0)(dmid[N_DEC // 2:]), dm[N_DEC // 2:])
                row["mlp_gain"] = float((dm.norm(dim=1) / dmid.norm(dim=1).clamp(min=1e-12)).mean()); row["step_rel"] = float((dmid.norm(dim=1) / midA.norm(dim=1)).mean())
                # how much of the block-1 input change is the token's own (block-0) vs the attention sublayer's
                row["share_resid_in_dmid"] = float(((xB - aB) - (xA - aA)).norm(dim=1).mean() / dmid.norm(dim=1).mean()); row["share_attn_in_dmid"] = float(d_a.norm(dim=1).mean() / dmid.norm(dim=1).mean())
                for k, v in row.items(): add(k, v)
                rows[f"{A}->{B}"] = row
        mean = {k: float(np.mean(v)) for k, v in agg.items()}
        mean.update(share_attn=share(a_nat), share_mlp=share(m_nat), share_x2=share(x2_nat), share_x1=share(x2_nat - a_nat - m_nat))
        res[f"seed{s}"] = dict(mean=mean, by_pair=rows)
        print(f"seed{s}: " + " ".join(f"{k} {v:.3f}" for k, v in mean.items() if not k.startswith(("token", "sources", "all"))), flush=True)
        (HERE / "block1.json").write_text(json.dumps(res))
    report()


def report():
    res = json.loads((HERE / "block1.json").read_text()); runs = list(res.values())
    mm = lambda k, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([r_["mean"][k] for r_ in runs])
    names = [("query", "decision token's query"), ("hist_kv", "history tokens' keys and values"), ("hist_k", "history tokens' keys only"), ("hist_v", "history tokens' values only"), ("self_kv", "decision token's own key and value"), ("goal_kv", "goal token's key and value"), ("goal_k", "goal token's key only"), ("goal_v", "goal token's value only"),
             ("resid", "residual only (block-0 content carried, no attention change)"), ("token", "token's own block-0 state: query + self + residual"), ("sources", "attention sources: history + goal tokens"), ("all", "all (= the natural change)")]
    L = ["# Block 1: goal-dependent selection of the history, or transformation of what block 0 combined?", "",
         f"Generated by `block1.py` (exploratory). {N_DEC} test decisions after the first step; block 1's attention at the decision token recomputed with each source of the goal switched A → B alone (the rest at A), block 1's MLP applied to the result. Means over the 12 ordered goal pairs, medians (min–max) over six models.", "",
         "## 1. The attention pattern", "",
         f"The decision token's block-1 attention mass on the goal token under A: {mm('goal_mass_A')}. Redistribution over the history tokens: total variation, per head, between the route's and A's attention renormalised over the history tokens (goal token excluded).", "",
         "| goal source switched A → B | TV over history tokens | change of the goal token's mass |", "|---|---|---|"]
    for k, lab in names: L.append(f"| {lab} | {mm(k + '_tv_hist')} | {mm(k + '_goal_mass_change', 3)} |")
    L += ["", "## 2. Causal contribution of each route", "",
          "| goal source switched A → B | block-1 attention output: projection on the natural change | its interaction part, R² | state after block 1: projection | its interaction part, R² | held-out pair difference vs Δ_B, R² | its change from Δ_A vs the natural change, R² | decision switches to B's | logit transfer |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, lab in names:
        L.append(f"| {lab} | {mm(k + '_attn_proj')} | {mm(k + '_attn_I_r2')} | {mm(k + '_x2_proj')} | {mm(k + '_x2_I_r2')} | {mm(k + '_pair_r2_vs_B')} | {mm(k + '_pair_I_r2_vs_B')} | {mm(k + '_dec_switch')} | {mm(k + '_dec_transfer')} |")
    L += [f"| untransformed Δ_A / M(A→B) Δ_A (site 2) | | | | | {mm('pair_r2_identity')} / {mm('pair_r2_map')} | | | |",
          f"| random, the natural change's norm | | | | | | | {mm('random_dec_switch')} | {mm('random_dec_transfer')} |", "",
          f"Decisions where the model's moves under A and B differ: {mm('differ')}. The whole state under B patched at the input of block 2 is the 'all' row.", "",
          "## 3. What block 1's MLP adds", "",
          f"* Interaction (goal × history) shares: input of block 1 {mm('share_x1')}, block-1 attention output {mm('share_attn')}, MLP output {mm('share_mlp')}, state after block 1 {mm('share_x2')}.",
          f"* Of the goal change at the MLP's input, the part carried in from block 0 (residual) is {mm('share_resid_in_dmid')} of its norm and block 1's attention change {mm('share_attn_in_dmid')}. The step is {mm('step_rel')} of the state's norm.",
          f"* The MLP's change against its first-order response at A: R² {mm('mlp_first_order_r2')}; at the midpoint {mm('mlp_midpoint_r2')}; a single linear map over decisions {mm('mlp_linear_fit_r2')}; gain ‖Δmlp‖ / ‖Δmid‖ {mm('mlp_gain')}.",
          "* Each route's MLP change against the natural MLP change (R²): " + ", ".join(f"{lab.split(':')[0].split(' (')[0]} {mm(k + '_mlp_r2_vs_nat')}" for k, lab in names[:-1]) + "."]
    (HERE / "block1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
