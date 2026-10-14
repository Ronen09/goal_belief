"""Where is the goal x belief interaction that survives block 0's fixed gates made? Exploratory. The natural network
with block 0's gates fixed per goal and step (section 15; no affine MLPs, so the inputs stay the network's own), at
decision tokens after the first step, under goals A and B with the gates fixed in both. Block 1's attention at the
decision token is recomputed with the incoming decision-token residual held at A's and
    weights   B's attention weights (B's query and every key) with A's values,
    values    B's values of the other tokens (history and goal tokens) with A's weights,
    both      B's weights and B's other-token values,
beside the decision token's own key and value, the residual alone, and everything (= the natural change under B).
On same-step pairs of decisions (posteriors >= 0.5 apart; goal x step offsets cancel in a same-step difference) the
belief-dependent goal difference is (x(h2, B) - x(h1, B)) - (x(h2, A) - x(h1, A)); a route's is the same with the
route-swapped states in place of B's. Measured at block 1's attention output (before the MLP) and at the state after
block 1 (block 1's own MLP applied): projection on the natural belief-dependent difference and R². Propagated to
decisions: on the A-equivalent, B-different pairs the route's pair difference added to the state under B at the
input of block 2 (B's decision switched to the donor's; ceiling the direct B-written difference, floor the A-written
one), and the route's whole state patched under A (A's decision switched to B's). Also the interaction already at
block 1's input, through block 1's attention, MLP and output, with the gates fixed and natural.

    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/H_simplex/gate1.py        # writes gate1.md / gate1.json
"""

from __future__ import annotations

import copy, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C, equiv as E, block1 as B1, gate0 as G0          # noqa: E402
from block0 import r2, proj                                                                               # noqa: E402

N_DEC, N_PAIRS, CHUNK = 8192, 4096, 1024
ROUTES = {"weights": {"q", "hist_k", "self_k", "goal_k"}, "values": {"hist_v", "goal_v"}, "both": {"q", "hist_k", "self_k", "goal_k", "hist_v", "goal_v"},
          "self_kv": {"self"}, "attn_all": {"q", "hist", "self", "goal"}, "resid": {"res"}, "all": {"q", "hist", "self", "goal", "res"}}


def interS(Y, st):
    """Variance share of the goal x belief interaction beyond a goal x step offset: Y [N, K, d]."""
    hist = Y.mean(1); D = Y - hist[:, None]; Ms = torch.zeros_like(D)
    for s_ in st.unique(): Ms[st == s_] = D[st == s_].mean(0, keepdim=True)
    return float(((D - Ms) ** 2).sum() / ((Y - Y.mean((0, 1))) ** 2).sum())


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K = len(t.goal_cell)
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        t0 = time.time()
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        b1 = net.blocks[1]; b1d = copy.deepcopy(b1).double(); mlp_d = lambda X: b1d.mlp(b1d.ln2(X))
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        dt = R.decisions(t, net, r, K)
        mean_mass, _ = G0.goal_mass(net, rf["tok"], K)
        v = G0.GateFixed(net, mean_mass)                                                     # the natural network, block-0 gates fixed
        g = torch.Generator(device=dev); g.manual_seed(5)
        ok = torch.nonzero(dt["st"] >= 1).squeeze(1); ti = ok[torch.randperm(len(ok), device=dev, generator=g)[:N_DEC]]
        tok, p, st = r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, dt["st"][ti]
        X, Pfix, a_nat, mid_nat, m_nat, x2_nat = B1.inputs(v, tok, p, K)
        X1 = X[torch.arange(len(p), device=dev), :, p].double()                              # the decision token at the input of block 1
        _, Lt = C.states(v, tok, p, K, (2,))
        # the interaction through block 1, gates fixed and natural
        Xn, _, a_n, _, m_n, x2_n = B1.inputs(net, tok, p, K)
        X1n = Xn[torch.arange(len(p), device=dev), :, p].double()
        shares = dict(fixed_x1=interS(X1, st), fixed_attn1=interS(a_nat, st), fixed_mlp1=interS(m_nat, st), fixed_x2=interS(x2_nat, st),
                      natural_x1=interS(X1n, st), natural_attn1=interS(a_n, st), natural_mlp1=interS(m_n, st), natural_x2=interS(x2_n, st))
        del Xn, X1n, a_n, m_n, x2_n
        # same-step pairs among the test decisions (posteriors >= 0.5 apart)
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
        # A-equivalent, B-different pairs among the test decisions (equiv.py's rule)
        Qt = E.qvals(t, dt, ti); cQ = (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Qt.clamp(min=1e-9)) / np.log(t.gamma))
        sett = Qt >= Qt.max(-1, keepdim=True).values - 1e-6
        gq = torch.Generator(device=dev); gq.manual_seed(7); m = len(ti); cand = torch.randint(m, (m, E.CANDS), device=dev, generator=gq)
        same_step = st[:, None] == st[cand]
        dq = (cQ[:, None] - cQ[cand]).abs().amax(-1); same_set = (sett[:, None] == sett[cand]).all(-1); disjoint = ~(sett[:, None] & sett[cand]).any(-1)
        reach = Qt.amax(-1) >= E.MINQ
        eq = same_set & (dq < E.EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > E.DIFF)
        epairs = {}
        for A in range(K):
            for B in range(K):
                if A == B: continue
                okp = same_step & eq[..., A] & dif[..., B] & (cand != torch.arange(m, device=dev)[:, None])
                ii, cc = torch.nonzero(okp, as_tuple=True)
                if len(ii) < 30: continue
                _, first = np.unique(ii.cpu().numpy(), return_index=True); ii, cc = ii[torch.tensor(first, device=dev)], cc[torch.tensor(first, device=dev)]
                jj = cand[ii, cc]
                if len(ii) > 1500:
                    sel = torch.randperm(len(ii), device=dev, generator=gq)[:1500]; ii, jj = ii[sel], jj[sel]
                epairs[(A, B)] = (ii, jj)
        pos = torch.arange(X.shape[2], device=dev)[None].expand(len(p), -1); is_hist = (pos != 1) & (pos <= p[:, None])
        renorm = lambda P: (P * is_hist[:, None]) / (P * is_hist[:, None]).sum(-1, keepdim=True).clamp(min=1e-12)
        agg = {}; add = lambda k, val: agg.setdefault(k, []).append(val); rows = {}
        for A in range(K):
            for B in range(K):
                if A == B: continue
                XA, XB = X[:, A], X[:, B]
                aA, aB, midA, midB, xA, xB, x1A, x1B = a_nat[:, A], a_nat[:, B], mid_nat[:, A], mid_nat[:, B], x2_nat[:, A], x2_nat[:, B], X1[:, A], X1[:, B]
                pd = lambda Y: Y[q2] - Y[q1]                                                   # pair difference
                T_a, T_x = pd(aB) - pd(aA), pd(xB) - pd(xA)                                    # the natural belief-dependent goal difference
                T_mid, T_x1 = pd(midB) - pd(midA), pd(x1B) - pd(x1A)
                row = dict(bd_norm_x1=float(T_x1.norm(dim=1).mean() / pd(x1A).norm(dim=1).mean()), bd_norm_x2=float(T_x.norm(dim=1).mean() / pd(xA).norm(dim=1).mean()),
                           bd_norm_attn=float(T_a.norm(dim=1).mean() / pd(aA).norm(dim=1).mean()),
                           ident_r2_x1=r2(pd(x1A), pd(x1B)), ident_r2_x2=r2(pd(xA), pd(xB)),
                           x1_in_x2_proj=proj(T_x1, T_x)[0], x1_in_x2_r2=r2(T_x1, T_x), attn_in_x2_proj=proj(T_a, T_x)[0], mid_in_x2_r2=r2(T_mid, T_x))
                decA, decB = Lt[:, A].argmax(-1), Lt[:, B].argmax(-1); differ = decA != decB; LA, LB = Lt[:, A], Lt[:, B]
                def patched(x2_new):
                    ns = x2_nat.clone().float(); ns[:, A] = x2_new.float()
                    Le = C.run_edit(v, tok, p, K, 2, ns)[:, A]
                    return float((Le.argmax(-1) == decB)[differ].double().mean()), float((((Le - LA) * (LB - LA)).sum(1) / ((LB - LA) ** 2).sum(1).clamp(min=1e-12)).mean())
                ii, jj = epairs.get((A, B), (None, None))
                def switched(xr):
                    if ii is None: return float("nan")
                    dec1B, dec2B = Lt[ii, B].argmax(-1), Lt[jj, B].argmax(-1); dif_ = dec1B != dec2B
                    ns = x2_nat[ii].clone().float(); ns[:, B] = ns[:, B] + (xr[jj] - xr[ii]).float()
                    Le = C.run_edit(v, tok[ii], p[ii], K, 2, ns)
                    return float((Le[:, B].argmax(-1) == dec2B)[dif_].double().mean()) if int(dif_.sum()) else float("nan")
                for name, parts in ROUTES.items():
                    a_r, P_r, xres = [], [], []
                    for i in range(0, len(p), CHUNK):
                        o = B1.route_attn(b1, XA[i:i + CHUNK], XB[i:i + CHUNK], p[i:i + CHUNK], parts); a_r.append(o[0]); P_r.append(o[1]); xres.append(o[2])
                    a_r, P_r, xres = torch.cat(a_r), torch.cat(P_r), torch.cat(xres)
                    if name == "all": assert float((a_r - aB).abs().max()) < 1e-2, float((a_r - aB).abs().max())
                    mid_r = xres + a_r; x_r = mid_r + mlp_d(mid_r)
                    Ra, Rx = pd(a_r) - pd(aA), pd(x_r) - pd(xA)                                 # the route's belief-dependent goal difference
                    row[f"{name}_attn_proj"], row[f"{name}_attn_r2"] = proj(Ra, T_a)
                    row[f"{name}_x2_proj"], row[f"{name}_x2_r2"] = proj(Rx, T_x)
                    row[f"{name}_tv_hist"] = float((0.5 * (renorm(P_r) - renorm(Pfix[:, A])).abs().sum(-1)).mean())
                    row[f"{name}_switch_B"] = switched(x_r)
                    row[f"{name}_dec_switch"], row[f"{name}_dec_transfer"] = patched(x_r)
                row["direct_B_switch_B"], row["untransformed_A_switch_B"] = switched(xB), switched(xA)
                if ii is not None:
                    gr = torch.Generator(device=dev); gr.manual_seed(3)
                    dB_ = xB[jj] - xB[ii]; rnd = torch.randn(dB_.shape, device=dev, generator=gr, dtype=torch.float64); rnd = rnd / rnd.norm(dim=1, keepdim=True) * dB_.norm(dim=1, keepdim=True)
                    dec1B, dec2B = Lt[ii, B].argmax(-1), Lt[jj, B].argmax(-1); dif_ = dec1B != dec2B
                    ns = x2_nat[ii].clone().float(); ns[:, B] = ns[:, B] + rnd.float()
                    Le = C.run_edit(v, tok[ii], p[ii], K, 2, ns); row["random_switch_B"] = float((Le[:, B].argmax(-1) == dec2B)[dif_].double().mean())
                else:
                    row["random_switch_B"] = float("nan")
                gr = torch.Generator(device=dev); gr.manual_seed(3)
                rnd = torch.randn(xA.shape, device=dev, generator=gr, dtype=torch.float64); rnd = rnd / rnd.norm(dim=1, keepdim=True) * (xB - xA).norm(dim=1, keepdim=True)
                row["random_dec_switch"], row["random_dec_transfer"] = patched(xA + rnd)
                row["differ"] = float(differ.double().mean())
                # the MLP on the belief-dependent difference: a linear map of its input difference?
                lin = R.Ridge(T_mid[:N_PAIRS // 2], (pd(x2_nat[:, B] - mid_nat[:, B]) - pd(x2_nat[:, A] - mid_nat[:, A]))[:N_PAIRS // 2], torch.arange(min(N_PAIRS // 2, len(q1)), device=dev) % 10 == 0)
                Tm = pd(m_nat[:, B]) - pd(m_nat[:, A])
                row["mlp_bd_linear_r2"] = r2(lin(T_mid[N_PAIRS // 2:]), Tm[N_PAIRS // 2:]); row["mlp_bd_gain"] = float(Tm.norm(dim=1).mean() / T_mid.norm(dim=1).mean())
                for k, val in row.items(): add(k, val)
                rows[f"{A}->{B}"] = row
        mean = {k: float(np.nanmean(val)) for k, val in agg.items()}; mean.update(shares)
        res[f"seed{s}"] = dict(mean=mean, by_pair=rows)
        print(f"seed{s} ({time.time() - t0:.0f}s): " + " ".join(f"{k} {val:.3f}" for k, val in mean.items()), flush=True)
        (HERE / "gate1.json").write_text(json.dumps(res))
    report()


def report():
    res = json.loads((HERE / "gate1.json").read_text()); runs = list(res.values())
    mm = lambda k, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([r_["mean"][k] for r_ in runs])
    names = [("weights", "B's attention weights (query and every key), A's values"), ("values", "B's values of the other tokens (history and goal), A's weights"), ("both", "both: B's weights and other-token values"),
             ("self_kv", "decision token's own key and value from B"), ("attn_all", "the whole attention sublayer from B (weights, every value)"), ("resid", "residual only: the token's block-0 state from B, attention at A"), ("all", "all (= the natural change under B)")]
    L = ["# With block 0's gates fixed, where is the remaining goal × belief interaction made?", "",
         f"Generated by `gate1.py` (exploratory). The natural network with block 0's gates fixed per goal and step (section 15), {N_DEC} test decisions after the first step, {N_PAIRS} same-step pairs (posteriors ≥ 0.5 apart). The belief-dependent goal difference of a quantity is (y(h2, B) − y(h2, A)) − (y(h1, B) − y(h1, A)); a route's is the same with the route-swapped state in place of B's, the incoming decision-token residual held at A's. Means over the 12 ordered goal pairs, medians (min–max) over six models.", "",
         "## 1. The interaction at block 1's input and through block 1", "",
         "Variance share of the goal × belief interaction beyond a goal × step offset, at the decision token under every goal:", "",
         "| network | input of block 1 | block-1 attention output | block-1 MLP output | state after block 1 |", "|---|---|---|---|---|",
         f"| natural | {mm('natural_x1')} | {mm('natural_attn1')} | {mm('natural_mlp1')} | {mm('natural_x2')} |",
         f"| block-0 gates fixed | {mm('fixed_x1')} | {mm('fixed_attn1')} | {mm('fixed_mlp1')} | {mm('fixed_x2')} |", "",
         f"On the same-step pairs, gates fixed: the belief-dependent goal difference is {mm('bd_norm_x1')} of the A-written difference's norm at the input of block 1, {mm('bd_norm_attn')} at block 1's attention output, {mm('bd_norm_x2')} after block 1; "
         f"R² of the A-written difference for the B-written one {mm('ident_r2_x1')} at the input of block 1, {mm('ident_r2_x2')} after it. "
         f"The input's belief-dependent difference carried unchanged against the output's: projection {mm('x1_in_x2_proj')}, R² {mm('x1_in_x2_r2')}; the attention output's alone: projection {mm('attn_in_x2_proj')}; input plus attention (the MLP's input): R² {mm('mid_in_x2_r2')}. "
         f"Block 1's MLP on the belief-dependent difference: a linear map of its input difference across pairs has R² {mm('mlp_bd_linear_r2')}; gain {mm('mlp_bd_gain')}.", "",
         "## 2. Routes at block 1 (residual at A)", "",
         "| goal source switched A → B | pattern change: TV over history tokens | attention output: projection on the natural belief-dependent difference / R² | after block 1's MLP: projection / R² | B's decision switched by the route's pair difference | A's decision switched to B's by the route's state / logit transfer |",
         "|---|---|---|---|---|---|"]
    for k, lab in names:
        L.append(f"| {lab} | {mm(k + '_tv_hist')} | {mm(k + '_attn_proj')} / {mm(k + '_attn_r2')} | {mm(k + '_x2_proj')} / {mm(k + '_x2_r2')} | {mm(k + '_switch_B')} | {mm(k + '_dec_switch')} / {mm(k + '_dec_transfer')} |")
    L += [f"| direct B-written difference (ceiling) / untransformed A-written (floor) / random | | | | {mm('direct_B_switch_B')} / {mm('untransformed_A_switch_B')} / {mm('random_switch_B')} | random: {mm('random_dec_switch')} / {mm('random_dec_transfer')} |", "",
          f"Decisions where the gate-fixed model's moves under A and B differ: {mm('differ')}."]
    (HERE / "gate1.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
