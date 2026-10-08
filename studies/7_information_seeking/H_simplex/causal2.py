"""Through the posterior or in parallel, decided by decomposition. At sites 1, 2, 4 the goal-averaged state of the
decision token is split as x̄ = f(b) + r, f a nonlinear (MLP) encoder of the state from the posterior fitted on the fit
decisions. On the matched pairs of causal.py, under every goal: swap the posterior part (x_A + f(b_B) − f(b_A)), swap
the residual part (x_A + r_B − r_A), both (= whole goal-averaged difference), the linear encoder's edit for reference,
and a random direction of the posterior part's norm. Read-out as in causal.py.

    .venv/bin/python studies/7_information_seeking/H_simplex/causal2.py        # writes causal2.md / causal2.json
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C          # noqa: E402

SITES = (1, 2, 4)


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
        G = R.goal_bias(df, K, H_len)
        gsub = torch.Generator(device=dev); gsub.manual_seed(1)
        fi = torch.randperm(len(df["st"]), device=dev, generator=gsub)[:24576]
        Xf, _ = C.states(net, rf["tok"][df["ep"][fi]], df["st"][fi] + 1, K, SITES)
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:8192]
        Xt, _ = C.states(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K, SITES)
        S = lambda d, idx: R.step_onehot(d["st"][idx], H_len)
        va = df["ep"][fi] % 10 == 0
        # the same pairs as causal.py
        gp = torch.Generator(device=dev); gp.manual_seed(5)
        A = torch.randperm(len(dt["st"]), device=dev, generator=gp)[:C.N_PAIRS * 3]
        Bc = torch.randperm(len(dt["st"]), device=dev, generator=gp)[:C.N_PAIRS * 3]
        by_step = {}
        for i in Bc.tolist():
            by_step.setdefault(int(dt["st"][i]), []).append(i)
        pairs = []
        for i in A.tolist():
            cands = by_step.get(int(dt["st"][i]), [])
            for _ in range(8):
                if not cands: break
                j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
                if j != i and float((dt["b"][i] - dt["b"][j]).abs().sum()) >= C.MIN_DIFF:
                    pairs.append((i, j)); break
            if len(pairs) >= C.N_PAIRS: break
        ia, ib = torch.tensor([p_[0] for p_ in pairs], device=dev), torch.tensor([p_[1] for p_ in pairs], device=dev)
        ta, tb = r["tok"][dt["ep"][ia]], r["tok"][dt["ep"][ib]]
        pa = dt["st"][ia] + 1
        XA, LA = C.states(net, ta, pa, K, SITES); XB, LB_ = C.states(net, tb, pa, K, SITES)
        HA, HB = LA.mean(1), LB_.mean(1)
        dH = HB - HA; dH2 = (dH ** 2).sum(1).clamp(min=1e-9)
        ga = dt["goal"][ia]; ar = torch.arange(len(ia), device=dev)
        Gt = G[ga, dt["st"][ia].clamp(max=R.SB - 1)]
        decA, decB = (HA + Gt).argmax(1), (HB + Gt).argmax(1); differ = decA != decB
        row = {}
        gr = torch.Generator(device=dev); gr.manual_seed(11)
        for l in SITES:
            xf, xt = Xf[l].mean(1).double(), Xt[l].mean(1).double()
            Fb = lambda d, idx: torch.cat([S(d, idx), d["b"][idx], (d["b"][idx] + 1e-3).log()], 1)      # b and log b as inputs
            f = R.fit_mlp(Fb(df, fi), xf, va, steps=4000, d=256)
            lin = R.Ridge(torch.cat([S(df, fi), df["b"][fi]], 1), xf, va)
            enc_r2 = R.r2(f(Fb(dt, ti)), xt, xf.mean(0)); lin_r2 = R.r2(lin(torch.cat([S(dt, ti), dt["b"][ti]], 1)), xt, xf.mean(0))
            dec = R.Ridge(torch.cat([S(df, fi), xf], 1), df["b"][fi], va)
            xa_bar, xb_bar = XA[l].mean(1).double(), XB[l].mean(1).double()
            fa, fb = f(Fb(dt, ia)), f(Fb(dt, ib))
            ra, rb = xa_bar - fa, xb_bar - fb
            d_post, d_res = (fb - fa).float(), (rb - ra).float()
            d_lin = (lin(torch.cat([S(dt, ia), dt["b"][ib]], 1)) - lin(torch.cat([S(dt, ia), dt["b"][ia]], 1))).float()
            rnd = torch.randn(d_post.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * d_post.norm(dim=1, keepdim=True)
            kinds = {"posterior_part": d_post, "residual_part": d_res, "both_parts": d_post + d_res, "linear_edit": d_lin, "random_posterior_norm": rnd}
            out = dict(encoder_r2_mlp=enc_r2, encoder_r2_linear=lin_r2, residual_share_of_state_variance=float(1 - enc_r2),
                       residual_norm_over_posterior_norm=float((d_res.norm(dim=1) / d_post.norm(dim=1).clamp(min=1e-9)).median()))
            for k, dv in kinds.items():
                ns = XA[l] + dv[:, None]
                Le = C.run_edit(net, ta, pa, K, l, ns); He = Le.mean(1)
                tr = ((He - HA) * dH).sum(1) / dH2
                de = (He + Gt).argmax(1)
                out[k] = dict(H_transfer=float(tr.mean()), H_transfer_median=float(tr.median()),
                              decision_to_donor=float((de == decB)[differ].double().mean()), decision_kept=float((de == decA)[differ].double().mean()),
                              decoded_posterior_vs_donor_r2=R.r2(dec(torch.cat([S(dt, ia), ns.mean(1).double()], 1)), dt["b"][ib], df["b"][fi].mean(0)),
                              edit_norm_over_state=float((dv.norm(dim=1) / XA[l].mean(1).norm(dim=1)).mean()))
            # whole goal-wise swap for reference (as causal.py)
            Le = C.run_edit(net, ta, pa, K, l, XB[l]); He = Le.mean(1)
            out["whole_swap"] = dict(H_transfer=float((((He - HA) * dH).sum(1) / dH2).mean()), decision_to_donor=float((((He + Gt).argmax(1)) == decB)[differ].double().mean()))
            row[l] = out
            print(f"seed{s} site {l}: encoder mlp {enc_r2:.2f} lin {lin_r2:.2f} | " + " | ".join(f"{k} H {out[k]['H_transfer']:.2f} dec {out[k]['decision_to_donor']:.2f} post {out[k]['decoded_posterior_vs_donor_r2']:.2f}" for k in kinds)
                  + f" | whole {out['whole_swap']['H_transfer']:.2f}/{out['whole_swap']['decision_to_donor']:.2f}", flush=True)
        res[f"seed{s}"] = row
        (HERE / "causal2.json").write_text(json.dumps(res))
    res = json.loads((HERE / "causal2.json").read_text())
    seeds = list(res)
    mm = lambda f, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([f(res[s_]) for s_ in seeds])
    L = ["# Through the posterior or in parallel: posterior-part and residual-part swaps", "",
         "Generated by `causal2.py` (exploratory). The decision token's goal-averaged state at each site split as x̄ = f(b) + r, f an MLP encoder from the posterior (inputs b and log b, "
         "fitted on 24 576 fit decisions); on the matched pairs of `causal.py`, the recipient's state under every goal plus the donor-minus-recipient difference of one part. Medians (min–max) over six models.", "",
         "| site | encoder R² (MLP / linear) | edit | H transfer | decision to donor | decoded posterior vs donor | edit norm / state norm |", "|---|---|---|---|---|---|---|"]
    for l in SITES:
        for k in ("posterior_part", "residual_part", "both_parts", "linear_edit", "random_posterior_norm", "whole_swap"):
            e_ = lambda f_, k=k: mm(lambda r_: r_[str(l)][k][f_]) if f_ in res[seeds[0]][str(l)][k] else "—"
            L.append(f"| {l} | {mm(lambda r_: r_[str(l)]['encoder_r2_mlp'])} / {mm(lambda r_: r_[str(l)]['encoder_r2_linear'])} | {k} | {e_('H_transfer')} | {e_('decision_to_donor')} | {e_('decoded_posterior_vs_donor_r2')} | {e_('edit_norm_over_state')} |")
    L += ["", "Residual norm over posterior-part norm (median pair): " + ", ".join(f"site {l}: {mm(lambda r_, l=l: r_[str(l)]['residual_norm_over_posterior_norm'])}" for l in SITES) + "."]
    (HERE / "causal2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
