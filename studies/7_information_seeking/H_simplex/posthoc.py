"""Post hoc (after the registered run): what the nonlinearity of H in the belief is.
    log-mixture             H as a logit of a belief-weighted action probability: H ≈ log Σ_s b(s) p_s(a) (+ step bias),
                            centred — linear in b in probability space; fitted (i) by ridge of softmax(H) on b, then the
                            log, and (ii) by gradient on per-cell probability tables p_s; the fitted form also run in the
                            additive code offline and online;
    direction / magnitude   R² of the affine fit for H's unit direction; the magnitude ||H|| against the entropy and the
                            most likely cell's probability (mean by entropy bin, correlation);
    piecewise affine        H ≈ Σ_s b(s) h_s^(k) with a separate table per entropy bin k (8 bins): R²; are the tables
                            the same direction at different gains? (correlation and norm ratio between bins);
    other features          affine in log b, in sharpened beliefs b^τ (τ 2, 4, 8), and in [b, b ⊗ entropy bin].

    .venv/bin/python studies/7_information_seeking/H_simplex/posthoc.py        # writes posthoc.json, posthoc.md
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR          # noqa: E402

ENT_BINS = 8


def fit_logmix(d_fit, va, n, H_len, steps=4000, seed=0):
    """H ≈ centre(log(b @ softmax(theta)) + beta[step]); theta [n, 4] per-cell action logits, fitted by Adam on the fit
    rows with early stopping on the validation episodes."""
    torch.manual_seed(seed)
    dev = d_fit["b"].device
    theta = torch.zeros(n, 4, device=dev, requires_grad=True)
    beta = torch.zeros(R.STEP_BINS, 4, device=dev, requires_grad=True)
    b, Y = d_fit["b"].float(), d_fit["H"].float()
    sb = (d_fit["st"] * R.STEP_BINS // H_len).clamp(max=R.STEP_BINS - 1)
    tr, vv = ~va, va

    def model(b, sb):
        z = (b @ torch.softmax(theta, 1)).clamp(min=1e-6).log() + beta[sb]
        return z - z.mean(1, keepdim=True)

    opt = torch.optim.Adam([theta, beta], lr=0.05)
    best, state = float("inf"), None
    with torch.enable_grad():
        for i in range(steps):
            loss = ((model(b[tr], sb[tr]) - Y[tr]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if i % 50 == 0 or i == steps - 1:
                with torch.no_grad():
                    v = float(((model(b[vv], sb[vv]) - Y[vv]) ** 2).mean())
                if v < best:
                    best, state = v, (theta.detach().clone(), beta.detach().clone())
    theta, beta = state

    def pred(d):
        sb = (d["st"] * R.STEP_BINS // H_len).clamp(max=R.STEP_BINS - 1)
        z = (d["b"].float() @ torch.softmax(theta, 1)).clamp(min=1e-6).log() + beta[sb]
        return (z - z.mean(1, keepdim=True)).double()
    return pred, torch.softmax(theta, 1).detach()


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len = len(t.goal_cell), t.H
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        va = df["ep"] % 10 == 0
        S = lambda d: R.step_onehot(d["st"], H_len)
        Y, Yt, ym = df["H"], dt["H"], df["H"].mean(0)
        row = {}
        fit = lambda f: R.r2(R.Ridge(f(df), Y, va)(f(dt)), Yt, ym)
        # the log-mixture: H as the logit of a belief-weighted action probability
        Q, Qt = torch.softmax(Y, 1), torch.softmax(Yt, 1)
        mq = R.Ridge(torch.cat([S(df), df["b"]], 1), Q, va)
        logq = lambda d: (lambda z: z - z.mean(1, keepdim=True))(mq(torch.cat([S(d), d["b"]], 1)).clamp(min=1e-4).log())
        row["logmix_ridge_r2"] = R.r2(logq(dt), Yt, ym)
        row["prob_space_affine_r2"] = R.r2(mq(torch.cat([S(dt), dt["b"]], 1)), Qt, Q.mean(0))               # linear fit of the probabilities themselves
        pred_lm, P_cells = fit_logmix(df, va, t.n, H_len)
        row["logmix_r2"] = R.r2(pred_lm(dt), Yt, ym)
        G = R.goal_bias(df, K, H_len)
        pred = {"logmix": pred_lm, "logmix_ridge": logq}
        off = R.offline(dt, pred, G, ["logmix", "logmix_ridge"])
        row["offline"] = off
        H_policy = lambda env, s_: (MS.logits_goals(net, env, s_).mean(1) + G[env.goal, min(s_, R.SB - 1)]).argmax(-1)
        on = dict(natural=MS.episodes(t, MS.natural(net), goal, cell, 79)[0], additive=MS.episodes(t, H_policy, goal, cell, 79)[0],
                  goal_blind=MS.episodes(t, lambda env, s_: MS.logits_goals(net, env, s_).mean(1).argmax(-1), goal, cell, 79)[0],
                  logmix=MS.episodes(t, R.fitted_policy(pred_lm, G, t, H_len), goal, cell, 79)[0])
        rec = lambda x: (x - on["goal_blind"]["ret"]) / (on["natural"]["ret"] - on["goal_blind"]["ret"])
        row["online"] = on; row["recovery"] = {k: rec(v["ret"]) for k, v in on.items()}
        # the per-cell probability tables against the candidates
        popt, reach = R.cell_candidates(t)
        corr = lambda a, b_: float(torch.corrcoef(torch.stack([a.flatten(), b_.flatten()]))[0, 1])
        Pc = (P_cells.double() - P_cells.double().mean(1, keepdim=True))
        row["P_cells_corr_popt"] = corr(Pc, popt - popt.mean(1, keepdim=True))
        row["P_cells_top_optimal_some_goal"] = float((popt > 0).gather(1, P_cells.argmax(1, keepdim=True)).double().mean())
        row["P_cells"] = P_cells.cpu().tolist()
        # direction and magnitude
        norm = lambda d: d["H"].norm(dim=1, keepdim=True).clamp(min=1e-9)
        Yd, Ydt = Y / norm(df), Yt / norm(dt)
        row["direction_affine_r2"] = R.r2(R.Ridge(torch.cat([S(df), df["b"]], 1), Yd, va)(torch.cat([S(dt), dt["b"]], 1)), Ydt, Yd.mean(0))
        row["direction_mlp_r2"] = R.r2(R.fit_mlp(torch.cat([S(df), df["b"]], 1), Yd, va)(torch.cat([S(dt), dt["b"]], 1)), Ydt, Yd.mean(0))
        mag, ent, pmax = norm(dt).squeeze(1), dt["ent"], dt["b"].max(1).values
        corr1 = lambda a, b_: float(torch.corrcoef(torch.stack([a, b_]))[0, 1])
        row["magnitude_corr_entropy"], row["magnitude_corr_pmax"] = corr1(mag, ent), corr1(mag, pmax)
        eb = (ent / 4.0 * ENT_BINS).long().clamp(max=ENT_BINS - 1)
        row["magnitude_by_entropy_bin"] = [float(mag[eb == k].mean()) if (eb == k).any() else None for k in range(ENT_BINS)]
        row["decisions_by_entropy_bin"] = [int((eb == k).sum()) for k in range(ENT_BINS)]
        # magnitude predicted from the belief alone (ridge on b and entropy, and the MLP)
        fm = lambda d: torch.cat([S(d), d["b"], d["ent"][:, None], d["b"].max(1).values[:, None]], 1)
        m_fit = norm(df).squeeze(1)
        row["magnitude_r2_ridge"] = R.r2(R.Ridge(fm(df), m_fit[:, None], va)(fm(dt)).squeeze(1), mag, m_fit.mean())
        row["magnitude_r2_mlp"] = R.r2(R.fit_mlp(torch.cat([S(df), df["b"]], 1), m_fit[:, None], va)(torch.cat([S(dt), dt["b"]], 1)).squeeze(1), mag, m_fit.mean())
        # piecewise affine in entropy bins
        def pw(d):
            k = (d["ent"] / 4.0 * ENT_BINS).long().clamp(max=ENT_BINS - 1)
            oh = torch.nn.functional.one_hot(k, ENT_BINS).double()
            return torch.cat([S(d), (d["b"][:, :, None] * oh[:, None, :]).flatten(1), oh], 1)
        m = R.Ridge(pw(df), Y, va)
        row["piecewise_affine_r2"] = R.r2(m(pw(dt)), Yt, ym)
        Wb = m.W[R.STEP_BINS: R.STEP_BINS + t.n * ENT_BINS].reshape(t.n, ENT_BINS, 4)             # [n, bin, 4]
        Wb = Wb - Wb.mean(-1, keepdim=True)
        flat = Wb.permute(1, 0, 2).reshape(ENT_BINS, -1)
        C = torch.corrcoef(flat)
        row["piecewise_table_corr_between_bins"] = C.cpu().tolist()
        row["piecewise_table_norm_by_bin"] = flat.norm(dim=1).cpu().tolist()
        # other feature sets
        row["log_belief_r2"] = fit(lambda d: torch.cat([S(d), (d["b"] + 1e-3).log()], 1))
        for tau in (2, 4, 8):
            sh = lambda d, tau=tau: (d["b"] ** tau) / (d["b"] ** tau).sum(1, keepdim=True)
            row[f"sharp{tau}_r2"] = fit(lambda d, sh=sh: torch.cat([S(d), sh(d)], 1))
            row[f"affine+sharp{tau}_r2"] = fit(lambda d, sh=sh: torch.cat([S(d), d["b"], sh(d)], 1))
        row["affine+entropy_bins_r2"] = fit(lambda d: torch.cat([S(d), d["b"], torch.nn.functional.one_hot((d["ent"] / 4.0 * ENT_BINS).long().clamp(max=ENT_BINS - 1), ENT_BINS).double()], 1))
        res[f"seed{s}"] = row
        print(f"seed{s}: LOG-MIX R2 {row['logmix_r2']:.3f} (ridge {row['logmix_ridge_r2']:.3f}; prob-space affine {row['prob_space_affine_r2']:.3f}) offline {off['logmix']:.3f} vs H {off['H']:.3f}, "
              f"recovery {row['recovery']['logmix']:.2f} vs additive {row['recovery']['additive']:.2f}; P_cells top-ok {row['P_cells_top_optimal_some_goal']:.2f} corr popt {row['P_cells_corr_popt']:.2f}")
        print(f"seed{s}: direction affine {row['direction_affine_r2']:.3f} (mlp {row['direction_mlp_r2']:.3f}) | |H| corr entropy {row['magnitude_corr_entropy']:.2f} pmax {row['magnitude_corr_pmax']:.2f}, "
              f"|H| R2 ridge {row['magnitude_r2_ridge']:.2f} mlp {row['magnitude_r2_mlp']:.2f} | piecewise affine {row['piecewise_affine_r2']:.3f} | log b {row['log_belief_r2']:.3f} "
              f"sharp 2/4/8 {row['sharp2_r2']:.3f}/{row['sharp4_r2']:.3f}/{row['sharp8_r2']:.3f} affine+sharp8 {row['affine+sharp8_r2']:.3f} affine+ent bins {row['affine+entropy_bins_r2']:.3f}", flush=True)
    (HERE / "posthoc.json").write_text(json.dumps(res))
    med = lambda k: float(np.median([res[f'seed{s}'][k] for s in range(6)]))
    mm2 = lambda res, a, b: (lambda xs: f"{np.median(xs):.3f} ({min(xs):.3f}–{max(xs):.3f})")([res[f'seed{s}'][a][b] for s in range(6)])
    mm = lambda k: f"{med(k):.3f} ({min(res[f'seed{s}'][k] for s in range(6)):.3f}–{max(res[f'seed{s}'][k] for s in range(6)):.3f})"
    L = ["# H on the belief simplex: post hoc", "", "Generated by `posthoc.py` after the registered run. Medians (min–max) over the six models, held-out R² as in `tables.md`.", "",
         "| model of H | R² |", "|---|---|",
         f"| **log-mixture**: H ≈ centre(log Σ_s b(s) p_s(a) + step bias), per-cell probability tables fitted by gradient | **{mm('logmix_r2')}** |",
         f"| log-mixture by ridge: softmax(H) ≈ affine in b, then the log | {mm('logmix_ridge_r2')} |",
         f"| (the probabilities softmax(H) themselves, affine in b: R² in probability space) | {mm('prob_space_affine_r2')} |",
         f"| affine fit of H's **unit direction** | {mm('direction_affine_r2')} |", f"| MLP fit of the direction | {mm('direction_mlp_r2')} |",
         f"| **piecewise affine**: a per-cell table per entropy bin (8 bins) | {mm('piecewise_affine_r2')} |",
         f"| affine + entropy-bin intercepts | {mm('affine+entropy_bins_r2')} |",
         f"| affine in log b | {mm('log_belief_r2')} |",
         f"| affine in b² (normalised) | {mm('sharp2_r2')} |", f"| affine in b⁴ | {mm('sharp4_r2')} |", f"| affine in b⁸ | {mm('sharp8_r2')} |", f"| affine in [b, b⁸] | {mm('affine+sharp8_r2')} |", "",
         "Decisions with the log-mixture Ĥ in the additive code: offline agreement where the goal matters " + mm2(res, 'offline', 'logmix') + " against H " + mm2(res, 'offline', 'H') +
         "; online recovery " + mm2(res, 'recovery', 'logmix') + " against the additive code's " + mm2(res, 'recovery', 'additive') + ". Its per-cell tables p_s: top action a shortest-path move for some goal in " +
         mm('P_cells_top_optimal_some_goal') + " of cells; correlation with popt " + mm('P_cells_corr_popt') + ".", "",
         "| magnitude ‖H‖ | value |", "|---|---|",
         f"| correlation with the posterior entropy | {mm('magnitude_corr_entropy')} |", f"| correlation with the most likely cell's probability | {mm('magnitude_corr_pmax')} |",
         f"| R² from [b, entropy, max b] (ridge) | {mm('magnitude_r2_ridge')} |", f"| R² from b (MLP) | {mm('magnitude_r2_mlp')} |", "",
         "Mean ‖H‖ by entropy bin (bins of 0.5 nat from 0; seed medians): " + ", ".join(
             f"{np.median([res[f'seed{s}']['magnitude_by_entropy_bin'][k] or np.nan for s in range(6)]):.2f}" for k in range(ENT_BINS)) + ".", "",
         "Piecewise tables: correlation between the entropy bins' per-cell tables (seed 0, bins 0–7):", "", "```"] + [
         " ".join(f"{v:5.2f}" for v in rowc) for rowc in res["seed0"]["piecewise_table_corr_between_bins"]] + ["```", "",
         "Norm of each bin's table (seed 0): " + ", ".join(f"{v:.2f}" for v in res["seed0"]["piecewise_table_norm_by_bin"]) + "."]
    (HERE / "posthoc.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
