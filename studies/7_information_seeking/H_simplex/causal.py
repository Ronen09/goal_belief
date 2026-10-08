"""Does the network compute H through its encoded posterior, or are posterior and H parallel consequences of the
history? Exploratory (no registration). At the decision token's residual stream after each block (sites 0..4):
    decoders   of the posterior b and of log b from the goal-averaged state (ridge, held-out R²; MLP for b);
    edits      on matched pairs (A recipient, B donor; same step; ||b_A - b_B||_1 >= 0.5): x_A(g) + (b_B - b_A) W_enc under
               every goal g, W_enc the ridge map from b to the goal-averaged state; controls: a random direction of the
               same norm, and the whole-token swap x_A(g) -> x_B(g) (ceiling); also the edit in log b coordinates;
    read-out   transfer of H: <H_edit - H_A, H_B - H_A> / ||H_B - H_A||²; of the decision: among pairs whose decisions
               under A's goal differ, the share now taking the donor's; and the decoded posterior of the edited state
               against b_B (does the edit set the code?).

    .venv/bin/python studies/7_information_seeking/H_simplex/causal.py        # writes causal.md / causal.json
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR          # noqa: E402

N_PAIRS, MIN_DIFF, CHUNK = 4096, 0.5, 2048


@torch.no_grad()
def states(net, tok, p, K, sites):
    """Residual states at the decision token p [N] under every goal: {site: [N, K, d]} and the centred logits [N, K, 4]."""
    N, dev = len(tok), tok.device
    out = {l: [] for l in sites}; lg = []
    for i in range(0, N, CHUNK):
        tk = tok[i:i + CHUNK].repeat(K, 1, 1); pp = p[i:i + CHUNK].repeat(K)
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(len(tok[i:i + CHUNK])) + 1
        logits, _, rec = net(tk, record=True)
        ar = torch.arange(len(tk), device=dev)
        for l in sites:
            out[l].append(rec["resid"][l][ar, pp].view(K, -1, rec["resid"][l].shape[-1]).transpose(0, 1))
        z = logits[ar, pp].view(K, -1, 4).transpose(0, 1).double()
        lg.append(z - z.mean(-1, keepdim=True))
    return {l: torch.cat(v) for l, v in out.items()}, torch.cat(lg)


@torch.no_grad()
def run_edit(net, tok, p, K, site, new_state):
    """Logits at p under every goal with the decision token's residual at `site` replaced by new_state [N, K, d]."""
    N, dev = len(tok), tok.device
    lg = []
    for i in range(0, N, CHUNK):
        tk = tok[i:i + CHUNK].repeat(K, 1, 1); pp = p[i:i + CHUNK].repeat(K); n_ = len(tok[i:i + CHUNK])
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
        ns = new_state[i:i + CHUNK].transpose(0, 1).reshape(K * n_, -1)
        ar = torch.arange(len(tk), device=dev)

        def patch(x):
            x = x.clone(); x[ar, pp] = ns.to(x.dtype); return x
        logits, _ = net(tk, patch={("resid", site): patch})
        z = logits[ar, pp].view(K, n_, 4).transpose(0, 1).double()
        lg.append(z - z.mean(-1, keepdim=True))
    return torch.cat(lg)


def main():
    if "--tables" in sys.argv:
        return tables(json.loads((HERE / "causal.json").read_text()))
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
        sites = list(range(net.nl + 1))
        _, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        _, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        G = R.goal_bias(df, K, H_len)
        # states of the fit decisions (a subsample) for the decoders and encoders
        gsub = torch.Generator(device=dev); gsub.manual_seed(1)
        fi = torch.randperm(len(df["st"]), device=dev, generator=gsub)[:16384]
        Xf, _ = states(net, rf["tok"][df["ep"][fi]], df["st"][fi] + 1, K, sites)
        ti = torch.randperm(len(dt["st"]), device=dev, generator=gsub)[:8192]
        Xt, _ = states(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K, sites)
        S = lambda d, idx: R.step_onehot(d["st"][idx], H_len)
        LB = lambda b: (b + 1e-3).log()
        va = df["ep"][fi] % 10 == 0
        row = {"decode": {}, "edit": {}}
        enc, enc_log, dec = {}, {}, {}
        for l in sites:
            xf, xt = Xf[l].mean(1).double(), Xt[l].mean(1).double()                                  # goal-averaged states
            dec[l] = R.Ridge(torch.cat([S(df, fi), xf], 1), df["b"][fi], va)
            r2b = R.r2(dec[l](torch.cat([S(dt, ti), xt], 1)), dt["b"][ti], df["b"][fi].mean(0))
            r2log = R.r2(R.Ridge(torch.cat([S(df, fi), xf], 1), LB(df["b"][fi]), va)(torch.cat([S(dt, ti), xt], 1)), LB(dt["b"][ti]), LB(df["b"][fi]).mean(0))
            r2mlp = R.r2(R.fit_mlp(torch.cat([S(df, fi), xf], 1), df["b"][fi], va)(torch.cat([S(dt, ti), xt], 1)), dt["b"][ti], df["b"][fi].mean(0))
            enc[l] = R.Ridge(torch.cat([S(df, fi), df["b"][fi]], 1), xf, va)                          # x̄ ≈ step + b W
            enc_log[l] = R.Ridge(torch.cat([S(df, fi), LB(df["b"][fi])], 1), xf, va)
            r2enc = R.r2(enc[l](torch.cat([S(dt, ti), dt["b"][ti]], 1)), xt, xf.mean(0))
            row["decode"][l] = dict(posterior_linear=r2b, log_posterior_linear=r2log, posterior_mlp=r2mlp, state_from_posterior=r2enc)
        # matched pairs: same step, posteriors differing by >= MIN_DIFF
        gp = torch.Generator(device=dev); gp.manual_seed(5)
        A = torch.randperm(len(dt["st"]), device=dev, generator=gp)[:N_PAIRS * 3]
        Bc = torch.randperm(len(dt["st"]), device=dev, generator=gp)[:N_PAIRS * 3]
        by_step = {}
        for i in Bc.tolist():
            by_step.setdefault(int(dt["st"][i]), []).append(i)
        pairs = []
        for i in A.tolist():
            cands = by_step.get(int(dt["st"][i]), [])
            for _ in range(8):
                if not cands: break
                j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
                if j != i and float((dt["b"][i] - dt["b"][j]).abs().sum()) >= MIN_DIFF:
                    pairs.append((i, j)); break
            if len(pairs) >= N_PAIRS: break
        ia, ib = torch.tensor([p_[0] for p_ in pairs], device=dev), torch.tensor([p_[1] for p_ in pairs], device=dev)
        ta, tb = r["tok"][dt["ep"][ia]], r["tok"][dt["ep"][ib]]
        pa = dt["st"][ia] + 1
        XA, LA = states(net, ta, pa, K, sites); XB, LB_ = states(net, tb, pa, K, sites)
        HA, HB = LA.mean(1), LB_.mean(1)
        dH = HB - HA; dH2 = (dH ** 2).sum(1).clamp(min=1e-9)
        ga = dt["goal"][ia]; ar = torch.arange(len(ia), device=dev)
        Gt = G[ga, dt["st"][ia].clamp(max=R.SB - 1)]
        decA, decB = (HA + Gt).argmax(1), (HB + Gt).argmax(1)
        differ = decA != decB
        own = lambda L_: L_[ar, ga]                                                                    # logits under A's own goal
        decA_own, decB_own = own(LA).argmax(1), own(LB_).argmax(1)
        differ_own = decA_own != decB_own
        row["pairs"] = dict(n=len(pairs), decision_differs=float(differ.double().mean()), decision_differs_own_goal=float(differ_own.double().mean()),
                            belief_l1=float((dt["b"][ia] - dt["b"][ib]).abs().sum(1).mean()))
        gr = torch.Generator(device=dev); gr.manual_seed(11)
        for l in sites:
            d_lin = (enc[l](torch.cat([S(dt, ia), dt["b"][ib]], 1)) - enc[l](torch.cat([S(dt, ia), dt["b"][ia]], 1))).float()      # [N, d] the same under every goal
            d_log = (enc_log[l](torch.cat([S(dt, ia), LB(dt["b"][ib])], 1)) - enc_log[l](torch.cat([S(dt, ia), LB(dt["b"][ia])], 1))).float()
            rnd = torch.randn(d_lin.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * d_lin.norm(dim=1, keepdim=True)
            kinds = {"belief_edit": XA[l] + d_lin[:, None], "log_belief_edit": XA[l] + d_log[:, None], "random_same_norm": XA[l] + rnd[:, None], "whole_swap": XB[l]}
            out = {}
            for k, ns in kinds.items():
                Le = run_edit(net, ta, pa, K, l, ns)
                He = Le.mean(1)
                tr = ((He - HA) * dH).sum(1) / dH2
                de = (He + Gt).argmax(1); de_own = own(Le).argmax(1)
                xe = ns.mean(1).double()
                dec_r2 = R.r2(dec[l](torch.cat([S(dt, ia), xe], 1)), dt["b"][ib], df["b"][fi].mean(0))
                out[k] = dict(H_transfer=float(tr.mean()), H_transfer_median=float(tr.median()),
                              decision_to_donor=float((de == decB)[differ].double().mean()), decision_kept=float((de == decA)[differ].double().mean()),
                              own_goal_to_donor=float((de_own == decB_own)[differ_own].double().mean()),
                              decoded_posterior_vs_donor_r2=dec_r2, edit_norm=float((ns - XA[l]).norm(dim=-1).mean()), state_norm=float(XA[l].norm(dim=-1).mean()))
            row["edit"][l] = out
        res[f"seed{s}"] = row
        print(f"seed{s}: pairs {row['pairs']} | " + " | ".join(
            f"site {l}: dec b {row['decode'][l]['posterior_linear']:.2f} (mlp {row['decode'][l]['posterior_mlp']:.2f}, log {row['decode'][l]['log_posterior_linear']:.2f}); "
            f"edit H {row['edit'][l]['belief_edit']['H_transfer']:.2f} dec {row['edit'][l]['belief_edit']['decision_to_donor']:.2f} | log-edit {row['edit'][l]['log_belief_edit']['H_transfer']:.2f}/{row['edit'][l]['log_belief_edit']['decision_to_donor']:.2f} "
            f"| rnd {row['edit'][l]['random_same_norm']['H_transfer']:.2f}/{row['edit'][l]['random_same_norm']['decision_to_donor']:.2f} | swap {row['edit'][l]['whole_swap']['H_transfer']:.2f}/{row['edit'][l]['whole_swap']['decision_to_donor']:.2f}"
            for l in sites), flush=True)
        (HERE / "causal.json").write_text(json.dumps(res))
    tables(json.loads((HERE / "causal.json").read_text()))


def tables(res):
    seeds = list(res); sites = sorted(int(k) for k in res[seeds[0]]["decode"])
    mm = lambda f, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([f(res[s_]) for s_ in seeds])
    L = ["# Through the encoded posterior, or in parallel? Belief-encoding edits at the decision token", "",
         "Generated by `causal.py` (exploratory). Six maze10 models; sites: the decision token's residual stream entering block l (site 0: the embedding; site 4: the final residual). "
         f"Pairs: {mm(lambda r_: r_['pairs']['n'], 0)} per model, same step, posteriors differing by {mm(lambda r_: r_['pairs']['belief_l1'])} of mass (L1); their decisions under A's goal differ in "
         f"{mm(lambda r_: r_['pairs']['decision_differs'])} (H + G) / {mm(lambda r_: r_['pairs']['decision_differs_own_goal'])} (the network's own logits). Medians (min–max).", "",
         "## Where the posterior is encoded (held-out R² from the goal-averaged state)", "", "| site | posterior, linear | posterior, MLP | log posterior, linear | state from posterior (encoder) |", "|---|---|---|---|---|"]
    for l in sites:
        d = lambda k: mm(lambda r_: r_["decode"][str(l)][k])
        L.append(f"| {l} | {d('posterior_linear')} | {d('posterior_mlp')} | {d('log_posterior_linear')} | {d('state_from_posterior')} |")
    L += ["", "## Edits: transfer of H toward the donor's and of the decision to the donor's", "",
          "H transfer: the projection of (H_edit − H_A) on (H_B − H_A), 1 = the donor's H. Decision to donor: among pairs whose decisions differ, the share now taking the donor's (H + G under A's goal). "
          "Decoded posterior: R² of the edited state's decoded posterior against the donor's (does the edit set the code?).", "",
          "| site | edit | H transfer | decision to donor | own-goal decision to donor | decoded posterior vs donor | edit norm / state norm |", "|---|---|---|---|---|---|---|"]
    for l in sites:
        for k in ("belief_edit", "log_belief_edit", "random_same_norm", "whole_swap"):
            e_ = lambda f: mm(lambda r_: r_["edit"][str(l)][k][f])
            L.append(f"| {l} | {k} | {e_('H_transfer')} | {e_('decision_to_donor')} | {e_('own_goal_to_donor')} | {e_('decoded_posterior_vs_donor_r2')} | "
                     f"{mm(lambda r_: r_['edit'][str(l)][k]['edit_norm'] / r_['edit'][str(l)][k]['state_norm'])} |")
    (HERE / "causal.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
