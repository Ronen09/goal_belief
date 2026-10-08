"""Does the goal only add an offset to H(b), or change which features of the belief are read? Exploratory.
At the decision token of the maze10 models, under every goal (history fixed):
  A. decomposition x(h, g) = hist(h) + M(g) + I(h, g) at the residual sites and at every component output; the
     interaction's variance share; the posterior-predictable part: R² of a shared encoder f(b) + M(g) against per-goal
     encoders f_g(b) (MLP), and of the interaction I(h, g) from b per goal — where the goal first changes the belief code;
  B. the posterior-part edit at the input of block 2 (donor belief component, same vector under every recipient goal):
     the induced logit change per goal split into its shared and goal-specific parts, against the natural change
     L_B(g) − L_A(g); transfer of the goal-specific part;
  C. offline removal of one component's interaction at a time (output replaced by hist + M): decisions changed under
     the own goal, and the goal-dependence of the move that remains.

    .venv/bin/python studies/7_information_seeking/H_simplex/goal_belief.py        # writes goal_belief.md / .json
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

SITES = (1, 2, 3, 4)
N_FIT, N_TEST = 12288, 6144


@torch.no_grad()
def states_all(net, tok, p, K):
    """{('resid', l) | ('attn', l) | ('mlp', l): [N, K, d]} at the decision token under every goal, and centred logits [N, K, 4]."""
    N, dev = len(tok), tok.device
    out, lg = {}, []
    for i in range(0, N, C.CHUNK):
        tk = tok[i:i + C.CHUNK].repeat(K, 1, 1); pp = p[i:i + C.CHUNK].repeat(K); n_ = len(tok[i:i + C.CHUNK])
        tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
        logits, _, rec = net(tk, record=True)
        ar = torch.arange(len(tk), device=dev)
        for l in range(net.nl + 1):
            out.setdefault(("resid", l), []).append(rec["resid"][l][ar, pp].view(K, n_, -1).transpose(0, 1))
        for l in range(net.nl):
            out.setdefault(("attn", l), []).append(rec["attn"][l][ar, pp].view(K, n_, -1).transpose(0, 1))
            out.setdefault(("mlp", l), []).append(rec["mlp"][l][ar, pp].view(K, n_, -1).transpose(0, 1))
        z = logits[ar, pp].view(K, n_, 4).transpose(0, 1).double(); lg.append(z - z.mean(-1, keepdim=True))
    return {k: torch.cat(v).double() for k, v in out.items()}, torch.cat(lg)


def decompose(x):
    """x [N, K, d] -> hist [N, d], M [K, d], I [N, K, d]; variance shares of the three parts (over N, K, d, about the grand mean)."""
    hist = x.mean(1); M = (x - hist[:, None]).mean(0); I = x - hist[:, None] - M[None]
    tot = ((x - x.mean((0, 1))) ** 2).sum()
    return hist, M, I, dict(hist=float(((hist - hist.mean(0)) ** 2).sum() * x.shape[1] / tot), goal=float((M ** 2).sum() * x.shape[0] / tot), inter=float((I ** 2).sum() / tot))


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
        Xf, _ = states_all(net, rf["tok"][df["ep"][fi]], df["st"][fi] + 1, K)
        Xt, Lt = states_all(net, r["tok"][dt["ep"][ti]], dt["st"][ti] + 1, K)
        S = lambda d, idx: R.step_onehot(d["st"][idx], H_len)
        Fb = lambda d, idx: torch.cat([S(d, idx), d["b"][idx], (d["b"][idx] + 1e-3).log()], 1)
        va = df["ep"][fi] % 10 == 0
        keys = [("resid", l) for l in SITES] + [(c, l) for l in range(net.nl) for c in ("attn", "mlp")]
        row = {"A": {}, "B": {}, "C": {}}
        # ---- A
        for k in keys:
            xf, xt = Xf[k], Xt[k]
            hf, Mf, If, _ = decompose(xf); ht, Mt, It, shares = decompose(xt)
            f = R.fit_mlp(Fb(df, fi), hf, va, steps=2000, d=128)
            pred_shared = f(Fb(dt, ti))[:, None] + Mf[None]                                       # f(b) + M(g)
            tot = ((xt - xt.mean((0, 1))) ** 2).sum()
            r2_shared = float(1 - ((pred_shared - xt) ** 2).sum() / tot)
            pred_pg = torch.stack([R.fit_mlp(Fb(df, fi), xf[:, g], va, steps=2000, d=128, seed=g)(Fb(dt, ti)) for g in range(K)], 1)
            r2_pergoal = float(1 - ((pred_pg - xt) ** 2).sum() / tot)
            # the interaction from b, per goal
            r2_I = float(1 - ((torch.stack([R.fit_mlp(Fb(df, fi), If[:, g], va, steps=2000, d=128, seed=10 + g)(Fb(dt, ti)) for g in range(K)], 1) - It) ** 2).sum() / (It ** 2).sum())
            row["A"][f"{k[0]}{k[1]}"] = dict(shares=shares, r2_shared_plus_offset=r2_shared, r2_per_goal=r2_pergoal, gap=r2_pergoal - r2_shared, r2_interaction_from_b=r2_I,
                                             inter_over_hist_norm=float((It.norm(dim=-1).mean() / (ht - ht.mean(0)).norm(dim=-1).mean())))
            print(f"seed{s} A {k}: shares hist {shares['hist']:.2f} goal {shares['goal']:.2f} inter {shares['inter']:.2f} | R2 shared+offset {r2_shared:.3f} per-goal {r2_pergoal:.3f} gap {r2_pergoal - r2_shared:.3f} | I from b {r2_I:.2f}", flush=True)
        # ---- B: the posterior-part edit at site 2 under every goal
        gp = torch.Generator(device=dev); gp.manual_seed(5)
        A_ = torch.randperm(len(dt["st"]), device=dev, generator=gp)[:C.N_PAIRS * 3]; Bc = torch.randperm(len(dt["st"]), device=dev, generator=gp)[:C.N_PAIRS * 3]
        by_step = {}
        for i in Bc.tolist(): by_step.setdefault(int(dt["st"][i]), []).append(i)
        pairs = []
        for i in A_.tolist():
            cands = by_step.get(int(dt["st"][i]), [])
            for _ in range(8):
                if not cands: break
                j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
                if j != i and float((dt["b"][i] - dt["b"][j]).abs().sum()) >= C.MIN_DIFF: pairs.append((i, j)); break
            if len(pairs) >= C.N_PAIRS: break
        ia, ib = torch.tensor([p_[0] for p_ in pairs], device=dev), torch.tensor([p_[1] for p_ in pairs], device=dev)
        ta, tb = r["tok"][dt["ep"][ia]], r["tok"][dt["ep"][ib]]; pa = dt["st"][ia] + 1
        XA, LA = C.states(net, ta, pa, K, (2,)); XB, LB_ = C.states(net, tb, pa, K, (2,))
        f2 = R.fit_mlp(Fb(df, fi), Xf[("resid", 2)].mean(1), va, steps=4000, d=256)
        d_post = (f2(Fb(dt, ib)) - f2(Fb(dt, ia))).float()
        Le = C.run_edit(net, ta, pa, K, 2, XA[2] + d_post[:, None])
        D_edit, D_nat = Le - LA, LB_ - LA                                                      # [N, K, 4]
        split = lambda D: (D.mean(1), D - D.mean(1, keepdim=True))
        se, ge = split(D_edit); sn, gn = split(D_nat)
        gs = lambda D, g_: float((g_ ** 2).sum() / (D ** 2).sum())
        proj = lambda a, b_: float(((a * b_).sum() / (b_ ** 2).sum().clamp(min=1e-9)))
        ga = dt["goal"][ia]; ar = torch.arange(len(ia), device=dev)
        decA, decB, decE = LA.argmax(-1), LB_.argmax(-1), Le.argmax(-1)                          # [N, K] under every goal
        differ = decA != decB
        row["B"] = dict(goal_specific_share_edit=gs(D_edit, ge), goal_specific_share_natural=gs(D_nat, gn),
                        transfer_shared=proj(se, sn), transfer_goal_specific=proj(ge, gn),
                        cos_between_goals_edit=float(torch.nn.functional.cosine_similarity(D_edit[:, :, None], D_edit[:, None, :], dim=-1)[:, ~torch.eye(K, dtype=torch.bool, device=dev)].mean()),
                        cos_between_goals_natural=float(torch.nn.functional.cosine_similarity(D_nat[:, :, None], D_nat[:, None, :], dim=-1)[:, ~torch.eye(K, dtype=torch.bool, device=dev)].mean()),
                        decision_to_donor_every_goal=float((decE == decB)[differ].double().mean()),
                        decision_to_donor_by_goal=[float((decE[:, g] == decB[:, g])[differ[:, g]].double().mean()) for g in range(K)],
                        decision_to_donor_own_goal=float((decE == decB)[ar, ga][differ[ar, ga]].double().mean()))
        print(f"seed{s} B: goal-specific share edit {row['B']['goal_specific_share_edit']:.2f} natural {row['B']['goal_specific_share_natural']:.2f} | transfer shared {row['B']['transfer_shared']:.2f} goal-specific {row['B']['transfer_goal_specific']:.2f} | to donor every goal {row['B']['decision_to_donor_every_goal']:.2f} own {row['B']['decision_to_donor_own_goal']:.2f}", flush=True)
        # ---- C: offline removal of one component's interaction (output -> hist + M, M from the natural run)
        comps = [(c, l) for l in range(net.nl) for c in ("attn", "mlp")]
        Ms = {k: decompose(Xt[k])[1] for k in comps}
        tok_t, p_t = r["tok"][dt["ep"][ti]], dt["st"][ti] + 1
        own = dt["goal"][ti]; ar_t = torch.arange(len(ti), device=dev)
        nat_dec = Lt.argmax(-1)                                                                 # [N, K]
        gdep_nat = float((nat_dec != nat_dec[:, :1]).any(1).double().mean())

        def removed(which):
            out = []
            for i in range(0, len(ti), C.CHUNK):
                tk = tok_t[i:i + C.CHUNK].repeat(K, 1, 1); pp = p_t[i:i + C.CHUNK].repeat(K); n_ = len(tok_t[i:i + C.CHUNK])
                tk[:, 1, MM.F_GOAL] = torch.arange(K, device=dev).repeat_interleave(n_) + 1
                ar_ = torch.arange(len(tk), device=dev)
                def mk(k):
                    def patch(z):
                        z = z.clone(); v = z[ar_, pp].view(K, n_, -1)
                        new = v.mean(0, keepdim=True) + Ms[k][:, None].to(z.dtype)
                        z[ar_, pp] = new.reshape(K * n_, -1); return z
                    return patch
                logits, _ = net(tk, patch={k: mk(k) for k in which})
                out.append(logits[ar_, pp].view(K, n_, 4).transpose(0, 1))
            return torch.cat(out).argmax(-1)
        for name, which in [(f"{c}{l}", [(c, l)]) for c, l in comps] + [("all", comps)]:
            dec = removed(which)
            row["C"][name] = dict(decisions_changed_own=float((dec != nat_dec)[ar_t, own].double().mean()),
                                  goal_dependence=float((dec != dec[:, :1]).any(1).double().mean()), goal_dependence_natural=gdep_nat)
        print(f"seed{s} C: " + " ".join(f"{k} chg {v['decisions_changed_own']:.2f} gdep {v['goal_dependence']:.2f}" for k, v in row["C"].items()) + f" (natural gdep {gdep_nat:.2f})", flush=True)
        res[f"seed{s}"] = row
        (HERE / "goal_belief.json").write_text(json.dumps(res))
    res = json.loads((HERE / "goal_belief.json").read_text()); runs = list(res.values())
    mm = lambda f, nd=2: (lambda xs: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})")([f(r_) for r_ in runs])
    L = ["# Does the goal change what is read from the belief?", "", "Generated by `goal_belief.py` (exploratory); six maze10 models, medians (min–max).", "",
         "## A. History fixed, goal changed: the state at the decision token as hist(h) + M(g) + I(h, g)", "",
         "Variance shares of the three parts; R² of the state from a shared encoder of the posterior plus a goal offset (f(b) + M(g)) against per-goal encoders f_g(b) (MLPs); the gap is the goal × belief part that is a function of the belief; "
         "R² of the interaction I(h, g) itself from b, per goal.", "",
         "| site / component | hist | goal | interaction | R² shared + offset | R² per-goal encoders | gap | interaction from b |", "|---|---|---|---|---|---|---|---|"]
    for k in runs[0]["A"]:
        a = lambda f_: mm(lambda r_: f_(r_["A"][k]))
        L.append(f"| {k} | {a(lambda v: v['shares']['hist'])} | {a(lambda v: v['shares']['goal'])} | {a(lambda v: v['shares']['inter'])} | {a(lambda v: v['r2_shared_plus_offset'])} | {a(lambda v: v['r2_per_goal'])} | {a(lambda v: v['gap'])} | {a(lambda v: v['r2_interaction_from_b'])} |")
    b_ = lambda f_: mm(lambda r_: f_(r_["B"]))
    L += ["", "## B. One donor belief component, every recipient goal (posterior-part edit at the input of block 2)", "",
          "The induced logit change per goal, split into the part shared across goals and the goal-specific part; the natural change L_B(g) − L_A(g) likewise; transfer = projection of the edit's part on the natural part.", "",
          "| | goal-specific share of the change | mean cosine between goals' changes | transfer |", "|---|---|---|---|",
          f"| edit | {b_(lambda v: v['goal_specific_share_edit'])} | {b_(lambda v: v['cos_between_goals_edit'])} | shared {b_(lambda v: v['transfer_shared'])}, goal-specific {b_(lambda v: v['transfer_goal_specific'])} |",
          f"| natural | {b_(lambda v: v['goal_specific_share_natural'])} | {b_(lambda v: v['cos_between_goals_natural'])} | |", "",
          f"Decisions to the donor's where they differ: under every goal {b_(lambda v: v['decision_to_donor_every_goal'])}; by goal " + ", ".join(mm(lambda r_, g=g: r_['B']['decision_to_donor_by_goal'][g]) for g in range(4)) + f"; under the episode's own goal {b_(lambda v: v['decision_to_donor_own_goal'])}.", "",
          "## C. Removing one component's interaction (offline, output → hist + M at the decision token)", "",
          "| component | own-goal decisions changed | goal-dependence of the move after (natural " + mm(lambda r_: r_['C']['all']['goal_dependence_natural']) + ") |", "|---|---|---|"]
    for k in runs[0]["C"]:
        L.append(f"| {k} | {mm(lambda r_: r_['C'][k]['decisions_changed_own'])} | {mm(lambda r_: r_['C'][k]['goal_dependence'])} |")
    (HERE / "goal_belief.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
