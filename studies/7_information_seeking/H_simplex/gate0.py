"""Do block 0's history-dependent gates on the goal token make the goal-specific coordinates? Exploratory.
In block 0 the goal reaches a goal-free decision token only through the goal token's key and value (section 12): per
head, the attention mass on the goal token is a gate on fixed goal content, and the gate depends on the history (the
softmax competition with the history tokens' keys). In the simplified network of section 14 (all four MLPs replaced by
shared affine maps at every token) the attention softmaxes and the layer norms are the only nonlinearities left, so: at block 0, replace each
head's attention mass on the goal token with its mean for that goal and query position (step), rescaling the other
tokens' weights to keep their relative weights (the history weights' total is then 1 − mean, fixed per goal and step).
Goal content is kept; its history-dependent gating is removed. Measured at the decision token under every goal:
  * the goal × history interaction share of the state at the input of blocks 1 and 2 and at the end, as in section 6
    and with the goal offset allowed to depend on the step (section 6's interaction includes a goal × step offset);
  * the goal-specific coordinates: R² of a shared posterior encoder plus a goal (× step) offset against per-goal
    encoders (section 6's A); the R² of the untransformed A-written difference Δ_A for the B-written Δ_B on the A-equivalent,
    B-different pairs beside the fitted map M(A→B)'s; section 10's transport test at the input of block 2 (B's
    decision switched by Δ_B, by M(A→B) Δ_A, by Δ_A, by random);
  * the policy: recovery of the goal-directed return, agreement with the natural model's decisions.
Variants: the natural network; the simplified one; the simplified one with the gates fixed at every token / at the
decision token only; the natural one with the gates fixed; the simplified one with the gates fixed to their
goal-averaged mean (block 0 then adds exactly hist(h) + M(g): no interaction from block 0 at all).

    PYTHONPATH=. .venv/bin/python studies/7_information_seeking/H_simplex/gate0.py        # writes gate0.md / gate0.json
"""

from __future__ import annotations

import json, math, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bigmaze as BM, mazemodel as MM

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "maze10"))
import run as R, measure as MS, train as TR, causal as C, equiv as E, mlp_affine as MA          # noqa: E402

SITES, N_FIT, CHUNK = (1, 2, 4), 12288, 1024


@torch.no_grad()
def goal_mass(net, tok, K):
    """Block 0's attention mass on the goal token over the live tokens of `tok` under every goal: mean and std per
    goal, head and query position, [K, h, L] each (position 0, the first observation, cannot see the goal: 0)."""
    N, L, dev = len(tok), tok.shape[1], tok.device
    s1 = torch.zeros(K, net.nh, L, device=dev, dtype=torch.float64); s2 = torch.zeros_like(s1); cnt = torch.zeros(K, L, device=dev, dtype=torch.float64)
    for i in range(0, N, CHUNK):
        tk = tok[i:i + CHUNK].repeat(K, 1, 1); n_ = len(tok[i:i + CHUNK])
        gl = torch.arange(K, device=dev).repeat_interleave(n_); tk[:, 1, MM.F_GOAL] = gl + 1
        _, _, rec = net(tk, record=True)
        m = rec["pattern"][0][:, :, :, 1].double(); live = (tk[..., MM.F_TYPE] != MM.PAD).double()
        for g in range(K):
            w = live[gl == g]
            s1[g] += (m[gl == g] * w[:, None]).sum(0); s2[g] += (m[gl == g] ** 2 * w[:, None]).sum(0); cnt[g] += w.sum(0)
    cnt = cnt.clamp(min=1)[:, None]
    mean = s1 / cnt; std = (s2 / cnt - mean ** 2).clamp(min=0).sqrt()
    mean[:, :, 0] = 0; std[:, :, 0] = 0
    return mean, std


class GateFixed:
    """The network with block 0's attention mass on the goal token fixed, per head and query position, to mbar[goal]
    (mbar [K, h, L]); the other tokens' weights rescaled by (1 − mbar) / (1 − mass), keeping their relative weights.
    decision_only: at each row's last live token only. Wraps the base network or a Replaced one."""

    def __init__(self, inner, mbar, decision_only=False):
        self.inner, self.mbar, self.decision_only = inner, mbar, decision_only

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def __call__(self, tok, q=None, record=False, patch=None):
        patch = dict(patch or {}); store = {}
        b = self.inner.blocks[0]; goal = tok[:, 1, MM.F_GOAL] - 1
        N, L = tok.shape[:2]; dev = tok.device
        live = tok[..., MM.F_TYPE] != MM.PAD; last = live.long().cumsum(1).argmax(1); pos = torch.arange(L, device=dev)
        rows = (pos[None] == last[:, None]) if self.decision_only else (pos[None] >= 1).expand(N, L)

        def cap(x):
            store["x"] = x; return x

        def rep(a):
            x = store["x"]; d = x.shape[-1]; h = b.h; dh = d // h
            qq, kk, vv = b.qkv(b.ln1(x)).view(N, L, 3, h, dh).permute(2, 0, 3, 1, 4)
            s = (qq @ kk.transpose(-1, -2)) / math.sqrt(dh)
            s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1), -torch.inf)
            A = s.softmax(-1); m = A[..., 1]                                                   # [N, h, L]
            target = self.mbar[goal][:, :, :L].to(A.dtype)
            A2 = A * ((1 - target) / (1 - m).clamp(min=1e-6))[..., None]; A2[..., 1] = target
            A2 = torch.where(rows[:, None, :, None], A2, A)
            return b.o((A2 @ vv).transpose(1, 2).reshape(N, L, d))

        patch[("resid", 0)] = cap; patch[("attn", 0)] = rep
        return self.inner(tok, q, record=record, patch=patch)


def decompose(x):
    """x [N, K, d] -> hist [N, d], M [K, d], I [N, K, d] and the three parts' variance shares."""
    hist = x.mean(1); M = (x - hist[:, None]).mean(0); I = x - hist[:, None] - M[None]
    tot = ((x - x.mean((0, 1))) ** 2).sum()
    return hist, M, I, dict(hist=float(((hist - hist.mean(0)) ** 2).sum() * x.shape[1] / tot), goal=float((M ** 2).sum() * x.shape[0] / tot), inter=float((I ** 2).sum() / tot))


def r2(P, T):
    return float(1 - ((P - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    cfg = json.load(open(HERE.parent / "maze10" / "runs" / "ppo" / "task.json"))
    t = BM.Sim(BM.make(**cfg["task"]), dev); K, H_len = len(t.goal_cell), t.H
    gen = torch.Generator(device=dev); gen.manual_seed(78); e = BM.Env(t, R.N_EP, gen); goal, cell = e.goal, e.cell
    gen.manual_seed(178); e = BM.Env(t, R.N_EP, gen); fgoal, fcell = e.goal, e.cell
    res = {}
    for s in range(6):
        t0 = time.time()
        ck = sorted((HERE.parent / "maze10" / "runs" / "ppo" / f"seed{s}" / "ckpt").glob("u*.pt"))[-1]
        net = TR.build(t.n_sym, K, t.H, 128, 4); net.load_state_dict(torch.load(ck, map_location=dev)); net = net.to(dev).eval()
        nat, rf = MS.episodes(t, MS.natural(net), fgoal, fcell, 179, full=True)
        nat, r = MS.episodes(t, MS.natural(net), goal, cell, 79, full=True)
        gb = MS.episodes(t, lambda env, s_: MS.logits_goals(net, env, s_).mean(1).argmax(-1), goal, cell, 79)[0]
        df, dt = R.decisions(t, net, rf, K), R.decisions(t, net, r, K)
        # ---- the gates: mean and spread of the mass on the goal token, per goal, head and position, on the fit episodes
        mean_mass, std_mass = goal_mass(net, rf["tok"], K)
        mavg = mean_mass.mean(0, keepdim=True).expand(K, -1, -1).contiguous()
        wpos = torch.zeros_like(mean_mass[0, 0]); wpos[2:] = 1                                   # decision tokens after the first step
        live_pos = (rf["tok"][..., MM.F_TYPE] != MM.PAD).double().mean(0) * wpos                   # weight positions by how often they are live
        wm = lambda z: float((z * live_pos).sum() / (live_pos.sum() * z.shape[0] * z.shape[1]))
        struct = dict(mass_per_goal=[wm(mean_mass[g:g + 1]) for g in range(K)], mass_per_head=[wm(mean_mass[:, hh:hh + 1]) for hh in range(net.nh)],
                      std_within_cell=wm(std_mass), mean=wm(mean_mass), std_of_means_across_goals=wm(mean_mass.std(0, keepdim=True).expand(K, -1, -1)))
        # check: the wrapper with a row's own masses reproduces the natural attention output
        tk1 = rf["tok"][:1].repeat(K, 1, 1); tk1[:, 1, MM.F_GOAL] = torch.arange(K, device=dev) + 1
        _, _, rec1 = net(tk1, record=True); own = rec1["pattern"][0][:, :, :, 1].double()
        _, _, rec2 = GateFixed(net, own)(tk1, record=True)
        assert float((rec2["attn"][0] - rec1["attn"][0]).abs().max()) < 1e-4, float((rec2["attn"][0] - rec1["attn"][0]).abs().max())
        # ---- the simplified network: all four MLPs by shared affine maps (section 14's fits)
        data = MA.fit_data(net, rf["tok"], K, (0, 1, 2, 3))
        simp = {l: dict(kind="shared", on_ln=True, **MA.affine(Z, Y, ep % 10 == 0)) for l, (Z, X, Y, gl, ep, _) in data.items()}
        del data
        variants = [("natural", lambda: net),
                    ("simplified (all four MLPs shared affine)", lambda: MA.Replaced(net, simp)),
                    ("simplified + block-0 gates fixed, every token", lambda: GateFixed(MA.Replaced(net, simp), mean_mass)),
                    ("simplified + block-0 gates fixed, decision token only", lambda: GateFixed(MA.Replaced(net, simp), mean_mass, True)),
                    ("natural + block-0 gates fixed, every token", lambda: GateFixed(net, mean_mass)),
                    ("simplified + block-0 gates fixed to the goal-averaged mean", lambda: GateFixed(MA.Replaced(net, simp), mavg))]
        # ---- decisions: fit subsample (encoders, maps), test subsample (everything), equiv pairs (as mlp_affine)
        gq = torch.Generator(device=dev); gq.manual_seed(1)
        fi = torch.randperm(len(df["st"]), device=dev, generator=gq)[:N_FIT]
        gq.manual_seed(1); ti = torch.randperm(len(dt["st"]), device=dev, generator=gq)[:E.N_TEST]
        tokf, pf = MA.pad_after(rf["tok"][df["ep"][fi]], df["st"][fi] + 1), df["st"][fi] + 1
        tokp, p = MA.pad_after(r["tok"][dt["ep"][ti]], dt["st"][ti] + 1), dt["st"][ti] + 1
        S = lambda d, idx: R.step_onehot(d["st"][idx], H_len)
        Fb = lambda d, idx: torch.cat([S(d, idx), d["b"][idx], (d["b"][idx] + 1e-3).log()], 1)
        va_f = df["ep"][fi] % 10 == 0
        gp = torch.Generator(device=dev); gp.manual_seed(21)
        mf = len(df["st"]); a_ = torch.randint(mf, (MA.N_TRAIN_PAIRS * 4,), device=dev, generator=gp); by_step = {}
        for i in torch.randint(mf, (MA.N_TRAIN_PAIRS * 4,), device=dev, generator=gp).tolist(): by_step.setdefault(int(df["st"][i]), []).append(i)
        tp = []
        for i in a_.tolist():
            cands = by_step.get(int(df["st"][i]), [])
            if not cands: continue
            j = cands[int(torch.randint(len(cands), (1,), device=dev, generator=gp))]
            if j != i and float((df["b"][i] - df["b"][j]).abs().sum()) >= 0.5: tp.append((i, j))
            if len(tp) >= MA.N_TRAIN_PAIRS: break
        tp = torch.tensor(tp, device=dev); f1, f2 = tp[:, 0], tp[:, 1]; va = df["ep"][f1] % 10 == 0
        tokf1, pf1 = MA.pad_after(rf["tok"][df["ep"][f1]], df["st"][f1] + 1), df["st"][f1] + 1; tokf2, pf2 = MA.pad_after(rf["tok"][df["ep"][f2]], df["st"][f2] + 1), df["st"][f2] + 1
        Qt = E.qvals(t, dt, ti); cQ = (lambda st_: st_ - st_.mean(-1, keepdim=True))(torch.log(Qt.clamp(min=1e-9)) / np.log(t.gamma))
        sett = Qt >= Qt.max(-1, keepdim=True).values - 1e-6
        gq.manual_seed(7); m = len(ti); cand = torch.randint(m, (m, E.CANDS), device=dev, generator=gq)
        same_step = dt["st"][ti][:, None] == dt["st"][ti][cand]
        dq = (cQ[:, None] - cQ[cand]).abs().amax(-1); same_set = (sett[:, None] == sett[cand]).all(-1); disjoint = ~(sett[:, None] & sett[cand]).any(-1)
        reach = Qt.amax(-1) >= E.MINQ
        eq = same_set & (dq < E.EQ) & reach[:, None] & reach[cand]; dif = disjoint & (dq > E.DIFF)
        pairs = {}
        for A in range(K):
            for B in range(K):
                if A == B: continue
                ok = same_step & eq[..., A] & dif[..., B] & (cand != torch.arange(m, device=dev)[:, None])
                ii, cc = torch.nonzero(ok, as_tuple=True)
                if len(ii) < 30: continue
                _, first = np.unique(ii.cpu().numpy(), return_index=True); ii, cc = ii[torch.tensor(first, device=dev)], cc[torch.tensor(first, device=dev)]
                jj = cand[ii, cc]
                if len(ii) > 1500:
                    sel = torch.randperm(len(ii), device=dev, generator=gq)[:1500]; ii, jj = ii[sel], jj[sel]
                pairs[(A, B)] = (ii, jj)
        own = dt["goal"][ti]; nat_dec = dt["lc"][ti][torch.arange(m, device=dev), own].argmax(-1)
        nat_gd = (dt["lc"][ti].argmax(-1) != dt["lc"][ti].argmax(-1)[:, :1]).any(1)
        rows = {}
        for name, mk in variants:
            t1 = time.time(); v = mk()
            on = MS.episodes(t, MS.natural(v), goal, cell, 79)[0]
            dv = R.decisions(t, v, r, K)
            lv = dv["lc"][ti]; dec = lv[torch.arange(m, device=dev), own].argmax(-1)
            row = dict(ret=on["ret"], success=on["success"], recovery=(on["ret"] - gb["ret"]) / (nat["ret"] - gb["ret"]),
                       agree=float((dec == nat_dec).double().mean()), goal_dep=float((lv.argmax(-1) != lv.argmax(-1)[:, :1]).any(1).double().mean()),
                       agree_gd=float((dec == nat_dec)[nat_gd].double().mean()))
            Xf, _ = C.states(v, tokf, pf, K, SITES); Xt, Lt = C.states(v, tokp, p, K, SITES)
            XF1, _ = C.states(v, tokf1, pf1, K, SITES); XF2, _ = C.states(v, tokf2, pf2, K, SITES)
            Ff, Ft = Fb(df, fi), Fb(dt, ti)
            stf, stt = df["st"][fi], dt["st"][ti]
            for l in SITES:
                xf, xt = Xf[l].double(), Xt[l].double()
                hf, Mf, _, _ = decompose(xf); ht, Mt, It, shares = decompose(xt)
                tot = ((xt - xt.mean((0, 1))) ** 2).sum()
                # the goal offset allowed to depend on the step: M(g, step); the interaction beyond goal × step
                Df, Dt = xf - hf[:, None], xt - ht[:, None]
                Mstep_f = torch.stack([Df[stf == s_].mean(0) if bool((stf == s_).any()) else Mf for s_ in range(H_len)])        # [H, K, d] on the fit set
                Mstep_t = torch.stack([Dt[stt == s_].mean(0) if bool((stt == s_).any()) else Mt for s_ in range(H_len)])
                row[f"share_inter_{l}"] = shares["inter"]; row[f"share_goal_{l}"] = shares["goal"]
                row[f"share_goalstep_{l}"] = float((Mstep_t[stt] ** 2).sum() / tot); row[f"share_interS_{l}"] = float(((Dt - Mstep_t[stt]) ** 2).sum() / tot)
                f = R.fit_mlp(Ff, hf, va_f, steps=2000, d=128); fs = f(Ft)
                r2_shared = float(1 - ((fs[:, None] + Mf[None] - xt) ** 2).sum() / tot)
                r2_sharedS = float(1 - ((fs[:, None] + Mstep_f[stt] - xt) ** 2).sum() / tot)
                pred_pg = torch.stack([R.fit_mlp(Ff, xf[:, g], va_f, steps=2000, d=128, seed=g)(Ft) for g in range(K)], 1)
                r2_pg = float(1 - ((pred_pg - xt) ** 2).sum() / tot)
                row[f"r2_shared_{l}"] = r2_shared; row[f"r2_sharedS_{l}"] = r2_sharedS; row[f"r2_pergoal_{l}"] = r2_pg
                row[f"gap_{l}"] = r2_pg - r2_shared; row[f"gapS_{l}"] = r2_pg - r2_sharedS
                # coordinates on the equiv pairs: Δ_A for Δ_B untransformed and through the fitted map
                X1F, X2F = XF1[l].double(), XF2[l].double(); ident, lin_ = [], []
                for (A, B), (ii, jj) in pairs.items():
                    lin = R.Ridge(X2F[:, A] - X1F[:, A], X2F[:, B] - X1F[:, B], va)
                    dA, dB = xt[jj, A] - xt[ii, A], xt[jj, B] - xt[ii, B]
                    ident.append(r2(dA, dB)); lin_.append(r2(lin(dA), dB))
                row[f"r2_identity_{l}"] = float(np.mean(ident)); row[f"r2_linear_{l}"] = float(np.mean(lin_))
            # transport at the input of block 2 (section 10's test)
            X2, X1F, X2F = Xt[2].double(), XF1[2].double(), XF2[2].double()
            tr = {k: [] for k in ("direct_B", "linear_diff", "untransformed_A", "random")}
            for (A, B), (ii, jj) in pairs.items():
                lin = R.Ridge(X2F[:, A] - X1F[:, A], X2F[:, B] - X1F[:, B], va)
                X1, X2p = X2[ii], X2[jj]; dA, dB = X2p[:, A] - X1[:, A], X2p[:, B] - X1[:, B]
                dec1B, dec2B = Lt[ii, B].argmax(-1), Lt[jj, B].argmax(-1); differ = dec1B != dec2B
                gr = torch.Generator(device=dev); gr.manual_seed(3)
                rnd = torch.randn(dB.shape, device=dev, generator=gr); rnd = rnd / rnd.norm(dim=1, keepdim=True) * dB.norm(dim=1, keepdim=True)
                for k, vec in (("direct_B", dB), ("linear_diff", lin(dA)), ("untransformed_A", dA), ("random", rnd)):
                    ns = X1.clone(); ns[:, B] = ns[:, B] + vec
                    Le = C.run_edit(v, tokp[ii], p[ii], K, 2, ns)
                    tr[k].append(float((Le[:, B].argmax(-1) == dec2B)[differ].double().mean()) if int(differ.sum()) else float("nan"))
            row.update({k: float(np.nanmean(vv)) for k, vv in tr.items()})
            rows[name] = row
            print(f"  {name} ({time.time() - t1:.0f}s): " + " ".join(f"{k} {vv:.3f}" for k, vv in row.items()), flush=True)
        res[f"seed{s}"] = dict(rows=rows, struct=struct, goal_blind_ret=gb["ret"])
        print(f"seed{s} done in {time.time() - t0:.0f}s; gates {struct}", flush=True)
        (HERE / "gate0.json").write_text(json.dumps(res))
    report()


def report():
    res = json.loads((HERE / "gate0.json").read_text()); runs = list(res.values())
    names = list(runs[0]["rows"].keys())
    fmt = lambda xs, nd=2: f"{np.median(xs):.{nd}f} ({min(xs):.{nd}f}–{max(xs):.{nd}f})"
    mm = lambda n, k, nd=2: fmt([r_["rows"][n][k] for r_ in runs], nd)
    st = lambda f: fmt([f(r_["struct"]) for r_ in runs])
    L = ["# Do block 0's gates on the goal token make the goal-specific coordinates?", "",
         "Generated by `gate0.py` (exploratory). At block 0 each head's attention mass on the goal token is replaced by its mean for that goal and query position (fit episodes, every goal), the other tokens' weights rescaled to keep their relative weights; goal content is kept, its history-dependent gating removed. "
         "'Simplified' = section 14's network with all four MLPs replaced by shared affine maps at every token. At the decision token under every goal (6144 test decisions): the variance shares of the goal part M(g) and of the goal × history interaction I(h, g) at the input of block 1 (site 1), of block 2 (site 2) and at the end (site 4); R² of the state from a shared MLP encoder of the posterior plus a goal offset against per-goal encoders (the gap is the goal-specific coding of the belief); "
         "on the A-equivalent, B-different pairs, R² of the untransformed A-written difference Δ_A for the B-written Δ_B and of the fitted map's M(A→B) Δ_A (ridge on 12 288 fit pairs); the transport test at the input of block 2 (B's decision switched to the donor's by Δ_B, by M(A→B) Δ_A, by Δ_A, by random). Behaviour: recovery of the goal-directed return over the goal-blind floor; agreement with the natural model's own-goal decisions (all / goal-dependent). Medians (min–max) over six models.", "",
         "## The gates", "",
         f"Mass on the goal token at decision tokens after the first step: {st(lambda v: v['mean'])}; per goal " + ", ".join(st(lambda v, g=g: v['mass_per_goal'][g]) for g in range(4)) + "; per head " + ", ".join(st(lambda v, h=h: v['mass_per_head'][h]) for h in range(4)) + ".",
         f"Its spread across histories within one (goal, head, step): std {st(lambda v: v['std_within_cell'])}; the std of the means across goals {st(lambda v: v['std_of_means_across_goals'])}.", "",
         "## Behaviour and the interaction", "",
         "The state as hist(h) + M(g) + I(h, g) (section 6's decomposition) and, with the goal offset allowed to depend on the step, as hist(h) + M(g, step) + I'(h, g): the share of the goal × step offset and of the interaction beyond it.", "",
         "| variant | recovery | agreement: all / goal-dependent | goal-dependent decisions | interaction share I(h, g), section 6: site 1 / 2 / 4 | goal × step offset: 1 / 2 / 4 | interaction beyond goal × step, I': 1 / 2 / 4 |", "|---|---|---|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | {mm(n, 'recovery')} | {mm(n, 'agree')} / {mm(n, 'agree_gd')} | {mm(n, 'goal_dep')} | " + " / ".join(mm(n, f'share_inter_{l}') for l in SITES) + " | " + " / ".join(mm(n, f'share_goalstep_{l}') for l in SITES) + " | " + " / ".join(mm(n, f'share_interS_{l}') for l in SITES) + " |")
    L += ["", "## The goal-specific coordinates", "",
          "Encoders: a shared MLP f(step, b) plus a goal offset M(g), the same plus a goal × step offset M(g, step), per-goal MLPs f_g(step, b); the gap is per-goal minus shared with the goal × step offset. Pairs: the A-equivalent, B-different pairs; R² of the untransformed Δ_A for Δ_B, and of the fitted map's M(A→B) Δ_A.", "",
          "| variant | R² shared + M(g): site 1 / 2 / 4 | R² shared + M(g, step): 1 / 2 / 4 | R² per-goal encoders: 1 / 2 / 4 | gap: 1 / 2 / 4 | R² of Δ_A for Δ_B: 1 / 2 / 4 | R² of M(A→B) Δ_A: 1 / 2 / 4 |", "|---|---|---|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | " + " / ".join(mm(n, f'r2_shared_{l}') for l in SITES) + " | " + " / ".join(mm(n, f'r2_sharedS_{l}') for l in SITES) + " | " + " / ".join(mm(n, f'r2_pergoal_{l}') for l in SITES) + " | " + " / ".join(mm(n, f'gapS_{l}') for l in SITES)
                 + " | " + " / ".join(mm(n, f'r2_identity_{l}') for l in SITES) + " | " + " / ".join(mm(n, f'r2_linear_{l}') for l in SITES) + " |")
    L += ["", "## Transport at the input of block 2 (B's decision switched to the donor's)", "",
          "| variant | direct Δ_B | M(A→B) Δ_A | untransformed Δ_A | random |", "|---|---|---|---|---|"]
    for n in names:
        L.append(f"| {n} | {mm(n, 'direct_B')} | {mm(n, 'linear_diff')} | {mm(n, 'untransformed_A')} | {mm(n, 'random')} |")
    (HERE / "gate0.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    report() if "--report" in sys.argv else main()
