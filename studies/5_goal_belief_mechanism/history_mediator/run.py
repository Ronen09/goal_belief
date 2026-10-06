"""The history-mediator experiment: which history-derived variable Z carries the history's effect on the decision? The
observation-prediction experiment's reward models, frozen. Candidates, arms and decision rule: PLAN.md.

    .venv/bin/python studies/5_goal_belief_mechanism/history_mediator/run.py                         # writes results.json
    .venv/bin/python studies/5_goal_belief_mechanism/history_mediator/run.py --untrained --seeds 0   # smoke test (initial checkpoint)

At the goal token's four attention outputs k: history part xbar_k(h) (mean over goals), Xbar_k(L) its fit-side mean per
prefix length, d_k(h) = xbar_k(h) - Xbar_k(L). For a candidate Z, x_Z(h) = E[d | Z, L] on fit-side histories (a table
for discrete codes and functions of the posterior, an MLP for H and H2), r(h) = d(h) - x_Z(h). Arms add a vector to
the four attention outputs and let everything else run live.
"""

from __future__ import annotations

import argparse, importlib.util, json, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROUNDS = HERE.parent.parent                                             # studies/


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


RUN = load("aarun", ROUNDS / "5_goal_belief_mechanism" / "additive_ablation" / "run.py")
R43 = load("r43run", ROUNDS / "5_goal_belief_mechanism" / "what_is_H" / "run.py")
R36, R30, M26, outputs = RUN.R36, RUN.R30, RUN.M26, RUN.outputs
CHUNK, SEED, MIN_FIT = 4096, 45, 5
CANDS = ("none", "top", "rank2", "H2", "H", "mix", "b", "all")
TABLES = ("top", "rank2", "mix", "b")


def keys_of(F):
    """Integer key per row of a feature matrix (distinct rows after rounding)."""
    return torch.unique(torch.round(F.double() * 1e6), dim=0, return_inverse=True)[1]


def table(key, L, nL, D, fit):
    """E[D | key, L] from fit-side rows; and the fit count of each row's (key, L)."""
    k = key * nL + L
    nk = int(k.max()) + 1
    cnt = torch.bincount(k[fit], minlength=nk)
    S = torch.zeros(nk, D.shape[1], device=D.device).index_add_(0, k[fit], D[fit])
    return (S / cnt.clamp(min=1)[:, None].float())[k], cnt[k]


def mlp(F, L, nL, D, fit, gen):
    """E[D | F, L] by a one-hidden-layer MLP on fit-side rows (90 % train, 10 % early stopping)."""
    X = torch.cat([F.float(), torch.nn.functional.one_hot(L, nL).float()], 1)
    mu, sd = X[fit].mean(0), X[fit].std(0).clamp(min=1e-6)
    X = (X - mu) / sd
    idx = fit[torch.randperm(len(fit), device=fit.device, generator=gen)]
    nv = len(idx) // 10
    va, tr = idx[:nv], idx[nv:]
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, D.shape[1])).to(D.device)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    best, state, bad = float("inf"), None, 0
    with torch.enable_grad():
        for ep in range(200):
            perm = tr[torch.randperm(len(tr), device=tr.device, generator=gen)]
            for i in range(0, len(perm), 2048):
                b = perm[i:i + 2048]
                loss = ((net(X[b]) - D[b]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                v = float(((net(X[va]) - D[va]) ** 2).mean())
            if v < best - 1e-7:
                best, state, bad = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
    net.load_state_dict(state)
    with torch.no_grad():
        return torch.cat([net(X[i:i + 65536]) for i in range(0, len(X), 65536)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--untrained", action="store_true", help="smoke test on the initial checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--explore", action="store_true", help="post hoc (not in PLAN.md): swap_r rotated, for H, mix and b; writes explore.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    dev, t0 = a.device, time.time()
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    nh, nL = len(ctx.L), t.max_prefix + 1
    allh = torch.arange(nh, device=dev)
    fit = torch.nonzero(ctx.fit).squeeze(1)
    Lh = ctx.L
    bel = t.belief[ctx.node].double()
    pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)[1]
    C = R43.candidates(t, ctx)                                                                # [nh, K, 4]
    mixF = torch.cat([C[:, R43.CANDS.index("popt")], C[:, R43.CANDS.index("meanMDP")]], 1)   # [nh, 8]
    cells0 = R36.Cells(ctx)
    res = dict(runs={})
    out = HERE / ("smoke.json" if a.untrained else "explore.json" if a.explore else "results.json")
    for s in a.seeds:
        net, ck = M26.load("random" if a.untrained else "reward", s, t, dev, M26.R23 / "runs")
        A = [("attn", l) for l in range(net.nl)]
        d = net.d
        gen = torch.Generator(device=dev); gen.manual_seed(SEED + s)
        # natural attention outputs and logits at the goal token, every history, every goal
        xs, lgs = [], []
        for g in range(3):
            xa, la = [], []
            for i in range(0, nh, 8192):
                h = allh[i:i + 8192]
                o, lg = outputs(net, ctx, h, torch.full_like(h, g))
                xa.append(torch.cat([o[c] for c in A], 1)); la.append(lg)
            xs.append(torch.cat(xa)); lgs.append(torch.cat(la))
        xbar = torch.stack(xs, 1).mean(1)                                                     # [nh, 4d]
        lg_all = torch.stack(lgs, 1)                                                          # [nh, 3, 4]
        del xs, lgs
        cntL = torch.bincount(Lh[fit], minlength=nL).clamp(min=1).float()[:, None]
        Xb = torch.zeros(nL, xbar.shape[1], device=dev).index_add_(0, Lh[fit], xbar[fit]) / cntL
        D = xbar - Xb[Lh]
        # candidates
        lc = lg_all.double() - lg_all.double().mean(-1, keepdim=True)
        H = lc.mean(1)                                                                        # [nh, 4]
        order = H.argsort(-1, descending=True)
        Hf = H[fit] - H[fit].mean(0)
        pcs = torch.linalg.eigh(Hf.T @ Hf)[1][:, -2:]
        feats = dict(top=order[:, 0], rank2=order[:, 0] * 4 + order[:, 1], mix=keys_of(mixF), b=pid)
        XZ, cnt = {}, {}
        for k in TABLES:
            XZ[k], cnt[k] = table(feats[k], Lh, nL, D, fit)
        XZ["H"] = mlp(H, Lh, nL, D, fit, gen)
        XZ["H2"] = mlp((H - H[fit].mean(0)) @ pcs, Lh, nL, D, fit, gen)
        XZ["none"], XZ["all"] = torch.zeros_like(D), D
        # shared cell set: every table candidate's (Z, L) has >= MIN_FIT fit histories
        ok = torch.stack([cnt[k][cells0.h] >= MIN_FIT for k in TABLES]).all(0)
        ch, cg, cls = cells0.h[ok], cells0.g[ok], cells0.cls[ok]
        N = len(ch)
        rows = torch.arange(N, device=dev)
        cL = Lh[ch]
        optg = ctx.opt[ch, cg]
        nat = lg_all[ch, cg].argmax(-1)
        # donors: another held-out cell history of the same length (one draw for every candidate)
        donor = rows.clone()
        for l in range(nL):
            idx = torch.nonzero(cL == l).squeeze(1)
            if len(idx) > 1:
                donor[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
        dh = ch[donor]
        inf = (dh != ch) & (lg_all[dh, cg].argmax(-1) != nat)
        dnat = lg_all[dh, cg].argmax(-1)
        # matched donors: same (Z, L) value, a different history
        matched = {}
        for k in TABLES:
            key = feats[k][ch] * nL + cL
            md = rows.clone()
            for v in torch.unique(key):
                idx = torch.nonzero(key == v).squeeze(1)
                if len(idx) > 1:
                    md[idx] = idx[torch.randperm(len(idx), device=dev, generator=gen)]
            matched[k] = (md, ch[md] != ch)
        B = torch.zeros(3, nL, dtype=torch.long, device=dev)
        for l in range(nL):
            m = fit[Lh[fit] == l]
            if len(m):
                B[:, l] = ctx.opt[m].double().mean(0).argmax(-1)
        ceil = float(optg[rows, B[cg, cL]].float().mean())
        grp = cg * nL + cL

        def hist_dep(act):
            c_ = torch.zeros(3 * nL, 4, device=dev).index_put_((grp, act), torch.ones(N, device=dev), accumulate=True)
            return float(1 - c_.max(-1).values.sum() / N)

        def run(vec):
            """Greedy actions with vec [N, 4d] added to the four attention outputs at the goal token."""
            acts = []
            for i in range(0, N, CHUNK):
                h, g, v = ch[i:i + CHUNK], cg[i:i + CHUNK], vec[i:i + CHUNK]
                n = torch.arange(len(h), device=dev)
                p = 1 + Lh[h]

                def adder(delta):
                    def f(z):
                        z = z.clone(); z[n, p] = z[n, p] + delta
                        return z
                    return f
                patch = {c: adder(v[:, j * d:(j + 1) * d]) for j, c in enumerate(A)}
                acts.append(net(R30.seq(ctx, h, g), patch=patch)[0][n, p].argmax(-1))
            return torch.cat(acts)

        def rot(vec):
            out_ = torch.empty_like(vec)
            for j in range(len(A)):
                v = vec[:, j * d:(j + 1) * d]
                r = torch.randn(v.shape, device=dev, generator=gen)
                out_[:, j * d:(j + 1) * d] = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-12) * v.norm(dim=-1, keepdim=True)
            return out_

        assert torch.equal(run(torch.zeros(N, D.shape[1], device=dev)), nat)                  # zero edit reproduces natural
        row = dict(checkpoint=ck, cells=N, cells_dropped=int((~ok).sum()), informative_pairs=int(inf.sum()), ceiling=ceil,
                   natural=dict(optimal=float(optg[rows, nat].float().mean()), history_dependence=hist_dep(nat)), cands={})
        Dc = D[ch]
        test_h = torch.unique(ch)
        for k in (("H", "mix", "b") if a.explore else CANDS):
            xz = XZ[k][ch]
            r = Dc - xz
            arms = {}
            if k != "none":
                arms.update(remove_Z=-xz, swap_Z=xz[donor] - xz, swap_Z_rot=rot(xz[donor] - xz))
            if k != "all":
                arms.update(remove_r=-r, swap_r=r[donor] - r)
            if a.explore:
                arms = dict(swap_r=r[donor] - r, swap_r_rot=rot(r[donor] - r))
            elif k in TABLES:
                md, mv = matched[k]
                arms["swap_r_matched"] = torch.where(mv[:, None], r[md] - r, torch.zeros_like(r))
            cr = dict(r2=float(1 - ((D[test_h] - XZ[k][test_h]) ** 2).sum() / (D[test_h] ** 2).sum()), arms={})
            for name, vec in arms.items():
                act = run(vec)
                m = dict(optimal=float(optg[rows, act].float().mean()), history_dependence=hist_dep(act),
                         follow=float((act == dnat)[inf].float().mean()), change=float((act != nat)[inf].float().mean()))
                if name == "swap_r_matched":
                    mv = matched[k][1]
                    m.update(matched_pairs=int(mv.sum()), matched_change=float((act != nat)[mv].float().mean()))
                cr["arms"][name] = m
            row["cands"][k] = cr
            print(f"seed{s} {k}: R2 {cr['r2']:.3f} " + " ".join(f"{n_} opt {m['optimal']:.3f} hd {m['history_dependence']:.3f} fol {m['follow']:.3f} chg {m['change']:.3f}"
                                                            + (f" mchg {m['matched_change']:.3f}" if "matched_change" in m else "") + " |" for n_, m in cr["arms"].items())
                  + f" {time.time() - t0:.0f}s", flush=True)
        res["runs"][f"seed{s}"] = row
        print(f"seed{s} {ck} cells {N} (dropped {row['cells_dropped']}) informative {row['informative_pairs']} ceiling {ceil:.3f} natural {row['natural']}", flush=True)
        out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
