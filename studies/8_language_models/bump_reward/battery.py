"""The policy study's battery on a small model of the bump maze (PLAN.md): a reward-trained ActorCritic
(runs/<kind>/seed<k>/model.pt) or a supervised StepTransformer / StepGRU (../bump_maze/runs/<kind>_100000/model.pt).
A1 set-sufficiency, A2 the table over all reachable sets (by the imitation data's coverage and the agent's own), A3 the
table's value against the online greedy return, B1 probes by block held out by set, C1 same-set / different-set swaps,
D2 the prefix patch (transformer). Writes battery_<name>.json.

    PYTHONPATH=/workspace/goal_belief:studies/8_language_models/bump_policy .venv/bin/python studies/8_language_models/bump_reward/battery.py --kind tfm --model runs/tfm/seed0/model.pt --name ppo_tfm_seed0
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "bump_policy")); sys.path.insert(0, str(HERE.parent / "bump_maze"))
from train_reward import ActorCritic, rollout, summarize                              # noqa: E402
from common import world, load_sets, key, size_bins, boot_mean, SIZES                   # noqa: E402
from policy_table import policy_value, fit_readout                                      # noqa: E402
from probes import fit, exact_share, r2, targets                                        # noqa: E402
from train import test_starts, step_tokens                                              # noqa: E402

POL = HERE.parent / "bump_policy"


def load_model(kind, path, H, device):
    m = ActorCritic(H, kind).to(device); sd = torch.load(path, map_location=device)
    if not any(k.startswith("body.") for k in sd): sd = {("body." + k): v for k, v in sd.items()}; sd.update({"value.weight": torch.zeros(1, 128, device=device), "value.bias": torch.zeros(1, device=device)})
    m.load_state_dict(sd); return m.eval()


@torch.no_grad()
def run(model, hists, H, device, record=False, patch=None, batch=2048):
    """Centred logits [n, 3] at the decision position (= number of steps); record: residuals entering each block and
    the final at the decision position [n, L + 1, d]; patch: list of (layer, position-index-per-row [n], values [n, d])."""
    Z, R = [], []
    for b in range(0, len(hists), batch):
        chunk = hists[b:b + batch]; n = len(chunk); acts = torch.zeros(n, H, dtype=torch.long); obss = torch.zeros_like(acts); ln = torch.tensor([len(a) for a, _ in chunk])
        for i, (a, o) in enumerate(chunk): acts[i, :len(a)] = torch.tensor(a); obss[i, :len(o)] = torch.tensor(o)
        tok = step_tokens(acts, obss, ln, H).to(device); ln = ln.to(device); ar = torch.arange(n, device=device)
        pt = [(l, p[b:b + batch].to(device), v[b:b + batch].to(device)) for l, p, v in patch] if patch else None
        out = model(tok, record=record, patch=pt); lg = out[0][ar, ln].float(); Z.append((lg - lg.mean(1, keepdim=True)).cpu())
        if record: R.append(torch.stack([r[ar, ln].float() for r in out[2]], 1).cpu())
    return torch.cat(Z), (torch.cat(R) if record else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", choices=["tfm", "gru"], required=True); ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--skip_a", action="store_true"); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); H = w.H; S = load_sets(); n = len(S["keys"]); masks = S["masks"]; Q = S["Q"].double(); V = Q.max(1).values; opt = Q >= V[:, None] - 1e-9; size = masks.sum(1).numpy()
    cnt_opt = S["cnt_opt"].numpy(); cnt100 = S["cnt_train_100000"].numpy(); hist = torch.load(POL / "histories.pt")
    mp = Path(a.model); mp = mp if mp.exists() else HERE / a.model; model = load_model(a.kind, mp, H, a.device); res = dict(name=a.name, kind=a.kind)
    # online greedy return on the fixed test starts, and the agent's own coverage from 100 000 sampled episodes
    egen = torch.Generator(device=a.device); egen.manual_seed(5); res["online"] = summarize(rollout(model, w, 2048, egen, a.device, greedy=True, starts=test_starts(w).to(a.device)), w.gamma)
    own = np.zeros(n, dtype=np.int64); idx = {k: i for i, k in enumerate(S["keys"])}
    for rep in range(25):
        gen = torch.Generator(device=a.device); gen.manual_seed(100 + rep); env = BMZ.Env(w, 4096, gen, device=a.device); tok = torch.zeros(4096, H + 1, dtype=torch.long, device=a.device)
        for s in range(H):
            live = ~env.done
            if not live.any(): break
            pk = np.packbits(env.filter.mask[live].cpu().numpy(), axis=1)
            for row in pk:
                j = idx.get(row.tobytes())
                if j is not None: own[j] += 1
            lg, _ = model(tok[:, : s + 1]); a_ = torch.distributions.Categorical(logits=lg[:, s]).sample(); _, o = env.step(a_); tok[:, s + 1] = torch.where(live, 1 + 2 * a_ + o, torch.zeros_like(a_))
    res["own_coverage_sets"] = int((own > 0).sum()); print(f"{a.name}: online greedy return {res['online']['ret']:.3f} success {res['online']['success']:.3f}; own coverage {res['own_coverage_sets']} sets {time.time() - t0:.0f}s", flush=True)
    hs = [(S["acts"][i], S["obss"][i]) for i in range(n)]; z0, _ = run(model, hs, H, a.device); act0 = z0.argmax(1).numpy(); ok = opt[torch.arange(n), torch.as_tensor(act0)].numpy(); regret = ((V - Q[torch.arange(n), torch.as_tensor(act0)]) / V.clamp(min=1e-9)).numpy()
    bins = size_bins(size)
    groups = {"optimum-visited": cnt_opt > 0, "imitation-training-visited, not by the optimum": (cnt100 > 0) & (cnt_opt == 0), "never visited by the imitation data": cnt100 == 0, "visited by the agent's own episodes": own > 0, "never visited by the agent": own == 0}
    res["table"] = {g: dict(n=int(m.sum()), optimal=float(ok[m].mean()), optimal_ci=boot_mean(ok[m]), regret=float(regret[m].mean()), by_size={b: (float(ok[m & bm].mean()) if (m & bm).any() else None) for b, bm in bins.items()}) for g, m in groups.items()}
    res["table"]["all"] = dict(n=n, optimal=float(ok.mean()), regret=float(regret.mean()))
    print(f"  table: all {ok.mean():.3f}; " + "; ".join(f"{g} {v['optimal']:.3f} ({v['n']})" for g, v in res["table"].items() if g != "all"), flush=True)
    # A1
    multi = [i for i in range(n) if len(hist[S["keys"][i]]) >= 2]; extra = [(i, h) for i in multi for h in hist[S["keys"][i]][1:]]
    ze, _ = run(model, [h for _, h in extra], H, a.device); acte = ze.argmax(1).numpy(); aall = {i: [int(act0[i])] for i in multi}
    for (i, _), ac in zip(extra, acte): aall[i].append(int(ac))
    same = np.array([len(set(aall[i])) == 1 for i in multi]); marr = np.array(multi); rng = np.random.default_rng(0); base = []
    for _ in range(len(multi)):
        i = marr[rng.integers(len(marr))]; cand = np.nonzero(size == size[i])[0]; j = cand[rng.integers(len(cand))]
        if j != i: base.append(act0[i] == act0[j])
    res["sufficiency"] = dict(all=dict(n=len(multi), same=float(same.mean()), same_ci=boot_mean(same), baseline_same_size=float(np.mean(base))), **{g: dict(n=int(m[marr].sum()), same=float(same[m[marr]].mean()) if m[marr].any() else None) for g, m in groups.items()})
    print(f"  set-sufficiency {same.mean():.3f} (baseline {np.mean(base):.3f}); " + "; ".join(f"{g} {v['same']:.3f}" for g, v in res['sufficiency'].items() if g != 'all' and v['same'] is not None), flush=True)
    # A3
    maj = act0.copy()
    for i in multi: maj[i] = np.bincount(aall[i], minlength=3).argmax()
    start = int(np.argmax(size)); Vn = V.numpy()
    for lab, acts_ in (("shortest", act0), ("majority", maj)):
        Vp = policy_value(w, sol, S, acts_); ratio = Vp / np.maximum(Vn, 1e-9)
        res[f"value_{lab}"] = dict(start=float(Vp[start]), start_optimal=float(Vn[start]), start_ratio=float(Vp[start] / Vn[start]), vs_online=float(Vp[start] / max(res["online"]["ret"], 1e-9)), mean_ratio_all=float(ratio.mean()), mean_ratio_ci=boot_mean(ratio), lost=float((ratio < 0.5).mean()))
        print(f"  table value ({lab}): start {Vp[start]:.4f} = {Vp[start] / Vn[start]:.3f} of V*, {Vp[start] / max(res['online']['ret'], 1e-9):.3f} of the online return; mean ratio {ratio.mean():.3f}; lost {float((ratio < 0.5).mean()):.3f} {time.time() - t0:.0f}s", flush=True)
    torch.save(dict(logits=z0, actions=torch.as_tensor(act0), majority=torch.as_tensor(maj), own=torch.as_tensor(own)), HERE / f"table_{a.name}.pt")
    # B1: probes by block, uniform over sets, held out by set
    rng = np.random.default_rng(0); pick = rng.choice(n, 8192, replace=False); hp = [hs[i] for i in pick]; te = torch.as_tensor(np.arange(8192) % 8 == 0); tr = ~te
    _, X = run(model, hp, H, a.device, record=True); M = masks[pick]; Qp = Q[pick].float(); tg = targets(w, M); rows = []
    for l in range(X.shape[1]):
        P, _ = fit(X[:, l], M, tr, te, "logistic"); row = dict(layer=l, set_exact=exact_share(P, M[te]), set_jaccard=float((((P > 0) & M[te]).sum(1) / ((P > 0) | M[te]).sum(1).clamp(min=1)).mean()))
        P, _ = fit(X[:, l], Qp, tr, te, "ridge"); row["Q_r2"] = r2(P, Qp[te]); P, _ = fit(X[:, l], tg["pbump"], tr, te, "ridge"); row["pbump_r2"] = r2(P, tg["pbump"][te]); rows.append(row)
        print(f"  probes block-entrance {l}: set exact {row['set_exact']:.3f} jaccard {row['set_jaccard']:.2f} | Q R² {row['Q_r2']:.3f} | P(bump) R² {row['pbump_r2']:.3f}", flush=True)
    res["probes"] = rows
    # C1: swaps at the decision position
    same_pairs = [(i, int(rng.integers(1, len(hist[S["keys"][i]])))) for i in rng.choice(multi, 1024, replace=False)]; hA = [hs[i] for i, _ in same_pairs]; hB = [hist[S["keys"][i]][j] for i, j in same_pairs]
    diff = []
    while len(diff) < 1024:
        i = int(rng.integers(n)); cand = np.nonzero(size == size[i])[0]; j = int(cand[rng.integers(len(cand))])
        if j != i: diff.append((i, j))
    hC = [hs[i] for i, _ in diff]; hD = [hs[j] for _, j in diff]
    zA, rA = run(model, hA, H, a.device, record=True); zB, rB = run(model, hB, H, a.device, record=True); zC, rC = run(model, hC, H, a.device, record=True); zD, rD = run(model, hD, H, a.device, record=True)
    dA, dB, dC, dD = zA.argmax(1), zB.argmax(1), zC.argmax(1), zD.argmax(1); lnA = torch.tensor([len(x[0]) for x in hA]); lnC = torch.tensor([len(x[0]) for x in hC]); c1 = []
    nl = rA.shape[1] if a.kind == "tfm" else 1
    for l in range(nl):
        if a.kind == "tfm":
            zp, _ = run(model, hA, H, a.device, patch=[(l, lnA, rB[:, l])]); zq, _ = run(model, hC, H, a.device, patch=[(l, lnC, rD[:, l])])
        else:
            # GRU: the hidden state at the decision is the whole state; swap it by running the head on the donor's state
            zp = model.body.head(rB[:, 0].to(a.device)).float().cpu(); zp = zp - zp.mean(1, keepdim=True); zq = model.body.head(rD[:, 0].to(a.device)).float().cpu(); zq = zq - zq.mean(1, keepdim=True)
        dp, dq = zp.argmax(1), zq.argmax(1); inf = dC != dD
        c1.append(dict(layer=l, same_changed=float((dp != dA).double().mean()), same_changed_ci=boot_mean((dp != dA).numpy()), same_to_donor=float((dp == dB)[dA != dB].double().mean()) if (dA != dB).any() else None, diff_to_donor=float((dq == dD)[inf].double().mean()), diff_to_donor_ci=boot_mean((dq == dD)[inf].numpy())))
        print(f"  swaps block-entrance {l}: same-set changed {c1[-1]['same_changed']:.3f}; different-set to donor {c1[-1]['diff_to_donor']:.3f}", flush=True)
    res["swaps"] = dict(same_natural_disagree=float((dA != dB).double().mean()), diff_natural_disagree=float((dC != dD).double().mean()), layers=c1)
    # D2: the prefix patch (transformer)
    if a.kind == "tfm":
        from prefix import run_hist
        Qd = {k: sol["Q"][sol["sets"][k]] for k in S["keys"]}; pool = [h for k in S["keys"] for h in hist[k]]; long = [h for h in pool if len(h[0]) >= 7]; byp = {}
        for h in pool:
            for p in range(4, 13):
                if len(h[0]) >= p: byp.setdefault(p, []).append((h[0][:p], h[1][:p]))
        pairs = []; tries = 0
        while len(pairs) < 1024 and tries < 200000:
            tries += 1; h = long[rng.integers(len(long))]; p = int(rng.integers(4, min(12, len(h[0]) - 3) + 1)); A = (h[0][: p + 3], h[1][: p + 3]); suf = (A[0][p:], A[1][p:]); pre = byp[p][rng.integers(len(byp[p]))]
            if pre == (A[0][:p], A[1][:p]): continue
            mA = run_hist(w, *A); B = (pre[0] + suf[0], pre[1] + suf[1]); mB = run_hist(w, *B)
            if mA is None or mB is None or key(mA[p - 1]) == key(mB[p - 1]): continue
            qa, qb = Qd[key(mA[-1])], Qd[key(mB[-1])]; oa, ob = qa >= qa.max() - 1e-9, qb >= qb.max() - 1e-9
            if (oa & ob).any(): continue
            pairs.append(dict(A=A, B=B, p=p, optB=ob))
        hA = [q["A"] for q in pairs]; hB = [q["B"] for q in pairs]; pp = torch.tensor([q["p"] for q in pairs]); zA, _ = run(model, hA, H, a.device); zB, rB_all = run_all(model, hB, H, a.device)
        dA, dB = zA.argmax(1), zB.argmax(1); inf = dA != dB; optB = torch.as_tensor(np.stack([q["optB"] for q in pairs])); d2 = []
        for l in range(rB_all.shape[1]):
            row = dict(layer=l)
            for name, poss in (("last_prefix_token", [pp]), ("all_prefix_tokens", [torch.full_like(pp, j) for j in range(1, 13)]), ("middle_prefix_token", [pp // 2])):
                patch = [(l, torch.minimum(pos, pp), rB_all[torch.arange(len(pairs)), l, torch.minimum(pos, pp)]) for pos in poss]
                zp, _ = run(model, hA, H, a.device, patch=patch); dp = zp.argmax(1); row[name] = dict(to_B=float((dp == dB)[inf].double().mean()), to_B_ci=boot_mean((dp == dB)[inf].numpy()), into_opt_B=float(optB[torch.arange(len(pairs)), dp].double().mean()))
            d2.append(row); print(f"  prefix patch block-entrance {l}: last token → B {row['last_prefix_token']['to_B']:.3f}; all prefix tokens {row['all_prefix_tokens']['to_B']:.3f}; middle {row['middle_prefix_token']['to_B']:.3f}", flush=True)
        res["prefix"] = dict(n_pairs=len(pairs), natural_differ=float(inf.double().mean()), layers=d2)
    (HERE / f"battery_{a.name}.json").write_text(json.dumps(res, indent=1)); print(f"done {time.time() - t0:.0f}s")


@torch.no_grad()
def run_all(model, hists, H, device, batch=2048):
    """Logits at the decision and the residuals entering each block at every position [n, L + 1, H + 1, d]."""
    Z, R = [], []
    for b in range(0, len(hists), batch):
        chunk = hists[b:b + batch]; n = len(chunk); acts = torch.zeros(n, H, dtype=torch.long); obss = torch.zeros_like(acts); ln = torch.tensor([len(a) for a, _ in chunk])
        for i, (a, o) in enumerate(chunk): acts[i, :len(a)] = torch.tensor(a); obss[i, :len(o)] = torch.tensor(o)
        tok = step_tokens(acts, obss, ln, H).to(device); out = model(tok, record=True); lg = out[0][torch.arange(n, device=device), ln.to(device)].float(); Z.append((lg - lg.mean(1, keepdim=True)).cpu()); R.append(torch.stack([r.float() for r in out[2]], 1).cpu())
    return torch.cat(Z), torch.cat(R)


if __name__ == "__main__":
    main()
