"""Parts B and D1 of PLAN.md: linear probes by layer at the decision token for the set, Q*, V*, the optimal action,
P(bump | forward), the cell and heading marginals (B1); the form of the readout from membership to the model's logits
over all reachable sets (B2); probes at the step tokens for the set so far (D1). The 100k model. Writes probes.json and
probe_weights.pt (the set probes by layer, for the edit test).

    PYTHONPATH=/workspace/goal_belief:. .venv/bin/python studies/8_language_models/bump_policy/probes.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from goalgeo import bumpmaze as BMZ
from common import HERE, RUNS, Runner, world, load_sets, boot_mean, SIZES, size_bins

N_DEC, N_STEP = 8192, 2048
STEP_LAYERS = [4, 8, 12, 14, 16, 18, 20, 22, 24]


def fit(X, Y, tr, te, kind, hidden=0, steps=800, seed=0, dev="cuda"):
    """A probe from X to Y fitted by LBFGS on tr. kind: 'logistic' (multi-label), 'ridge', 'softmax' (Y long labels).
    Returns the predictions on te and the fitted (net, mu, sd)."""
    torch.manual_seed(seed); X = X.float().to(dev); mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-3); Z = (X - mu) / sd; Y = Y.to(dev)
    k = 3 if kind == "softmax" else Y.shape[1]
    net = (nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.GELU(), nn.Linear(hidden, k)) if hidden else nn.Linear(Z.shape[1], k)).to(dev)
    lf = {"logistic": lambda p, y: nn.functional.binary_cross_entropy_with_logits(p, y.float()), "ridge": lambda p, y: ((p - y.float()) ** 2).mean(), "softmax": lambda p, y: nn.functional.cross_entropy(p, y)}[kind]
    opt = torch.optim.LBFGS(net.parameters(), max_iter=steps, line_search_fn="strong_wolfe", history_size=20)
    with torch.enable_grad():
        def closure():
            opt.zero_grad(); l = lf(net(Z[tr]), Y[tr]) + 1e-4 * sum((p ** 2).sum() for p in net.parameters()); l.backward(); return l
        opt.step(closure)
    with torch.no_grad(): return net(Z[te]).cpu(), (net, mu.cpu(), sd.cpu())


def exact_share(P, M): return float(((P > 0) == M).all(1).double().mean())
def r2(P, Y): Y = Y.float(); return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def targets(w, masks):
    """Per set: Q*, V*, the optimal set, P(bump | forward), the cell marginal, the heading marginal."""
    m = masks.float(); p = m / m.sum(1, keepdim=True)
    bump = torch.as_tensor(w.bump).float(); cell = torch.zeros(w.n_states, w.n_cells); cell[torch.arange(w.n_states), torch.arange(w.n_states) // 4] = 1
    head = torch.zeros(w.n_states, 4); head[torch.arange(w.n_states), torch.arange(w.n_states) % 4] = 1
    return dict(pbump=(p @ bump)[:, None], cell=p @ cell, heading=p @ head)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--model", default="llm_100000"); ap.add_argument("--n_dec", type=int, default=N_DEC); ap.add_argument("--n_step", type=int, default=N_STEP); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); S = load_sets(); n = len(S["keys"]); masks = S["masks"]; Q = S["Q"].double(); V = Q.max(1).values; opt = Q >= V[:, None] - 1e-9
    R = Runner(str(RUNS / a.model / "model"), w, a.device)
    rng = np.random.default_rng(0); pick = rng.choice(n, min(a.n_dec, n), replace=False); hists = [(S["acts"][i], S["obss"][i]) for i in pick]
    te = torch.as_tensor(np.arange(len(pick)) % 8 == 0); tr = ~te; layers = list(range(R.n_layers + 1))
    z, rec = R.run(hists, record=layers, pos=[-1]); X = torch.stack([rec[l][:, 0] for l in layers], 1).half()        # [n, L+1, d]
    print(f"{len(pick)} decisions (uniform over sets), states recorded {time.time() - t0:.0f}s", flush=True)
    M = masks[pick]; Qp = Q[pick].float(); Vp = V[pick].float()[:, None]; optp = opt[pick]; y_act = Q[pick].argmax(1); tg = targets(w, M)
    res = dict(model=a.model, n=len(pick), layers=[], probe_weights_layers=[]); weights = {}
    for l in layers:
        Xl = X[:, l].float(); row = dict(layer=l)
        P, (net, mu, sd) = fit(Xl, M, tr, te, "logistic"); row["set_exact"] = exact_share(P, M[te]); row["set_jaccard"] = float((((P > 0) & M[te]).sum(1) / ((P > 0) | M[te]).sum(1).clamp(min=1)).mean())
        if l >= 10: weights[l] = dict(W=net.weight.detach().cpu(), b=net.bias.detach().cpu(), mu=mu, sd=sd)
        P, _ = fit(Xl, Qp, tr, te, "ridge"); row["Q_r2"] = r2(P, Qp[te]); row["Q_argmax_optimal"] = float(optp[te][torch.arange(int(te.sum())), P.argmax(1)].double().mean())
        P, _ = fit(Xl, Vp, tr, te, "ridge"); row["V_r2"] = r2(P, Vp[te])
        P, _ = fit(Xl, y_act, tr, te, "softmax"); row["action_optimal"] = float(optp[te][torch.arange(int(te.sum())), P.argmax(1)].double().mean())
        for k in ("pbump", "cell", "heading"):
            P, _ = fit(Xl, tg[k], tr, te, "ridge"); row[k + "_r2"] = r2(P, tg[k][te])
        res["layers"].append(row); print(f"  layer {l}: set exact {row['set_exact']:.3f} | Q R² {row['Q_r2']:.3f} (argmax optimal {row['Q_argmax_optimal']:.3f}) V R² {row['V_r2']:.3f} action {row['action_optimal']:.3f} | P(bump) {row['pbump_r2']:.3f} cell {row['cell_r2']:.3f} heading {row['heading_r2']:.3f} {time.time() - t0:.0f}s", flush=True)
    best = {k: int(max(res["layers"], key=lambda r: r[k])["layer"]) for k in ("set_exact", "Q_r2", "V_r2", "action_optimal", "pbump_r2", "cell_r2", "heading_r2")}; res["best_layer"] = best
    # the model's own decision on the test decisions as the reference for the action probe
    res["model_optimal_on_these"] = float(optp[torch.arange(len(pick)), z.argmax(1)].double().mean())
    torch.save(weights, HERE / "probe_weights.pt"); (HERE / "probes.json").write_text(json.dumps(res, indent=1)); print("best layers", best, flush=True)
    # B2: the readout's form over all sets
    tp = HERE / f"table_{a.model}.pt"
    if tp.exists(): zall = torch.load(tp)["logits"]
    else: zall, _ = R.run([(S["acts"][i], S["obss"][i]) for i in range(n)])
    half = torch.as_tensor(np.arange(n) % 2 == 0); Xm = masks.float(); act_model = zall.argmax(1)
    form = {}
    for name, hid in (("linear", 0), ("mlp256", 256), ("mlp1024", 1024)):
        P, _ = fit(Xm, zall, half, ~half, "ridge", hidden=hid, steps=1500); form[name] = dict(r2=r2(P, zall[~half]), agreement=float((P.argmax(1) == act_model[~half]).double().mean()), agreement_ci=boot_mean((P.argmax(1) == act_model[~half]).numpy()), optimal=float(opt[~half][torch.arange(int((~half).sum())), P.argmax(1)].double().mean()))
        print(f"B2 {name}: R² {form[name]['r2']:.3f}, agreement with the model's decision {form[name]['agreement']:.3f}, optimal {form[name]['optimal']:.3f} {time.time() - t0:.0f}s", flush=True)
    res["readout_form"] = form; res["model_optimal_all_sets"] = float(opt[torch.arange(n), act_model].double().mean()); (HERE / "probes.json").write_text(json.dumps(res, indent=1))
    # D1: the set so far at the step tokens
    steps = np.array([len(S["acts"][i]) for i in pick]); sel = np.nonzero(steps >= 4)[0][: a.n_step]; hs = [hists[i] for i in sel]; kmax = int(steps[sel].max())
    pos = torch.full((len(sel), kmax), -10 ** 6, dtype=torch.long)
    for r, i in enumerate(sel):
        pp = R.positions(S["acts"][pick[i]]); pos[r, : len(pp)] = torch.tensor(pp)
    _, rec = R.run(hs, record=STEP_LAYERS, pos=pos)
    print(f"D1: {len(sel)} decisions with ≥ 4 steps, step-token states recorded {time.time() - t0:.0f}s", flush=True)
    sets_after = {}
    for r, i in enumerate(sel):
        acts, obss = hists[i]; m = np.zeros(w.n_states, dtype=bool); m[w.start_states] = True; row = []
        for a_, o_ in zip(acts, obss): m = BMZ.step_belief(w, m, a_, o_); row.append(m.copy())
        sets_after[r] = row
    te_s = torch.as_tensor(sel % 8 == 0); tr_s = ~te_s; d1 = []
    for i in range(1, min(kmax, 16) + 1):
        has = torch.as_tensor(steps[sel] >= i); Mi = torch.stack([torch.as_tensor(sets_after[r][i - 1]) if has[r] else torch.zeros(w.n_states, dtype=torch.bool) for r in range(len(sel))])
        trr, tee = tr_s & has, te_s & has
        if int(tee.sum()) < 50: break
        row = dict(step=i, n_test=int(tee.sum()))
        for l in STEP_LAYERS:
            P, _ = fit(rec[l][:, i - 1], Mi, trr, tee, "logistic", steps=400); row[f"L{l}"] = exact_share(P, Mi[tee])
        d1.append(row); print(f"  D1 step {i}: exact recovery by layer " + " ".join(f"{l}:{row[f'L{l}']:.2f}" for l in STEP_LAYERS) + f" {time.time() - t0:.0f}s", flush=True)
    res["step_tokens"] = dict(layers=STEP_LAYERS, rows=d1, best_mean_from_4={f"L{l}": float(np.mean([r[f"L{l}"] for r in d1 if r["step"] >= 4])) for l in STEP_LAYERS})
    (HERE / "probes.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
