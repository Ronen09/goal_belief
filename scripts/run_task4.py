"""TASK4: what makes information geometrically prominent? Sweeps relevance frequency r,
strength delta, loss weight lambda and delay k on the parametrised HMM, trains the GRU
with checkpoints, and regresses prominence on decodability, forgetting cost and gradients.

    .venv/bin/python scripts/run_task4.py [--quick] [--jobs 6] [--grid] [--out results4]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr
from goalgeo.analysis import to_jsonable

BASE = dict(r=1.0, delta=0.4, k=1, lam=1.0)
SWEEPS = {"frequency": [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0], "strength": [0, 0.02, 0.05, 0.1, 0.2, 0.4],
          "weight": [0, 0.1, 0.25, 0.5, 1, 2, 5], "delay": [1, 2, 4, 8, 16]}
GRID_R, GRID_D = [0.1, 0.25, 0.5, 1.0], [0.05, 0.1, 0.2, 0.4]
QUICK = {"frequency": [0, 1.0], "strength": [0, 0.4], "weight": [0, 5], "delay": [1, 4]}
PARAM = {"frequency": "r", "strength": "delta", "weight": "lam", "delay": "k", "delay_matched": "k"}
INT_KEYS = ("G_rel", "Gpar_rel", "G_imm", "Gpar_imm")


def key_of(c):
    return (float(c["r"]), float(c["delta"]), int(c["k"]), float(c["lam"]))


def matched_lambda(k):
    f1 = H4.relevant_fraction(H4.make_hmm4(1.0, 0.4, 1)); fk = H4.relevant_fraction(H4.make_hmm4(1.0, 0.4, k))
    return float(f1 / fk)


def conditions(quick, grid=False):
    C = {}
    def add(sweep, **kw):
        c = {**BASE, **kw}; c["k"] = int(c["k"]); c["r"], c["delta"], c["lam"] = float(c["r"]), float(c["delta"]), float(c["lam"])
        C.setdefault(key_of(c), {"cond": c, "sweeps": []})["sweeps"].append(sweep)
    sw = QUICK if quick else SWEEPS
    for name, vals in sw.items():
        for v in vals:
            add(name, **{PARAM[name]: v})
    for k in sw["delay"]:
        if k != 1:
            add("delay_matched", k=k, lam=matched_lambda(k))
    if grid:
        for r in GRID_R:
            for d in GRID_D:
                add("grid", r=r, delta=d)
    return C


DENSE_CKPTS = (0, 10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250, 300, 400, 500, 700, 1000, 1500)
DENSE_CONDS = {"base (r=1, δ=0.4, k=1)": dict(BASE), "δ=0.05": {**BASE, "delta": 0.05}, "r=0": {**BASE, "r": 0.0}, "k=8": {**BASE, "k": 8}}


def run_one(args):
    cond, seed, steps, quick = args[:4]; ckpts = args[4] if len(args) > 4 else None
    torch.set_num_threads(1)
    m = H4.make_hmm4(cond["r"], cond["delta"], cond["k"])
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed)
    if ckpts is None:
        ckpts = tuple(s for s in Sm.CHECKPOINTS if s <= steps) if not quick else (0, steps // 2, steps)
    hist, ckpt = Sm.train_weighted(net, m, lam=cond["lam"], steps=steps, seed=seed, checkpoints=ckpts)
    ev = Pr.make_eval(m, n=2000 if not quick else 300, T=48)
    dyn = {}
    for step, sd in ckpt.items():
        net.load_state_dict(sd); dyn[step] = Pr.measure(net, m, ev, n_rsa=800 if not quick else 200, n_dec=3000 if not quick else 500)
    fc = H4.forgetting_cost(m, ev["X"], ev["Z"])
    integ = {f"{k}_int": float(np.mean([dyn[s][k] for s in dyn])) for k in INT_KEYS}
    return {"cond": cond, "seed": seed, "init": dyn[0], "final": dyn[max(dyn)], "integrated": integ, "forget": fc,
            "cost": float(cond["lam"] * fc["per_step"]), "dynamics": dyn, "loss_hist": hist[::10]}


def run_conditions(C, seeds, steps, quick, jobs, done):
    todo = [(C[key]["cond"], s, steps, quick) for key in C for s in seeds if (key, s) not in done]
    print(f"running {len(todo)} runs on {jobs} workers ...", flush=True)
    t0 = time.time(); out = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            r["sweeps"] = list(C[key_of(r["cond"])]["sweeps"]); out.append(r)
            f = r["final"]; c = r["cond"]
            print(f"[{i + 1}/{len(todo)}] r={c['r']:.2f} d={c['delta']:.2f} k={c['k']:2d} lam={c['lam']:.2f} seed={r['seed']}  "
                  f"P_metric={f['P_metric']:.2f} pre={f['P_metric_pre']:.2f} R2={f['belief_r2']:.2f} acc={f['branch_acc_pre']:.2f} "
                  f"Gpar_int={r['integrated']['Gpar_rel_int']:.3f} kl={f['kl']:.3f}  {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------- analysis ----------------
def target_gap(cond):
    """Required logit-space separation at t*: L2 norm of the log-odds difference between the two
    branches' next-token predictives at the pre-relevant position (0 when the step carries no loss)."""
    if cond["lam"] == 0:
        return 0.0
    m = H4.make_hmm4(cond["r"], cond["delta"], cond["k"])
    B = np.zeros((2, len(m.pi))); B[0, m.S[f"FA{m.k}"]] = 1; B[1, m.S[f"FB{m.k}"]] = 1
    P = H4.next_token(m, B); ok = (P > 1e-12).all(0)
    return float(np.linalg.norm(np.log(P[0, ok]) - np.log(P[1, ok])))


def rows_of(runs):
    return [{"P_metric": r["final"]["P_metric"], "belief_r2": r["final"]["belief_r2"], "branch_acc_pre": r["final"]["branch_acc_pre"],
             "cost": r["cost"], "cost_per_event": r["cond"]["lam"] * r["forget"]["per_event"],
             "G_rel_int": r["integrated"]["G_rel_int"], "Gpar_rel_int": r["integrated"]["Gpar_rel_int"],
             "applied_G": r["cond"]["lam"] * r["integrated"]["G_rel_int"], "applied_Gpar": r["cond"]["lam"] * r["integrated"]["Gpar_rel_int"],
             "target_gap": target_gap(r["cond"]), "sweeps": r["sweeps"], "cond": r["cond"]} for r in runs]


def ols(Xm, y):
    Z = np.column_stack([np.ones(len(y)), Xm]); beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    pred = Z @ beta; r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return beta, float(r2)


def regression(rows, names=("belief_r2", "cost", "G_rel_int", "Gpar_rel_int")):
    Xm = np.array([[r[n] for n in names] for r in rows]); y = np.array([r["P_metric"] for r in rows])
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-12; Xs = (Xm - mu) / sd; ys = (y - y.mean()) / (y.std() + 1e-12)
    beta, r2 = ols(Xs, ys)
    out = {"n": len(rows), "r2_full": r2, "beta": {n: float(b) for n, b in zip(names, beta[1:])},
           "spearman": {n: float(spearmanr(Xm[:, j], y).correlation) for j, n in enumerate(names)},
           "r2_single": {n: ols(Xs[:, [j]], ys)[1] for j, n in enumerate(names)}}
    for subset, label in ((("belief_r2",), "decodability_only"), (("cost",), "cost_only"), (("cost", "Gpar_rel_int"), "cost_plus_gradient"),
                          (("belief_r2", "cost"), "decodability_plus_cost")):
        idx = [n for n in subset if n in names]
        if len(idx) == len(subset):
            idx = [names.index(n) for n in subset]; out[f"r2_{label}"] = ols(Xs[:, idx], ys)[1]
    # leave-one-sweep-out out-of-sample R^2
    loso = {}
    for sweep in sorted({s for r in rows for s in r["sweeps"]}):
        te = np.array([sweep in r["sweeps"] for r in rows]); tr = ~te
        if te.sum() < 3 or tr.sum() < 5:
            continue
        m_, s_ = Xm[tr].mean(0), Xm[tr].std(0) + 1e-12
        beta, _ = ols((Xm[tr] - m_) / s_, y[tr]); pred = np.column_stack([np.ones(te.sum()), (Xm[te] - m_) / s_]) @ beta
        loso[sweep] = float(1 - ((y[te] - pred) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-12))
    out["loso_r2"] = loso
    return out


def dynamics_summary(run):
    dyn = run["dynamics"]; steps = sorted(dyn)
    def t90(key):
        v0, v1 = dyn[steps[0]][key], dyn[steps[-1]][key]; tot = abs(v1 - v0)
        if tot < 1e-9:
            return None
        return next(s for s in steps if abs(dyn[s][key] - v0) >= 0.9 * tot)
    peak = max(steps, key=lambda s: dyn[s]["G_rel"])
    return {"t90_branch_acc_pre": t90("branch_acc_pre"), "t90_belief_r2": t90("belief_r2"), "t90_P_metric": t90("P_metric"),
            "t90_D_delay": t90("D_delay"), "t90_kl_relevant": t90("kl_relevant"), "t_peak_G_rel": peak, "t_peak_Gpar_rel": max(steps, key=lambda s: dyn[s]["Gpar_rel"]),
            "curves": {k: [dyn[s][k] for s in steps] for k in ("branch_acc_pre", "belief_r2", "P_metric", "P_metric_pre", "D_delay", "D_control", "D_pre", "G_rel", "Gpar_rel", "kl_relevant", "kl")},
            "steps": steps}


def run_dense_dynamics(out, seeds, jobs, quick=False):
    """Dense-checkpoint runs of a few conditions for the training-dynamics figure."""
    steps = DENSE_CKPTS[-1] if not quick else 100
    ck = DENSE_CKPTS if not quick else (0, 10, 20, 50, 100)
    todo = [(c, s, steps, False, ck) for c in DENSE_CONDS.values() for s in seeds]
    print(f"dense dynamics: {len(todo)} runs ...", flush=True); t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(run_one, todo))
    D = {}
    for (label, _), r in zip([(l, s) for l in DENSE_CONDS for s in seeds], res):
        D.setdefault(label, {})[r["seed"]] = dynamics_summary(r)
    (out / "dynamics_dense.json").write_text(json.dumps(to_jsonable(D), indent=1))
    print(f"dense dynamics done in {time.time() - t0:.0f}s", flush=True)
    return D


def aggregate(runs):
    """Per condition: mean and std over seeds of the final, init and integrated quantities."""
    C = {}
    for r in runs:
        C.setdefault(key_of(r["cond"]), []).append(r)
    agg = {}
    for key, rr in C.items():
        a = {"cond": rr[0]["cond"], "sweeps": rr[0]["sweeps"], "n_seeds": len(rr), "cost": float(np.mean([r["cost"] for r in rr])),
             "forget_per_event": float(np.mean([r["forget"]["per_event"] for r in rr])), "relevant_frac": rr[0]["forget"]["relevant_frac"]}
        for k in rr[0]["final"]:
            v = [r["final"][k] for r in rr]; a[k] = float(np.mean(v)); a[f"{k}_std"] = float(np.std(v))
            a[f"init_{k}"] = float(np.mean([r["init"][k] for r in rr]))
        for k in INT_KEYS:
            v = [r["integrated"][f"{k}_int"] for r in rr]; a[f"{k}_int"] = float(np.mean(v)); a[f"{k}_int_std"] = float(np.std(v))
        a["target_gap"] = target_gap(rr[0]["cond"]); a["target_gap_std"] = 0.0
        a["applied_Gpar"] = rr[0]["cond"]["lam"] * a["Gpar_rel_int"]; a["applied_Gpar_std"] = rr[0]["cond"]["lam"] * a["Gpar_rel_int_std"]
        agg[key] = a
    return agg


def sweep_conditions(agg, sweep):
    rows = [a for a in agg.values() if sweep in a["sweeps"]]
    if sweep == "grid":
        return sorted(rows, key=lambda a: (a["cond"]["r"], a["cond"]["delta"]))
    return sorted(rows, key=lambda a: a["cond"][PARAM[sweep]])


def sweep_effect(agg, sweep):
    rows = sweep_conditions(agg, sweep)
    if len(rows) < 3:
        return float("nan")
    return float(spearmanr([a["cond"][PARAM[sweep]] for a in rows], [a["P_metric"] for a in rows]).correlation)


def write_tables(R, out):
    agg = R["agg"]; L = ["# TASK4 tables: geometric prominence\n"]
    cols = [("P_metric", "P_metric"), ("P_metric_pre", "P_metric (pre-relevant)"), ("D_delay_over_median", "d/median"), ("var_ratio_delayed_vs_random", "var ratio vs random dir"),
            ("belief_r2", "belief R²"), ("branch_acc_pre", "branch acc (pre)"), ("rsa_belief", "RSA belief"), ("rsa_P1", "RSA 1-step"), ("rsa_P2", "RSA 2-step"), ("rsa_future", "RSA future"),
            ("G_rel_int", "∫G_t"), ("Gpar_rel_int", "∫G_∥"), ("G_rel", "G_t final"), ("kl", "KL"), ("kl_relevant", "KL at t*")]
    for sweep in ("frequency", "strength", "weight", "delay", "delay_matched", "grid"):
        rows = sweep_conditions(agg, sweep)
        if not rows:
            continue
        L += [f"## {sweep} sweep", "", "| r | δ | k | λ | cost λ·ΔL_forget | ΔL per event | target gap | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (7 + len(cols))]
        for a in rows:
            c = a["cond"]
            L.append(f"| {c['r']:.2f} | {c['delta']:.2f} | {c['k']} | {c['lam']:.2f} | {a['cost']:.4f} | {a['forget_per_event']:.4f} | {a['target_gap']:.3f} | " +
                     " | ".join(f"{a[k]:.3f}±{a[k + '_std']:.3f}" if k + "_std" in a else f"{a[k]:.3f}" for k, _ in cols) + " |")
        a = rows[0]
        L.append("| init | | | | | | | " + " | ".join(f"{a['init_' + k]:.3f}" if "init_" + k in a else "" for k, _ in cols) + " |")
        L.append("")
    reg = R["regression"]
    L += ["## Regression of P_metric (per run, standardised)", "", f"n = {reg['n']}, R² full = {reg['r2_full']:.3f}", "",
          "| predictor | beta | Spearman | R² alone |", "|---|---|---|---|"]
    for n in reg["beta"]:
        L.append(f"| {n} | {reg['beta'][n]:+.3f} | {reg['spearman'][n]:+.3f} | {reg['r2_single'][n]:.3f} |")
    L += ["", "| model | R² |", "|---|---|"] + [f"| {k[3:]} | {v:.3f} |" for k, v in reg.items() if k.startswith("r2_") and k not in ("r2_full", "r2_single")]
    L += ["", "Leave-one-sweep-out out-of-sample R²: " + ", ".join(f"{k} {v:.3f}" for k, v in reg["loso_r2"].items()), ""]
    reg = R["regression_extended"]
    L += ["## Extended regression (adds per-event cost, λ-applied gradients, required target gap)", "", f"n = {reg['n']}, R² full = {reg['r2_full']:.3f}", "",
          "| predictor | beta | Spearman | R² alone |", "|---|---|---|---|"]
    for n in reg["beta"]:
        L.append(f"| {n} | {reg['beta'][n]:+.3f} | {reg['spearman'][n]:+.3f} | {reg['r2_single'][n]:.3f} |")
    L += ["", "Leave-one-sweep-out out-of-sample R²: " + ", ".join(f"{k} {v:.3f}" for k, v in reg["loso_r2"].items()), "",
          f"Target gap alone: R² {R['regression_gap_only']['r2_full']:.3f} over all runs; "
          f"{R['regression_gap_frequency_strength_weight']['r2_full']:.3f} over the frequency, strength, weight (and grid) sweeps (n = {R['regression_gap_frequency_strength_weight']['n']}); "
          "leave-one-sweep-out " + ", ".join(f"{k} {v:.3f}" for k, v in R["regression_gap_only"]["loso_r2"].items()), ""]
    L += ["## Training dynamics (step at which each quantity has made 90 % of its total change; G: step of its peak)", "",
          "| condition | seed | branch acc | belief R² | D_delay | P_metric | peak G_t | peak G_∥ | KL at t* |", "|---|---|---|---|---|---|---|---|---|"]
    dyn_sets = {"base (coarse checkpoints)": R["dynamics"]["base"], **R.get("dynamics_dense", {})}
    for label, D in dyn_sets.items():
        for seed, d in D.items():
            L.append(f"| {label} | {seed} | {d['t90_branch_acc_pre']} | {d['t90_belief_r2']} | {d.get('t90_D_delay')} | {d['t90_P_metric']} | {d['t_peak_G_rel']} | {d.get('t_peak_Gpar_rel')} | {d['t90_kl_relevant']} |")
    L += ["", "sweep effects (Spearman of P_metric with the swept parameter): " + ", ".join(f"{k} {v:+.2f}" for k, v in R["sweep_effect"].items()), ""]
    (out / "tables4.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results4")
    ap.add_argument("--jobs", type=int, default=6); ap.add_argument("--grid", action="store_true", help="force the r x delta grid")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="reuse runs already in <out>/results4.json")
    ap.add_argument("--dynamics", action="store_true", help="also run the dense-checkpoint dynamics conditions")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(exist_ok=True, parents=True); t0 = time.time()
    seeds = [0, 1, 2] if not args.quick else [0]; steps = args.steps or (3000 if not args.quick else 200)
    C = conditions(args.quick)
    prev = []
    if args.resume and (out / "results4.json").exists():
        for r in json.loads((out / "results4.json").read_text())["runs"]:
            r["dynamics"] = {int(k): v for k, v in r["dynamics"].items()}; r["sweeps"] = list(C.get(key_of(r["cond"]), {"sweeps": r["sweeps"]})["sweeps"]); prev.append(r)
        print(f"resumed {len(prev)} runs")
    done = {(key_of(r["cond"]), r["seed"]) for r in prev}
    runs = prev + run_conditions(C, seeds, steps, args.quick, args.jobs, done)
    agg = aggregate(runs)
    effects = {s: sweep_effect(agg, s) for s in ("frequency", "strength", "weight", "delay")}
    grid_ran = False
    if not args.quick and (args.grid or (effects["frequency"] > 0.5 and effects["strength"] > 0.5)):
        Cg = {k: v for k, v in conditions(False, grid=True).items() if "grid" in v["sweeps"]}
        done = {(key_of(r["cond"]), r["seed"]) for r in runs}
        for r in runs:                      # conditions already run just get the grid label
            if key_of(r["cond"]) in Cg and "grid" not in r["sweeps"]:
                r["sweeps"].append("grid")
        runs += run_conditions(Cg, seeds, steps, False, args.jobs, done); grid_ran = True
        agg = aggregate(runs)
    base_key = key_of(BASE)
    rows = rows_of(runs)
    R = {"runs": runs, "agg": agg, "regression": regression(rows), "sweep_effect": effects, "grid_ran": grid_ran,
         "regression_extended": regression(rows, names=("belief_r2", "cost", "cost_per_event", "applied_G", "applied_Gpar", "target_gap")),
         "regression_gap_only": regression(rows, names=("target_gap",)),
         "regression_gap_frequency_strength_weight": regression([r for r in rows if set(r["sweeps"]) & {"frequency", "strength", "weight", "grid"}], names=("target_gap",)),
         "dynamics": {"base": {r["seed"]: dynamics_summary(r) for r in runs if key_of(r["cond"]) == base_key},
                      "all": {f"{key_of(r['cond'])}/{r['seed']}": {k: v for k, v in dynamics_summary(r).items() if k != "curves"} for r in runs}}}
    if args.dynamics:
        R["dynamics_dense"] = run_dense_dynamics(out, seeds, args.jobs, args.quick)
    elif (out / "dynamics_dense.json").exists():
        R["dynamics_dense"] = json.loads((out / "dynamics_dense.json").read_text())
    write_tables(R, out)
    J = to_jsonable({**R, "agg": {str(k): v for k, v in agg.items()}})
    (out / "results4.json").write_text(json.dumps(J, indent=1))
    try:
        from goalgeo.plotting4 import make_figures
        make_figures(R, out)
    except ImportError as e:
        print(f"figures skipped: {e}")
    print(f"done in {time.time() - t0:.0f}s -> {out}/  sweep effects {effects}  regression R² {R['regression']['r2_full']:.3f}")


if __name__ == "__main__":
    main()
