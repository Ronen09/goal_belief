"""TASK7: which representation measures are invariant to function-preserving changes of
coordinates and optimisation history, and still sensitive to the function?

    .venv/bin/python rounds/r08_invariants/run.py [--quick] [--jobs 8] [--out results7]
"""
from __future__ import annotations

import argparse, itertools, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr, invariants as Iv
from goalgeo.analysis import to_jsonable

LR = 3e-3
HORIZON_CKPTS = (0, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000)
EPS_KL = 0.003
TRANSFORMS = [("scalar 0.1", "scalar", {"scale": 0.1}), ("scalar 10", "scalar", {"scale": 10.0}), ("diagonal", "diagonal", {}), ("orthogonal", "orthogonal", {}),
              ("full cond 1", "full", {"cond": 1.0}), ("full cond 3", "full", {"cond": 3.0}), ("full cond 10", "full", {"cond": 10.0}), ("full cond 30", "full", {"cond": 30.0})]


def conditions(quick):
    C = []
    def add(label, family, seeds, **kw):
        c = {"delta": 0.4, "gain": None, "lr_ratio": 1.0, "init_scale": 1.0, "steps": 3000, "ckpts": None, **kw}
        C.append({"label": label, "family": family, "seeds": list(seeds), "cond": c})
    if quick:
        add("base", "base", range(2)); add("gain0.5", "gain", range(1), gain=0.5); add("lr10", "lr", range(1), lr_ratio=10.0)
        add("horizon", "horizon", range(1), steps=300, ckpts=(0, 100, 200, 300)); add("d0.1_lr1", "sens", range(1), delta=0.1)
        return C
    add("base", "base", range(10))
    for c in (0.5, 1.0, 2.0, 4.0, 8.0):
        add(f"gain{c:g}", "gain", range(3), gain=c)
    for l in (0.1, 0.3, 3.0, 10.0):
        add(f"lr{l:g}", "lr", range(3), lr_ratio=l)
    for s in (0.1, 10.0):
        add(f"init{s:g}", "init", range(3), init_scale=s)
    add("horizon", "horizon", range(3), steps=10000, ckpts=HORIZON_CKPTS)
    for d in (0.1, 0.2):
        for l in (0.1, 1.0, 10.0):
            add(f"d{d:g}_lr{l:g}", "sens", range(3), delta=d, lr_ratio=l)
    return C


def eval_indices(n_states, quick):
    rng = np.random.default_rng(7)
    return rng.choice(n_states, 800 if not quick else 150, replace=False), rng.choice(n_states, 3000 if not quick else 400, replace=False)


def kl_to_targets(net, ev):
    with torch.no_grad():
        logp = torch.log_softmax(net.forward_all(torch.as_tensor(ev["X"])), -1)[:, :-1].numpy()
    Y = ev["Y"][:, :-1]
    return float((Y * (np.log(Y + 1e-12) - logp)).sum(-1).mean())


def run_one(args):
    C, seed, quick, out = args
    torch.set_num_threads(1); c = C["cond"]
    m = H4.make_hmm4(1.0, c["delta"], 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed, gain=c["gain"], out_scale=c["init_scale"])
    steps = c["steps"] if not quick else min(c["steps"], 200)
    ckpts = c["ckpts"] if c["ckpts"] else (0, steps)
    hist, ckpt = Sm.train_weighted(net, m, steps=steps, seed=seed, checkpoints=ckpts, lr=LR, lr_out=LR * c["lr_ratio"])
    torch.save(ckpt[steps], out / "models" / f"{C['label']}_s{seed}.pt")
    ev = Pr.make_eval(m, n=2000 if not quick else 300, T=48)
    sub, dec = eval_indices(ev["X"].size, quick)
    res = {"label": C["label"], "family": C["family"], "cond": c, "seed": seed, "horizon": {}}
    for step, sd in ckpt.items():
        net.load_state_dict(sd)
        ex = Iv.extract(net, m, ev, sub, dec, n_group=400 if not quick else 60, n_anchor=200 if not quick else 40, seed=seed)
        r, arrays = Iv.metrics_from_arrays(ex); r["kl"] = kl_to_targets(net, ev)
        if step == steps:
            res.update({"metrics": r, "Q": arrays["Q"], "RDM": arrays["RDM"], "logp": ex["logp"]})
            if C["label"] == "base" and seed == 0:
                res["ex"] = {k: v for k, v in ex.items() if k != "logp"}
        if c["ckpts"]:
            res["horizon"][step] = r
    return res


# ---------------- analyses ----------------
def converged(runs):
    return [r for r in runs if r["metrics"]["kl"] < EPS_KL]


def stability(rows, keys=Iv.ALL):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        if len(v) < 2:
            continue
        out[k] = {"n": len(v), "mean": float(v.mean()), "cv": float(v.std() / (abs(v.mean()) + 1e-12)), "fold": float(v.max() / v.min()) if v.min() > 0 else float("nan")}
    return out


def families(runs):
    conv = converged(runs); d04 = [r for r in conv if r["cond"]["delta"] == 0.4]
    base3 = [r for r in d04 if r["label"] == "base" and r["seed"] < 3]
    F = {"gain": [r for r in d04 if r["family"] == "gain"],
         "lr": [r for r in d04 if r["family"] == "lr"] + base3,
         "init": [r for r in d04 if r["family"] == "init"] + base3,
         "seeds": [r for r in d04 if r["label"] == "base"],
         "all": [r for r in d04 if r["family"] != "horizon"]}
    hz = [r for r in runs if r["family"] == "horizon"]
    F["horizon"] = [dict(metrics=v) for r in hz for s, v in r["horizon"].items() if s >= 1000 and v["kl"] < EPS_KL]
    return F


def pairwise_output_kl(runs):
    rows = [r for r in converged(runs) if r["cond"]["delta"] == 0.4 and r["family"] != "horizon"]
    out = {}
    for a, b in itertools.combinations(rows, 2):
        pa, pb = np.exp(a["logp"]), a["logp"] - b["logp"]
        kl = (pa * pb).sum(-1)
        out[f"{a['label']}_s{a['seed']}|{b['label']}_s{b['seed']}"] = (float(kl.mean()), float(kl.max()))
    vals = np.array(list(out.values()))
    return {"n_pairs": len(out), "mean_of_mean": float(vals[:, 0].mean()), "max_of_mean": float(vals[:, 0].max()), "max_of_max": float(vals[:, 1].max()),
            "worst_pairs": sorted(out.items(), key=lambda kv: -kv[1][0])[:5]}


def transform_test(ex, seed=0):
    rng = np.random.default_rng(seed); d = ex["W"].shape[1]
    base, _ = Iv.metrics_from_arrays(ex); Q0 = Iv.subspace_basis(ex["H_sub"])
    out = {}
    for name, kind, kw in TRANSFORMS:
        A = Iv.random_transform(kind, d, rng, **kw); ex2 = Iv.transform(ex, A)
        r, arrays = Iv.metrics_from_arrays(ex2)
        logit_dev = float(np.abs(ex2["H_sub"] @ ex2["W"].T - ex["H_sub"] @ ex["W"].T).max())
        out[name] = {"ratio": {k: (r[k] / base[k] if abs(base[k]) > 1e-12 else float("nan")) for k in Iv.ALL},
                     "P_frob": float(np.linalg.norm(Iv.projection(arrays["Q"]) - Iv.projection(Q0))), "logit_dev": logit_dev, "cond": float(np.linalg.cond(A))}
    return {"base": base, "transforms": out}


def two_model(runs):
    def mean_metrics(label):
        rr = [r for r in converged(runs) if r["label"] == label]
        return {k: float(np.mean([r["metrics"][k] for r in rr])) for k in Iv.ALL + ("kl",)}
    return {"lr0.1": mean_metrics("lr0.1"), "lr10": mean_metrics("lr10"), "base": mean_metrics("base")}


def seed_agreement(runs):
    rows = [r for r in converged(runs) if r["label"] == "base"]
    rsa, ov, dF = [], [], []
    for a, b in itertools.combinations(rows, 2):
        rsa.append(float(spearmanr(a["RDM"][np.triu_indices(len(a["RDM"]), 1)], b["RDM"][np.triu_indices(len(b["RDM"]), 1)]).correlation))
        ov.append(Iv.subspace_overlap(a["Q"], b["Q"]))
        dF.append(abs(a["metrics"]["D_F"] - b["metrics"]["D_F"]) / (0.5 * (a["metrics"]["D_F"] + b["metrics"]["D_F"])))
    return {"n_seeds": len(rows), "rsa_between_seeds": float(np.mean(rsa)), "subspace_overlap_between_seeds": float(np.mean(ov)), "rel_diff_D_F": float(np.mean(dF)),
            "cv": stability([r["metrics"] for r in rows])}


def sensitivity(runs):
    """F-ratio per measure: variance of the delta-group means over mean within-group variance,
    groups = delta in {0.1, 0.2, 0.4} each with lr ratios {0.1, 1, 10} x 3 seeds."""
    rows = [r for r in converged(runs) if (r["family"] == "sens") or (r["cond"]["delta"] == 0.4 and r["label"] in ("lr0.1", "lr10")) or (r["label"] == "base" and r["seed"] < 3)]
    groups = {}
    for r in rows:
        groups.setdefault(r["cond"]["delta"], []).append(r["metrics"])
    out = {}
    for k in Iv.ALL:
        means = {d: np.mean([m[k] for m in g]) for d, g in groups.items()}
        within = np.mean([np.var([m[k] for m in g]) for g in groups.values()])
        between = np.var(list(means.values()))
        out[k] = {"F": float(between / (within + 1e-12)), "means": {str(d): float(v) for d, v in means.items()}}
    return out, {str(d): [dict(cond=r["cond"], seed=r["seed"], **r["metrics"]) for r in rows if r["cond"]["delta"] == d] for d in groups}


def classify(R):
    tr = R["transform"]["transforms"]; st = R["stability"]; se = R["sensitivity"]
    out = {}
    for k in Iv.ALL:
        ratios = [abs(t["ratio"][k] - 1) for t in tr.values() if np.isfinite(t["ratio"][k])]
        exact = bool(ratios) and max(ratios) < 1e-5
        cv_all = st["all"].get(k, {}).get("cv", float("nan")); F = se[k]["F"]
        g = lambda f: st[f].get(k, {}).get("cv", float("nan"))   # noqa: E731
        out[k] = {"exact_coord_invariant": exact, "max_transform_dev": float(max(ratios)) if ratios else float("nan"), "cv_all": cv_all, "cv_lr": g("lr"),
                  "cv_gain": g("gain"), "cv_seeds": g("seeds"), "F_delta": F,
                  "class": "exact invariant" if exact and cv_all < 0.15 else ("empirical invariant" if cv_all < 0.15 and F > 10 else ("non-invariant descriptor" if cv_all >= 0.15 else "constant / insensitive"))}
    return out


def write_tables(R, out):
    L = ["# TASK7 tables: representation invariants\n"]
    L += ["## Equivalence set", "", f"converged (KL to targets < {EPS_KL}): {R['n_converged']} of {R['n_runs']} runs; excluded: {R['excluded']}", "",
          f"pairwise output KL over 300 sequences, δ=0.4 converged models: {R['pairwise_kl']['n_pairs']} pairs, mean {R['pairwise_kl']['mean_of_mean']:.5f}, worst mean {R['pairwise_kl']['max_of_mean']:.5f}, worst max {R['pairwise_kl']['max_of_max']:.4f}", ""]
    fams = ["gain", "lr", "init", "horizon", "seeds", "all"]
    L += ["## Stability across equivalent models (CV = std/|mean|; fold = max/min)", "", "| measure | " + " | ".join(f"{f} CV | {f} fold" for f in fams) + " |", "|" + "---|" * (1 + 2 * len(fams))]
    for k in Iv.ALL:
        L.append(f"| {k} | " + " | ".join(f"{R['stability'][f][k]['cv']:.3f} | {R['stability'][f][k]['fold']:.2f}" if k in R["stability"][f] else " | " for f in fams) + " |")
    L += ["", "## Exact coordinate transforms on the base seed-0 model: M(AH)/M(H)", "", "| measure | " + " | ".join(t[0] for t in TRANSFORMS) + " |", "|" + "---|" * (1 + len(TRANSFORMS))]
    tr = R["transform"]["transforms"]
    for k in Iv.ALL:
        L.append(f"| {k} | " + " | ".join(f"{tr[t[0]]['ratio'][k]:.6f}" for t in TRANSFORMS) + " |")
    L.append("| ‖P′−P‖_F | " + " | ".join(f"{tr[t[0]]['P_frob']:.2e}" for t in TRANSFORMS) + " |")
    L.append("| max logit deviation | " + " | ".join(f"{tr[t[0]]['logit_dev']:.2e}" for t in TRANSFORMS) + " |")
    L += ["", "## Two models, same function (seed means)", "", "| measure | lr ×0.1 | base | lr ×10 | ratio ×0.1 / ×10 |", "|---|---|---|---|---|"]
    tm = R["two_model"]
    for k in Iv.ALL + ("kl",):
        L.append(f"| {k} | {tm['lr0.1'][k]:.4f} | {tm['base'][k]:.4f} | {tm['lr10'][k]:.4f} | {tm['lr0.1'][k] / tm['lr10'][k] if abs(tm['lr10'][k]) > 1e-12 else float('nan'):.3f} |")
    sa = R["seed_agreement"]
    L += ["", "## Seeds", "", f"{sa['n_seeds']} base seeds: between-seed Euclidean RSA {sa['rsa_between_seeds']:.3f}, subspace overlap {sa['subspace_overlap_between_seeds']:.3f}, relative |ΔD_F| {sa['rel_diff_D_F']:.3f}", ""]
    L += ["## Functional sensitivity (δ ∈ {0.1, 0.2, 0.4}, each with lr ratios 0.1/1/10 × 3 seeds)", "", "| measure | mean δ=0.1 | mean δ=0.2 | mean δ=0.4 | F between/within |", "|---|---|---|---|---|"]
    for k in Iv.ALL:
        s = R["sensitivity"][k]; L.append(f"| {k} | {s['means'].get('0.1', float('nan')):.4f} | {s['means'].get('0.2', float('nan')):.4f} | {s['means'].get('0.4', float('nan')):.4f} | {s['F']:.1f} |")
    L += ["", "## Classification", "", "| measure | exact coordinate invariant | max transform deviation | CV all | CV lr | CV gain | CV seeds | F(δ) | class |", "|---|---|---|---|---|---|---|---|---|"]
    for k, c in R["classification"].items():
        L.append(f"| {k} | {c['exact_coord_invariant']} | {c['max_transform_dev']:.1e} | {c['cv_all']:.3f} | {c['cv_lr']:.3f} | {c['cv_gain']:.3f} | {c['cv_seeds']:.3f} | {c['F_delta']:.1f} | {c['class']} |")
    (out / "tables.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="rounds/r08_invariants"); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(); out = Path(args.out); (out / "models").mkdir(exist_ok=True, parents=True); t0 = time.time()
    C = conditions(args.quick)
    todo = [(c, s, args.quick, out) for c in sorted(C, key=lambda c: -c["cond"]["steps"]) for s in c["seeds"]]
    print(f"running {len(todo)} runs on {args.jobs} workers ...", flush=True); runs = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            runs.append(r); mt = r["metrics"]
            print(f"[{i + 1}/{len(todo)}] {r['label']:10s} seed={r['seed']}  kl={mt['kl']:.4f} sep={mt['sep_cue']:.2f} g={mt['g_eff']:.2f} D_future={mt['D_future']:.3f} D_F={mt['D_F']:.3f} "
                  f"JS={mt['JS_future']:.4f} R2={mt['r2_belief']:.3f} rank={mt['rank']}  {time.time() - t0:.0f}s", flush=True)
    F = families(runs)
    R = {"n_runs": len(runs), "n_converged": len(converged(runs)), "excluded": [f"{r['label']}_s{r['seed']} kl={r['metrics']['kl']:.4f}" for r in runs if r["metrics"]["kl"] >= EPS_KL],
         "stability": {f: stability([r["metrics"] for r in rows]) for f, rows in F.items()},
         "pairwise_kl": pairwise_output_kl(runs), "two_model": two_model(runs), "seed_agreement": seed_agreement(runs)}
    base0 = next(r for r in runs if r["label"] == "base" and r["seed"] == 0)
    R["transform"] = transform_test(base0["ex"])
    R["sensitivity"], R["sensitivity_rows"] = sensitivity(runs)
    R["horizon"] = {r["seed"]: {int(s): v for s, v in r["horizon"].items()} for r in runs if r["family"] == "horizon"}
    R["classification"] = classify(R)
    R["runs"] = [{k: v for k, v in r.items() if k not in ("Q", "RDM", "logp", "ex")} for r in runs]
    write_tables(R, out)
    (out / "results7.json").write_text(json.dumps(to_jsonable(R), indent=1))
    from plots import make_figures
    make_figures(R, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
