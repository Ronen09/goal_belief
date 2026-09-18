"""TASK9: reproduce the TASK4 dissociation, the readout-scale allocation and the functional
invariants on a 2-layer pre-LN causal transformer, with and without the final LayerNorm.

    .venv/bin/python scripts/run_task9.py [--quick] [--jobs 8] [--out results9]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr, tfm as Tf, tfm_measure as Tm
from goalgeo.analysis import to_jsonable

LR = 1e-3
KEYS_CV = ("sep_cue", "hidden_norm_l1", "D_delay", "P_metric", "rsa_euclid_l1", "belief_r2_l1", "branch_acc_pre_postnorm", "g_eff", "sep_pre_postnorm", "sep_pre_final",
           "cos_theta_postnorm", "achieved_gap_postnorm", "D_future", "D_F", "JS_future", "rank_l1")


def conditions(quick):
    C = []
    def add(label, family, seeds, **kw):
        c = {"delta": 0.4, "r": 1.0, "lam": 1.0, "final_ln": True, "gain": None, "lr_ratio": 1.0, "init_scale": 1.0, **kw}
        C.append({"label": label + ("" if c["final_ln"] else "_noln"), "family": family, "seeds": list(seeds), "cond": c})
    seeds = range(3) if not quick else range(1)
    for ln in (True, False):
        add("base", "core", seeds, final_ln=ln)
        if quick:
            add("r0", "core", seeds, r=0.0, final_ln=ln); add("gain8", "alloc", seeds, gain=8.0, final_ln=ln); continue
        for d in (0.1, 0.2):
            add(f"d{d:g}", "core", seeds, delta=d, final_ln=ln)
        add("r0", "core", seeds, r=0.0, final_ln=ln); add("lam0", "core", seeds, lam=0.0, final_ln=ln)
        for c in (0.5, 2.0, 8.0):
            add(f"gain{c:g}", "alloc", seeds, gain=c, final_ln=ln)
        for l in (0.1, 10.0):
            add(f"lr{l:g}", "alloc", seeds, lr_ratio=l, final_ln=ln)
    return C


def run_one(args):
    C, seed, quick, out, device, steps = args
    torch.set_num_threads(1); c = C["cond"]
    m = H4.make_hmm4(c["r"], c["delta"], 1)
    net = Tf.CausalTransformer(n_vocab=H4.V, final_ln=c["final_ln"], seed=seed, gain=c["gain"], out_scale=c["init_scale"]).to(device)
    steps = steps if not quick else 200
    hist, ckpt = Sm.train_weighted(net, m, lam=c["lam"], steps=steps, seed=seed, checkpoints=(0, steps), lr=LR, lr_out=LR * c["lr_ratio"])
    torch.save(ckpt[steps], out / "models" / f"{C['label']}_s{seed}.pt")
    net = net.cpu(); ev = Pr.make_eval(m, n=2000 if not quick else 300, T=48); net.eval()   # measurements run on CPU
    res = {"label": C["label"], "family": C["family"], "cond": c, "seed": seed}
    res["final"] = Tm.measure_tfm(net, m, ev, n_dec=3000 if not quick else 400, n_anchor=200 if not quick else 40)
    net.load_state_dict(ckpt[0]); res["init"] = Tm.measure_tfm(net, m, ev, n_dec=3000 if not quick else 400, n_anchor=200 if not quick else 40)
    return res


def aggregate(runs):
    A = {}
    for r in runs:
        A.setdefault(r["label"], []).append(r)
    agg = {}
    for label, rr in A.items():
        a = {"label": label, "family": rr[0]["family"], "cond": rr[0]["cond"], "n": len(rr)}
        for k, v in rr[0]["final"].items():
            if isinstance(v, list):
                a[k] = np.mean([r["final"][k] for r in rr], 0).tolist(); continue
            vals = [r["final"][k] for r in rr]; a[k] = float(np.mean(vals)); a[f"{k}_std"] = float(np.std(vals)); a[f"init_{k}"] = float(np.mean([r["init"][k] for r in rr]))
        agg[label] = a
    return agg


def cv(runs, keys):
    out = {}
    for k in keys:
        v = np.array([r["final"][k] for r in runs], float); out[k] = float(v.std() / (abs(v.mean()) + 1e-12)) if len(v) > 1 else float("nan")
    return out


def write_tables(R, out):
    agg = R["agg"]; L = ["# TASK9 tables: transformer reproduction\n"]
    core = [("P_metric", "P_metric (l1 cue)"), ("P_metric_pre_l1", "P_metric (l1, t*−1)"), ("branch_acc_cue_l1", "branch acc cue (l1)"), ("branch_acc_pre_l1", "branch acc t*−1 (l1)"),
            ("branch_acc_pre_final", "branch acc t*−1 (final res.)"), ("branch_acc_pre_postnorm", "branch acc t*−1 (post-norm)"), ("belief_r2_l1", "belief R² (l1)"), ("JS_future", "JS (patch)"), ("kl", "KL")]
    alloc = [("g_eff", "‖w_x−w_y‖"), ("ln_gain_norm", "‖γ‖"), ("sep_pre_postnorm", "‖Δh̄‖ post-norm"), ("cos_theta_postnorm", "cos θ post-norm"), ("achieved_gap_postnorm", "achieved gap"),
             ("sep_pre_final", "‖Δh̄‖ final residual"), ("cos_theta_final", "cos θ final residual"), ("sep_cue", "‖Δh̄‖ l1 cue"), ("hidden_norm_l1", "‖h‖ l1"), ("required_gap", "required gap"), ("kl", "KL")]
    for title, fam, cols in (("Core: decodability vs prominence", "core", core), ("Allocation: readout scale", "alloc", alloc)):
        for ln in (True, False):
            rows = [a for a in agg.values() if (a["family"] == fam or a["label"] in ("base", "base_noln")) and a["cond"]["final_ln"] == ln]
            L += [f"## {title}, {'with' if ln else 'without'} final LayerNorm", "", "| condition | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (1 + len(cols))]
            for a in rows:
                L.append(f"| {a['label']} | " + " | ".join(f"{a[k]:.3f}±{a[k + '_std']:.3f}" if k + "_std" in a else f"{a[k]:.3f}" for k, _ in cols) + " |")
            a = rows[0]; L.append("| init (base) | " + " | ".join(f"{a['init_' + k]:.3f}" if "init_" + k in a else "" for k, _ in cols) + " |"); L.append("")
    L += ["## Invariants: CV across functionally equivalent δ=0.4 models (base seeds + lr ×0.1/×10 + gains, KL < 0.003)", "", "| measure | with final LN | without final LN |", "|---|---|---|"]
    for k in KEYS_CV:
        L.append(f"| {k} | {R['cv_ln'][k]:.3f} | {R['cv_noln'][k]:.3f} |")
    L += ["", "## Propagation depth (patched B state into A sequence; JS of predictions and raw final-residual difference at positions t..t+4)", "", "| model | quantity | k=0 | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|---|"]
    for label in ("base", "lr0.1", "lr10", "base_noln", "lr0.1_noln", "lr10_noln"):
        if label in agg:
            L.append(f"| {label} | JS | " + " | ".join(f"{x:.3f}" for x in agg[label]["depth_js"]) + " |")
            L.append(f"| {label} | raw | " + " | ".join(f"{x:.3f}" for x in agg[label]["depth_raw_final"]) + " |")
    L += ["", "excluded from CV (KL ≥ 0.003): " + ", ".join(R["excluded"]), ""]
    (out / "tables9.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results9"); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="training device; the transformer trains ~16x faster on the GPU")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--only", default=None, help="comma-separated label_s<seed> ids to (re)run")
    args = ap.parse_args(); out = Path(args.out); (out / "models").mkdir(parents=True, exist_ok=True); t0 = time.time()
    C = conditions(args.quick); todo = [(c, s, args.quick, out, args.device, args.steps) for c in C for s in c["seeds"]]
    prev = []
    if args.only:
        keep = set(args.only.split(",")); todo = [t for t in todo if f"{t[0]['label']}_s{t[1]}" in keep]
        if (out / "results9.json").exists():   # keep every earlier run except the ones being redone
            prev = [r for r in json.loads((out / "results9.json").read_text())["runs"] if f"{r['label']}_s{r['seed']}" not in keep]
            print(f"resumed {len(prev)} runs", flush=True)
    print(f"running {len(todo)} runs on {args.jobs} workers ({args.device}) ...", flush=True); runs = list(prev)
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            runs.append(r); f = r["final"]
            print(f"[{i + 1}/{len(todo)}] {r['label']:12s} s{r['seed']}  kl={f['kl']:.4f} P={f['P_metric']:.2f} acc_pre(l1/final/pn)={f['branch_acc_pre_l1']:.2f}/{f['branch_acc_pre_final']:.2f}/{f['branch_acc_pre_postnorm']:.2f} "
                  f"g={f['g_eff']:.2f} sep_pn={f['sep_pre_postnorm']:.2f} sep_fin={f['sep_pre_final']:.2f} gap={f['achieved_gap_postnorm']:.2f} JS={f['JS_future']:.3f} D_F={f['D_F']:.2f}  {time.time() - t0:.0f}s", flush=True)
    eq = lambda ln: [r for r in runs if r["cond"]["delta"] == 0.4 and r["cond"]["r"] == 1 and r["cond"]["lam"] == 1 and r["cond"]["final_ln"] == ln and r["final"]["kl"] < 0.003]   # noqa: E731
    R = {"runs": runs, "agg": aggregate(runs), "cv_ln": cv(eq(True), KEYS_CV), "cv_noln": cv(eq(False), KEYS_CV), "n_eq": [len(eq(True)), len(eq(False))],
         "excluded": [f"{r['label']}_s{r['seed']} kl={r['final']['kl']:.4f}" for r in runs if r["final"]["kl"] >= 0.003]}
    write_tables(R, out); (out / "results9.json").write_text(json.dumps(to_jsonable(R), indent=1))
    from goalgeo.plotting9 import make_figures
    make_figures(R, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
