"""TASK6 (allocation): what sets the optimiser's split of the required logit gap between
readout gain and hidden separation? Three manipulations at delta=0.4 with a free readout:
readout learning-rate ratio, readout initialisation scale, and the control branch's contrast.

    .venv/bin/python scripts/run_task6.py [--quick] [--jobs 8] [--out results6]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr
from goalgeo.analysis import to_jsonable

LR = 3e-3
SWEEPS = {"lr_ratio": [0.1, 0.3, 1.0, 3.0, 10.0], "init_scale": [0.1, 1.0, 10.0], "control": [0.9, 0.75, 0.6, 0.5]}
QUICK = {"lr_ratio": [0.1, 1.0, 10.0], "init_scale": [10.0], "control": [0.5]}
KEYS = ("mean_sep_pre", "mean_sep_control", "mean_sep_cue", "D_pre", "D_delay", "D_control", "P_metric", "g_eff", "achieved_gap_pre", "achieved_gap_control",
        "cos_readout_pre", "cos_readout_control", "readout_frobenius", "required_gap", "kl", "kl_relevant", "branch_acc_pre", "belief_r2")


def conditions(quick):
    sw = QUICK if quick else SWEEPS; C = {}
    def add(sweep, **kw):
        c = {"lr_ratio": 1.0, "init_scale": 1.0, "control": 0.9, **kw}; key = (c["lr_ratio"], c["init_scale"], c["control"])
        C.setdefault(key, {"cond": c, "sweeps": []})["sweeps"].append(sweep)
    for name, vals in sw.items():
        for v in vals:
            add(name, **{name: v})
    return C


def run_one(args):
    cond, seed, steps, quick = args
    torch.set_num_threads(1)
    m = H4.make_hmm4(1.0, 0.4, 1, ctrl=cond["control"])
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed, out_scale=cond["init_scale"])
    ckpts = tuple(s for s in Sm.CHECKPOINTS if s <= steps) if not quick else (0, steps // 2, steps)
    hist, ckpt = Sm.train_weighted(net, m, lam=1.0, steps=steps, seed=seed, checkpoints=ckpts, lr=LR, lr_out=LR * cond["lr_ratio"])
    ev = Pr.make_eval(m, n=2000 if not quick else 300, T=48)
    dyn = {}
    for step, sd in ckpt.items():
        net.load_state_dict(sd); dyn[step] = Pr.measure(net, m, ev, n_rsa=800 if not quick else 200, n_dec=3000 if not quick else 500)
    return {"cond": cond, "seed": seed, "init": dyn[0], "final": dyn[max(dyn)],
            "dynamics": {s: {k: d[k] for k in KEYS} for s, d in dyn.items()}, "loss_hist": hist[::10]}


def aggregate(runs):
    C = {}
    for r in runs:
        C.setdefault(tuple(r["cond"].values()), []).append(r)
    agg = {}
    for key, rr in C.items():
        a = {"cond": rr[0]["cond"], "sweeps": rr[0]["sweeps"], "n": len(rr)}
        for k in KEYS:
            v = [r["final"][k] for r in rr]; a[k] = float(np.mean(v)); a[f"{k}_std"] = float(np.std(v)); a[f"init_{k}"] = float(np.mean([r["init"][k] for r in rr]))
        agg[key] = a
    return agg


def write_tables(R, out):
    agg = R["agg"]; L = ["# TASK6 tables: what sets the readout / representation split\n"]
    cols = [("g_eff", "‖w_x−w_y‖"), ("init_g_eff", "‖w_x−w_y‖ init"), ("mean_sep_pre", "‖h̄_C−h̄_D‖ at t*−1"), ("mean_sep_control", "‖h̄_U−h̄_V‖"), ("mean_sep_cue", "‖h̄_A−h̄_B‖ cue"),
            ("cos_readout_pre", "cos θ"), ("achieved_gap_pre", "achieved gap"), ("achieved_gap_control", "achieved gap control"), ("P_metric", "P_metric"), ("readout_frobenius", "‖W‖_F"), ("kl", "KL")]
    for sweep, xl in (("lr_ratio", "readout lr / recurrent lr"), ("init_scale", "readout init scale"), ("control", "control contrast P(x|A′)")):
        rows = sorted([a for a in agg.values() if sweep in a["sweeps"]], key=lambda a: a["cond"][sweep])
        L += [f"## {sweep} sweep (δ = 0.4, free readout)", "", f"| {xl} | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (1 + len(cols))]
        for a in rows:
            L.append(f"| {a['cond'][sweep]:g} | " + " | ".join(f"{a[k]:.3f}±{a[k + '_std']:.3f}" if k + "_std" in a else f"{a[k]:.3f}" for k, _ in cols) + " |")
        L.append("")
    (out / "tables6.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results6"); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(exist_ok=True, parents=True); t0 = time.time()
    seeds = [0, 1, 2] if not args.quick else [0]; steps = 3000 if not args.quick else 200
    C = conditions(args.quick)
    todo = [(C[k]["cond"], s, steps, args.quick) for k in C for s in seeds]
    print(f"running {len(todo)} runs on {args.jobs} workers ...", flush=True); runs = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            r["sweeps"] = C[tuple(r["cond"].values())]["sweeps"]; runs.append(r); f = r["final"]; c = r["cond"]
            print(f"[{i + 1}/{len(todo)}] lr×{c['lr_ratio']:g} init×{c['init_scale']:g} ctrl={c['control']:g} seed={r['seed']}  g_eff={f['g_eff']:.2f} (init {r['init']['g_eff']:.2f}) "
                  f"sep_pre={f['mean_sep_pre']:.2f} sep_ctrl={f['mean_sep_control']:.2f} cos={f['cos_readout_pre']:.2f} gap={f['achieved_gap_pre']:.2f} kl={f['kl']:.3f}  {time.time() - t0:.0f}s", flush=True)
    R = {"runs": runs, "agg": aggregate(runs)}
    write_tables(R, out)
    (out / "results6.json").write_text(json.dumps(to_jsonable({**R, "agg": {str(k): v for k, v in R["agg"].items()}}), indent=1))
    from goalgeo.plotting6 import make_figures
    make_figures(R, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
