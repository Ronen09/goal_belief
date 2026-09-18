"""TASK5 (readout scale): is the required logit gap implemented by hidden-state separation
or by readout norm? Fix the readout gain c (weight-normalised output layer) and see whether
the hidden separation at t*-1 compensates as 1/c.

    .venv/bin/python scripts/run_task5.py [--quick] [--jobs 8] [--out results5]
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

GAINS = [None, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DELTAS = [0.1, 0.4]
KEYS = ("mean_sep_pre", "mean_sep_control", "mean_sep_cue", "D_pre", "D_delay", "D_control", "P_metric", "g_eff", "achieved_gap_pre", "achieved_gap_control",
        "cos_readout_pre", "cos_readout_control", "readout_frobenius", "required_gap", "kl", "kl_relevant", "branch_acc_pre", "belief_r2", "D_delay_over_median")


def run_one(args):
    delta, gain, seed, steps, quick = args
    torch.set_num_threads(1)
    m = H4.make_hmm4(1.0, delta, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed, gain=gain)
    ckpts = tuple(s for s in Sm.CHECKPOINTS if s <= steps) if not quick else (0, steps // 2, steps)
    hist, ckpt = Sm.train_weighted(net, m, lam=1.0, steps=steps, seed=seed, checkpoints=ckpts)
    ev = Pr.make_eval(m, n=2000 if not quick else 300, T=48)
    dyn = {}
    for step, sd in ckpt.items():
        net.load_state_dict(sd); dyn[step] = Pr.measure(net, m, ev, n_rsa=800 if not quick else 200, n_dec=3000 if not quick else 500)
    return {"delta": delta, "gain": gain, "seed": seed, "init": dyn[0], "final": dyn[max(dyn)],
            "dynamics": {s: {k: d[k] for k in KEYS} for s, d in dyn.items()}, "loss_hist": hist[::10]}


def label(g):
    return "free" if g is None else f"{g:g}"


def aggregate(runs):
    agg = {}
    for r in runs:
        agg.setdefault((r["delta"], label(r["gain"])), []).append(r)
    out = {}
    for key, rr in agg.items():
        a = {"delta": key[0], "gain": key[1], "n": len(rr)}
        for k in KEYS:
            v = [r["final"][k] for r in rr]; a[k] = float(np.mean(v)); a[f"{k}_std"] = float(np.std(v)); a[f"init_{k}"] = float(np.mean([r["init"][k] for r in rr]))
        out[key] = a
    return out


def slope_fit(runs, delta, kl_max=0.01):
    """Slope of log mean_sep_pre against log g_eff over constrained, converged runs at one delta."""
    pts = [(np.log(r["final"]["g_eff"]), np.log(r["final"]["mean_sep_pre"])) for r in runs if r["delta"] == delta and r["gain"] is not None and r["final"]["kl"] < kl_max]
    if len(pts) < 3:
        return {"n": len(pts), "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    x, y = np.array(pts).T; A = np.column_stack([np.ones_like(x), x]); beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta; r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return {"n": len(pts), "slope": float(beta[1]), "intercept": float(beta[0]), "r2": float(r2)}


def write_tables(R, out):
    agg = R["agg"]; L = ["# TASK5 tables: readout scale vs hidden separation\n"]
    cols = [("g_eff", "‖w_x−w_y‖"), ("mean_sep_pre", "‖h̄_C−h̄_D‖ at t*−1"), ("mean_sep_control", "‖h̄_U−h̄_V‖"), ("mean_sep_cue", "‖h̄_A−h̄_B‖ at cue"),
            ("cos_readout_pre", "cos(w, h_C−h_D)"), ("achieved_gap_pre", "achieved gap"), ("required_gap", "required gap"), ("achieved_gap_control", "achieved gap control"),
            ("D_pre", "D_pre (mean pairwise)"), ("P_metric", "P_metric"), ("D_delay_over_median", "d/median"), ("branch_acc_pre", "branch acc"), ("kl", "KL"), ("kl_relevant", "KL at t*")]
    for delta in DELTAS:
        L += [f"## δ = {delta}", "", "| gain c | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (1 + len(cols))]
        for g in GAINS:
            a = agg.get((delta, label(g)))
            if a is None:
                continue
            L.append(f"| {label(g)} | " + " | ".join(f"{a[k]:.3f}±{a[k + '_std']:.3f}" for k, _ in cols) + " |")
        a = agg[(delta, "free")]
        L.append("| init (free) | " + " | ".join(f"{a['init_' + k]:.3f}" for k, _ in cols) + " |")
        f = R["slopes"][str(delta)]
        L += ["", f"log ‖h̄_C−h̄_D‖ vs log ‖w_x−w_y‖ over constrained converged runs: slope {f['slope']:.2f}, R² {f['r2']:.2f}, n = {f['n']} (prediction −1)", ""]
    (out / "tables5.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="results5"); ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(exist_ok=True, parents=True); t0 = time.time()
    seeds = [0, 1, 2] if not args.quick else [0]; steps = 3000 if not args.quick else 200
    gains = GAINS if not args.quick else [None, 0.5, 4.0]
    todo = [(d, g, s, steps, args.quick) for d in DELTAS for g in gains for s in seeds]
    print(f"running {len(todo)} runs on {args.jobs} workers ...", flush=True); runs = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for i, r in enumerate(ex.map(run_one, todo)):
            runs.append(r); f = r["final"]
            print(f"[{i + 1}/{len(todo)}] delta={r['delta']} gain={label(r['gain'])} seed={r['seed']}  g_eff={f['g_eff']:.2f} sep_pre={f['mean_sep_pre']:.2f} sep_ctrl={f['mean_sep_control']:.2f} "
                  f"gap={f['achieved_gap_pre']:.2f}/{f['required_gap']:.2f} cos={f['cos_readout_pre']:.2f} kl={f['kl']:.3f}  {time.time() - t0:.0f}s", flush=True)
    R = {"runs": runs, "agg": aggregate(runs), "slopes": {str(d): slope_fit(runs, d) for d in DELTAS}}
    write_tables(R, out)
    (out / "results5.json").write_text(json.dumps(to_jsonable({**R, "agg": {f"{k[0]}/{k[1]}": v for k, v in R["agg"].items()}}), indent=1))
    from goalgeo.plotting5 import make_figures
    make_figures(R, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/  slopes {R['slopes']}")


if __name__ == "__main__":
    main()
