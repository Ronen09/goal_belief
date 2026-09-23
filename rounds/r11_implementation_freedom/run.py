"""TASK10: two experiments designed to falsify the claims of rounds/r11_implementation_freedom/THEORY.md.

  exp 1 (routing):       force the cue's route with attention masks or an attention penalty and
                         ask whether single-node effects move while complete-cut effects do not.
  exp 2 (factorization): vary the readout gain budget under three final-norm variants and test
                         the pre-registered prediction of which term of C = g*D*cos(theta)
                         absorbs the change, including the frozen-LayerNorm ceiling and where
                         the empirical failure boundary actually falls.
  controls:              the same frozen models at a 3x learning rate and at 4x the steps, which
                         separate "the ceiling forbids it" from "the optimiser did not get there".

Every grid is trained as ONE stacked model on the GPU (goalgeo/tfm_batched.py): the models of a
grid step together, which is ~4x the throughput of one process per model and actually loads the
device. Measurement runs on the unstacked models in plain fp32.

    .venv/bin/python rounds/r11_implementation_freedom/run.py --exp all --steps 25000      # ~1.5 h on an A100
    .venv/bin/python rounds/r11_implementation_freedom/run.py --exp routing --quick        # smoke test
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from goalgeo import hmm4 as H4, prominence as Pr, cuts as Ct, factorize as Fz, tfm_batched as Tb
from goalgeo.analysis import to_jsonable

LR = 1e-3
D_MODEL = 64
MU = 3.0                     # attention-penalty coefficient
DEFAULT_STEPS = 25000
ROUTING = [
    ("free",    {}),
    ("l1_diag", {"attn_diag": (0,)}),                      # block 1 moves nothing: route via res1(t)
    ("l2_diag", {"attn_diag": (1,)}),                      # block 2 is local: route via res1(t+1)
    ("pen_l2",  {"penalty": "pen_l2", "mu": 3.0}),         # soft version of l2_diag's pressure
    ("pen_l1",  {"penalty": "pen_l1", "mu": 3.0}),         # soft version of l1_diag's pressure
]


# ---------------------------------------------------------------- grids

GAINS_D4 = (0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 2.4)
GAINS_D2 = (0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3)
NORMS = ("none", "frozen", "learn")


def routing_specs(seeds):
    S = []
    for lab, kw in ROUTING:
        for s in seeds:
            S.append(Tb.Spec(seed=s, delta=0.4, r=1.0, norm="learn", label=lab,
                             attn_diag=kw.get("attn_diag", ()),
                             mu_l1=MU if kw.get("penalty") == "pen_l1" else 0.0,
                             mu_l2=MU if kw.get("penalty") == "pen_l2" else 0.0))
    return S


def factor_specs(seeds, gains4=GAINS_D4, gains2=GAINS_D2):
    S = [Tb.Spec(seed=s, delta=0.4, gain=c, norm=n, label=f"{n}_c{c:g}_d0.4")
         for n in NORMS for c in gains4 for s in seeds]
    S += [Tb.Spec(seed=s, delta=0.2, gain=c, norm=n, label=f"{n}_c{c:g}_d0.2")
          for n in ("frozen", "none") for c in gains2 for s in seeds]
    return S


def control_specs(seeds, gains=(0.2, 0.3, 0.4, 0.6)):
    """Frozen models at and above the empirical boundary, for the two optimiser controls."""
    return [Tb.Spec(seed=s, delta=0.4, gain=c, norm="frozen", label=f"frozen_c{c:g}_d0.4")
            for c in gains for s in seeds]


# ---------------------------------------------------------------- run one stacked grid

def train_grid(specs, steps, lr, device, out, tag, pool=4000, batch=128, T=48):
    t0 = time.time()
    bt = Tb.BatchedTransformer(specs, d=D_MODEL, T=T, device=device)
    Xp, Yp = Tb.make_data(specs, pool=pool, T=T, device=device)
    prog = lambda st, losses: print(f"  [{tag}] step {st}/{steps}  loss mean {np.mean(losses):.4f} "   # noqa: E731
                                    f"max {np.max(losses):.4f}  {time.time() - t0:.0f}s", flush=True)
    hist = Tb.train_batched(bt, Xp, Yp, steps=steps, batch=batch, lr=lr, log_every=max(steps // 10, 1), progress=prog)
    nets = bt.to_nets()
    del bt, Xp, Yp
    torch.cuda.empty_cache() if device == "cuda" else None
    return nets, hist


def measure_grid(specs, nets, hist, kind, steps, lr, device, out, quick, control=False):
    runs = []
    for j, (spec, net) in enumerate(zip(specs, nets)):
        m = H4.make_hmm4(spec.r, spec.delta, 1)
        net = net.to(device).eval()
        ev = Pr.make_eval(m, n=400 if quick else 2000, T=48)
        r = {"exp": kind, "label": spec.label, "seed": spec.seed, "steps": steps, "lr": lr,
             "control": control, "loss": float(hist[-50:, j].mean())}
        if kind == "routing":
            r.update(Ct.measure(net, m, ev, n_anchor=40 if quick else 200))
        else:
            r.update({"norm": spec.norm, "gain": spec.gain, "delta": spec.delta})
            r.update(Fz.measure(net, m, ev, spec.gain))
        runs.append(r)
        name = (f"{kind}_{spec.label}_s{spec.seed}" + (f"_lr{lr:g}" if lr != LR else "")
                + (f"_st{steps}" if steps != DEFAULT_STEPS else "") + ".pt")
        torch.save(net.state_dict(), Path(out) / "models" / name)
        net.cpu()
    return runs


def report_routing(r):
    return (f"{r['label']:8s} s{r['seed']} kl={r['kl']:.4f} behav={r['phi_behav']:.4f} "
            f"cut={r['phi_L1{t,t+1}']:.4f} t={r['phi_L1{t}']:.4f} t+1={r['phi_L1{t+1}']:.4f} "
            f"a1={r['attn1_t1_to_t']:.2f} a2={r['attn2_t1_to_t']:.2f} dev={max(r[f'dev_{k}'] for k in Ct.COMPLETE):.1e}")


def report_factor(r):
    return (f"{r['label']:18s} s{r['seed']} kl={r['kl']:.4f} klrel={r['kl_relevant']:.4f} "
            f"C={r['C']:.3f}/{r['C_required']:.3f} g={r['g']:.3f} D={r['D']:.2f} cos={r['cos_theta']:.3f} "
            f"|g|={r['gamma_norm']:.2f} conv={r['converged']:.0f}")


# ---------------------------------------------------------------- analysis

def _cv(v):
    v = np.asarray(v, float)
    return float(v.std() / (abs(v.mean()) + 1e-12)) if len(v) > 1 else float("nan")


def _slope(x, y):
    """OLS slope of log y on log x."""
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    if len(x) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def by(runs, *keys):
    g = {}
    for r in runs:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def analyse_routing(runs, kl_tol=0.003):
    eq = [r for r in runs if r["kl"] < kl_tol]
    A = {"n": len(runs), "n_equivalent": len(eq), "kl_tol": kl_tol,
         "excluded": [f"{r['label']}_s{r['seed']} kl={r['kl']:.4f}" for r in runs if r["kl"] >= kl_tol]}
    keys = [f"phi_{k}" for k in Ct.CUT_SETS] + ["phi_behav"]
    A["cv"] = {k: _cv([r[k] for r in eq]) for k in keys}
    A["mean"] = {k: float(np.mean([r[k] for r in eq])) for k in keys}
    A["max_dev_complete"] = float(max((r[f"dev_{k}"] for r in runs for k in Ct.COMPLETE), default=np.nan))
    A["max_dev_incomplete"] = float(max((r[f"dev_{k}"] for r in runs for k in Ct.CUT_SETS if k not in Ct.COMPLETE), default=np.nan))
    # only functionally equivalent models: a condition's mean must not mix in a model that
    # computes a different function (the masked conditions each lost a seed to a bad basin)
    A["per_condition"] = {lab[0]: {**{k: float(np.mean([r[k] for r in rr]))
                                      for k in keys + ["attn1_t1_to_t", "attn2_t1_to_t", "kl"]},
                                   "n": len(rr), "n_all": len([r for r in runs if r["label"] == lab[0]])}
                          for lab, rr in by(eq, "label").items()}
    if len(eq) > 1:
        A["corr_attn2_vs_phi_t"] = float(np.corrcoef([r["attn2_t1_to_t"] for r in eq], [r["phi_L1{t}"] for r in eq])[0, 1])
        A["corr_attn1_vs_phi_t1"] = float(np.corrcoef([r["attn1_t1_to_t"] for r in eq], [r["phi_L1{t+1}"] for r in eq])[0, 1])
        A["cv_sum_routes"] = _cv([r["phi_L1{t}"] + r["phi_L1{t+1}"] for r in eq])
    # pre-registered predictions
    cvc = [A["cv"][f"phi_{k}"] for k in Ct.COMPLETE]
    A["predictions"] = {
        "P1 max|phi_complete - phi_behav| < 1e-5": bool(A["max_dev_complete"] < 1e-5),
        "P2 CV(phi_complete) <= 0.02": bool(max(cvc) <= 0.02),
        "P3 CV(phi_L1{t}) >= 0.5": bool(A["cv"]["phi_L1{t}"] >= 0.5),
        "P3b phi_L1{t} < 0.02 in l2_diag": bool(A["per_condition"].get("l2_diag", {}).get("phi_L1{t}", 1) < 0.02),
        "P4 CV(phi_L1{t+1}) >= 0.3": bool(A["cv"]["phi_L1{t+1}"] >= 0.3),
        "P5 corr(attn2, phi_L1{t}) > 0.8": bool(A.get("corr_attn2_vs_phi_t", 0) > 0.8),
        "P6 CV(phi_t + phi_t+1) >= 0.1": bool(A.get("cv_sum_routes", 0) >= 0.1),
    }
    A["values"] = {"max_dev_complete": A["max_dev_complete"], "cv_complete": cvc,
                   "cv_single_t": A["cv"]["phi_L1{t}"], "cv_single_t1": A["cv"]["phi_L1{t+1}"],
                   "corr_attn2_phi_t": A.get("corr_attn2_vs_phi_t"), "cv_sum_routes": A.get("cv_sum_routes")}
    return A


def analyse_factor(all_runs):
    runs = [r for r in all_runs if not r.get("control", False)]
    A = {"n": len(runs), "controls": [{k: r[k] for k in ("label", "seed", "lr", "steps", "C", "C_required", "D", "g",
                                                         "cos_theta", "kl", "kl_relevant", "converged")}
                                      for r in all_runs if r.get("control", False)]}
    cells = {f"{k[0]}|c{k[1]:g}|d{k[2]:g}": {kk: float(np.mean([r[kk] for r in rr]))
                                             for kk in ("C", "g", "D", "cos_theta", "gamma_norm", "kl", "kl_relevant",
                                                        "C_required", "ceiling", "C_over_required", "converged",
                                                        "postnorm_norm", "prenorm_norm", "sep_prenorm")}
             for k, rr in by(runs, "norm", "gain", "delta").items()}
    A["cells"] = cells
    conv = [r for r in runs if r["converged"] and r["kl"] < 0.003]
    A["n_converged"] = len(conv)
    for d in (0.4, 0.2):
        cd = [r for r in conv if r["delta"] == d]
        if cd:
            A[f"cv_C_d{d:g}"] = _cv([r["C"] for r in cd]); A[f"cv_D_d{d:g}"] = _cv([r["D"] for r in cd])
            A[f"max_rel_err_C_d{d:g}"] = float(max(abs(r["C"] / r["C_required"] - 1) for r in cd))
    c_crit = {d: Fz.required_gap(d) / (4 * np.sqrt(D_MODEL)) for d in (0.4, 0.2)}
    A["c_crit"] = c_crit
    A["slopes"] = {}
    for norm in NORMS:
        sel = [r for r in conv if r["norm"] == norm and r["delta"] == 0.4]
        if len(sel) > 2:
            A["slopes"][f"{norm}_logD_vs_logc"] = _slope([r["gain"] for r in sel], [r["D"] for r in sel])
            A["slopes"][f"{norm}_logcos_vs_logc"] = _slope([r["gain"] for r in sel], [abs(r["cos_theta"]) for r in sel])
            if norm == "learn":
                A["slopes"]["learn_loggamma_vs_logc"] = _slope([r["gain"] for r in sel if r["gain"] <= 0.3],
                                                               [r["gamma_norm"] for r in sel if r["gain"] <= 0.3])
    frac = lambda sel: float(np.mean([r["converged"] for r in sel])) if sel else float("nan")   # noqa: E731
    pick = lambda norm, d, lo, hi: [r for r in runs if r["norm"] == norm and r["delta"] == d and lo <= r["gain"] <= hi]
    A["boundary"] = {
        "frozen_d0.4_below": frac(pick("frozen", 0.4, 0, 0.1373)), "frozen_d0.4_above": frac(pick("frozen", 0.4, 0.2, 9)),
        "none_d0.4_below": frac(pick("none", 0.4, 0, 0.1373)), "learn_d0.4_below": frac(pick("learn", 0.4, 0, 0.1373)),
        "frozen_d0.2_below": frac(pick("frozen", 0.2, 0, 0.04)), "frozen_d0.2_above": frac(pick("frozen", 0.2, 0.07, 9)),
        "none_d0.2_below": frac(pick("none", 0.2, 0, 0.04)),
    }
    below = [r for r in pick("frozen", 0.4, 0, 0.1373) if not r["converged"]]
    A["ceiling_tracking"] = float(np.mean([abs(r["C"] / r["ceiling"] - 1) for r in below])) if below else float("nan")
    # the formal ceiling assumes antipodal readout rows (g = 2c) and antipodal states (D = 2 sqrt d);
    # record what training actually attains, and where the frozen models actually stop converging
    A["attained"] = {}; A["boundary_observed"] = {}
    for d_ in (0.4, 0.2):
        near = [r for r in runs if r["norm"] == "frozen" and r["delta"] == d_]
        # the attained ceiling must be read off the models that are AT it: frozen models that
        # missed the floor with alignment saturated (cos ~ 1) have spent every factor they can.
        capped = [r for r in near if not r["converged"] and abs(r["cos_theta"]) > 0.99]
        if capped:
            # kappa = (g/c) * D is the attained ceiling slope; it is only meaningful where it is
            # flat, so report its spread and estimate it from the two gains next to the boundary
            # (small gains include models that are degenerate rather than capped).
            gains = sorted({r["gain"] for r in capped})
            kap = {g: float(np.mean([r["g"] / r["gain"] for r in capped if r["gain"] == g])
                            * np.mean([r["D"] for r in capped if r["gain"] == g])) for g in gains}
            edge = gains[-2:]
            gc = float(np.mean([r["g"] / r["gain"] for r in capped if r["gain"] in edge]))
            Dh = float(np.mean([r["D"] for r in capped if r["gain"] in edge]))
            req = Fz.required_gap(d_)
            A["attained"][f"d{d_:g}"] = {"g_over_c": gc, "D": Dh, "kappa": gc * Dh, "n": len(capped),
                                         "kappa_by_gain": kap, "kappa_spread": max(kap.values()) / min(kap.values()),
                                         "frac_of_formal": gc * Dh / (4 * np.sqrt(D_MODEL)),
                                         "c_attained": req / (gc * Dh)}
        if near:
            gains = sorted({r["gain"] for r in near})
            ok = [g for g in gains if all(r["converged"] for r in near if r["gain"] == g)]
            A["boundary_observed"][f"frozen_d{d_:g}"] = float(min(ok)) if ok else float("nan")
    maxD_frozen = max((r["D"] for r in runs if r["norm"] == "frozen"), default=float("nan"))
    A["max_D_frozen"] = float(maxD_frozen)
    sel04 = lambda norm: [r for r in conv if r["norm"] == norm and r["delta"] == 0.4]   # noqa: E731
    cos_frozen = sorted(((r["gain"], abs(r["cos_theta"])) for r in sel04("frozen")))
    A["cos_frozen_by_gain"] = [[g, float(np.mean([c for gg, c in cos_frozen if gg == g]))] for g in sorted({g for g, _ in cos_frozen})]
    A["predictions"] = {
        "P7 |C/C*-1| <= 0.03 and CV(C)<=0.03 < CV(D)": bool(A.get("max_rel_err_C_d0.4", 1) <= 0.03 and A.get("cv_C_d0.4", 1) <= 0.03 and A.get("cv_D_d0.4", 0) >= 0.5),
        "P8 none: slope logD ~ -1 +- 0.15": bool(abs(A["slopes"].get("none_logD_vs_logc", 0) + 1) <= 0.15),
        "P9a frozen: D <= 2*sqrt(d) = 16": bool(A["max_D_frozen"] <= 2 * np.sqrt(D_MODEL) + 1e-6),
        "P9b frozen: slope logD >= -0.5": bool(A["slopes"].get("frozen_logD_vs_logc", -9) >= -0.5),
        "P9c frozen: cos rises as c falls (slope < 0), >= 0.9 at smallest converged c":
            bool(A["slopes"].get("frozen_logcos_vs_logc", 1) < 0 and (A["cos_frozen_by_gain"][0][1] >= 0.9 if A["cos_frozen_by_gain"] else False)),
        "P10 frozen fails below c_crit(0.4), converges above; none/learn converge below":
            bool(A["boundary"]["frozen_d0.4_below"] == 0 and A["boundary"]["frozen_d0.4_above"] == 1
                 and A["boundary"]["none_d0.4_below"] == 1 and A["boundary"]["learn_d0.4_below"] == 1),
        "P10b ceiling tracking within 25%": bool(A["ceiling_tracking"] <= 0.25),
        "P11 boundary moves with delta (frozen d0.2: fail <=0.04, pass >=0.07)":
            bool(A["boundary"]["frozen_d0.2_below"] == 0 and A["boundary"]["frozen_d0.2_above"] == 1),
        "P12 learn: slope log||gamma|| ~ -1 +- 0.3 for c <= 0.3": bool(abs(A["slopes"].get("learn_loggamma_vs_logc", 0) + 1) <= 0.3),
    }
    return A


# ---------------------------------------------------------------- tables

def write_tables(R, out):
    L = ["# TASK10 tables: implementation freedom (transformer, GPU)\n"]
    if "routing" in R:
        A = R["routing"]; keys = ["phi_behav"] + [f"phi_{k}" for k in Ct.CUT_SETS]
        L += ["## Experiment 1 — forced routing", "",
              f"{A['n_equivalent']}/{A['n']} models within {A['kl_tol']} nats of the Bayes floor. "
              f"Complete cuts marked *.", "",
              "| condition | " + " | ".join(k.replace("phi_", "") + ("*" if k.replace("phi_", "") in Ct.COMPLETE else "") for k in keys)
              + " | attn1 t+1->t | attn2 t+1->t | KL |", "|" + "---|" * (len(keys) + 4)]
        for lab, v in A["per_condition"].items():
            L.append(f"| {lab} | " + " | ".join(f"{v[k]:.4f}" for k in keys)
                     + f" | {v['attn1_t1_to_t']:.3f} | {v['attn2_t1_to_t']:.3f} | {v['kl']:.4f} |")
        L += ["", "| measure | value |", "|---|---|",
              f"| CV across equivalent models, complete cuts | " + ", ".join(f"{A['cv'][f'phi_{k}']:.4f}" for k in Ct.COMPLETE) + " |",
              f"| CV, single node L1{{t}} | {A['cv']['phi_L1{t}']:.3f} |",
              f"| CV, single node L1{{t+1}} | {A['cv']['phi_L1{t+1}']:.3f} |",
              f"| CV, incomplete pair L1{{t,t+2}} | {A['cv']['phi_L1{t,t+2}']:.3f} |",
              f"| CV, sum of the two single-node effects | {A.get('cv_sum_routes', float('nan')):.3f} |",
              f"| max deviation from behaviour, complete cuts | {A['max_dev_complete']:.2e} |",
              f"| max deviation from behaviour, incomplete cuts | {A['max_dev_incomplete']:.2e} |",
              f"| corr(attn2 t+1->t, phi L1{{t}}) | {A.get('corr_attn2_vs_phi_t', float('nan')):.3f} |",
              f"| corr(attn1 t+1->t, phi L1{{t+1}}) | {A.get('corr_attn1_vs_phi_t1', float('nan')):.3f} |", ""]
        L += ["| pre-registered prediction | outcome |", "|---|---|"]
        L += [f"| {k} | {'HELD' if v else 'FAILED'} |" for k, v in A["predictions"].items()]
        L += ["", "excluded: " + (", ".join(A["excluded"]) or "none"), ""]
    if "factor" in R:
        A = R["factor"]
        L += ["## Experiment 2 — factorisation of C = g·D·cos θ", "",
              f"c_crit(0.4) = {A['c_crit'][0.4]:.4f}, c_crit(0.2) = {A['c_crit'][0.2]:.4f}; "
              f"{A['n_converged']}/{A['n']} models reached the floor.", "",
              "| norm | gain c | δ | C | C* | C/C* | g | D | cos θ | ‖γ‖ | ceiling 4c√d | KL | KL(rel) | converged |",
              "|" + "---|" * 14]
        for key in sorted(A["cells"], key=lambda k: (k.split("|")[2], k.split("|")[0], float(k.split("|")[1][1:]))):
            n, c, d = key.split("|"); v = A["cells"][key]
            L.append(f"| {n} | {c[1:]} | {d[1:]} | {v['C']:.3f} | {v['C_required']:.3f} | {v['C_over_required']:.3f} | {v['g']:.3f} | "
                     f"{v['D']:.3f} | {v['cos_theta']:.3f} | {v['gamma_norm']:.2f} | "
                     f"{v['ceiling']:.2f} | {v['kl']:.4f} | {v['kl_relevant']:.4f} | {v['converged']:.2f} |")
        L += ["", "| measure | value |", "|---|---|"]
        for k, v in A["slopes"].items():
            L.append(f"| slope {k} | {v:.3f} |")
        for k in ("cv_C_d0.4", "cv_D_d0.4", "max_rel_err_C_d0.4", "max_D_frozen", "ceiling_tracking"):
            if k in A:
                L.append(f"| {k} | {A[k]:.4f} |")
        for d_, v in A.get("attained", {}).items():
            L.append(f"| attained g/c, {d_} | {v['g_over_c']:.3f} |")
            L.append(f"| attained D, {d_} | {v['D']:.2f} |")
            L.append(f"| attained ceiling slope kappa (C = kappa c), {d_} | {v['kappa']:.2f} |")
            L.append(f"| kappa spread across capped gains, {d_} | {v['kappa_spread']:.2f}x |")
            L.append(f"| attained ceiling / formal ceiling, {d_} | {v['frac_of_formal']:.3f} |")
            L.append(f"| c where attained ceiling = C*, {d_} | {v['c_attained']:.3f} |")
        for k, v in A.get("boundary_observed", {}).items():
            L.append(f"| smallest gain where every seed converges, {k} | {v:.3f} |")
        if A.get("controls"):
            L += ["", "### Optimiser controls (3x learning rate, 4x steps)", "",
                  "| model | seed | lr | steps | C | C* | D | cos θ | KL | KL(rel) | converged |", "|" + "---|" * 11]
            for r in A["controls"]:
                L.append(f"| {r['label']} | {r['seed']} | {r['lr']:g} | {r['steps']} | {r['C']:.3f} | {r['C_required']:.3f} | "
                         f"{r['D']:.2f} | {r['cos_theta']:.3f} | {r['kl']:.4f} | {r['kl_relevant']:.4f} | {r['converged']:.0f} |")
        for k, v in A["boundary"].items():
            L.append(f"| converged fraction, {k} | {v:.2f} |")
        L += ["", "| pre-registered prediction | outcome |", "|---|---|"]
        L += [f"| {k} | {'HELD' if v else 'FAILED'} |" for k, v in A["predictions"].items()]
        L.append("")
    (out / "tables.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["routing", "factorization", "controls", "all"])
    ap.add_argument("--out", default="rounds/r11_implementation_freedom"); ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--seeds", type=int, default=None); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lr-control", type=float, default=3e-3)
    ap.add_argument("--long-control", type=int, default=100000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out); (out / "models").mkdir(parents=True, exist_ok=True); t0 = time.time()
    steps = 300 if args.quick else args.steps
    R = json.loads((out / "results10.json").read_text()) if (out / "results10.json").exists() else {}

    def phase(specs, kind, st, lr, tag, control=False):
        print(f"[{tag}] {len(specs)} models, {st} steps, lr={lr:g}, one stacked model on {args.device}", flush=True)
        nets, hist = train_grid(specs, st, lr, args.device, out, tag)
        runs = measure_grid(specs, nets, hist, kind, st, lr, args.device, out, args.quick, control)
        fmt = report_routing if kind == "routing" else report_factor
        for r in runs:
            print("  " + fmt(r), flush=True)
        print(f"[{tag}] done at {time.time() - t0:.0f}s", flush=True)
        return runs

    if args.exp in ("routing", "all"):
        seeds = range(args.seeds or (1 if args.quick else 4))
        runs = phase(routing_specs(seeds), "routing", steps, LR, "routing")
        R["routing_runs"] = runs; R["routing"] = analyse_routing(runs)
        (out / "results10.json").write_text(json.dumps(to_jsonable(R), indent=1))

    if args.exp in ("factorization", "all"):
        seeds = range(args.seeds or (1 if args.quick else 3))
        specs = (factor_specs(seeds, (0.1, 0.3), (0.04,)) if args.quick else factor_specs(seeds))
        runs = phase(specs, "factor", steps, LR, "factorization")
        R["factor_runs"] = runs; R["factor"] = analyse_factor(runs)
        (out / "results10.json").write_text(json.dumps(to_jsonable(R), indent=1))

    if args.exp in ("controls", "all"):
        seeds = range(args.seeds or (1 if args.quick else 2))
        lr_runs = phase(control_specs(seeds), "factor", steps, args.lr_control, "lr control", control=True)
        long_steps = 600 if args.quick else args.long_control
        long_runs = phase(control_specs(seeds, gains=(0.3, 0.4)), "factor", long_steps, LR, "long control", control=True)
        R["control_runs"] = lr_runs + long_runs
        R["factor_runs"] = R.get("factor_runs", []) + lr_runs + long_runs
        R["factor"] = analyse_factor(R["factor_runs"])

    write_tables(R, out); (out / "results10.json").write_text(json.dumps(to_jsonable(R), indent=1))
    try:
        from plots import make_figures
        make_figures(R, out)
    except Exception as e:
        print("figures skipped:", e)
    for exp in ("routing", "factor"):
        if exp in R:
            print(f"\n{exp}:")
            for k, v in R[exp]["predictions"].items():
                print(("  HELD   " if v else "  FAILED ") + k)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
