"""TASK8: intervention equivalence, on/off-manifold local metrics and propagation depth on the
functionally equivalent models saved by TASK7 (plus six new delay-4 models).

    .venv/bin/python rounds/r09_interventions/run.py [--quick] [--jobs 6] [--out results8]
"""
from __future__ import annotations

import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from goalgeo import hmm4 as H4, seqmodels as Sm, prominence as Pr, invariants as Iv, steering as St, geometry as G
from goalgeo.analysis import to_jsonable

MODELS7 = Path("rounds/r08_invariants/models")
ALPHAS = np.linspace(0, 1.5, 16)
SCALES = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5]
LR = 3e-3


def load7(label, seed):
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed)
    net.load_state_dict(torch.load(MODELS7 / f"{label}_s{seed}.pt")); return net.eval()


def setup(m, K, n_anchor, seed=0, n=2000):
    ev = Pr.make_eval(m, n=n, T=48)
    P, anchors = Pr.positions(m, ev["Z"]); Z = ev["Z"]; T = ev["X"].shape[1]
    anchors = anchors[anchors[:, 1] + K <= T - 1]
    aA = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["P"]]; aB = anchors[Z[anchors[:, 0], anchors[:, 1]] == m.S["Q"]]
    rng = np.random.default_rng(seed); aA = aA[rng.choice(len(aA), min(n_anchor, len(aA)), replace=False)]; jB = aB[rng.integers(0, len(aB), len(aA))]
    win = np.stack([ev["X"][i, t + 1:t + 1 + K] for i, t in aA])
    sub = rng.choice(ev["X"].size, 800, replace=False)
    return {"ev": ev, "P": P, "aA": aA, "jB": jB, "win": win, "sub": sub}


def states(net, S):
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(S["ev"]["X"])).numpy().astype(np.float64)
    P = S["P"]; v = Hs[P["cueB"]].mean(0) - Hs[P["cueA"]].mean(0)
    flat = Hs.reshape(-1, Hs.shape[-1])
    return {"hA": Hs[S["aA"][:, 0], S["aA"][:, 1]], "hB": Hs[S["jB"][:, 0], S["jB"][:, 1]], "v": v, "H_sub": flat[S["sub"]],
            "median": float(np.median(G.upper(G.rdm(flat[S["sub"][:400]]))))}


# ---------------- A: intervention equivalence ----------------
def steering_pair(net1, net2, S, name1, name2):
    s1, s2 = states(net1, S), states(net2, S); win = S["win"][:, :1]
    out = {"pair": f"{name1} | {name2}", "norm_v": [float(np.linalg.norm(s1["v"])), float(np.linalg.norm(s2["v"]))],
           "norm_v_over_median": [float(np.linalg.norm(s1["v"]) / s1["median"]), float(np.linalg.norm(s2["v"]) / s2["median"])]}
    B, r2 = St.align_states(s1["H_sub"], s2["H_sub"]); out["align_r2"] = r2; out["cos_v1_mapped_v2"] = St.cosine(s1["v"] @ B, s2["v"])
    curves = {}
    for tag, (net, s) in (("1", (net1, s1)), ("2", (net2, s2))):
        g = St.gradient_direction(net, s["hA"], win, H4.TOK["y"], H4.TOK["x"], step=1)
        g = g / np.linalg.norm(g, axis=1, keepdims=True) * np.linalg.norm(s["v"])
        out[f"cos_grad_vs_meandiff_{tag}"] = float(np.mean([St.cosine(gi, s["v"]) for gi in g]))
        for kind, v in (("meandiff", s["v"]), ("gradient", g)):
            js, Pc, p0 = St.effect_curve(net, s["hA"], win, v, ALPHAS, step=1); curves[(tag, kind)] = (js, Pc, p0, net, s, v)
    out["curves"] = {f"{t}_{k}": c[0].tolist() for (t, k), c in curves.items()}
    p01, p02 = curves[("1", "meandiff")][2], curves[("2", "meandiff")][2]
    out["cross_js_alpha0"] = float(Iv.js_divergence(p01, p02).mean())
    out["matched"] = {}
    for kind in ("meandiff", "gradient"):
        js1, P1, _, _, _, _ = curves[("1", kind)]; js2, P2, _, net2_, s2_, v2 = curves[("2", kind)]
        rows = []
        for a1 in (0.25, 0.5, 0.75, 1.0):
            i1 = int(np.argmin(np.abs(ALPHAS - a1))); a2 = St.match_alpha(js1[i1], ALPHAS, js2)
            _, P2m, _ = St.effect_curve(net2_, s2_["hA"], win, v2, [a2], step=1)
            rows.append({"alpha1": a1, "alpha2": a2, "effect1": float(js1[i1]), "effect2_matched": float(Iv.js_divergence(P2m[0], curves[("2", kind)][2]).mean()),
                         "cross_js_matched": float(Iv.js_divergence(P1[i1], P2m[0]).mean()), "cross_js_same_alpha": float(Iv.js_divergence(P1[i1], P2[i1]).mean())})
        out["matched"][kind] = rows
    # steering-vector geometry seen through the same downstream test: cross-model JS at alpha=1 for both kinds
    return out


# ---------------- B: on / off manifold ----------------
def local_metrics(net, S, name, seed=0):
    s = states(net, S); win = S["win"][:, :1]; rng = np.random.default_rng(seed)
    H = s["H_sub"]; Vt = np.linalg.svd(H - H.mean(0), full_matrices=False)[2]
    dn = float(np.linalg.norm(s["v"]))
    dirs = {"meandiff": [s["v"] / dn], "top_pc": [Vt[i] for i in range(3)], "bottom_pc": [Vt[-1 - i] for i in range(3)],
            "random": [u / np.linalg.norm(u) for u in rng.standard_normal((5, H.shape[1]))]}
    _, p0 = St.propagate(net, s["hA"], win); p0 = p0[:, 1]
    res = {"name": name, "dh_norm": dn, "finite": {}, "linear": {}}
    for kind, us in dirs.items():
        fin = np.zeros(len(SCALES)); lin = np.zeros(len(SCALES))
        for u in us:
            dz = St.directional_logit_derivative(net, s["hA"], win, u)[:, 1]      # per unit step along u
            for j, sc in enumerate(SCALES):
                for sign in (1, -1):
                    _, p = St.propagate(net, s["hA"] + sign * sc * dn * u, win)
                    fin[j] += Iv.js_divergence(p[:, 1], p0).mean() / (2 * len(us))
                    lin[j] += St.js_linear_prediction(sign * sc * dn * dz, p0).mean() / (2 * len(us))
        res["finite"][kind] = fin.tolist(); res["linear"][kind] = lin.tolist()
    return res


# ---------------- C: propagation depth ----------------
def depth(net, S, name):
    s = states(net, S); d = St.depth_curves(net, s["hA"], s["hB"], S["win"])
    return {"name": name, **{k: v.tolist() for k, v in d.items()}}


def train_k4(args):
    lr_ratio, seed, out, quick = args
    torch.set_num_threads(1); m = H4.make_hmm4(1.0, 0.4, 4)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=seed)
    Sm.train_weighted(net, m, steps=3000 if not quick else 200, seed=seed, checkpoints=(0,), lr=LR, lr_out=LR * lr_ratio)
    ev = Pr.make_eval(m, n=500, T=48)
    with torch.no_grad():
        logp = torch.log_softmax(net.forward_all(torch.as_tensor(ev["X"])), -1)[:, :-1].numpy()
    kl = float((ev["Y"][:, :-1] * (np.log(ev["Y"][:, :-1] + 1e-12) - logp)).sum(-1).mean())
    torch.save(net.state_dict(), out / "models" / f"k4_lr{lr_ratio:g}_s{seed}.pt"); return {"lr_ratio": lr_ratio, "seed": seed, "kl": kl}


def cv_table(rows, key, n):
    """CV across models at each index of rows[i][key]."""
    A = np.array([r[key] for r in rows]); return (A.std(0) / (np.abs(A.mean(0)) + 1e-12)).tolist()[:n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="rounds/r09_interventions"); ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args(); out = Path(args.out); (out / "models").mkdir(parents=True, exist_ok=True); t0 = time.time()
    seeds = [0, 1, 2] if not args.quick else [0]; n_anchor = 200 if not args.quick else 40
    # k=4 models (background) ------------------------------------------------------------
    todo = [(l, s, out, args.quick) for l in (0.1, 10.0) for s in seeds]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        fut = ex.map(train_k4, todo)
        # A + B + C on k=1 models meanwhile ----------------------------------------------
        m1 = H4.make_hmm4(1.0, 0.4, 1); S1 = setup(m1, K=4, n_anchor=n_anchor, n=2000 if not args.quick else 400)
        R = {"A": [], "B": [], "C_k1": [], "C_k4": []}
        for s in seeds:
            R["A"].append(steering_pair(load7("lr0.1", s), load7("lr10", s), S1, f"lr0.1_s{s}", f"lr10_s{s}"))
            print(f"A pair {s}: cross JS a=0 {R['A'][-1]['cross_js_alpha0']:.4f}  matched {[round(r['cross_js_matched'], 4) for r in R['A'][-1]['matched']['meandiff']]}  "
                  f"same-alpha {[round(r['cross_js_same_alpha'], 4) for r in R['A'][-1]['matched']['meandiff']]}  cos(v1->v2) {R['A'][-1]['cos_v1_mapped_v2']:.2f}  {time.time() - t0:.0f}s", flush=True)
        R["A"].append(steering_pair(load7("base", 0), load7("base", 1), S1, "base_s0", "base_s1"))
        for label in ("lr0.1", "base", "lr10"):
            for s in seeds:
                net = load7(label, s); name = f"{label}_s{s}"
                R["B"].append(local_metrics(net, S1, name)); R["C_k1"].append(depth(net, S1, name))
                print(f"B/C {name}: lin err meandiff s=1 {abs(R['B'][-1]['finite']['meandiff'][4] - R['B'][-1]['linear']['meandiff'][4]) / R['B'][-1]['finite']['meandiff'][4]:.2f}  "
                      f"depth JS {[round(x, 3) for x in R['C_k1'][-1]['js']]}  {time.time() - t0:.0f}s", flush=True)
        R["k4_train"] = list(fut)
    m4 = H4.make_hmm4(1.0, 0.4, 4); S4 = setup(m4, K=8, n_anchor=n_anchor, n=2000 if not args.quick else 400)
    for l in (0.1, 10.0):
        for s in seeds:
            net = Sm.SeqNet(n_vocab=H4.V, hidden=64, emb=16, out_dim=H4.V, seed=s); net.load_state_dict(torch.load(out / "models" / f"k4_lr{l:g}_s{s}.pt")); net.eval()
            R["C_k4"].append({"lr_ratio": l, **depth(net, S4, f"k4_lr{l:g}_s{s}")})
            print(f"C k=4 lr{l:g} s{s}: JS {[round(x, 3) for x in R['C_k4'][-1]['js']]} raw {[round(x, 2) for x in R['C_k4'][-1]['raw']]}", flush=True)
    # summaries ------------------------------------------------------------------------
    R["B_summary"] = {kind: {"cv_finite": cv_table([{"f": r["finite"][kind]} for r in R["B"]], "f", len(SCALES)), "cv_linear": cv_table([{"f": r["linear"][kind]} for r in R["B"]], "f", len(SCALES)),
                             "lin_err": np.mean([[abs(f - l) / max(f, 1e-12) for f, l in zip(r["finite"][kind], r["linear"][kind])] for r in R["B"]], 0).tolist()} for kind in ("meandiff", "top_pc", "bottom_pc", "random")}
    R["C_summary"] = {"k1": {k: cv_table(R["C_k1"], k, 5) for k in ("js", "raw", "lin", "finite_logit")}, "k4": {k: cv_table(R["C_k4"], k, 9) for k in ("js", "raw", "lin", "finite_logit")}}
    write_tables(R, out)
    (out / "results8.json").write_text(json.dumps(to_jsonable(R), indent=1))
    from plots import make_figures
    make_figures(R, out)
    print(f"done in {time.time() - t0:.0f}s -> {out}/")


def write_tables(R, out):
    L = ["# TASK8 tables: intervention equivalence, on/off-manifold local metrics, propagation depth\n", "## A. Intervention equivalence (steering at the cue, effect read one step later)", ""]
    L += ["| pair | ‖v₁‖ | ‖v₂‖ | ‖v‖/median (1, 2) | alignment R² | cos(v₁→v₂) | cos(grad, meandiff) (1, 2) | cross JS at α=0 |", "|---|---|---|---|---|---|---|---|"]
    for a in R["A"]:
        L.append(f"| {a['pair']} | {a['norm_v'][0]:.2f} | {a['norm_v'][1]:.2f} | {a['norm_v_over_median'][0]:.2f}, {a['norm_v_over_median'][1]:.2f} | {a['align_r2']:.3f} | {a['cos_v1_mapped_v2']:.3f} | {a['cos_grad_vs_meandiff_1']:.2f}, {a['cos_grad_vs_meandiff_2']:.2f} | {a['cross_js_alpha0']:.5f} |")
    for kind in ("meandiff", "gradient"):
        L += ["", f"### {kind} steering: matched-effect cross-model divergence", "", "| pair | α₁ | α₂ (matched) | effect₁ | effect₂ | cross JS matched | cross JS same α |", "|---|---|---|---|---|---|---|"]
        for a in R["A"]:
            for r in a["matched"][kind]:
                L.append(f"| {a['pair']} | {r['alpha1']:.2f} | {r['alpha2']:.2f} | {r['effect1']:.4f} | {r['effect2_matched']:.4f} | {r['cross_js_matched']:.5f} | {r['cross_js_same_alpha']:.5f} |")
    L += ["", "## B. Local metrics on and off the manifold (9 models: lr ×0.1, ×1, ×10 × 3 seeds; effect at one step)", "",
          "| direction | " + " | ".join(f"s={s}" for s in SCALES) + " |", "|" + "---|" * (1 + len(SCALES))]
    for kind, d in R["B_summary"].items():
        L.append(f"| {kind}: linearisation error | " + " | ".join(f"{x:.2f}" for x in d["lin_err"]) + " |")
        L.append(f"| {kind}: CV of finite effect across models | " + " | ".join(f"{x:.2f}" for x in d["cv_finite"]) + " |")
        L.append(f"| {kind}: CV of linear prediction across models | " + " | ".join(f"{x:.2f}" for x in d["cv_linear"]) + " |")
    L += ["", "scale s is in units of each model's own branch-mean separation ‖Δh‖; effects averaged over ±direction.", ""]
    for tag, rows, K in (("k=1 HMM (9 models)", R["C_k1"], 4), ("k=4 HMM (6 models)", R["C_k4"], 8)):
        L += [f"## C. Propagation depth, {tag}", "", "| model | quantity | " + " | ".join(f"k={k}" for k in range(K + 1)) + " |", "|" + "---|" * (K + 3)]
        for r in rows:
            for q in ("raw", "lin", "finite_logit", "js"):
                L.append(f"| {r['name']} | {q} | " + " | ".join(f"{x:.3f}" for x in r[q][:K + 1]) + " |")
        key = "k1" if K == 4 else "k4"
        for q in ("raw", "lin", "finite_logit", "js"):
            L.append(f"| **CV across models** | {q} | " + " | ".join(f"{x:.2f}" for x in R["C_summary"][key][q]) + " |")
        L.append("")
    L += ["k=4 training: " + ", ".join(f"lr{r['lr_ratio']:g}_s{r['seed']} kl={r['kl']:.4f}" for r in R["k4_train"]), ""]
    (out / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
