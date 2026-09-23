"""TASK4: one-step vs k-step vs sequential objectives on an HMM with one-step-equivalent beliefs.

    .venv/bin/python rounds/r04_hmm_objectives/run.py [--quick] [--out results_hmm]
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from goalgeo import hmm as Hm, seqmodels as Sm, geometry as G
from goalgeo.analysis import to_jsonable
from goalgeo.plotting import SERIES, MUTED, _save

L = 6
OBJECTIVES = [("one-step", {}), ("k-step", {"k": 2}), ("k-step", {"k": 3}), ("k-step", {"k": 2, "kind": "marginals"}),
              ("sequential", {"targets": "exact"}), ("sequential", {"targets": "sampled"})]


def oname(obj, kw):
    if obj == "one-step": return "one-step"
    if obj == "k-step": return f"{kw['k']}-step " + ("marginals" if kw.get("kind") == "marginals" else "joint")
    return f"sequential ({kw['targets']})"


def out_dim(obj, kw):
    if obj == "one-step": return 4
    if obj == "k-step": return 4 * kw["k"] if kw.get("kind") == "marginals" else 4 ** kw["k"]
    return 4


def references(m, X_seen):
    B = Hm.beliefs_of(m, X_seen)
    P1 = np.stack([Hm.predictive(m, b, 1) for b in B]); P2 = np.stack([Hm.predictive(m, b, 2) for b in B]); P3 = np.stack([Hm.predictive(m, b, 3) for b in B])
    return {"belief": B, "P1": P1, "P2": P2, "P3": P3}


def analyse(h, X, refs, sub):
    """h: [N, H] hidden states; X: [N, 12] full histories; refs on the seen tokens."""
    last, prev = X[:, -1], X[:, -2]
    gA, gB = last == Hm.TOK["p"], last == Hm.TOK["q"]
    gC, gD = (prev == Hm.TOK["p"]) & (last >= 2), (prev == Hm.TOK["q"]) & (last >= 2)
    def cross(a, b):
        return float(np.mean(np.linalg.norm(h[a][:, None] - h[b][None], axis=-1)))
    def within(a):
        R = G.rdm(h[a][:200]); return float(G.upper(R).mean())
    med = float(np.median(G.upper(G.rdm(h[sub]))))
    res = {"d_AB": cross(gA, gB) / med, "d_CD": cross(gC, gD) / med, "within_A": within(gA) / med, "within_C": within(gC) / med,
           "ratio_AB_over_CD": cross(gA, gB) / cross(gC, gD), "n_A": int(gA.sum()), "n_C": int(gC.sum())}
    Rh = G.rdm(h[sub])
    for k, v in refs.items():
        res[f"rsa_{k}"] = G.rsa(Rh, G.rdm(v[sub]))
    res["decode_belief_r2"] = G.ridge_cv_r2(h, refs["belief"], alpha=1.0)
    res["decode_P2_r2"] = G.ridge_cv_r2(h, refs["P2"], alpha=1.0)
    cls = np.unique(np.round(refs["P1"], 2), axis=0, return_inverse=True)[1].ravel()
    wb = G.rdm(h[sub]); iu = np.triu_indices(len(sub), 1); same = cls[sub][iu[0]] == cls[sub][iu[1]]
    res["within_over_between_onestep_classes"] = float(wb[iu][same].mean() / wb[iu][~same].mean())
    res["n_onestep_classes"] = int(cls.max() + 1)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default="rounds/r04_hmm_objectives")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(exist_ok=True); t0 = time.time()
    dev = torch.device(args.device); print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")
    seeds = [0, 1, 2] if not args.quick else [0, 1]; steps = 1500 if not args.quick else 200
    R = {}
    for kind in ("clean", "noisy"):
        m = Hm.make_hmm(kind)
        X_eval, _ = Hm.sample(m, n=3000 if not args.quick else 600, T=12, seed=123)
        rng = np.random.default_rng(0); sub = rng.choice(len(X_eval), min(800, len(X_eval)), replace=False)
        refs_win = references(m, X_eval[:, -L:]); refs_full = references(m, X_eval)
        R[kind] = {}
        for arch in ("gru", "mlp"):
            for obj, kw in OBJECTIVES:
                if arch == "mlp" and (obj == "sequential" or kw.get("k") == 3 or kw.get("kind") == "marginals"):
                    continue
                name = f"{arch}/{oname(obj, kw)}"; R[kind][name] = []
                for seed in seeds:
                    net = (Sm.SeqNet(hidden=64, emb=16, out_dim=out_dim(obj, kw), seed=seed) if arch == "gru"
                           else Sm.WindowMLP(L=L, hidden=64, out_dim=out_dim(obj, kw), seed=seed)).to(dev)
                    hist = Sm.train_objective(net, m, obj, steps=steps, seed=seed, L=L, **kw)
                    X_seen = X_eval if (obj == "sequential") else X_eval[:, -L:]
                    h = net.hidden(torch.as_tensor(X_seen)).numpy()
                    r = analyse(h, X_eval, refs_full if obj == "sequential" else refs_win, sub); r["final_loss"] = float(np.mean(hist[-50:]))
                    # random-init control
                    R[kind][name].append(r)
                rr = R[kind][name]
                print(f"[{kind}] {name:26s} d(AB)/d(CD)={np.mean([r['ratio_AB_over_CD'] for r in rr]):.2f}  d_AB/med={np.mean([r['d_AB'] for r in rr]):.2f} d_CD/med={np.mean([r['d_CD'] for r in rr]):.2f} "
                      f"RSA belief/P1/P2/P3={np.mean([r['rsa_belief'] for r in rr]):.2f}/{np.mean([r['rsa_P1'] for r in rr]):.2f}/{np.mean([r['rsa_P2'] for r in rr]):.2f}/{np.mean([r['rsa_P3'] for r in rr]):.2f} "
                      f"decode belief R²={np.mean([r['decode_belief_r2'] for r in rr]):.2f} w/b 1-step classes={np.mean([r['within_over_between_onestep_classes'] for r in rr]):.2f}  {time.time()-t0:.0f}s")
            # random-init reference (GRU)
        for name, mk, seen, rf in (("gru/random init", lambda s: Sm.SeqNet(hidden=64, emb=16, out_dim=4, seed=100 + s), X_eval, refs_full),
                                   ("mlp/random init", lambda s: Sm.WindowMLP(L=L, hidden=64, out_dim=4, seed=100 + s), X_eval[:, -L:], refs_win)):
            R[kind][name] = []
            for seed in seeds:
                h = mk(seed).to(dev).hidden(torch.as_tensor(seen)).numpy()
                R[kind][name].append(analyse(h, X_eval, rf, sub))
        rr = R[kind]["gru/random init"]; name = "gru/random init"
        print(f"[{kind}] {name:26s} d(AB)/d(CD)={np.mean([r['ratio_AB_over_CD'] for r in rr]):.2f}  d_AB/med={np.mean([r['d_AB'] for r in rr]):.2f} d_CD/med={np.mean([r['d_CD'] for r in rr]):.2f} RSA belief/P1/P2/P3={np.mean([r['rsa_belief'] for r in rr]):.2f}/{np.mean([r['rsa_P1'] for r in rr]):.2f}/{np.mean([r['rsa_P2'] for r in rr]):.2f}/{np.mean([r['rsa_P3'] for r in rr]):.2f}")
        # reference geometry facts
        R[kind]["_refs"] = {"rsa_P1_P2": G.rsa(G.rdm(refs_win["P1"][sub]), G.rdm(refs_win["P2"][sub])), "rsa_belief_P2": G.rsa(G.rdm(refs_win["belief"][sub]), G.rdm(refs_win["P2"][sub])),
                            "rsa_belief_P1": G.rsa(G.rdm(refs_win["belief"][sub]), G.rdm(refs_win["P1"][sub]))}
    # ---- figure
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(12, 3.6), sharey=True)
    for ax, kind in zip(axes, ("clean", "noisy")):
        names = [n for n in R[kind] if not n.startswith("_")]
        x = np.arange(len(names)); w = 0.38
        ax.bar(x - w / 2, [np.mean([r["d_AB"] for r in R[kind][n]]) for n in names], w, color=SERIES[0], edgecolor="white", label="b₁ vs b₂ (one-step equal, two-step differ)")
        ax.errorbar(x - w / 2, [np.mean([r["d_AB"] for r in R[kind][n]]) for n in names], [np.std([r["d_AB"] for r in R[kind][n]]) for n in names], fmt="none", ecolor=MUTED, capsize=2)
        ax.bar(x + w / 2, [np.mean([r["d_CD"] for r in R[kind][n]]) for n in names], w, color=SERIES[1], edgecolor="white", label="control: C vs D (one-step differ)")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5); ax.set_title(f"{kind} HMM: hidden distance / median"); ax.grid(True, axis="y")
    axes[0].legend(fontsize=7.5)
    _save(fig, out / "hmm_key_pair.png")
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(12, 3.6), sharey=True)
    for ax, kind in zip(axes, ("clean", "noisy")):
        names = [n for n in R[kind] if not n.startswith("_")]
        x = np.arange(len(names)); w = 0.2
        for k, (key, lab, col) in enumerate([("rsa_belief", "belief", SERIES[4]), ("rsa_P1", "1-step predictive", SERIES[0]), ("rsa_P2", "2-step predictive", SERIES[2]), ("rsa_P3", "3-step predictive", SERIES[3])]):
            ax.bar(x + (k - 1.5) * w, [np.mean([r[key] for r in R[kind][n]]) for n in names], w, color=col, edgecolor="white", label=lab)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.5); ax.set_title(f"{kind} HMM: RSA of hidden geometry"); ax.grid(True, axis="y"); ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].legend(fontsize=7.5)
    _save(fig, out / "hmm_rsa.png")
    # ---- tables
    lines = ["# TASK4 tables: HMM objectives\n"]
    for kind in ("clean", "noisy"):
        rf = R[kind]["_refs"]
        lines += [f"## {kind} HMM", f"- reference RSA: P1~P2 {rf['rsa_P1_P2']:.2f}, belief~P1 {rf['rsa_belief_P1']:.2f}, belief~P2 {rf['rsa_belief_P2']:.2f}", "",
                  "| model / objective | d(b1,b2)/median | d(C,D)/median | d(b1,b2)/d(C,D) | within b1 / median | RSA belief | RSA P1 | RSA P2 | RSA P3 | decode belief R² | decode P2 R² | within/between one-step classes | final loss |", "|---|" + "---|" * 12]
        for n, rr in R[kind].items():
            if n.startswith("_"): continue
            g = lambda k: float(np.mean([r[k] for r in rr]))
            lines.append(f"| {n} | {g('d_AB'):.2f} | {g('d_CD'):.2f} | {g('ratio_AB_over_CD'):.2f} | {g('within_A'):.2f} | {g('rsa_belief'):+.2f} | {g('rsa_P1'):+.2f} | {g('rsa_P2'):+.2f} | {g('rsa_P3'):+.2f} | {g('decode_belief_r2'):.2f} | {g('decode_P2_r2'):.2f} | {g('within_over_between_onestep_classes'):.2f} | {g('final_loss') if 'final_loss' in rr[0] else float('nan'):.3f} |")
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    Path(out / "results_hmm.json").write_text(json.dumps(to_jsonable(R), indent=1))
    print(f"done in {time.time()-t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
