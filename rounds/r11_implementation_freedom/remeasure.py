"""Re-measure the saved TASK10 models without retraining them.

The grids are expensive to train and cheap to measure, so the measurement pass is separable:
this rebuilds `factor_runs` / `routing_runs` from `rounds/r11_implementation_freedom/models/*.pt`. Use it after a change
to the measurement code, or when a run crashed after training.

    .venv/bin/python rounds/r11_implementation_freedom/remeasure.py [--kind factor|routing|both]
"""
from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from goalgeo import cuts as Ct, factorize as Fz, hmm4 as H4, prominence as Pr, tfm as Tf
from goalgeo.analysis import to_jsonable
from run import DEFAULT_STEPS, LR, ROUTING, analyse_factor, analyse_routing, write_tables

FACTOR = re.compile(r"^factor_(?P<norm>none|frozen|learn)_c(?P<gain>[\d.]+)_d(?P<delta>[\d.]+)_s(?P<seed>\d+)"
                    r"(?:_lr(?P<lr>[\d.e-]+))?(?:_st(?P<steps>\d+))?\.pt$")
ROUTE = re.compile(r"^routing_(?P<label>[a-z0-9_]+)_s(?P<seed>\d+)\.pt$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rounds/r11_implementation_freedom"); ap.add_argument("--kind", default="both", choices=["factor", "routing", "both"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(); out = Path(args.out); t0 = time.time()
    R = json.loads((out / "results10.json").read_text()) if (out / "results10.json").exists() else {}
    kw = dict(ROUTING)

    if args.kind in ("factor", "both"):
        runs = []
        for f in sorted((out / "models").glob("factor_*.pt")):
            m_ = FACTOR.match(f.name)
            if not m_:
                print("skip", f.name); continue
            gain, delta, seed = float(m_["gain"]), float(m_["delta"]), int(m_["seed"])
            lr = float(m_["lr"]) if m_["lr"] else LR
            steps = int(m_["steps"]) if m_["steps"] else DEFAULT_STEPS
            hmm = H4.make_hmm4(1.0, delta, 1)
            net = Tf.CausalTransformer(n_vocab=H4.V, d=64, seed=seed, gain=gain, final_norm=m_["norm"])
            net.load_state_dict(torch.load(f)); net = net.to(args.device).eval()
            ev = Pr.make_eval(hmm, n=2000, T=48)
            r = {"exp": "factor", "label": f"{m_['norm']}_c{gain:g}_d{delta:g}", "seed": seed, "steps": steps, "lr": lr,
                 "control": lr != LR or steps != DEFAULT_STEPS, "norm": m_["norm"], "gain": gain, "delta": delta}
            r.update(Fz.measure(net, hmm, ev, gain)); runs.append(r)
            print(f"  {r['label']:18s} s{seed} kl={r['kl']:.4f} C={r['C']:.3f}/{r['C_required']:.3f} "
                  f"g={r['g']:.3f} D={r['D']:.2f} cos={r['cos_theta']:.3f} conv={r['converged']:.0f}", flush=True)
        R["factor_runs"] = runs; R["factor"] = analyse_factor(runs)

    if args.kind in ("routing", "both"):
        runs = []
        for f in sorted((out / "models").glob("routing_*.pt")):
            m_ = ROUTE.match(f.name)
            label, seed = m_["label"], int(m_["seed"])
            hmm = H4.make_hmm4(1.0, 0.4, 1)
            net = Tf.CausalTransformer(n_vocab=H4.V, d=64, seed=seed, attn_diag=kw[label].get("attn_diag", ()))
            net.load_state_dict(torch.load(f)); net = net.to(args.device).eval()
            ev = Pr.make_eval(hmm, n=2000, T=48)
            r = {"exp": "routing", "label": label, "seed": seed}
            r.update(Ct.measure(net, hmm, ev, n_anchor=200)); runs.append(r)
            print(f"  {label:8s} s{seed} kl={r['kl']:.4f} cut={r['phi_L1{t,t+1}']:.4f} "
                  f"t={r['phi_L1{t}']:.4f} t+1={r['phi_L1{t+1}']:.4f}", flush=True)
        R["routing_runs"] = runs; R["routing"] = analyse_routing(runs)

    write_tables(R, out); (out / "results10.json").write_text(json.dumps(to_jsonable(R), indent=1))
    from plots import make_figures
    make_figures(R, out)
    print(f"re-measured in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
