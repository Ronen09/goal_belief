"""TASK11: tables, the pre-registered prediction checks and figures from rounds/r12_hidden_goal/results11.json.

    .venv/bin/python rounds/r12_hidden_goal/tables.py [--out results11]
"""

from __future__ import annotations

import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OBJ = ("goal", "act_soft", "act_hard", "next_obs")
KINDS = ("iid", "channel")
ARCHS = ("tfm", "gru")
IFACE = {"tfm": "u", "gru": "h"}
TSITES = ("res0", "res1", "res2", "u")


class R:
    def __init__(self, path):
        d = json.load(open(path)); self.runs = d["runs"]; self.base = d["baselines"]

    def sel(self, arch, kind, K=4, obj=None, netho=False, conv=True):
        out = [r for r in self.runs if r["arch"] == arch and r["kind"] == kind and r["K"] == K
               and (obj is None or r["objective"] == obj) and r["netho"] == netho]
        if conv and obj != "init":
            out = [r for r in out if r.get("converged", True)]
        return out

    def vals(self, arch, kind, obj, key, site=None, K=4, netho=False):
        site = site or IFACE[arch]
        return np.array([r["sites"][site][key] for r in self.sel(arch, kind, K, obj, netho)])

    def m(self, *a, **k):
        v = self.vals(*a, **k); return float(v.mean()) if len(v) else float("nan")

    def best_site(self, arch, kind, obj, key="IID_r2y", K=4):
        sites = ("res1", "res2", "u") if arch == "tfm" else ("h",)
        return max(sites, key=lambda s: self.m(arch, kind, obj, key, site=s, K=K))


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


# ---------------------------------------------------------------- tables
def tables(Rs: R) -> str:
    L = ["# TASK11 tables (round 12: hidden goal)", "",
         "Seed means over converged models; K = 4 unless stated. Probe: unregularised least squares on [h, 1].",
         "R²_y: log-odds; R²_b: posterior probabilities; G_count = 1 − SSE(probe)/SSE(best affine-in-counts predictor).", ""]

    L += ["## 1. Convergence", "",
          "| arch | env | K | objective | pool | converged | KL to target (nats) | argmax agrees with optimal action |",
          "|---|---|---|---|---|---|---|---|"]
    cells = sorted({(r["arch"], r["kind"], r["K"], r["objective"], r["netho"]) for r in Rs.runs if r["objective"] != "init"})
    for a, k, K, o, ho in cells:
        rs = Rs.sel(a, k, K, o, ho, conv=False)
        kl = np.mean([r["behaviour"]["kl"] for r in rs])
        ag = [r["behaviour"].get("act_agree") for r in rs if "act_agree" in r["behaviour"]]
        L.append(f"| {a} | {k} | {K} | {o} | {'no-conflict' if ho else 'full'} | {sum(r['converged'] for r in rs)}/{len(rs)} "
                 f"| {f(kl, 4)} | {f(np.mean(ag), 4) if ag else '—'} |")

    L += ["", "## 2. The readout interface (tfm: post-final-LN u; GRU: h)", "",
          "| arch | env | objective | IID R²_y | IID R²_b | EXT R²_y | EXT R²_b | CONF R²_y | TIME R²_y | IID RMSE (nats) | probe KL (nats) | G_count | action acc |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for k in KINDS:
            for o in OBJ + ("init",):
                g = lambda key: Rs.m(a, k, o, key)       # noqa: E731
                L.append(f"| {a} | {k} | {o} | {f(g('IID_r2y'))} | {f(g('IID_r2b'))} | {f(g('EXT_r2y'))} | {f(g('EXT_r2b'))} "
                         f"| {f(g('CONF_r2y'))} | {f(g('TIME_r2y'))} | {f(g('IID_rmse'))} | {f(g('IID_kl'), 4)} "
                         f"| {f(g('G_count')) if k == 'channel' else '—'} | {f(g('action_acc'))} |")
    L += ["", "Baselines (features instead of a network):", "",
          "| env | features | IID R²_y | IID R²_b | EXT R²_y | CONF R²_y | TIME R²_y | action acc |", "|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for name, s in Rs.base[f"{k}_K4"].items():
            L.append(f"| {k} | {name} | {f(s['IID_r2y'])} | {f(s['IID_r2b'])} | {f(s['EXT_r2y'])} | {f(s['CONF_r2y'])} "
                     f"| {f(s['TIME_r2y'])} | {f(s['action_acc'])} |")

    L += ["", "## 3. Transformer sites", "", "IID R²_y (TIME R²_y) [G_count in the channel env]. res0 TIME is ill-posed: positions t > 12 lie outside the span of the fitted positional embeddings, and the least-squares extrapolation diverges (R² ~ −1e10).", "",
          "| env | objective | res0 | res1 | res2 | u |", "|---|---|---|---|---|---|"]
    for k in KINDS:
        for o in OBJ + ("init",):
            cells = []
            for s in TSITES:
                tv = Rs.m('tfm', k, o, 'TIME_r2y', site=s)
                c = f"{f(Rs.m('tfm', k, o, 'IID_r2y', site=s))} ({'n/a' if s == 'res0' else f(tv)})"
                if k == "channel":
                    c += f" [{f(Rs.m('tfm', k, o, 'G_count', site=s))}]"
                cells.append(c)
            L.append(f"| {k} | {o} | " + " | ".join(cells) + " |")

    L += ["", "## 4. Network-level held-out: models trained without any sequence that enters the conflict region", "",
          "Probe fit outside the region, scored inside it. KL: the model's own output against its exact target.", "",
          "| arch | env | objective | CONF R²_y, full pool | CONF R²_y, no-conflict pool | KL in / out, full | KL in / out, no-conflict |",
          "|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for k in KINDS:
            for o in OBJ:
                beh = lambda ho, key: np.mean([r["behaviour"][key] for r in Rs.sel(a, k, 4, o, ho)]) if Rs.sel(a, k, 4, o, ho) else np.nan   # noqa: E731
                L.append(f"| {a} | {k} | {o} | {f(Rs.m(a, k, o, 'CONF_r2y'))} | {f(Rs.m(a, k, o, 'CONF_r2y', netho=True))} "
                         f"| {f(beh(False, 'kl_in'), 4)} / {f(beh(False, 'kl_out'), 4)} | {f(beh(True, 'kl_in'), 4)} / {f(beh(True, 'kl_out'), 4)} |")

    L += ["", "## 5. Number of goals", "", "| arch | env | K | objective | IID R²_y | EXT R²_y | CONF R²_y | TIME R²_y | G_count (best site) | counts-baseline R²_y |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for k in KINDS:
            for K in (3, 4, 5):
                for o in ("goal", "act_hard", "init"):
                    bs = Rs.best_site(a, k, o, "G_count", K=K) if k == "channel" else None
                    L.append(f"| {a} | {k} | {K} | {o} | {f(Rs.m(a, k, o, 'IID_r2y', K=K))} | {f(Rs.m(a, k, o, 'EXT_r2y', K=K))} "
                             f"| {f(Rs.m(a, k, o, 'CONF_r2y', K=K))} | {f(Rs.m(a, k, o, 'TIME_r2y', K=K))} "
                             f"| {f(Rs.m(a, k, o, 'G_count', site=bs, K=K)) if bs else '—'} | {f(Rs.base[f'{k}_K{K}']['counts']['IID_r2y'])} |")

    L += ["", "## 6. Seed stability of interface IID R²_y (CV across seeds)", "", "| arch | env | " + " | ".join(OBJ) + " |", "|---|---|" + "---|" * len(OBJ)]
    for a in ARCHS:
        for k in KINDS:
            L.append(f"| {a} | {k} | " + " | ".join(f(cv(Rs.vals(a, k, o, 'IID_r2y')), 4) for o in OBJ) + " |")
    return "\n".join(L) + "\n"


def cv(v):
    return float(v.std() / abs(v.mean())) if len(v) > 1 else float("nan")


# ---------------------------------------------------------------- predictions
def predictions(Rs: R):
    P = []

    def add(name, ok, detail):
        P.append((name, bool(ok), detail))

    cells = [(a, k) for a in ARCHS for k in KINDS]
    # P1
    det = []; ok = True
    for a, k in cells:
        v = {s: Rs.m(a, k, "goal", f"{s}_r2y") for s in ("IID", "EXT", "CONF", "TIME")}
        ok &= v["IID"] >= 0.99 and min(v["EXT"], v["CONF"], v["TIME"]) >= 0.97
        det.append(f"{a}/{k}: " + ", ".join(f"{s} {f(x)}" for s, x in v.items()))
    add("P1 forced identity", ok, "; ".join(det))
    # P2
    det = []; ok = True
    for a, k in cells:
        gy, gb = Rs.m(a, k, "goal", "IID_r2y"), Rs.m(a, k, "goal", "IID_r2b")
        sy, sb = Rs.m(a, k, "act_soft", "IID_r2y"), Rs.m(a, k, "act_soft", "IID_r2b")
        ok &= gy > gb and sb > sy
        det.append(f"{a}/{k}: goal y {f(gy)} vs b {f(gb)}; act_soft y {f(sy)} vs b {f(sb)}")
    add("P2 coordinates follow the target", ok, "; ".join(det))
    # P3
    det = []; ok = True
    for a, k in cells:
        g, s = Rs.m(a, k, "goal", "EXT_r2y"), Rs.m(a, k, "act_soft", "EXT_r2y")
        ok &= g - s >= 0.2; det.append(f"{a}/{k}: goal {f(g)} − act_soft {f(s)} = {f(g - s)}")
    add("P3 extrapolation exposes the coordinate", ok, "; ".join(det))
    # P4
    det = []; ok = True
    for a, k in cells:
        g, h, acc = Rs.m(a, k, "goal", "IID_r2y"), Rs.m(a, k, "act_hard", "IID_r2y"), Rs.m(a, k, "act_hard", "action_acc")
        ok &= g - h >= 0.1 and acc >= 0.95
        det.append(f"{a}/{k}: goal {f(g)} − act_hard {f(h)} = {f(g - h)}, action acc {f(acc)}")
    add("P4 hard targets keep the region, not the posterior", ok, "; ".join(det))
    # P5
    det = []; ok = True
    for o in OBJ:
        s = Rs.best_site("tfm", "iid", o); v = Rs.m("tfm", "iid", o, "IID_r2y", site=s)
        ok &= v >= 0.95; det.append(f"{o}: {s} {f(v)}")
    add("P5 iid is not diagnostic (tfm)", ok, "; ".join(det))
    # P6
    det = []; ok = True
    for a in ARCHS:
        for o in ("goal", "next_obs"):
            s = Rs.best_site(a, "channel", o, "G_count"); v = Rs.m(a, "channel", o, "G_count", site=s)
            ok &= v >= 0.7; det.append(f"{a} {o}: {s} {f(v)}")
        h, g = Rs.m(a, "channel", "act_hard", "G_count"), Rs.m(a, "channel", "goal", "G_count")
        ok &= h < g; det.append(f"{a} act_hard {f(h)} < goal {f(g)}")
    add("P6 channel is diagnostic", ok, "; ".join(det))
    # P7
    det = []; ok = True
    for a, k in cells:
        sites = TSITES if a == "tfm" else ("h",)
        mx = max(Rs.m(a, k, "init", "IID_r2y", site=s) for s in sites)
        ok &= mx <= 0.8; d = f"{a}/{k}: max R²_y {f(mx)}"
        if k == "channel":
            g = max(Rs.m(a, k, "init", "G_count", site=s) for s in sites); ok &= g <= 0.3; d += f", max G {f(g)}"
        det.append(d)
    add("P7 init control", ok, "; ".join(det))
    # P8
    det = []; ok = True
    for a, k in cells:
        rs = Rs.sel(a, k, 4, "goal", True)
        c = Rs.m(a, k, "goal", "CONF_r2y", netho=True)
        ratio = np.mean([r["behaviour"]["kl_in"] / max(r["behaviour"]["kl_out"], 1e-12) for r in rs]) if rs else np.nan
        full = Rs.sel(a, k, 4, "goal", False)
        rfull = np.mean([r["behaviour"]["kl_in"] / max(r["behaviour"]["kl_out"], 1e-12) for r in full]) if full else np.nan
        ok &= c >= 0.9 and ratio <= 3
        det.append(f"{a}/{k}: CONF R²_y {f(c)}, KL in/out {f(ratio, 2)} (full-pool models, not part of the test: {f(rfull, 2)})")
    add("P8 network-level held-out", ok, "; ".join(det))
    # P9
    i, t = Rs.m("tfm", "iid", "goal", "IID_r2y", site="res1"), Rs.m("tfm", "iid", "goal", "TIME_r2y", site="res1")
    add("P9 block 1 codes a running mean", i - t >= 0.1, f"res1 IID {f(i)} − TIME {f(t)} = {f(i - t)}")
    # P10
    det = []; ok = True
    for a, k in cells:
        for o in OBJ:
            c = cv(Rs.vals(a, k, o, "IID_r2y")); ok &= not c > 0.02
            if c > 0.02:
                det.append(f"{a}/{k}/{o} {f(c, 4)}")
    add("P10 seed stability", ok, "violations: " + (", ".join(det) if det else "none"))
    # P11
    det = []; ok = True
    for K in (3, 5):
        for a in ARCHS:
            for k in KINDS:
                v = {s: Rs.m(a, k, "goal", f"{s}_r2y", K=K) for s in ("IID", "EXT", "CONF", "TIME")}
                good = v["IID"] >= 0.99 and min(v["EXT"], v["CONF"], v["TIME"]) >= 0.97
                if k == "channel":
                    s = Rs.best_site(a, k, "goal", "G_count", K=K); g = Rs.m(a, k, "goal", "G_count", site=s, K=K)
                    good &= g >= 0.7
                ok &= good
                if not good:
                    det.append(f"K={K} {a}/{k}: " + ", ".join(f"{s} {f(x)}" for s, x in v.items()))
    add("P11 K = 3, 5", ok, "failures: " + ("; ".join(det) if det else "none"))
    return P


def followup_table(Lr: R, Rs: R) -> str:
    """Channel-env act_hard at 4x the steps (rounds/r12_hidden_goal/followup.py; not pre-registered)."""
    L = ["", "## 8. Follow-up (not pre-registered): channel-env act_hard at 4× the steps", "",
         "Transformers 80k steps, GRUs 60k (main grid: 20k / 15k). All models shown, converged or not; "
         "the goal row is the main grid's for comparison.", "",
         "| arch | pool | steps | converged | argmax agrees | IID R²_y | IID R²_b | EXT R²_y | CONF R²_y | G_count | action acc |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARCHS:
        for ho in (False, True):
            for lab, src in (("main", Rs), ("4×", Lr)):
                rs = src.sel(a, "channel", 4, "act_hard", ho, conv=False)
                if not rs:
                    continue
                g = lambda key: np.mean([r["sites"][IFACE[a]][key] for r in rs])      # noqa: E731
                L.append(f"| {a} | {'no-conflict' if ho else 'full'} | {lab} | {sum(r['converged'] for r in rs)}/{len(rs)} "
                         f"| {f(np.mean([r['behaviour']['act_agree'] for r in rs]), 4)} | {f(g('IID_r2y'))} | {f(g('IID_r2b'))} "
                         f"| {f(g('EXT_r2y'))} | {f(g('CONF_r2y'))} | {f(g('G_count'))} | {f(g('action_acc'))} |")
        L.append(f"| {a} | full | goal (main) | — | — | {f(Rs.m(a, 'channel', 'goal', 'IID_r2y'))} | {f(Rs.m(a, 'channel', 'goal', 'IID_r2b'))} "
                 f"| {f(Rs.m(a, 'channel', 'goal', 'EXT_r2y'))} | {f(Rs.m(a, 'channel', 'goal', 'CONF_r2y'))} | {f(Rs.m(a, 'channel', 'goal', 'G_count'))} "
                 f"| {f(Rs.m(a, 'channel', 'goal', 'action_acc'))} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="rounds/r12_hidden_goal"); a = ap.parse_args()
    out = Path(a.out); Rs = R(out / "results11.json")
    txt = tables(Rs)
    P = predictions(Rs)
    txt += "\n## 7. Pre-registered predictions (rounds/r12_hidden_goal/THEORY.md §5)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {d} |\n" for n, ok, d in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    long = out / "long" / "results11.json"
    if long.exists():
        txt += followup_table(R(long), Rs)
    (out / "tables.md").write_text(txt)
    print(txt[-3000:])
    import plots as plotting11
    plotting11.all_figures(Rs, out)


if __name__ == "__main__":
    main()
