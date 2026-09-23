"""TASK15 (round 16): tables, prediction checks and figure from results15/results15.json.

    .venv/bin/python scripts/task15_tables.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from goalgeo.plotting11 import INK, MUTED, _style  # noqa: E402

OUT = Path("results15")
PS = (0.0, 0.5, 0.75, 0.9, 0.97, 1.0)
TS = (4, 8, 16, 23)
RP = json.load(open("results14/recompute_prediction.json"))


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


class D:
    def __init__(self, *paths):
        self.runs = [r for p in paths if Path(p).exists() for r in json.load(open(p))["runs"]]

    def ps(self, fam):
        return sorted({r["p"] for r in self.runs if r["family"] == fam})

    def sel(self, fam, p):
        return [r for r in self.runs if r["family"] == fam and r["p"] == p]

    def cal(self, fam, p, t, cell, regime="full"):
        """Calibrated λ of a cell (R1 = 0, R4 = 1), per seed then averaged."""
        vals = []
        for r in self.sel(fam, p):
            c = r["t"][str(t)][regime]
            vals.append((c[cell] - c["R1"]) / (c["R4"] - c["R1"]))
        return float(np.mean(vals))

    def kl(self, fam, p, key):
        return float(np.mean([r[key] for r in self.sel(fam, p)]))


def tables(d: D) -> str:
    L = ["# TASK15 tables (round 16: inducing recurrence by K/V dropout)", "",
         "Prior weight = calibrated λ of R2 (position t's exports from B, A's tokens) at the output of t+1; R3 = older positions "
         "from B, t from A; 'recompute' = round 15's exact pure-recomputation prediction for R3. Full = full attention at test; "
         "drop = the model's own training dropout at test (query t+1 always sees t). 3 seeds, seed means.", ""]
    for fam in ("carry", "plain"):
        L += [f"## {fam}", "", "| p | KL full | KL drop | " + " | ".join(f"prior t={t} (full)" for t in TS) + " | "
              + " | ".join(f"prior t={t} (drop)" for t in TS) + " | R3 t=16 (full) | recompute t=16 |",
              "|---|---|---|" + "---|" * (2 * len(TS) + 2)]
        for p in PS:
            L.append(f"| {p} | {f(d.kl(fam, p, 'kl_full'), 4)} | {f(d.kl(fam, p, 'kl_drop'), 4)} | "
                     + " | ".join(f(d.cal(fam, p, t, 'R2')) for t in TS) + " | "
                     + " | ".join(f(d.cal(fam, p, t, 'R2', 'drop')) for t in TS)
                     + f" | {f(d.cal(fam, p, 16, 'R3'))} | {f(RP['24_16'])} |")
        L.append("")
    fine = [p for fam in ("carry", "plain") for p in d.ps(fam) if p not in PS]
    if fine:
        L += ["## Follow-up (not pre-registered): the finer dropout grid, merged", "",
              "Carry p ∈ {0.05, 0.1, 0.2, 0.3} and plain p ∈ {0.1, 0.25} were added after the main grid showed the carry "
              "transition lies between p = 0 and 0.5 (`results15/fine/`). Prior weight at t = 16.", "",
              "| family | p | KL full | KL drop | prior (full) | prior (drop) | R3 (full) |", "|---|---|---|---|---|---|---|"]
        for fam in ("carry", "plain"):
            for p in d.ps(fam):
                L.append(f"| {fam} | {p} | {f(d.kl(fam, p, 'kl_full'), 4)} | {f(d.kl(fam, p, 'kl_drop'), 4)} | {f(d.cal(fam, p, 16, 'R2'))} "
                         f"| {f(d.cal(fam, p, 16, 'R2', 'drop'))} | {f(d.cal(fam, p, 16, 'R3'))} |")
    return "\n".join(L) + "\n"


def predictions(d: D):
    P = []
    add = lambda n, ok, x: P.append((n, bool(ok), x))                   # noqa: E731
    v = [d.cal("carry", p, 16, "R2") for p in PS]
    mid = sum(0.1 < x < 0.95 for x in v[1:-1])
    add("I1 carry continuum: ≤ 0.1 at p=0, ≥ 0.95 at p=1, non-decreasing, ≥ 2 intermediate",
        v[0] <= 0.1 and v[-1] >= 0.95 and all(b >= a - 1e-9 for a, b in zip(v, v[1:])) and mid >= 2, " → ".join(f"p={p}: {f(x)}" for p, x in zip(PS, v)))
    v = {p: d.kl("carry", p, "kl_drop") for p in PS}
    add("I2 carry: KL < 0.01 under own dropout at every p", max(v.values()) < 0.01, "; ".join(f"p={p} {f(x, 4)}" for p, x in v.items()))
    x = d.cal("carry", 0.9, 16, "R2")
    add("I3 carry p=0.9: prior weight under full access ≥ 0.3", x >= 0.3, f(x))
    ok, det = True, []
    for fam in ("carry", "plain"):
        for p in PS[1:-1]:
            a, b = d.cal(fam, p, 16, "R2", "drop"), d.cal(fam, p, 16, "R2")
            ok &= a >= b; det.append(f"{fam} p={p}: drop {f(a)} vs full {f(b)}")
    add("I4 dropout at test shifts weight to the prior", ok, "; ".join(det))
    v = [d.cal("plain", p, 16, "R2") for p in PS]
    add("I5 plain: rises from ≤ 0.05 at p=0, > 0.2 at p=0.9", v[0] <= 0.05 and v[3] > 0.2, " → ".join(f"p={p}: {f(x)}" for p, x in zip(PS, v)))
    x = d.kl("plain", 1.0, "kl_drop")
    add("I6 plain p=1: KL under own dropout > 0.05", x > 0.05, f(x, 4))
    ok, det = True, []
    for p in (0.75, 0.9, 0.97, 1.0):
        pw, r3 = d.cal("carry", p, 16, "R2"), d.cal("carry", p, 16, "R3")
        ok &= r3 <= RP["24_16"] - 0.5 * pw; det.append(f"p={p}: R3 {f(r3)} vs bound {f(RP['24_16'] - 0.5 * pw)}")
    add("I7 carry p ≥ 0.75: R3 ≤ recompute − 0.5 × prior weight", ok, "; ".join(det))
    return P


def figure(d: D):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    col = {"carry": "#2a78d6", "plain": "#eb6834"}
    ax = axes[0]; _style(ax)
    for fam in ("carry", "plain"):
        ps = d.ps(fam)
        ax.plot(ps, [d.cal(fam, p, 16, "R2") for p in ps], color=col[fam], lw=2, marker="o", ms=5, label=f"{fam}, full access at test")
        ax.plot(ps, [d.cal(fam, p, 16, "R2", "drop") for p in ps], color=col[fam], lw=1.5, ls="--", marker="o", ms=3, label=f"{fam}, own dropout at test")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("weight on position t's state (λ, t = 16)", fontsize=8, color=INK)
    ax.set_title("the prior's weight vs training K/V dropout", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=7, loc="center right")
    ax = axes[1]; _style(ax)
    for fam in ("carry", "plain"):
        ps = d.ps(fam)
        ax.plot(ps, [d.kl(fam, p, "kl_full") for p in ps], color=col[fam], lw=2, marker="o", ms=5, label=f"{fam}, full access")
        ax.plot(ps, [d.kl(fam, p, "kl_drop") for p in ps], color=col[fam], lw=1.5, ls="--", marker="o", ms=3, label=f"{fam}, own dropout")
    ax.axhline(0.01, color=MUTED, ls=":", lw=1); ax.set_yscale("log")
    ax.set_title("KL to the exact predictive", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=7)
    for a in axes[:2]:
        a.set_xlabel("training dropout p of historical K/V", fontsize=8, color=INK)
    ax = axes[2]; _style(ax)
    ps = d.ps("carry"); cmap = plt.get_cmap("Blues")
    for j, p in enumerate(ps):
        ax.plot(TS, [d.cal("carry", p, t, "R2") for t in TS], color=cmap(0.3 + 0.7 * j / max(1, len(ps) - 1)), lw=2, marker="o", ms=4, label=f"p = {p}")
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel("position t", fontsize=8, color=INK)
    ax.set_title("carry models: prior weight by position (full access)", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig1_incentive.png", dpi=150); plt.close(fig)


def main():
    d = D(OUT / "results15.json", OUT / "fine" / "results15.json")
    txt = tables(d); P = predictions(d)
    txt += "\n## Pre-registered predictions (docs/task15_incentive_theory.md §4)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {x} |\n" for n, ok, x in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    (OUT / "tables15.md").write_text(txt); print(txt)
    figure(d)


if __name__ == "__main__":
    main()
