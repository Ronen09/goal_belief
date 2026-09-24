"""TASK16 (round 17): tables, prediction checks and figure from rounds/r17_read_cost/results16.json.

    .venv/bin/python rounds/r17_read_cost/tables.py
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from goalgeo.style import INK, MUTED, _style  # noqa: E402

OUT = Path("rounds/r17_read_cost")
CS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3)
TS = (4, 8, 16, 23)


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


class D:
    def __init__(self, *paths):
        self.runs = [r for p in paths if Path(p).exists() for r in json.load(open(p))["runs"]]

    def cs(self, fam):
        return sorted({r["c"] for r in self.runs if r["family"] == fam})

    def sel(self, fam, c):
        return [r for r in self.runs if r["family"] == fam and r["c"] == c]

    def cal(self, fam, c, t, cell, regime="open"):
        """Calibrated λ of a cell (R1 = 0, R4 = 1), per seed then averaged."""
        vals = []
        for r in self.sel(fam, c):
            x = r["t"][str(t)][regime]
            vals.append((x[cell]["lam"] - x["R1"]["lam"]) / (x["R4"]["lam"] - x["R1"]["lam"]))
        return float(np.mean(vals))

    def mean(self, fam, c, key):
        return float(np.nanmean([r[key] for r in self.sel(fam, c)]))

    def rate_pos(self, fam, c):
        return np.mean([r["rate_pos"] for r in self.sel(fam, c)], 0)


def crossover(d: D, fam):
    """Smallest grid price with seed-mean read rate <= 0.5 (inf if none)."""
    for c in d.cs(fam):
        if d.mean(fam, c, "read_rate") <= 0.5:
            return c
    return float("inf")


def qualifying(d: D):
    """The intermediate carry price for R4/R5: seed-mean read rate in [0.1, 0.6], else the nearest above 0.6."""
    rates = {c: d.mean("carry", c, "read_rate") for c in d.cs("carry")}
    mid = [c for c, r in rates.items() if 0.1 <= r <= 0.6]
    if mid:
        return min(mid, key=lambda c: abs(rates[c] - 0.35))
    above = [c for c, r in rates.items() if r > 0.6]
    return min(above, key=lambda c: rates[c]) if above else max(rates, key=rates.get)


def tables(d: D) -> str:
    L = ["# TASK16 tables (round 17: pricing recomputation with a learned read gate)", "",
         "Read rate = fraction of (block, position ≥ 2) with an open deterministic gate on 3000 held-out sequences. Prior weight = "
         "calibrated λ of R2 (position t's exports from B, A's tokens) at the output of t+1; 'open' = every gate forced open at "
         "test, 'own' = the learned gates. Entropy / movement split = mean exact-filter entropy / KL(J_u ‖ J_{u−1}) at open minus "
         "closed (block, position ≥ 8) pairs. 3 seeds, seed means. The carry model has no gate at u = 0 and an unpriced gate at u = 1; plain gates at u = 0–1 change nothing; the read rate, and so 'u=2–4', counts u ≥ 2 only.", ""]
    for fam in ("carry", "plain"):
        L += [f"## {fam}", "", "| c | read rate | KL own | KL open | " + " | ".join(f"prior t={t} (open)" for t in TS) + " | "
              + " | ".join(f"prior t={t} (own)" for t in TS) + " | R3 t=16 (open) | entropy split | movement split | rate u=2–4 | rate u≥16 |",
              "|---|---|---|---|" + "---|" * (2 * len(TS) + 5)]
        for c in d.cs(fam):
            rp = d.rate_pos(fam, c)
            L.append(f"| {c} | {f(d.mean(fam, c, 'read_rate'))} | {f(d.mean(fam, c, 'kl_own'), 4)} | {f(d.mean(fam, c, 'kl_open'), 4)} | "
                     + " | ".join(f(d.cal(fam, c, t, 'R2')) for t in TS) + " | "
                     + " | ".join(f(d.cal(fam, c, t, 'R2', 'own')) for t in TS)
                     + f" | {f(d.cal(fam, c, 16, 'R3'))} | {f(d.mean(fam, c, 'entropy_split'))} | {f(d.mean(fam, c, 'movement_split'))} "
                     f"| {f(float(rp[2:5].mean()))} | {f(float(rp[16:].mean()))} |")
        L += ["", f"crossover c* ({fam}) = {crossover(d, fam)}", ""]
    L += [f"Qualifying intermediate carry price for R4/R5: c = {qualifying(d)}", ""]
    return "\n".join(L) + "\n"


def predictions(d: D):
    P = []
    add = lambda n, ok, x: P.append((n, bool(ok), x))                   # noqa: E731
    cs = [c for c in CS if c in d.cs("carry")]
    v = [d.mean("carry", c, "read_rate") for c in cs]
    add("R1 carry: read rate non-increasing in c, ≥ 0.5 at c=0, ≤ 0.05 at c=0.3",
        v[0] >= 0.5 and v[-1] <= 0.05 and all(b <= a + 1e-9 for a, b in zip(v, v[1:])), " → ".join(f"c={c}: {f(x)}" for c, x in zip(cs, v)))
    kl = {c: d.mean("carry", c, "kl_own") for c in cs}
    add("R2 carry: KL under own gates < 0.01 at every c", max(kl.values()) < 0.01, "; ".join(f"c={c} {f(x, 4)}" for c, x in kl.items()))
    closed = [c for c in cs if d.mean("carry", c, "read_rate") <= 0.05]
    ok = bool(closed); det = []
    for c in closed:
        a, b = d.cal("carry", c, 16, "R2", "own"), d.cal("carry", c, 16, "R2")
        ok &= a >= 0.95 and b >= 0.8; det.append(f"c={c}: own {f(a)}, open {f(b)}")
    add("R3 carry, read rate ≤ 0.05: prior weight own ≥ 0.95 and open ≥ 0.8", ok, "; ".join(det) or "no closed model")
    q = qualifying(d); e, m = d.mean("carry", q, "entropy_split"), d.mean("carry", q, "movement_split")
    add("R4 carry, intermediate c: entropy split ≥ +0.05, movement split ≥ +0.02", e >= 0.05 and m >= 0.02, f"c={q}: entropy {f(e)}, movement {f(m)}")
    rp = d.rate_pos("carry", q); early, late = float(rp[2:5].mean()), float(rp[16:].mean())
    add("R5 carry, intermediate c: read rate at u=2–4 exceeds u≥16 by ≥ 0.2", early - late >= 0.2, f"c={q}: {f(early)} vs {f(late)}")
    pcs = [c for c in CS if c in d.cs("plain")]
    v = {c: d.cal("plain", c, 16, "R2") for c in pcs}
    add("R6 plain: prior weight under forced-open ≤ 0.4 at every c", max(v.values()) <= 0.4, "; ".join(f"c={c} {f(x)}" for c, x in v.items()))
    low = [c for c in pcs if d.mean("plain", c, "read_rate") <= 0.1]
    ok = bool(low); det = []
    for c in low:
        x = d.mean("plain", c, "kl_own"); ok &= x >= 0.03; det.append(f"c={c}: {f(x, 4)}")
    add("R7 plain, read rate ≤ 0.1: KL under own gates ≥ 0.03", ok, "; ".join(det) or "no closed plain model")
    cc, cp = crossover(d, "carry"), crossover(d, "plain"); r01 = d.mean("plain", 0.01, "read_rate") if 0.01 in pcs else float("nan")
    add("R8 c*_carry < c*_plain; plain read rate ≥ 0.5 at c=0.01", cc < cp and r01 >= 0.5, f"c* carry {cc}, plain {cp}; plain rate at 0.01: {f(r01)}")
    return P


def figure(d: D, out: Path = OUT):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    col = {"carry": "#2a78d6", "plain": "#eb6834"}
    xs = lambda cs: [c if c > 0 else 1e-3 / 3 for c in cs]                # noqa: E731  c = 0 drawn left of the log axis
    ax = axes[0]; _style(ax)
    for fam in ("carry", "plain"):
        cs = d.cs(fam)
        ax.plot(xs(cs), [d.mean(fam, c, "read_rate") for c in cs], color=col[fam], lw=2, marker="o", ms=5, label=f"{fam}: read rate")
        ax.plot(xs(cs), [d.cal(fam, c, 16, "R2") for c in cs], color=col[fam], lw=1.5, ls="--", marker="o", ms=3, label=f"{fam}: prior weight (open)")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.05); ax.set_title("reads and the prior's weight vs price", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]; _style(ax)
    for fam in ("carry", "plain"):
        cs = d.cs(fam)
        ax.plot(xs(cs), [d.mean(fam, c, "kl_own") for c in cs], color=col[fam], lw=2, marker="o", ms=5, label=f"{fam}: own gates")
        ax.plot(xs(cs), [d.mean(fam, c, "kl_open") for c in cs], color=col[fam], lw=1.5, ls="--", marker="o", ms=3, label=f"{fam}: forced open")
    ax.axhline(0.01, color=MUTED, ls=":", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("KL to the exact predictive", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=7)
    for a in axes[:2]:
        a.set_xlabel("price c per read (nats / token; leftmost point is c = 0)", fontsize=8, color=INK)
    ax = axes[2]; _style(ax)
    cs = d.cs("carry"); cmap = plt.get_cmap("Blues")
    for j, c in enumerate(cs):
        ax.plot(range(25), d.rate_pos("carry", c), color=cmap(0.3 + 0.7 * j / max(1, len(cs) - 1)), lw=2, label=f"c = {c}")
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel("position u", fontsize=8, color=INK)
    ax.set_title("carry models: read rate by position", fontsize=9, color=INK); ax.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(out / "fig1_price.png", dpi=150); plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(OUT)); a = ap.parse_args()
    out = Path(a.out)
    d = D(out / "results16.json")
    txt = tables(d); P = predictions(d)
    txt += "\n## Pre-registered predictions (rounds/r17_read_cost/THEORY.md §4)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {x} |\n" for n, ok, x in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    (out / "tables.md").write_text(txt); print(txt)
    figure(d, out)


if __name__ == "__main__":
    main()
