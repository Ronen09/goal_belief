"""TASK13 (round 14): tables, prediction checks and figures from results13/results13.json.

    .venv/bin/python scripts/task13_tables.py
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

OUT = Path("results13")
WINDOWS = (1, 2, 4, 8, 25)
EDITS = ("swap_u", "probe_u", "swap_res1")


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


class D:
    def __init__(self, path):
        self.runs = json.load(open(path))["runs"]

    def sel(self, kind, l, carry):
        return [r for r in self.runs if r["kind"] == kind and r["window"] == l and r["carry"] == carry]

    def kl(self, kind, l, carry):
        return float(np.mean([r["kl"] for r in self.sel(kind, l, carry)]))

    def conv(self, kind, l, carry):
        return sum(r["kl"] < 0.01 for r in self.sel(kind, l, carry)), len(self.sel(kind, l, carry))

    def F(self, kind, l, carry, edit, t, part="future"):
        """Gap to the genuine-B future closed by an edit, pooled over offsets (nan if there is no gap)."""
        vals = []
        for r in self.sel(kind, l, carry):
            s = r["steer"][f"t{t}"]; e, b = np.array(s[edit]), np.array(s["base"])
            e, b = (e[1:], b[1:]) if part == "future" else (e[:1], b[:1])
            vals.append(np.nan if b.sum() < 1e-9 else 1 - e.sum() / b.sum())
        return float(np.nanmean(vals)) if not all(np.isnan(vals)) else float("nan")

    def moved(self, kind, l, carry, edit, t):
        """Largest change of the future divergence caused by the edit (0 = the edit moved nothing)."""
        return float(max(np.abs(np.array(r["steer"][f"t{t}"][edit][1:]) - np.array(r["steer"][f"t{t}"]["base"][1:])).max()
                         for r in self.sel(kind, l, carry)))

    def m(self, kind, l, carry, fn):
        return float(np.mean([fn(r) for r in self.sel(kind, l, carry)]))


def tables(d: D) -> str:
    L = ["# TASK13 tables (round 14: windowed transformers)", "",
         "Window l = attention to the last l positions (itself included) in both layers; carry = the previous position's "
         "readout interface u added to the input. Objective `goal`, K = 4, 3 seeds, seed means. F = gap to the genuine-B "
         "future closed by a single-site edit at position t, pooled over future offsets k ≥ 1; 'no gap' where the unedited "
         "future already equals B's.", ""]
    L += ["## 1. Accuracy and recoverability", "",
          "| env | l | carry | converged (KL < 0.01) | KL | KL at positions 1–7 | KL at positions 17–24 | goal R² from u | channel R² from u |",
          "|---|---|---|---|---|---|---|---|---|"]
    for kind in ("iid", "channel"):
        for l in WINDOWS:
            for c in (False, True):
                n, N = d.conv(kind, l, c)
                early = d.m(kind, l, c, lambda r: np.mean(r["kl_pos"][1:8])); late = d.m(kind, l, c, lambda r: np.mean(r["kl_pos"][17:25]))
                ch = f(d.m(kind, l, c, lambda r: r["recover_u"]["IID"]["channel"])) if kind == "channel" else "—"
                L.append(f"| {kind} | {l} | {'yes' if c else 'no'} | {n}/{N} | {f(d.kl(kind, l, c), 4)} | {f(early, 4)} | {f(late, 4)} "
                         f"| {f(d.m(kind, l, c, lambda r: r['r2_goal']))} | {ch} |")
    L += ["", "## 2. Steerability: how much of the future one edit at position t controls", "",
          "| env | l | carry | SWAP u_t, t=6 | t=12 | PROBE-E u_t, t=6 | t=12 | SWAP res1(t), t=6 | t=12 | PROBE-E u_t, immediate (k=0), t=6 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("iid", "channel"):
        for l in WINDOWS:
            for c in (False, True):
                g = lambda e, t: d.F(kind, l, c, e, t)             # noqa: E731
                cells = [g("swap_u", 6), g("swap_u", 12), g("probe_u", 6), g("probe_u", 12), g("swap_res1", 6), g("swap_res1", 12),
                         d.F(kind, l, c, "probe_u", 6, "k0")]
                L.append(f"| {kind} | {l} | {'yes' if c else 'no'} | " + " | ".join("no gap" if np.isnan(x) else f(x) for x in cells) + " |")
    L += ["", "Reference: round-12 GRU (complete cut) SWAP 1.000; round-12 full-attention transformer, SWAP res1(t) ≤ 0.16.", "",
          "## 3. Equal full state at lengths 8 vs 16 (channel)", "",
          "| l | carry | model divergence | Bayes divergence | model ÷ Bayes | model ÷ cross-length random |", "|---|---|---|---|---|---|"]
    for l in WINDOWS:
        for c in (False, True):
            rs = d.sel("channel", l, c)
            mod = np.mean([r["equal"]["Ex"]["model"] for r in rs]); bay = np.mean([r["equal"]["Ex"]["bayes"] for r in rs])
            rnd = np.mean([r["equal"]["Xr"]["model"] for r in rs])
            L.append(f"| {l} | {'yes' if c else 'no'} | {f(mod, 6)} | {f(bay, 6)} | {f(mod / bay, 2)} | {f(mod / rnd, 5) if rnd > 0 else '—'} |")
    return "\n".join(L) + "\n"


def predictions(d: D):
    P = []
    add = lambda n, ok, x: P.append((n, bool(ok), x))                   # noqa: E731
    v = {(k, l): d.kl(k, l, False) for k in ("iid", "channel") for l in WINDOWS}
    add("W1 no carry: l ≤ 8 fail (KL ≥ 0.01), full window converges",
        all(v[(k, l)] >= 0.01 for k in ("iid", "channel") for l in (1, 2, 4, 8)) and all(v[(k, 25)] < 0.01 for k in ("iid", "channel")),
        "; ".join(f"{k} l={l} {f(x, 4)}" for (k, l), x in v.items()))
    v = {(k, l): d.kl(k, l, True) for k in ("iid", "channel") for l in WINDOWS}
    add("W2 carry: every window converges", all(x < 0.01 for x in v.values()), "; ".join(f"{k} l={l} {f(x, 4)}" for (k, l), x in v.items()))
    v = {(k, t): (d.F(k, 1, True, "swap_u", t), d.F(k, 1, True, "probe_u", t)) for k in ("iid", "channel") for t in (6, 12)}
    add("W3 carry l = 1: SWAP u_t ≥ 0.95, PROBE-E ≥ 0.9", all(a >= 0.95 and b >= 0.9 for a, b in v.values()),
        "; ".join(f"{k} t={t}: SWAP {f(a)}, PROBE-E {f(b)}" for (k, t), (a, b) in v.items()))
    ok = True; det = []
    for k in ("iid", "channel"):
        for t in (6, 12):
            s = [d.F(k, l, True, "swap_u", t) for l in WINDOWS]
            ok &= all(a > b for a, b in zip(s, s[1:])) and s[-1] <= 0.5
            det.append(f"{k} t={t}: " + " > ".join(f(x) for x in s))
    add("W4 carry: SWAP u_t decreases with l, ≤ 0.5 at l = 25", ok, "; ".join(det))
    mv = max(d.moved(k, l, False, "swap_u", t) for k in ("iid", "channel") for l in WINDOWS for t in (6, 12))
    conv = [(k, l) for k in ("iid", "channel") for l in WINDOWS if d.kl(k, l, False) < 0.01]
    r1 = {(k, l, t): d.F(k, l, False, "swap_res1", t) for k, l in conv for t in (6, 12)}
    add("W5 no carry: u_t edit moves nothing; SWAP res1 ≤ 0.3 (converged models)",
        mv < 1e-12 and all(x <= 0.3 for x in r1.values()),
        f"largest change from the u_t edit {mv:.1e}; SWAP res1: " + "; ".join(f"{k} l={l} t={t} {f(x)}" for (k, l, t), x in r1.items()))
    rs = d.sel("channel", 1, True)
    ratio = np.mean([r["equal"]["Ex"]["model"] for r in rs]) / np.mean([r["equal"]["Ex"]["bayes"] for r in rs])
    add("W6 carry l = 1 channel: cross-length equal state ≤ 3× Bayes", ratio <= 3, f"{f(ratio, 2)}×")
    g = d.m("channel", 1, True, lambda r: r["recover_u"]["IID"]["goal"]); c = d.m("channel", 1, True, lambda r: r["recover_u"]["IID"]["channel"])
    add("W7 carry l = 1 channel: goal R² ≥ 0.98, channel ≥ 0.9 from u", g >= 0.98 and c >= 0.9, f"goal {f(g)}, channel {f(c)}")
    return P


def figures(d: D):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    cols = {1: "#2a78d6", 2: "#1baf7a", 4: "#eda100", 8: "#e87ba4", 25: "#eb6834"}
    for ax, c in zip(axes[:2], (False, True)):
        _style(ax)
        for l in WINDOWS:
            k = np.mean([r["kl_pos"] for r in d.sel("channel", l, c)], 0)
            ax.plot(range(1, 25), np.maximum(k[1:], 1e-6), color=cols[l], lw=2, label=f"l = {l if l < 25 else 'full'}")
        ax.axhline(0.01, color=MUTED, ls=":", lw=1); ax.set_yscale("log"); ax.set_ylim(1e-5, 2)
        ax.set_xlabel("position t", fontsize=8, color=INK)
        ax.set_title(f"channel, {'with' if c else 'no'} carry: KL to the posterior by position", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    ax = axes[2]; _style(ax); x = np.arange(len(WINDOWS))
    for kind, ls in (("iid", "-"), ("channel", "--")):
        ax.plot(x, [d.F(kind, l, True, "swap_u", 6) for l in WINDOWS], color="#2a78d6", ls=ls, lw=2, marker="o", ms=4, label=f"carry, SWAP u_t ({kind})")
        ax.plot(x, [d.F(kind, l, True, "swap_res1", 6) for l in WINDOWS], color="#1baf7a", ls=ls, lw=2, marker="o", ms=4, label=f"carry, SWAP res1(t) ({kind})")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1); ax.axhline(0.16, color="#eb6834", ls=":", lw=1)
    ax.text(0.02, 0.18, "round-12 full-attention transformer, res1 edit", fontsize=6, color="#eb6834")
    ax.set_xticks(x, [str(l) if l < 25 else "full" for l in WINDOWS]); ax.set_xlabel("attention window l", fontsize=8, color=INK)
    ax.set_ylim(-0.05, 1.05); ax.set_title("future controlled by one edit at t = 6", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig1_window.png", dpi=150); plt.close(fig)


def main():
    d = D(OUT / "results13.json")
    txt = tables(d); P = predictions(d)
    txt += "\n## 4. Pre-registered predictions (docs/task13_window_theory.md §4)\n\n| prediction | held | numbers |\n|---|---|---|\n"
    txt += "".join(f"| {n} | {'yes' if ok else '**no**'} | {x} |\n" for n, ok, x in P)
    txt += f"\n{sum(ok for _, ok, _ in P)} of {len(P)} held.\n"
    (OUT / "tables13.md").write_text(txt); print(txt)
    figures(d)


if __name__ == "__main__":
    main()
