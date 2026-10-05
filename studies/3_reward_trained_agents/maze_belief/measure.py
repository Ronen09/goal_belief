"""The maze-belief experiment: every measurement on saved checkpoints. One run directory (a seed of a condition) in, measures.json out.

On the fixed bank, per checkpoint:
  policy     expected local regret of the model's action distribution, by goal, at the reveal and after
  decode     held-out affine decoding at 9 sites of the exact posterior over cells, of P(left arm), and of the exact
             action values; at decision tokens and at prefix tokens (where the goal is not yet known)
  transfer   a posterior decoder fitted on decisions for one goal, tested on the other goals (at the reveal)
  swap       the same history with another goal: decoded posterior and action distribution
  patches    at key checkpoints: components at the goal token and the residual stream at prefix tokens, from
             donors with other evidence and the same or another goal
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from goalgeo import mazemeasure as MS, mazemodel as MM, mazeppo as P
from goalgeo.navprobe import Affine, site_names

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as T                                                    # noqa: E402

BANK_SEED = 180018
KEY = (30, 60, 100, 150, 250, 400, 700)


class Context:
    def __init__(self, device, quick=False):
        self.t = t = T.tables(device, quick)
        self.bank, self.fit_h = MS.make_bank(t, BANK_SEED, 4000 if quick else 40000)
        self.d = MS.decisions(t, self.bank, self.fit_h)
        self.p = MS.prefix_rows(t, self.bank, self.fit_h)
        n = 600 if quick else 30000
        self.pairs = MS.build_pairs(t, self.bank, self.fit_h, n, 3, min_prefix=1 if quick else 2)                # held-out histories
        self.discovery = MS.build_pairs(t, self.bank, ~self.fit_h, n, 4, min_prefix=1 if quick else 2)           # the fitting histories


def decode(acts, rows, names, extra=None):
    fit, test = rows["fit"], ~rows["fit"]
    out = {}
    for i, n in enumerate(names):
        pr = Affine(acts[i][fit], rows["belief"][fit])
        s = MS.belief_scores(pr(acts[i][test]), rows["belief"][test])
        for k, y in (extra or {}).items():
            y2 = y if y.dim() == 2 else y[:, None]
            s[k] = MS.scalar_r2(Affine(acts[i][fit], y2[fit])(acts[i][test]), y2[test])
        out[n] = s
    return out


@torch.no_grad()
def measure(net, ctx, full):
    t, b, d, p = ctx.t, ctx.bank, ctx.d, ctx.p
    names = site_names(net.nl)
    acts, lg = MS.collect(net, b.tok, d)
    pi = lg.softmax(-1)
    reg = (pi * (d["q_star"].max(1, keepdim=True).values - d["q_star"])).sum(1)
    om = (pi * d["optimal"]).sum(1)
    at0 = d["step"] == 0
    m = dict(policy=dict(regret=reg.mean().item(), optimal_mass=om.mean().item(), regret_reveal=reg[at0].mean().item(),
                         optimal_mass_reveal=om[at0].mean().item(),
                         **{f"regret_G{g + 1}": reg[d["goal"] == g].mean().item() for g in range(3)},
                         **{f"regret_reveal_G{g + 1}": reg[at0 & (d["goal"] == g)].mean().item() for g in range(3)}))
    m["decode"] = decode(acts, d, names, dict(p_left=d["p_left"], q_star=d["q_star"], advantage=d["q_star"] - d["q_star"].max(1, keepdim=True).values))
    sub = lambda rows, acts_, sel: ({k: v[sel] for k, v in rows.items()}, acts_[:, sel])
    r0, a0 = sub(d, acts, at0 & (d["prefix"] >= 2))
    m["decode_reveal"] = decode(a0, r0, names, dict(p_left=r0["p_left"], q_star=r0["q_star"]))
    # decoders fitted without one goal, tested on it: raw, and with each goal's mean activation removed
    m["transfer"] = {}
    gm = torch.stack([a0[:, r0["goal"] == g].mean(1) for g in range(3)], 1)              # [sites, goals, d]
    for i, n in enumerate(names):
        cen = a0[i] - gm[i][r0["goal"]]
        row = {}
        for g in range(3):
            fit_, test_ = r0["fit"] & (r0["goal"] != g), ~r0["fit"] & (r0["goal"] == g)
            own = r0["fit"] & (r0["goal"] == g)
            row[f"G{g + 1}"] = dict(raw=MS.belief_scores(Affine(a0[i][fit_], r0["belief"][fit_])(a0[i][test_]), r0["belief"][test_])["r2"],
                                    centred=MS.belief_scores(Affine(cen[fit_], r0["belief"][fit_])(cen[test_]), r0["belief"][test_])["r2"],
                                    own=MS.belief_scores(Affine(a0[i][own], r0["belief"][own])(a0[i][test_]), r0["belief"][test_])["r2"])
        m["transfer"][n] = row
    pa, _ = MS.collect(net, b.tok, p)
    m["decode_prefix"] = decode(pa, p, names, dict(p_left=p["belief"] @ torch.tensor([c <= 4 for _, c in t.cells], device=t.dev, dtype=torch.float32)))
    rl, al = sub(p, pa, p["last"] & (p["pos"] >= 2))
    m["decode_last_prefix"] = decode(al, rl, names)
    # probes of the residual stream at the reveal, for the swap and the patches
    probes = [Affine(a0[2 * l][r0["fit"]], r0["belief"][r0["fit"]]) for l in range(net.nl + 1)]
    m["swap"] = MS.goal_swap(net, ctx.pairs, probes)
    if full:
        disc = MS.interventions(net, ctx.discovery, probes[-1])
        groups = MS.head_groups(net, disc)
        m["groups"] = {k: [MS.name(c) for c in v] for k, v in groups.items()}
        m["patches"] = MS.interventions(net, ctx.pairs, probes[-1], groups=groups)
        m["patches_discovery"] = {k: v for k, v in disc.items() if k.startswith("L0.")}
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--only-final", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ctx = Context(a.device, a.quick)
    for run in a.runs:
        run = Path(run)
        layers = json.loads((run / "log.json").read_text())["args"]["layers"]
        t = ctx.t
        net = MM.Net(t.n_sym, len(t.goal_cell), t.H, t.max_prefix, layers=layers).to(a.device).eval()
        cks = sorted((run / "ckpt").glob("u*.pt"))
        if a.only_final:
            cks = [cks[0], cks[-1]]
        us = [int(c.stem[1:]) for c in cks]
        key = {us[0], us[-1]} | {min(us, key=lambda u: abs(u - k)) for k in KEY}
        res = dict(sites=site_names(layers), pairs=dict(n=ctx.pairs["n"]), checkpoints={})
        for ck in cks:
            net.load_state_dict(torch.load(ck))
            u = int(ck.stem[1:])
            m = measure(net, ctx, u in key)
            res["checkpoints"][u] = m
            print(run, ck.stem, f"bank regret {m['policy']['regret']:.4f} belief R2 {max(v['r2'] for v in m['decode'].values()):.3f}", flush=True)
        (run / "measures.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
