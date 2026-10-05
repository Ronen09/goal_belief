"""The navigate-commit experiment: a model's on-policy regret for every pair of station reliabilities on the grid (final checkpoint),
and how much its action distribution depends on the reliabilities it is told. Writes qgen.json in each run."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch

from goalgeo import navbank as B, navmodel as NM, navppo as P

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train as T                                                    # noqa: E402


@torch.no_grad()
def q_sensitivity(net, tab, bank, n=6000):
    """The same histories told (0.6, 0.6) and (0.95, 0.95): total variation between the two action distributions
    at decisions with at least one clue, and the exact policies' disagreement there (share of decisions whose
    optimal sets are disjoint)."""
    b = bank["broad"]
    lay = (b.cfg // (tab.nq * tab.nq))[:n]
    out = {}
    pis, opts = [], []
    for qi in (0, tab.nq - 1):
        cfg = tab.config(lay, torch.full_like(lay, qi), torch.full_like(lay, qi))
        rep = b.tok[:n, :, NM.F_REP]
        yh, yv = (rep == NM.REP_R).any(1).long(), (rep == NM.REP_B).any(1).long()
        r = B.replay(tab, cfg, b.state[:n, 0, 1], b.act[:n], b.alive[:n], yh, yv)
        d = B.decisions(tab, r)
        lg, _ = net(r.tok, r.q)
        pis.append(lg[d["hist"], d["pos"]].masked_fill(~d["legal"], -1e9).softmax(-1)); opts.append(d["optimal"])
        clue = (d["sh"] > 0) | (d["sv"] > 0)
    tv = 0.5 * (pis[0] - pis[1]).abs().sum(1)
    return dict(tv=tv[clue].mean().item(), exact_disjoint=(~(opts[0] & opts[1]).any(1))[clue].float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    tab = P.Tables(T.ENV["n"], T.ENV["H"], T.ENV["c"], T.Q_GRID, dev)
    bank = B.make_bank(tab, T.q_pairs("grid", dev, "all")[0], 170018, n_broad=6000, dense_layouts=2, dense_per=10)
    sets = {k: {tuple(p) for p in T.q_pairs("grid", dev, k)[0].tolist()} for k in ("train", "held_value", "held_combo")}
    for run in a.runs:
        run = Path(run)
        layers = json.loads((run / "log.json").read_text())["args"]["layers"]
        net = NM.Net(T.ENV["n"] ** 2, T.ENV["H"], layers=layers).to(dev).eval()
        net.load_state_dict(torch.load(sorted((run / "ckpt").glob("u*.pt"))[-1]))
        rows = []
        for i in range(tab.nq):
            for j in range(tab.nq):
                g = torch.Generator(device=dev); g.manual_seed(500 + 8 * i + j)
                cfg, start = P.sample_configs(tab, a.n, torch.tensor([[i, j]], device=dev), g)
                s = P.summarize(P.rollout(net, tab, cfg, start, gen=g, greedy=True))
                rows.append(dict(qh=T.Q_GRID[i], qv=T.Q_GRID[j], set=next(k for k, v in sets.items() if (i, j) in v),
                                 regret=s["regret"], queries=s["queries"], v_star=tab.V[cfg, tab.H, start, 0, 0].mean().item(),
                                 err_query=s["err_query"], err_commit=s["err_commit"], err_route=s["err_route"]))
        res = dict(pairs=rows, sensitivity=q_sensitivity(net, tab, bank))
        (run / "qgen.json").write_text(json.dumps(res, indent=1))
        by = {k: sum(r["regret"] for r in rows if r["set"] == k) / sum(r["set"] == k for r in rows) for k in sets}
        print(run, {k: round(v, 4) for k, v in by.items()}, res["sensitivity"], flush=True)


if __name__ == "__main__":
    main()
