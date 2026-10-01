"""Round 29, post hoc (after the registered results): is interface reliance set by the pair or by the goal?

    .venv/bin/python rounds/r29_goal_route_selection/posthoc.py            # writes posthoc.json

On pairs with at least two eligible goals: the variance of a_interface between pairs (pair means) and within pairs
(across goals), and the association of direct-route adequacy with a_interface at each level (pair-mean adequacy
against pair-mean reliance, and the within-pair coefficient of the registered regression, for comparison).
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R                                                       # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    t = R.M26.T18.tables("cuda"); ctx = R.M26.Ctx(t); pairs = R.R27.Pairs(ctx)
    out = {}
    for s in range(10):
        net, _ = R.M26.load("reward", s, t, "cuda", R.M26.R23 / "runs")
        gen = torch.Generator(device="cuda"); gen.manual_seed(2900 + s)
        itf = R.R28.Interface(net, ctx, gen); probes = R.Probes(net, ctx)
        row = {}
        for name, pr in (("main", pairs.main), ("onestep", pairs.onestep)):
            a, b = pr["r"], pr["d"]
            c, _ = R.cells(net, ctx, itf, a, b)
            elig = c["tv"] >= R.MIN_TV
            multi = elig.sum(1) >= 2
            y = c["a_interface"]; X = (probes.direct_ok[a] & probes.direct_ok[b]).float()
            w = (elig & multi[:, None]).float()
            cnt = w.sum(1)
            pm = (y * w).sum(1) / cnt.clamp(min=1); xm = (X * w).sum(1) / cnt.clamp(min=1)
            sel = multi
            within = (((y - pm[:, None]) ** 2) * w).sum() / w.sum()
            grand = (pm[sel] * cnt[sel]).sum() / cnt[sel].sum()
            between = (((pm[sel] - grand) ** 2) * cnt[sel]).sum() / cnt[sel].sum()
            # pair level: reliance of pairs whose direct route is inadequate under every eligible goal, against adequate under every one
            allbad, allgood = sel & (xm == 0), sel & (xm == 1)
            row[name] = dict(var_between=float(between), var_within=float(within), share_between=float(between / (between + within)),
                             pairs_all_inadequate=int(allbad.sum()), pairs_all_adequate=int(allgood.sum()),
                             a_interface_pairs_all_inadequate=float(pm[allbad].mean()), a_interface_pairs_all_adequate=float(pm[allgood].mean()),
                             pair_level_corr=float(torch.corrcoef(torch.stack([pm[sel], xm[sel]]))[0, 1]))
        out[f"seed{s}"] = row
        print(s, {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in row.items()}, flush=True)
    (HERE / "posthoc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
