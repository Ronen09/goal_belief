"""The goal-swap-components experiment, post hoc (after the registered results): ratio-of-means transfer on the test cells,
1 - mean TV(patched, donor) / mean TV(recipient, donor). The registered per-cell mean is dominated by cells where the
two goals' runs barely differ, where single-head patches give large negative values.

    .venv/bin/python studies/5_goal_belief_mechanism/goal_swap_components/posthoc.py            # writes posthoc.json
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R                                                       # noqa: E402

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    t = R.M26.T18.tables("cuda"); ctx = R.M26.Ctx(t)
    test = R.Cells(ctx, ctx.test, 31)
    res = json.load(open(HERE / "results.json"))["runs"]
    out = {}
    for s in range(10):
        net, _ = R.M26.load("reward", s, t, "cuda", R.M26.R23 / "runs"); net.eval()
        after = R.components(net, 1)
        S = [c for c in after if R.name(c) in res[f"seed{s}"]["selected"]]
        groups = dict(all_blocks_1_3=after, selected=S, complement=[c for c in after if c not in S],
                      attention_1_3=[c for c in after if c[0] == "head"], mlp_1_3=[c for c in after if c[0] == "mlp"])
        items = [(R.name(c), [c]) for c in after] + [(k, v) for k, v in groups.items() if v]
        acc = {k: torch.zeros(3, dtype=torch.float64, device="cuda") for k, _ in items}
        for st in range(0, test.n, R.CHUNK):
            h, gr, gd = test.hist[st:st + R.CHUNK], test.gr[st:st + R.CHUNK], test.gd[st:st + R.CHUNK]
            n = torch.arange(len(h), device="cuda"); pos = 1 + ctx.L[h]
            tR, tD = R.seq(ctx, h, gr), R.seq(ctx, h, gd)
            lR, _, rR = net(tR, record=True); lD, _, rD = net(tD, record=True)
            pR, pD = lR[n, pos].softmax(-1), lD[n, pos].softmax(-1)
            den = R.tv(pR, pD)
            for k, cs in items:
                pp = net(tR, patch=R.make_patch(cs, rD, n, pos))[0][n, pos].softmax(-1)
                pq = net(tD, patch=R.make_patch(cs, rR, n, pos))[0][n, pos].softmax(-1)
                acc[k] += torch.stack([R.tv(pp, pD).sum(), R.tv(pq, pR).sum(), den.sum()]).double()
        out[f"seed{s}"] = {k: dict(patch=float(1 - v[0] / v[2]), restore=float(1 - v[1] / v[2])) for k, v in acc.items()}
        print(s, {k: round(v["patch"], 2) for k, v in out[f"seed{s}"].items()}, flush=True)
    (HERE / "posthoc.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
