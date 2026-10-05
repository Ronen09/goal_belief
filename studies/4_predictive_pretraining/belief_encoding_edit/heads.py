"""The belief-encoding-edit experiment: the predictive-transfer experiment's goal-conditioned heads, rebuilt with the predictive-transfer experiment's code, seeds and layout and kept.

    .venv/bin/python studies/4_predictive_pretraining/belief_encoding_edit/heads.py            # writes heads/*.pt, heads.json

The predictive-transfer experiment trained the heads of all forty backbones together, one layer at a time, and did not keep them. The
minibatch draws are shared across heads, so the same layout (four conditions x ten seeds, the predictive-transfer experiment's order) is
rebuilt here for each layer used (site 4, and site 1 for the random backbone) and N = 1 000 and 100 000. Every
head's held-out regret is compared with the one the predictive-transfer experiment recorded.
"""

from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazepred as PR

HERE = Path(__file__).resolve().parent
R26 = HERE.parent.parent / "4_predictive_pretraining" / "predictive_transfer"
spec = importlib.util.spec_from_file_location("r26measure", R26 / "measure.py")
M26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(M26)

SITES = (4, 1)
NS = (1000, 100000)
HIDDEN = 16


def main():
    dev = "cuda"
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    old = json.load(open(R26 / "results.json"))["heads"][f"h{HIDDEN}"]
    keys, feats = [], []
    for cond in M26.CONDS:
        for s in range(10):
            net, _ = M26.load(cond, s, t, dev, R26 / "runs")
            feats.append(PR.features(net, ctx.bank.tok, ctx.bank.prefix).half()); keys.append((cond, s))
    nb = len(keys)
    xmap = torch.arange(nb, device=dev).repeat(len(M26.HEAD_SEEDS))
    report = {}
    for l in SITES:
        X = torch.stack([PR.layer_norm(f[l].float()) for f in feats])
        for n in NS:
            hd = M26.fit_heads(X, xmap, ctx, HIDDEN, n)
            ev = M26.evaluate(PR.apply_heads(hd, X, xmap, ctx.test_hist), ctx)
            diff = []
            for j in range(len(xmap)):
                c, s = keys[int(xmap[j])]
                diff.append(abs(ev["regret"][j] - old[c][f"seed{s}"][str(l)][str(n)]["regret"][j // nb]))
            report[f"site{l}_n{n}"] = dict(max_abs_regret_difference=float(max(diff)), mean_regret=float(np.mean(ev["regret"])))
            torch.save(dict(params=[p.detach().cpu() for p in hd.params], hidden=HIDDEN, site=l, n=n, keys=keys, xmap=xmap.cpu(),
                            head_seeds=list(M26.HEAD_SEEDS)), HERE / "heads" / f"site{l}_n{n}.pt")
            print(l, n, report[f"site{l}_n{n}"], flush=True)
    (HERE / "heads.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
