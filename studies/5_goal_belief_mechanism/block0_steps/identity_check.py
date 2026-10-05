"""The block-0-steps experiment: the registered check N compared edits at attn and mid built from each step's own fit. Affine is a ridge
regression that also shrinks the (goal, length) offsets, which differ between the steps by the embedding, so the two
fits differ slightly (a deviation from the plan, found after the run). This check applies the identical edit vectors
(mid's tables) at both patch points.

    .venv/bin/python studies/5_goal_belief_mechanism/block0_steps/identity_check.py            # writes identity_check.json
"""

from __future__ import annotations

import importlib.util, json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r35run", HERE / "run.py")
R35 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R35)
R32, R27, M26 = R35.R32, R35.R27, R35.M26


class Shifted:
    """mid's tables, applied to the attention output: the same edit vectors, states shifted by the embedding."""

    def __init__(self, T, Za):
        self.T, self.Za = T, Za.double()

    def edits(self, a, b, g):
        e = self.T.edits(a, b, g)
        shift = self.T.Z[a, g] - self.Za[a, g]
        sh = lambda x: x - shift if not isinstance(x, list) else [y - shift for y in x]
        out = {k: sh(v) for k, v in e.items()}
        out["whole"] = self.Za[b, g]
        return out


def main():
    dev = "cuda"
    R32.KINDS = R35.KINDS
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    ra, rb = pairs.main["r"], pairs.main["d"]
    bel = t.belief[ctx.node].double()
    _, pid = torch.unique(torch.round(bel * 1e9), dim=0, return_inverse=True)
    all_h = torch.arange(len(ctx.L), device=dev)
    res = {}
    for s in range(10):
        net, _ = M26.load("reward", s, t, dev, M26.R23 / "runs")
        st = [R35.steps(net, ctx, all_h, g) for g in range(3)]
        Za, Zm = torch.stack([x["attn"] for x in st], 1), torch.stack([x["mid"] for x in st], 1)
        T = R35.Tables(Zm, ctx, pid)
        R32.SITE = ("resid_mid", 0); Pm = R32.outputs(net, ctx, T, ra, rb)[0]
        R32.SITE = ("attn", 0); Pa = R32.outputs(net, ctx, Shifted(T, Za), ra, rb)[0]
        res[f"seed{s}"] = max(float((Pa[k] - Pm[k]).abs().max()) for k in R35.KINDS)
        print(f"seed{s} identical edit vectors at attn and mid: largest |Δp| {res[f'seed{s}']:.2e}", flush=True)
    (HERE / "identity_check.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
