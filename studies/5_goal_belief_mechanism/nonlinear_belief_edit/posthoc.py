"""The nonlinear-belief-edit experiment, post hoc (after the registered results): is the goal-conditioned encodings' advantage on goal-dependent
pairs specific to the recipient's goal?

    .venv/bin/python studies/5_goal_belief_mechanism/nonlinear_belief_edit/posthoc.py            # writes posthoc.json

The encodings are refitted exactly as in run.py (same seeds). Edits under goal g:

    mlp_goal          f(b_B, g) - f(b_A, g)                  the recipient's goal (as in run.py)
    mlp_goal_wrong    f(b_B, g') - f(b_A, g') for both g' != g (averaged)       another goal's map
    table_goal_wrong  the same with the per-posterior-and-goal table
If the advantage is a goal x belief interaction that the policy reads, another goal's map should lose it, and may do
worse than the shared map.
"""

from __future__ import annotations

import importlib.util, json, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r34run", HERE / "run.py")
R34 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R34)
R32, R27, M26 = R34.R32, R34.R27, R34.M26
KINDS = ("none", "whole", "mlp", "mlp_goal", "mlp_goal_wrong", "table", "table_goal", "table_goal_wrong")


class Wrong:
    def __init__(self, enc):
        self.enc = enc

    def edits(self, a, b, g):
        e = self.enc
        zA = e.Z[a, g]
        wrong = [h for h in range(3) if h != g]
        return dict(none=zA, whole=e.Z[b, g], mlp=zA + e.code[b] - e.code[a], mlp_goal=zA + e.code_g[b, g] - e.code_g[a, g],
                    mlp_goal_wrong=[zA + e.code_g[b, h] - e.code_g[a, h] for h in wrong], table=zA + e.mu[b] - e.mu[a],
                    table_goal=zA + e.mu_g[b, g] - e.mu_g[a, g], table_goal_wrong=[zA + e.mu_g[b, h] - e.mu_g[a, h] for h in wrong])


def main():
    dev, t0 = "cuda", time.time()
    R32.KINDS = KINDS
    t = M26.T18.tables(dev)
    ctx = M26.Ctx(t)
    pairs = R27.Pairs(ctx)
    m = pairs.main
    ra, rb = m["r"], m["d"]
    res = {}
    for s in range(10):
        net, ck = M26.load("reward", s, t, dev, M26.R23 / "runs")
        enc = R34.Nonlinear(net, ctx, torch.Generator(device=dev).manual_seed(3300 + s), 3400 + 10 * s)
        P, num, den = R32.outputs(net, ctx, Wrong(enc), ra, rb)
        dj = pairs.disjoint(ra, rb)
        res[f"seed{s}"] = dict(fit_mlp_goal=enc.fit_stats["mlp_goal"]["r2_within_goal_length"], change=R32.summarise(P, num, den, ctx, ra, rb, dj),
                               goal_dependent=R32.goal_dependent(P, ctx, ra, rb))
        gd = res[f"seed{s}"]["goal_dependent"]
        print(f"seed{s} R² mlp_goal {res[f'seed{s}']['fit_mlp_goal']:.3f} | goal-dep: " + " ".join(f"{k} {gd[k]['right_every_change_goal']:.3f}" for k in KINDS) + f" | {time.time() - t0:.0f}s", flush=True)
    (HERE / "posthoc.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
