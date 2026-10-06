"""The interior-goals experiment, post hoc (not in PLAN.md): does the collect arm's policy take its two goals in a
fixed order? For each pair of goals, the share of episodes whose first pickup is that pair's most common first goal
(0.5: no preference; 1: always the same goal first), and how the choice depends on which goal is nearer to the spawn.

    .venv/bin/python studies/7_information_seeking/interior_goals/order.py            # writes order.json
"""
import json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure_collect as MC                                          # noqa: E402
from goalgeo import bigcollect as BC                                  # noqa: E402


def stats(t, r, pair):
    first = r["first"]
    ok = first >= 0
    P = len(t.pairs)
    a, b = t.pairs[pair, 0], t.pairs[pair, 1]
    took_a = (first == a)[ok].float()
    cnt = torch.zeros(P, device=t.dev).index_add_(0, pair[ok], torch.ones_like(took_a))
    sa = torch.zeros(P, device=t.dev).index_add_(0, pair[ok], took_a) / cnt.clamp(min=1)
    fixed = (torch.maximum(sa, 1 - sa) * cnt).sum() / cnt.sum()
    d = t.dist[t.pairs[pair].T, r["spawn"][None]]                                              # [2, N]
    gap = (d[1] - d[0])[ok]                                                                    # > 0: the first-listed goal is nearer
    modal_is_a = (sa >= 0.5)[pair][ok]
    took_modal = torch.where(modal_is_a, took_a, 1 - took_a)
    modal_nearer = torch.where(modal_is_a, gap > 0, gap < 0)
    far = gap.abs() >= 4                                                                       # one goal clearly nearer (4 moves or more)
    f = lambda x, w: float(x[w].mean()) if w.any() else None
    return dict(order_fixedness=float(fixed), takes_modal_when_modal_is_clearly_nearer=f(took_modal, far & modal_nearer),
                takes_modal_when_other_is_clearly_nearer=f(took_modal, far & ~modal_nearer),
                nearer_first_when_clear=f(torch.where(gap > 0, took_a, 1 - took_a), far))


def main():
    torch.set_grad_enabled(False)
    dev = "cuda"
    runs = HERE / "runs" / "collect"
    task = json.load(open(runs / "task.json"))["task"]
    t, _, build = MC.TR.setup("collect", task, dev)
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BC.Env(t, MC.N_EVAL, gen)
    pair, cell = e.pair, e.cell
    res = dict(references={}, runs={})
    for name, fn in (("oracle", BC.oracle), ("qmdp2", BC.qmdp2), ("qmdp", BC.qmdp), ("mls", BC.mls)):
        res["references"][name] = stats(t, MC.episodes(t, MC.behaviour(fn), pair, cell, 79)[1], pair)
        print(name, {k: round(v, 3) for k, v in res["references"][name].items()}, flush=True)
    for s in range(6):
        net = build()
        net.load_state_dict(torch.load(sorted((runs / f"seed{s}" / "ckpt").glob("u*.pt"))[-1], map_location=dev))
        net = net.to(dev).eval()
        res["runs"][f"seed{s}"] = stats(t, MC.episodes(t, MC.natural(net), pair, cell, 79)[1], pair)
        print(f"seed{s}", {k: round(v, 3) for k, v in res["runs"][f"seed{s}"].items()}, flush=True)
    (HERE / "order.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
