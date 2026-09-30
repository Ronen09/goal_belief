"""Round 24: the prediction head on identical-posterior histories (round 22's pairs, round 23's models).

    .venv/bin/python rounds/r24_head_consistency/run.py            # writes results.json
"""

from __future__ import annotations

import importlib.util, json, sys
from pathlib import Path

import torch

from goalgeo import mazeaux as AX, mazebelief as MB, mazegraph as MG

HERE = Path(__file__).resolve().parent
R18, R23 = HERE.parent / "r18_maze_belief", HERE.parent / "r23_obs_prediction"
sys.path.insert(0, str(R18))
import train as T18                                                  # noqa: E402

spec = importlib.util.spec_from_file_location("r22run", HERE.parent / "r22_pair_types" / "run.py")
R22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R22)

ARMS = ("aux1", "aux01")
tv = R22.tv


def normed(p):
    z = p.sum(-1, keepdim=True)
    return p / z.clamp(min=1e-12), z.squeeze(-1) > 1e-6


@torch.no_grad()
def analyse(net, data, p_obs):
    t = data.t
    out = {}
    for name in ("A", "C"):
        v = data.types[name]
        rows = torch.arange(len(v["r"]), device=data.dev)
        gpos, last = 1 + v["L"], v["L"]
        acc = {k: [] for k in ("d_head", "d_exact", "err", "pol", "w", "greedy_diff", "onp")}
        for g in range(3):
            gg = torch.full_like(v["nA"], g)
            lgA, _, obA = net(data.seq(v["r"], v["L"], g), aux=True)
            lgB, _, obB = net(data.seq(v["d"], v["L"], g), aux=True)
            hA, hB = obA[rows, gpos].softmax(-1), obB[rows, gpos].softmax(-1)                  # [N, 4, symbols]
            eA, okA = normed(p_obs[t.reveal[v["nA"], gg]]); eB, okB = normed(p_obs[t.reveal[v["nB"], gg]])
            pA, pB = lgA[rows, gpos].softmax(-1), lgB[rows, gpos].softmax(-1)
            acc["d_head"].append(tv(hA, hB)); acc["d_exact"].append(tv(eA, eB)); acc["err"].append(0.5 * (tv(hA, eA) + tv(hB, eB)))
            acc["w"].append((okA & okB).float()); acc["pol"].append(tv(pA, pB)); acc["greedy_diff"].append(pA.argmax(1) != pB.argmax(1))
            acc["onp"].append(torch.nn.functional.one_hot(pA.argmax(1), 4).float())     # post hoc: the recipient's greedy move
            if g == 0:                                                   # the last prefix token does not depend on the goal
                qA, qB = obA[rows, last].softmax(-1), obB[rows, last].softmax(-1)
                xA, kA = normed(p_obs[v["nA"]]); xB, kB = normed(p_obs[v["nB"]])
                pre = {}
                for label, sel in (("lengths_2_3", v["L"] < t.max_prefix), ("length_4_untrained", v["L"] == t.max_prefix)):
                    w = (kA & kB).float() * sel[:, None] if label == "lengths_2_3" else sel[:, None].float().expand(-1, 4)
                    m = lambda x: float((x * w).sum() / w.sum().clamp(min=1))
                    pre[label] = dict(n=int(sel.sum()), d_head=m(tv(qA, qB)), **(dict(d_exact=m(tv(xA, xB)), error=m(0.5 * (tv(qA, xA) + tv(qB, xB)))) if label == "lengths_2_3" else {}))
        d, e, err, w = (torch.cat(acc[k]) for k in ("d_head", "d_exact", "err", "w"))
        pol, gd = torch.cat(acc["pol"]), torch.cat(acc["greedy_diff"])
        m = lambda x, ww=w: float((x * ww).sum() / ww.sum().clamp(min=1))
        res = dict(cells=int(len(pol)), d_head=m(d), d_exact=m(e), error=m(err), policy_tv=float(pol.mean()), greedy_differs=float(gd.float().mean()),
                   d_head_p95=float(torch.quantile((d * w).sum(1) / w.sum(1).clamp(min=1), 0.95)), prefix_token=pre,
                   d_head_where_greedy_differs=m(d, w * gd[:, None]), d_head_where_greedy_same=m(d, w * ~gd[:, None]),
                   error_where_greedy_differs=m(err, w * gd[:, None]), error_where_greedy_same=m(err, w * ~gd[:, None]))
        # post hoc (the head is trained only on moves that were taken; after the reveal those are the policy's own):
        # the same measures for the recipient's greedy move alone, and for the other three candidates
        on = torch.cat(acc["onp"])
        res["greedy_move"] = dict(d_head=m(d, w * on), d_exact=m(e, w * on), error=m(err, w * on), d_head_same_greedy=m(d, w * on * ~gd[:, None]))
        res["other_moves"] = dict(d_head=m(d, w * (1 - on)), d_exact=m(e, w * (1 - on)), error=m(err, w * (1 - on)))
        out[name] = res
    out["relative_inconsistency"] = dict(head=out["A"]["d_head"] / out["C"]["d_head"], policy=out["A"]["policy_tv"] / out["C"]["policy_tv"])
    return out


def main():
    dev = "cuda"
    t = T18.tables(dev)
    p_obs = torch.tensor(MG.cached(MB.cross_maze(), R18 / "cache" / "graph.npz").p_obs, dtype=torch.float32, device=dev)
    data = R22.Data(t)
    res = dict(types=data.info, runs={})
    for arm in ARMS:
        res["runs"][arm] = {}
        for run in sorted((R23 / "runs" / arm).glob("seed*"), key=lambda p: int(p.name[4:])):
            net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(dev).eval()
            net.load_state_dict(torch.load(sorted((run / "ckpt").glob("u*.pt"))[-1]))
            res["runs"][arm][run.name] = r = analyse(net, data, p_obs)
            print(arm, run.name, f"A: D_same {r['A']['d_head']:.4f} error {r['A']['error']:.4f} policy {r['A']['policy_tv']:.3f} | C: D {r['C']['d_head']:.3f} exact {r['C']['d_exact']:.3f} policy {r['C']['policy_tv']:.3f} | "
                  f"relative head {r['relative_inconsistency']['head']:.3f} policy {r['relative_inconsistency']['policy']:.3f}", flush=True)
            (HERE / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
