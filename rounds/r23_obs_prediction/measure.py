"""Round 23: the fixed evaluation (round 22's pairs and code) on every arm and seed.

    .venv/bin/python rounds/r23_obs_prediction/measure.py            # writes results.json

Final checkpoint: round 22's full analysis (natural access and direct route removed; hybrid, whole and rank-13 PCA
patches). Every checkpoint from update 200: the two behavioural outcomes, against the logged greedy regret.
"""

from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import torch

from goalgeo import mazeaux as AX, mazemeasure as MS

HERE = Path(__file__).resolve().parent
R18 = HERE.parent / "r18_maze_belief"
sys.path.insert(0, str(R18))
import measure as M18, train as T18                                  # noqa: E402

spec = importlib.util.spec_from_file_location("r22run", HERE.parent / "r22_pair_types" / "run.py")
R22 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R22)

ARMS = (("reward", 0.0), ("aux1", 1.0), ("aux01", 0.1))
FROM = 200
RANK = 13


@torch.no_grad()
def pca_projector(net, t, bank, fit):
    """The top-13 principal directions of the prefix states entering block 1, on round 18's fitting histories (as round 21)."""
    pr = MS.prefix_rows(t, bank, fit)
    X = MS.collect(net, bank.tok, pr)[0][2 * R22.L_IF][pr["fit"]].double()
    Xc = X - X.mean(0)
    V = torch.linalg.eigh(Xc.T @ Xc / len(Xc))[1].flip(1)[:, :RANK]
    return V @ V.T


@torch.no_grad()
def outcomes(net, data):
    """O1 and O2 under natural access, with no patch: base against hybrid."""
    out = {}
    for name in ("A", "B"):
        v = data.types[name]
        rows = torch.arange(len(v["r"]), device=data.dev)
        gpos = 1 + v["L"]
        tvs, chg, cross, lost, same = [], [], [], [], []
        for g in range(3):
            pb = net(data.seq(v["r"], v["L"], g))[0][rows, gpos].softmax(-1)
            ph = net(data.seq(v["d"], v["L"], g))[0][rows, gpos].softmax(-1)
            opt = R22.opt_set(data.t.Q[data.t.reveal[v["nB"], torch.full_like(v["nB"], g)]])
            s = v["cls"][:, g] == 0
            tvs.append(R22.tv(pb, ph)[s]); chg.append((pb.argmax(1) != ph.argmax(1))[s].float())
            cross.append(((pb - ph) * opt).sum(1).abs()[s])
            lost.append((opt[rows, pb.argmax(1)] & ~opt[rows, ph.argmax(1)])[s].float())
            same.append((pb * opt).sum(1)[s])
        out[name] = dict(tv=float(torch.cat(tvs).mean()), greedy_changed=float(torch.cat(chg).mean()), crossing=float(torch.cat(cross).mean()),
                         greedy_lost=float(torch.cat(lost).mean()), on_optimal=float(torch.cat(same).mean()))
    return out


def slim(r):
    """Round 22's analysis without the per-goal and per-length tables."""
    return {c: {ty: {k: v for k, v in x.items() if k in ("all_goals", "chosen_same", "chosen_disjoint")} for ty, x in r[c].items()} for c in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    t = T18.tables(dev)
    bank, fit = MS.make_bank(t, M18.BANK_SEED, 40000)
    data = R22.Data(t)
    res = dict(types=data.info, runs={})
    for arm, coef in ARMS:
        res["runs"][arm] = {}
        for run in sorted((Path(a.runs) / arm).glob("seed*"), key=lambda p: int(p.name[4:])):
            log = json.load(open(run / "log.json"))["log"]
            net = AX.NetAux(t.n_sym, len(t.goal_cell), t.H, t.max_prefix).to(dev).eval()
            curve = []
            for row in log:
                if row["update"] < FROM:
                    continue
                net.load_state_dict(torch.load(run / "ckpt" / f"u{row['update']:06d}.pt"))
                curve.append(dict(update=row["update"], regret=row["greedy_regret"], **{f"regret_G{g}": row[f"greedy_regret_G{g}"] for g in (1, 2, 3)},
                                  excess_nll=row["obs_nll"] - row["obs_nll_exact"], **{f"{k}_{m}": x for k, v in outcomes(net, data).items() for m, x in v.items()}))
            final = slim(R22.analyse(net, data, pca_projector(net, t, bank, fit)))            # the last checkpoint is loaded
            res["runs"][arm][run.name] = dict(coef=coef, curve=curve, final=final)
            c = curve[-1]
            print(arm, run.name, f"regret {c['regret']:.4f} (G2 {c['regret_G2']:.4f}) excess nll {c['excess_nll']:.3f} O1 {c['A_tv']:.3f} O2 {c['B_greedy_changed']:.3f}", flush=True)
            (Path(a.out) / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
