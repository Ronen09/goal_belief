"""Post hoc (not registered): why does the probe edit close two thirds of the gap? Encoders fitted from other
parametrisations of the belief (log-odds y, probabilities b, both, Q*), the belief-node mean as the ceiling of any
belief-only edit, and the residual by stratum. The trained models of results.json; no training.

    .venv/bin/python studies/2_belief_state/reward_bandit/edit_followup.py            # writes edit_followup.json, edit_followup.md
"""

from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bandit as BD, beliefprobe as BP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as ME                                                 # noqa: E402
import train as TR                                                   # noqa: E402


class Encoder:
    """Decoder H -> Y and reverse-regression encoder Y -> H for any affine target Y; probe_e moves h along the
    encoder until the decoder reads y* (least squares where the target is rank-deficient)."""

    def __init__(self, H, Y):
        W = BP._fit(H, Y); self.Wd, self.cd = W[:-1], W[-1]
        E = BP._fit(Y, H); self.We, self.ce = E[:-1], E[-1]
        self.M_pinv = np.linalg.pinv(self.We @ self.Wd)

    def z(self, h):
        return h @ self.Wd + self.cd

    def probe_e(self, h, y_star):
        return h + (y_star - self.z(h)) @ self.M_pinv.T @ self.We


def gap(net, hA, hE, pB, base):
    p = ME.heads(net, hE)
    return float(1 - ME.js(p, pB).mean() / base.mean()), p


def main():
    dev = "cuda"
    s = BD.Spec(**TR.TASK); t = BD.Sim(BD.Graph(s), dev)
    gen = torch.Generator(device=dev); gen.manual_seed(78)
    e = BD.Env(t, ME.N_EVAL, gen)
    goal, cues, u = e.goal, e.cues, e.u
    res = {}
    for arch in ("tfm", "gru"):
        args = json.load(open(HERE / "runs" / arch / "task.json"))["args"]
        for seed in range(6):
            net = BD.build(arch, s, args.get("d", 128), args.get("layers", 2))
            net.load_state_dict(torch.load(sorted((HERE / "runs" / arch / f"seed{seed}" / "ckpt").glob("u*.pt"))[-1], map_location=dev)); net = net.to(dev).eval()
            r = BD.rollout(net, t, len(goal), None, greedy=True, goal=goal, cues=cues, u=u)
            D = ME.Decisions(t, r, seed)
            S = ME.sites(net, r.tok, torch.as_tensor(D.step[: D.T] + t.n_cue, device=dev), arch)
            H = S["res2" if arch == "tfm" else "h"]
            rng = np.random.default_rng(400 + seed)
            nd = np.nonzero(D.nondeg)[0]
            perm = rng.permutation(len(nd)); A, B = nd, nd[perm]
            same = D.step[A] == D.step[B]; A, B = A[same], B[same]
            hA, hB = H[A], H[B]
            pA, pB = ME.heads(net, hA), ME.heads(net, hB)
            base = ME.js(pA, pB)
            targets = {"y (log-odds)": D.y, "b (probabilities)": D.b[:, :2], "y and b": np.c_[D.y, D.b[:, :2]], "Q*": D.q, "y, b and Q*": np.c_[D.y, D.b[:, :2], D.q]}
            row = {"pairs": int(len(A))}
            edits = {}
            for name, Y in targets.items():
                enc = Encoder(H[nd], Y[nd])
                he = enc.probe_e(hA, Y[B])
                row[f"probe_e from {name}"], edits[name] = gap(net, hA, he, pB, base)
                row[f"probe_e from {name}: decoded target R²"] = BP.r2(Y[B], enc.z(he))
            # the belief-node mean: the ceiling of any belief-only edit
            _, inv = np.unique(D.node, return_inverse=True)
            means = np.zeros((inv.max() + 1, H.shape[1])); cnt = np.zeros(inv.max() + 1)
            np.add.at(means, inv, H); np.add.at(cnt, inv, 1); means /= cnt[:, None]
            row["node mean of B"], p_nm = gap(net, hA, means[inv[B]], pB, base)
            row["A + (node mean of B − node mean of A)"], _ = gap(net, hA, hA + means[inv[B]] - means[inv[A]], pB, base)
            row["swap (B's state)"], _ = gap(net, hA, hB, pB, base)
            # where the residual lies, for the y-encoder edit
            p_e = edits["y (log-odds)"]
            left = ME.js(p_e, pB)
            strata = {"optimal action same at A and B": D.action[A] == D.action[B], "optimal action differs": D.action[A] != D.action[B],
                      "B certain (max b ≥ 0.8)": D.b[B].max(1) >= 0.8, "B uncertain (max b < 0.6)": D.b[B].max(1) < 0.6,
                      "step 0": D.step[B] == 0, "steps 1–2": (D.step[B] >= 1) & (D.step[B] <= 2), "steps 3–5": D.step[B] >= 3}
            row["strata"] = {k: dict(share=float(m.mean()), gap_closed=float(1 - left[m].mean() / base[m].mean()), base_js=float(base[m].mean())) for k, m in strata.items()}
            # the norm of the move: the edit against the true difference of states
            row["edit length / true difference"] = float(np.linalg.norm(edits["y (log-odds)"] - pA, axis=1).mean() / max(np.linalg.norm(pB - pA, axis=1).mean(), 1e-9)) if False else float(
                np.linalg.norm(Encoder(H[nd], D.y[nd]).probe_e(hA, D.y[B]) - hA, axis=1).mean() / np.linalg.norm(hB - hA, axis=1).mean())
            res[f"{arch}/seed{seed}"] = row
            print(f"{arch} seed{seed}: " + " | ".join(f"{k} {v:.2f}" for k, v in row.items() if isinstance(v, float) and "R²" not in k), flush=True)
    (HERE / "edit_followup.json").write_text(json.dumps(res, indent=1))
    f = lambda x: f"{x:.2f}"
    med = lambda arch, k: f(np.median([res[f'{arch}/seed{s}'][k] for s in range(6)])) + " (" + f(min(res[f'{arch}/seed{s}'][k] for s in range(6))) + "–" + f(max(res[f'{arch}/seed{s}'][k] for s in range(6))) + ")"
    L = ["# Why the probe edit closes two thirds of the gap (post hoc)", "", "Generated by `edit_followup.py`. Gap closed = 1 − JS(edited, model on B) / JS(model on A, model on B); median (range) over six seeds.", "",
         "| edit of A's state at the state the heads read | transformer | GRU |", "|---|---|---|"]
    for k in [k for k in res["tfm/seed0"] if isinstance(res["tfm/seed0"][k], float) and "R²" not in k and k != "edit length / true difference"]:
        L.append(f"| {k} | {med('tfm', k)} | {med('gru', k)} |")
    L.append(f"| length of the y-encoder move / length of the true difference hB − hA | {med('tfm', 'edit length / true difference')} | {med('gru', 'edit length / true difference')} |")
    L += ["", "The y-encoder edit by stratum of the pair (gap closed; share of pairs):", "", "| stratum | transformer | GRU |", "|---|---|---|"]
    for k in res["tfm/seed0"]["strata"]:
        g = lambda arch: f(np.median([res[f'{arch}/seed{s}']['strata'][k]['gap_closed'] for s in range(6)])) + f" ({np.median([res[f'{arch}/seed{s}']['strata'][k]['share'] for s in range(6)]):.2f} of pairs)"
        L.append(f"| {k} | {g('tfm')} | {g('gru')} |")
    (HERE / "edit_followup.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
