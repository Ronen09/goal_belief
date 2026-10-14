"""Part D2 of PLAN.md: the prefix patch (prior vs recompute). Pairs of histories A, B with prefixes of the same length
(and the same token count) whose sets differ, the same three-step suffix, and different optimal actions at the end.
In A's run the residual of the newline token ending the prefix is replaced by B's, at several layers, against
replacing all prefix tokens and one middle prefix token. The 100k model. Writes prefix.json.

    PYTHONPATH=/workspace/goal_belief:. .venv/bin/python studies/8_language_models/bump_policy/prefix.py
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ
from common import HERE, RUNS, Runner, world, load_sets, boot_mean, key

LAYERS = [4, 8, 12, 16, 18, 20, 22]
N_PAIRS, SUFFIX = 1024, 3


def run_hist(w, acts, obss):
    m = np.zeros(w.n_states, dtype=bool); m[w.start_states] = True; ms = []
    for a_, o_ in zip(acts, obss):
        m = BMZ.step_belief(w, m, a_, o_); ms.append(m.copy())
        if not m.any(): return None
    return ms


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda"); ap.add_argument("--model", default="llm_100000"); a = ap.parse_args()
    t0 = time.time(); w, sol = world(); S = load_sets(); n = len(S["keys"]); Qd = {k: sol["Q"][sol["sets"][k]] for k in S["keys"]}
    hist = torch.load(HERE / "histories.pt") if (HERE / "histories.pt").exists() else None
    pool = [h for k in S["keys"] for h in (hist[k] if hist else [(S["acts"][S["keys"].index(k)], S["obss"][S["keys"].index(k)])])]; rng = np.random.default_rng(0)
    long = [h for h in pool if len(h[0]) >= 4 + SUFFIX]; byp = {}
    for h in pool:
        for p in range(4, 13):
            if len(h[0]) >= p: byp.setdefault((p, sum(1 for x in h[0][:p] if x == 2)), []).append((h[0][:p], h[1][:p]))
    pairs = []; tries = 0
    while len(pairs) < N_PAIRS and tries < 200000:
        tries += 1; h = long[rng.integers(len(long))]; p = int(rng.integers(4, min(12, len(h[0]) - SUFFIX) + 1)); A = (h[0][: p + SUFFIX], h[1][: p + SUFFIX]); suf = (A[0][p:], A[1][p:])
        cands = byp.get((p, sum(1 for x in A[0][:p] if x == 2)), [])
        if not cands: continue
        pre = cands[rng.integers(len(cands))]
        if pre == (A[0][:p], A[1][:p]): continue
        mA = run_hist(w, *A); B = (pre[0] + suf[0], pre[1] + suf[1]); mB = run_hist(w, *B)
        if mA is None or mB is None or key(mA[p - 1]) == key(mB[p - 1]): continue
        qa, qb = Qd[key(mA[-1])], Qd[key(mB[-1])]; oa, ob = qa >= qa.max() - 1e-9, qb >= qb.max() - 1e-9
        if (oa & ob).any(): continue
        pairs.append(dict(A=A, B=B, p=p, optA=oa, optB=ob))
    print(f"{len(pairs)} pairs (prefix length 4–12, suffix {SUFFIX}; sets after the prefix differ, optimal sets at the end disjoint) {time.time() - t0:.0f}s", flush=True)
    R = Runner(str(RUNS / a.model / "model"), w, a.device)
    hA = [q["A"] for q in pairs]; hB = [q["B"] for q in pairs]
    # positions from the end: every token of the transcript prefix (steps 1..p) and the newline ending step p, the middle step's newline
    posA = [R.positions(q["A"][0]) for q in pairs]                         # newline of each step, from the end
    def prefix_positions(q, pp):
        f = R.fmt; L = len(f.prefix); pos = []
        T = L + sum(len(f.step_ids[i]) + 1 + (len(f.obs_ids[0]) if a_ == 2 else 0) + 1 for i, a_ in enumerate(q["A"][0])) + len(f.step_ids[len(q["A"][0])])
        t = L
        for i in range(q["p"]):
            k = len(f.step_ids[i]) + 1 + (len(f.obs_ids[0]) if q["A"][0][i] == 2 else 0) + 1; pos += [t + j - T for j in range(k)]; t += k
        return pos
    allpos = [prefix_positions(q, pp) for q, pp in zip(pairs, posA)]; kmax = max(len(x) for x in allpos)
    last = torch.tensor([pp[q["p"] - 1] for q, pp in zip(pairs, posA)]); mid = torch.tensor([pp[q["p"] // 2 - 1] for q, pp in zip(pairs, posA)])
    posT = torch.full((len(pairs), kmax), -10 ** 6, dtype=torch.long)
    for r, x in enumerate(allpos): posT[r, : len(x)] = torch.tensor(x)
    zA, _ = R.run(hA); zB, rB = R.run(hB, record=LAYERS, pos=posT); dA, dB = zA.argmax(1), zB.argmax(1); optB = torch.as_tensor(np.stack([q["optB"] for q in pairs])); optA = torch.as_tensor(np.stack([q["optA"] for q in pairs]))
    inf = dA != dB; res = dict(n_pairs=len(pairs), natural_differ=float(inf.double().mean()), model_optimal_A=float(optA[torch.arange(len(pairs)), dA].double().mean()), model_optimal_B=float(optB[torch.arange(len(pairs)), dB].double().mean()), layers=[])
    print(f"natural decisions differ in {float(inf.double().mean()):.3f}; model optimal on A {res['model_optimal_A']:.3f}, on B {res['model_optimal_B']:.3f} {time.time() - t0:.0f}s", flush=True)
    for l in LAYERS:
        row = dict(layer=l)
        for name in ("last_prefix_token", "all_prefix_tokens", "middle_prefix_token"):
            if name == "last_prefix_token":
                sel = [int(allpos[r].index(int(last[r]))) for r in range(len(pairs))]; vals = rB[l][torch.arange(len(pairs)), torch.tensor(sel)]
                # positions differ per row: patch row by row via per-row position support -> emulate with groups of equal position
                zp = patch_rows(R, hA, l, last, vals)
            elif name == "middle_prefix_token":
                sel = [int(allpos[r].index(int(mid[r]))) for r in range(len(pairs))]; vals = rB[l][torch.arange(len(pairs)), torch.tensor(sel)]; zp = patch_rows(R, hA, l, mid, vals)
            else:
                zp = patch_rows_multi(R, hA, l, allpos, rB[l])
            dp = zp.argmax(1); toB = (dp == dB)[inf]; into = optB[torch.arange(len(pairs)), dp]; tr = (((zp - zA) * (zB - zA)).sum(1) / ((zB - zA) ** 2).sum(1).clamp(min=1e-9))[inf].mean()
            row[name] = dict(to_B=float(toB.double().mean()), to_B_ci=boot_mean(toB.numpy()), into_opt_B=float(into.double().mean()), transfer=float(tr))
        res["layers"].append(row); print(f"  layer {l}: last prefix token → B's decision {row['last_prefix_token']['to_B']:.3f} (into π*(B) {row['last_prefix_token']['into_opt_B']:.3f}); all prefix tokens {row['all_prefix_tokens']['to_B']:.3f}; middle token {row['middle_prefix_token']['to_B']:.3f} {time.time() - t0:.0f}s", flush=True)
        (HERE / "prefix.json").write_text(json.dumps(res, indent=1))


def patch_rows(R, hA, l, pos, vals):
    """Patch one position (from the end, per row) at layer l; rows grouped by position."""
    out = torch.zeros(len(hA), 3); pos = pos.tolist()
    for p in sorted(set(pos)):
        idx = [r for r in range(len(hA)) if pos[r] == p]; z, _ = R.run([hA[r] for r in idx], patch=[(l, p, vals[idx])]); out[idx] = z
    return out


def patch_rows_multi(R, hA, l, allpos, rec):
    """Patch every prefix token (per-row position lists) at layer l; rows grouped by identical position lists."""
    out = torch.zeros(len(hA), 3); groups = {}
    for r, x in enumerate(allpos): groups.setdefault(tuple(x), []).append(r)
    for x, idx in groups.items():
        patch = [(l, p, rec[idx][:, j]) for j, p in enumerate(x)]; z, _ = R.run([hA[r] for r in idx], patch=patch); out[idx] = z
    return out


if __name__ == "__main__":
    main()
