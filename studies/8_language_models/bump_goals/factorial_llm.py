"""Goal swaps in the fine-tuned language model's residual stream (PLAN.md, part B): quartets of two same-step
histories x two goals; the decision token's residual entering every layer patched from each other cell (12
directions), scored as in the factorial-patch experiment; at the best layer for the goal swap, the state-level mean
goal offset per step and a linear map from the state under A to the state under B against the natural swap.

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_goals/factorial_llm.py --run runs/llm_40000
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze")); sys.path.insert(0, str(HERE))
from design import worlds_and_solutions, K                                           # noqa: E402
from train_goals import LLMFormatGoals, pad_left                                      # noqa: E402
from additive import decisions_with_all_goals                                         # noqa: E402

N_PAIRS, N_FIT, BATCH = 1024, 6144, 32
ROLES = ("h1A", "h1B", "h2A", "h2B")


class Patched:
    """Run the model with the decision token's (last position's) residual entering layer l replaced; records the
    residual entering every layer at the last position."""
    def __init__(self, model, fmt, device):
        self.model, self.fmt, self.dev = model, fmt, device
        self.n_layers = len(model.model.layers); self._patch = None; self._rec = None
        model.model.embed_tokens.register_forward_hook(lambda m, i, o: self._hook(0, o))
        for l, layer in enumerate(model.model.layers): layer.register_forward_hook(lambda m, i, o, l=l: self._hook(l + 1, o))

    def _hook(self, l, out):
        x = out[0] if isinstance(out, tuple) else out
        if self._rec is not None: self._rec[l] = x[:, -1].float().clone()
        if self._patch is not None and self._patch[0] == l:
            x = x.clone(); x[:, -1] = self._patch[1].to(x.dtype)
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return None

    @torch.no_grad()
    def run(self, ids, att, patch=None, record=False):
        self._patch, self._rec = patch, ({} if record else None)
        h = self.model.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state[:, -1]
        z = self.model.lm_head(h).float()[:, self.fmt.act_tok.to(self.dev)].double(); z = z - z.mean(1, keepdim=True)
        rec = self._rec; self._patch, self._rec = None, None
        return z.cpu(), (torch.stack([rec[l] for l in range(self.n_layers + 1)], 1).cpu() if record else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="runs/llm_40000"); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--n_pairs", type=int, default=N_PAIRS); ap.add_argument("--n_fit", type=int, default=N_FIT); ap.add_argument("--layers", type=int, nargs="*", default=None); a = ap.parse_args()
    run = HERE / a.run; out = Path(a.out) if a.out else run / "factorial.json"; t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    free, goals, worlds, sols = worlds_and_solutions(); H = worlds[0].H
    tok = AutoTokenizer.from_pretrained(run / "model"); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(run / "model", dtype=torch.bfloat16, device_map=a.device).eval(); fmt = LLMFormatGoals(tok, worlds); P = Patched(model, fmt, a.device)
    # decisions after the first step, same-step pairs with different goal-free sets
    dec = [d for d in decisions_with_all_goals(worlds, sols, 256, 700) if d["step"] >= 1]; rng = np.random.default_rng(1)
    by_step = {}
    for i, d in enumerate(dec): by_step.setdefault(d["step"], []).append(i)
    pairs = []
    while len(pairs) < a.n_pairs + a.n_fit:
        i = int(rng.integers(len(dec))); c = by_step[dec[i]["step"]]; j = c[int(rng.integers(len(c)))]
        if j != i and (dec[i]["acts"], dec[i]["obss"]) != (dec[j]["acts"], dec[j]["obss"]): pairs.append((i, j))
    fit_pairs, test_pairs = pairs[: a.n_fit], pairs[a.n_fit:]
    def states(idx, g):
        """Natural logits [n, 3] and residuals entering every layer at the decision token [n, L + 1, d]."""
        Z, X = [], []
        for b in range(0, len(idx), BATCH):
            ids, att = pad_left([fmt.prompt(dec[i]["acts"], dec[i]["obss"], g) for i in idx[b:b + BATCH]], tok.pad_token_id); z, x = P.run(ids, att, record=True); Z.append(z); X.append(x)
        return torch.cat(Z), torch.cat(X)
    def patched_logits(idx, g, l, new):
        Z = []
        for b in range(0, len(idx), BATCH):
            ids, att = pad_left([fmt.prompt(dec[i]["acts"], dec[i]["obss"], g) for i in idx[b:b + BATCH]], tok.pad_token_id); z, _ = P.run(ids, att, patch=(l, new[b:b + BATCH])); Z.append(z)
        return torch.cat(Z)
    gpairs = [(A, B) for A in range(K) for B in range(A + 1, K)]
    i1 = [p[0] for p in test_pairs]; i2 = [p[1] for p in test_pairs]; n = len(test_pairs)
    nat = {}                                                                       # (history index set name, goal) -> (logits, residuals)
    for g in range(K): nat[("h1", g)] = states(i1, g); nat[("h2", g)] = states(i2, g)
    print(f"{n} test quartet pairs x {len(gpairs)} goal pairs; natural runs recorded {time.time() - t0:.0f}s", flush=True)
    Lsites = P.n_layers + 1
    res = dict(n_pairs=n, layers=Lsites, by_layer=[])
    agg = {}
    def add(k, v): agg.setdefault(k, []).append(v)
    for l in (a.layers if a.layers is not None else range(Lsites)):
        row = {}
        for (A, B) in gpairs:
            cells = {0: ("h1", A), 1: ("h1", B), 2: ("h2", A), 3: ("h2", B)}
            Lc = {r: nat[c][0] for r, c in cells.items()}; dc = {r: L_.argmax(1) for r, L_ in Lc.items()}; Xc = {r: nat[c][1][:, l] for r, c in cells.items()}
            d4 = (dc[0] != dc[1]) & (dc[0] != dc[2]) & (dc[0] != dc[3]) & (dc[1] != dc[2]) & (dc[1] != dc[3]) & (dc[2] != dc[3])
            for r in range(4):
                hidx = i1 if r < 2 else i2; rg = cells[r][1]
                for s in range(4):
                    if s == r: continue
                    ty = "same_goal" if (r & 1) == (s & 1) else ("same_hist" if (r >> 1) == (s >> 1) else "both")
                    Lhat = patched_logits(hidx, rg, l, Xc[s]); dh = Lhat.argmax(1)
                    qh, qg = ((s >> 1) << 1) | (r & 1), ((r >> 1) << 1) | (s & 1); inf = dc[s] != dc[r]
                    if inf.any():
                        add((l, ty, "to_source"), float((dh == dc[s])[inf].double().mean())); add((l, ty, "kept"), float((dh == dc[r])[inf].double().mean()))
                        diff = Lc[s] - Lc[r]; nn_ = (diff ** 2).sum(1); ok = nn_ > 1e-9; add((l, ty, "transfer"), float((((Lhat - Lc[r]) * diff).sum(1) / nn_.clamp(min=1e-9))[ok].mean()))
                    if ty == "both":
                        for name, q in (("qh", qh), ("qg", qg)):
                            m = dc[q] != dc[r]
                            if m.any(): add((l, ty, "to_" + name), float((dh == dc[q])[m].double().mean()))
                        if d4.sum() >= 5:
                            add((l, ty, "u_source"), float((dh == dc[s])[d4].double().mean())); add((l, ty, "u_qh"), float((dh == dc[qh])[d4].double().mean())); add((l, ty, "u_qg"), float((dh == dc[qg])[d4].double().mean())); add((l, ty, "u_kept"), float((dh == dc[r])[d4].double().mean()))
        row = {f"{ty}_{st}": float(np.mean(agg[(l, ty, st)])) for (ll, ty, st) in list(agg) if ll == l}
        row["layer"] = l; res["by_layer"].append(row)
        print(f"layer {l}: goal swap to source {row.get('same_hist_to_source', float('nan')):.2f}, history swap {row.get('same_goal_to_source', float('nan')):.2f}, both {row.get('both_to_source', float('nan')):.2f} (q_h {row.get('both_to_qh', float('nan')):.2f}, q_g {row.get('both_to_qg', float('nan')):.2f}; unique source {row.get('both_u_source', float('nan')):.2f}) {time.time() - t0:.0f}s", flush=True)
        out.write_text(json.dumps(res, indent=1))
    # at the best layer for the goal swap: the mean offset per step and a linear map, fitted on the fit pairs' decisions
    done_layers = a.layers if a.layers is not None else list(range(Lsites))
    best = done_layers[int(np.argmax([r.get("same_hist_to_source", 0) for r in res["by_layer"]]))]; res["best_layer"] = best
    fidx = sorted(set([p[0] for p in fit_pairs] + [p[1] for p in fit_pairs]))[: a.n_fit]
    Xf = {g: states(fidx, g)[1][:, best] for g in range(K)}; sf = torch.tensor([min(dec[i]["step"], H - 1) for i in fidx])
    st1 = torch.tensor([min(dec[i]["step"], H - 1) for i in i1]); oh = lambda s_: torch.nn.functional.one_hot(s_, H).double()
    def ridge(X, Y, lam=1e-2):
        X1 = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1); return torch.linalg.solve(X1.T @ X1 + lam * torch.eye(X1.shape[1], dtype=X.dtype), X1.T @ Y)
    swaps = {}
    for (A, B) in gpairs:
        for (ra, rb) in ((A, B), (B, A)):
            Lr, Ls = nat[("h1", ra)][0], nat[("h1", rb)][0]; dr, ds = Lr.argmax(1), Ls.argmax(1); dep = dr != ds
            XA, XB = nat[("h1", ra)][1][:, best].double(), nat[("h1", rb)][1][:, best].double()
            W = ridge(torch.cat([Xf[ra].double(), oh(sf)], 1), Xf[rb].double()); pred = torch.cat([XA, oh(st1), torch.ones(n, 1, dtype=torch.float64)], 1) @ W
            Wg = ridge(oh(sf), Xf[rb].double() - Xf[ra].double()); offset = XA + torch.cat([oh(st1), torch.ones(n, 1, dtype=torch.float64)], 1) @ Wg
            gr = torch.Generator().manual_seed(3); rnd = torch.randn(XA.shape, generator=gr, dtype=torch.float64); rnd = XA + rnd / rnd.norm(dim=1, keepdim=True) * (XB - XA).norm(dim=1, keepdim=True)
            for name, ns in (("natural", XB), ("linear_map", pred), ("mean_offset", offset), ("random", rnd)):
                Lhat = patched_logits(i1, ra, best, ns.float()); r2 = float(1 - ((ns - XB) ** 2).sum() / ((XB - XB.mean(0)) ** 2).sum())
                swaps.setdefault(name, {"switch": [], "r2": [], "transfer": []})
                swaps[name]["switch"].append(float((Lhat.argmax(1) == ds)[dep].double().mean())); swaps[name]["r2"].append(r2)
                diff = Ls - Lr; swaps[name]["transfer"].append(float((((Lhat - Lr) * diff).sum(1) / (diff ** 2).sum(1).clamp(min=1e-9)).mean()))
    res["goal_swap_at_best"] = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in swaps.items()}
    print(f"best layer {best}: " + " | ".join(f"{k}: switch {v['switch']:.3f} transfer {v['transfer']:.2f} R² {v['r2']:.2f}" for k, v in res["goal_swap_at_best"].items()) + f" {time.time() - t0:.0f}s")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
