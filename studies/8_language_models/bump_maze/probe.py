"""Belief probes (PLAN.md, part B): from the residual stream at the last transcript token, at every layer, to the
consistent set (membership of each of the 140 states; multi-label logistic probe, AUC and exact-set recovery) and to
its size (ridge on log size), for the fine-tuned language model, the base model before fine-tuning, and the
from-scratch transformer; an MLP probe at the best layer as the ceiling. Decisions from the exact optimum's
trajectories, held out by episode.

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_maze/probe.py
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zero_shot import SEED, SIZE, LOOPS, optimum_decisions          # noqa: E402
from train import LLM, LLMFormat, StepTransformer, step_tokens          # noqa: E402

N_DEC = 8192


@torch.no_grad()
def llm_states(model_dir, w, dec, device, batch=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map=device).eval(); fmt = LLMFormat(tok, w)
    X = None
    for b in range(0, len(dec), batch):
        seqs = [fmt.prompt(d["acts"], d["obss"]) for d in dec[b:b + batch]]; L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id); att = torch.zeros(len(seqs), L, dtype=torch.long)
        for r, s in enumerate(seqs): ids[r, L - len(s):] = torch.tensor(s); att[r, L - len(s):] = 1
        hs = model(input_ids=ids.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
        last = torch.stack([h[:, -1].float() for h in hs], 1).cpu()           # [n, layers + 1, d]
        X = last if X is None else torch.cat([X, last])
    del model; torch.cuda.empty_cache(); return X


@torch.no_grad()
def tfm_states(path, w, dec, device):
    model = StepTransformer(w.H).to(device); model.load_state_dict(torch.load(path, map_location=device)); model.eval()
    acts = torch.zeros(len(dec), w.H, dtype=torch.long); obss = torch.zeros_like(acts); ln = torch.tensor([len(d["acts"]) for d in dec])
    for i, d in enumerate(dec): acts[i, :len(d["acts"])] = torch.tensor(d["acts"]); obss[i, :len(d["obss"])] = torch.tensor(d["obss"])
    _, rec = model(step_tokens(acts, obss, ln, w.H).to(device), record=True)
    return torch.stack([r[torch.arange(len(dec)), ln.to(device)].float() for r in rec], 1).cpu()


def fit_probe(X, Y, tr, te, kind="logistic", steps=800, hidden=0, seed=0):
    """A probe from X [n, d] to Y [n, k] (bool for logistic, float for ridge-like), fitted by LBFGS on tr, scored on te.
    Returns predictions on te."""
    torch.manual_seed(seed); dev = "cuda"
    X = X.float().to(dev); mu, sd = X[tr].mean(0), X[tr].std(0).clamp(min=1e-3); Z = (X - mu) / sd; Y = Y.to(dev)
    net = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.GELU(), nn.Linear(hidden, Y.shape[1])) if hidden else nn.Linear(Z.shape[1], Y.shape[1]); net = net.to(dev)
    loss_fn = (lambda p, y: nn.functional.binary_cross_entropy_with_logits(p, y.float())) if kind == "logistic" else (lambda p, y: ((p - y.float()) ** 2).mean())
    opt = torch.optim.LBFGS(net.parameters(), max_iter=steps, line_search_fn="strong_wolfe", history_size=20)
    with torch.enable_grad():
        def closure():
            opt.zero_grad(); l = loss_fn(net(Z[tr]), Y[tr]) + 1e-4 * sum((p ** 2).sum() for p in net.parameters()); l.backward(); return l
        opt.step(closure)
    with torch.no_grad(): return net(Z[te]).cpu()


def membership_scores(P, M):
    """P logits [n, S], M bool [n, S]: mean per-decision AUC (where both classes occur), the share of decisions whose
    thresholded set equals the exact set, and the mean Jaccard."""
    aucs, exact, jac = [], [], []
    for p, m in zip(P.numpy(), M.numpy()):
        if m.any() and (~m).any():
            pos, neg = p[m], p[~m]; aucs.append(((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()))
        pred = p > 0; exact.append(bool((pred == m).all())); jac.append((pred & m).sum() / max((pred | m).sum(), 1))
    return dict(auc=float(np.mean(aucs)), exact=float(np.mean(exact)), jaccard=float(np.mean(jac)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=str(HERE / "runs" / "llm_100000" / "model")); ap.add_argument("--tfm", default=str(HERE / "runs" / "tfm_100000" / "model.pt"))
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n_dec", type=int, default=N_DEC); ap.add_argument("--out", default=str(HERE / "probe.json")); ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    w = BMZ.make(SEED, SIZE, LOOPS); sol = BMZ.solve(w); t0 = time.time()
    dec_all = optimum_decisions(w, sol, 1024, 4000)
    # episode ids: optimum_decisions lists, for each step, the alive episodes in order; recover by replaying the alive pattern
    ep_ids = []; alive = np.ones(1024, dtype=bool); step_counts = {}
    for d in dec_all: step_counts.setdefault(d["step"], 0); step_counts[d["step"]] += 1
    # the same episodes in the same order per step: episode index = the k-th alive episode at that step
    e = BMZ.Env(w, 1024, torch.Generator().manual_seed(4000))
    for t in range(w.H):
        al = torch.nonzero(~e.done).squeeze(1).tolist(); ep_ids += al
        a_opt, _ = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); e.step(torch.as_tensor(a_opt))
        if e.done.all(): break
    assert len(ep_ids) == len(dec_all)
    rng = np.random.default_rng(0); pick = rng.choice(len(dec_all), min(a.n_dec, len(dec_all)), replace=False)
    dec = [dec_all[i] for i in pick]; ep = np.array([ep_ids[i] for i in pick]); te = torch.as_tensor(ep % 8 == 0); tr = ~te
    M = torch.stack([torch.as_tensor(d["mask"]) for d in dec]); size = M.sum(1).float(); logsize = size.log()[:, None]
    print(f"{len(dec)} decisions, {int(tr.sum())} fit / {int(te.sum())} test; set size mean {size.mean():.1f}, known {float((size == 1).float().mean()):.2f} {time.time() - t0:.0f}s", flush=True)
    res = dict(n=len(dec), models={})
    sources = {"llm_finetuned": lambda: llm_states(a.llm, w, dec, a.device), "llm_base": lambda: llm_states(LLM, w, dec, a.device), "tfm_scratch": lambda: tfm_states(a.tfm, w, dec, a.device)}
    for name, fn in sources.items():
        if name in a.skip: continue
        if name == "tfm_scratch" and not Path(a.tfm).exists(): continue
        if name == "llm_finetuned" and not Path(a.llm).exists(): continue
        X = fn(); rows = []
        for l in range(X.shape[1]):
            P = fit_probe(X[:, l], M, tr, te); m = membership_scores(P, M[te])
            S = fit_probe(X[:, l], logsize, tr, te, kind="ridge"); r2 = float(1 - ((S - logsize[te]) ** 2).sum() / ((logsize[te] - logsize[te].mean()) ** 2).sum())
            rows.append(dict(layer=l, **m, size_r2=r2)); print(f"  {name} layer {l}: membership AUC {m['auc']:.3f} exact {m['exact']:.3f} jaccard {m['jaccard']:.3f} | log size R² {r2:.3f} {time.time() - t0:.0f}s", flush=True)
        best = max(rows, key=lambda r: r["auc"])["layer"]
        P = fit_probe(X[:, best], M, tr, te, hidden=256); mm = membership_scores(P, M[te])
        res["models"][name] = dict(layers=rows, best_layer=best, mlp_at_best=mm); print(f"  {name}: best layer {best}; MLP probe there AUC {mm['auc']:.3f} exact {mm['exact']:.3f}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
