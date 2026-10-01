"""Round 26: prediction-only backbones. Round 18's transformer, trained only to predict the symbols that follow supplied
moves (k = 1: each of the four moves; k = 2: each of the sixteen move pairs) on goal-free random walks that fill the
context. No goal token, no reward; the policy and value heads are never trained.

    .venv/bin/python rounds/r26_predictive_transfer/train_pred.py --k 1 --seeds 0 1 2 3 4 5 6 7 8 9 --out rounds/r26_predictive_transfer/runs/predict1
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import mazemeasure as MS, mazepred as PR, mazeppo as P

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
HERE = Path(__file__).resolve().parent
R18 = HERE.parent / "r18_maze_belief"
sys.path.insert(0, str(R18))
import train as T18                                                  # noqa: E402

EVAL_BANK_SEED = 260026                                              # evaluation histories during training (not round 22's bank)


def tv(a, b):
    return 0.5 * (a - b).abs().sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1, choices=(1, 2))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024, help="walks per model per update")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-grad", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.quick:
        a.updates, a.batch, a.seeds = 30, 128, a.seeds[:2]
    out, dev, M = Path(a.out), a.device, len(a.seeds)
    t = T18.tables(dev, a.quick)
    nets = []
    for s in a.seeds:
        torch.manual_seed(s)
        nets.append(PR.NetPred(t.n_sym, len(t.goal_cell), t.H, t.max_prefix, k=a.k).to(dev))
        (out / f"seed{s}" / "ckpt").mkdir(parents=True, exist_ok=True)
    stk = PR.StackedPred(nets)
    params = stk.trainable()
    opt = torch.optim.Adam(params, lr=a.lr, eps=1e-5)
    gen = torch.Generator(device=dev); gen.manual_seed(2600 + 10 * a.k + a.seeds[0])
    # evaluation: prefix tokens of held-out histories (every prefix token, lengths 0-4), against the exact prediction
    bank, _ = MS.make_bank(t, EVAL_BANK_SEED, 1024 if a.quick else 8192)
    rows = MS.prefix_rows(t, bank, torch.ones(len(bank.prefix), dtype=torch.bool, device=dev))
    bel = t.belief[rows["node"]].double()
    exact = PR.predictive(t, bel, a.k).float()
    exact1 = PR.predictive(t, bel, 1).float()
    tok_eval = bank.tok.repeat(M, 1, 1)
    save_at = set(T18.checkpoints(a.updates))
    logs, t0 = [[] for _ in a.seeds], time.time()

    @torch.no_grad()
    def evaluate(u):
        lg = stk.pred(tok_eval).view(M, len(bank.prefix), t.L, -1, stk.base.n_out)
        for m, seed in enumerate(a.seeds):
            p = lg[m][rows["hist"], rows["pos"]].softmax(-1)                    # [rows, seq, out]
            row = dict(update=u, walks=u * a.batch, seconds=time.time() - t0, tv=float(tv(p, exact).mean()),
                       nll=float(-(exact * p.clamp(min=1e-12).log()).sum(-1).mean()), nll_exact=float(-(exact * exact.clamp(min=1e-12).log()).sum(-1).mean()),
                       tv_last=float(tv(p, exact)[rows["last"]].mean()))
            if a.k == 2:                                                        # the implied one-step prediction (marginal over the second symbol, first move of each pair)
                p1 = p.view(-1, 4, 4, t.n_sym, t.n_sym).sum(-1).mean(2)
                row["tv_1step"] = float(tv(p1, exact1).mean())
            logs[m].append(row)
            (out / f"seed{seed}" / "log.json").write_text(json.dumps(dict(args=vars(a), seed=seed, log=logs[m]), indent=1))
        print(f"[k={a.k}] u={u} {time.time() - t0:.0f}s TV " + " ".join(f"{l[-1]['tv']:.4f}" for l in logs) +
              " | excess nll " + " ".join(f"{l[-1]['nll'] - l[-1]['nll_exact']:.4f}" for l in logs), flush=True)

    for u in range(a.updates + 1):
        if u in save_at:
            for m, seed in enumerate(a.seeds):
                torch.save(stk.state_dict(m), out / f"seed{seed}" / "ckpt" / f"u{u:06d}.pt")
            evaluate(u)
        if u == a.updates:
            break
        for g_ in opt.param_groups:
            g_["lr"] = a.lr * min(1, (u + 1) / a.warmup)
        tok, _, target = PR.walks(t, a.batch * M, a.k, gen)
        nll = PR.pred_loss(stk.pred(tok), target)                               # [M * batch, L, seq]
        loss = nll.view(M, -1).mean(1).sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        P.clip_per_model(params, a.max_grad)
        opt.step()


if __name__ == "__main__":
    main()
