"""Training on the four-goal bump maze (PLAN.md, part A): Qwen2.5-0.5B full fine-tuning with the goal named in the
prompt, and a from-scratch transformer / GRU with the goal as the start token; supervision from the exact optimum
under the episode's goal. Evaluation per goal: closed loop on fixed starts and teacher-forced decision optimality.

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_goals/train_goals.py --model llm --n_train 40000 --save
"""

from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze")); sys.path.insert(0, str(HERE))        # this directory first: its design.py, not bump_maze's
from design import worlds_and_solutions, K                                           # noqa: E402
from zero_shot import CANDS, optimum_decisions                                        # noqa: E402
from train import LLM, StepTransformer, StepGRU, EPS, TEST_SEED                       # noqa: E402

N_TEST = 2048


# ------------------------------------------------------------------ data

def generate(worlds, sols, n_ep, seed, eps, batch=4096):
    H = worlds[0].H; A, O, Lb, Ln, G = [], [], [], [], []
    per = n_ep // len(worlds)
    for gi, (w, sol) in enumerate(zip(worlds, sols)):
        gen = torch.Generator().manual_seed(seed * 100 + gi)
        for b in range(0, per, batch):
            n = min(batch, per - b); e = BMZ.Env(w, n, gen)
            acts, obss, labs = torch.zeros(n, H, dtype=torch.long), torch.zeros(n, H, dtype=torch.long), torch.full((n, H), -1, dtype=torch.long)
            for t in range(H):
                a_opt = torch.as_tensor(BMZ.optimal_action(w, sol, e.filter.mask.numpy())[0])
                a = torch.where(torch.rand(n, generator=gen) < eps, torch.randint(3, (n,), generator=gen), a_opt)
                alive = ~e.done; labs[alive, t] = a_opt[alive]; _, o = e.step(a); acts[:, t] = a; obss[:, t] = o
                if e.done.all(): break
            A.append(acts); O.append(obss); Lb.append(labs); Ln.append((labs >= 0).sum(1)); G.append(torch.full((n,), gi))
    return dict(acts=torch.cat(A), obss=torch.cat(O), labels=torch.cat(Lb), length=torch.cat(Ln), goal=torch.cat(G))


def dataset(worlds, sols, n_train):
    p = HERE / "data" / f"train_{n_train}.pt"
    if p.exists(): return torch.load(p)
    p.parent.mkdir(exist_ok=True); d = generate(worlds, sols, n_train, 10, EPS); torch.save(d, p); return d


def test_starts(w, gi):
    return torch.as_tensor(w.start_states)[torch.randint(len(w.start_states), (N_TEST,), generator=torch.Generator().manual_seed(TEST_SEED + gi))]


def goal_sentence(w):
    return f"The goal is at row {w.goal[1]}, column {w.goal[0]}.\n"


# ------------------------------------------------------------------ the language model with a goal in the prompt

class LLMFormatGoals:
    def __init__(self, tok, worlds):
        self.tok = tok; w = worlds[0]
        self.prefix = [tok(BMZ.describe(w_) + goal_sentence(w_) + "\nTranscript:\n", add_special_tokens=False)["input_ids"] for w_ in worlds]
        self.act_ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in CANDS]; assert all(len(x) == 1 for x in self.act_ids)
        self.act_tok = torch.tensor([x[0] for x in self.act_ids])
        self.obs_ids = {0: tok(" -> moved", add_special_tokens=False)["input_ids"], 1: tok(" -> bumped", add_special_tokens=False)["input_ids"]}
        self.step_ids = [tok(f"{i + 1}.", add_special_tokens=False)["input_ids"] for i in range(w.H)]; self.nl = tok("\n", add_special_tokens=False)["input_ids"]

    def sequence(self, acts, obss, g, labels=None):
        ids = list(self.prefix[g]); tgt = [-100] * len(ids)
        for i, (a, o) in enumerate(zip(acts, obss)):
            ids += self.step_ids[i]; tgt += [-100] * len(self.step_ids[i])
            ids += self.act_ids[a]; tgt += [int(self.act_tok[labels[i]]) if labels is not None else -100]
            if a == 2: ids += self.obs_ids[o]; tgt += [-100] * len(self.obs_ids[o])
            ids += self.nl; tgt += [-100]
        return ids, tgt

    def prompt(self, acts, obss, g):
        return self.sequence(acts, obss, g)[0] + self.step_ids[len(acts)]


def pad_left(seqs, pad):
    L = max(len(s) for s in seqs); ids = torch.full((len(seqs), L), pad); att = torch.zeros(len(seqs), L, dtype=torch.long)
    for r, s in enumerate(seqs): ids[r, L - len(s):] = torch.tensor(s); att[r, L - len(s):] = 1
    return ids, att


class LLMPolicy:
    def __init__(self, model, fmt, device, batch=64): self.model, self.fmt, self.dev, self.batch = model, fmt, device, batch

    @torch.no_grad()
    def logits(self, hist):
        """hist: list of (acts, obss, goal); the centred logits over the three actions [n, 3] (double)."""
        out = []
        for b in range(0, len(hist), self.batch):
            ids, att = pad_left([self.fmt.prompt(a, o, g) for a, o, g in hist[b:b + self.batch]], self.fmt.tok.pad_token_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.model.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state[:, -1]
                z = self.model.lm_head(h).float()[:, self.fmt.act_tok.to(self.dev)].double()
            out.append((z - z.mean(1, keepdim=True)).cpu())
        return torch.cat(out)

    def act(self, hist): return self.logits(hist).argmax(1).tolist()


def train_llm(worlds, data, n_train, epochs, device, lr=2e-5, batch=64, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float32).to(device); model.gradient_checkpointing_enable(); model.config.use_cache = False
    fmt = LLMFormatGoals(tok, worlds)
    seqs = [fmt.sequence(data["acts"][i, : data["length"][i]].tolist(), data["obss"][i, : data["length"][i]].tolist(), int(data["goal"][i]), data["labels"][i, : data["length"][i]].tolist()) for i in range(n_train)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = epochs * math.ceil(n_train / batch); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(0); step, t0 = 0, time.time(); model.train()
    for ep in range(epochs):
        order = torch.randperm(n_train, generator=g).tolist()
        for b in range(0, n_train, batch):
            chunk = [seqs[i] for i in order[b:b + batch]]; L = max(len(s[0]) for s in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id); tgt = torch.full((len(chunk), L), -100); att = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, (s, t_) in enumerate(chunk): ids[r, :len(s)] = torch.tensor(s); tgt[r, :len(s)] = torch.tensor(t_); att[r, :len(s)] = 1
            tgt = tgt.to(device); sel = tgt[:, 1:] != -100
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
                logits = model.lm_head(h[:, :-1][sel])
            loss = nn.functional.cross_entropy(logits.float(), tgt[:, 1:][sel])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 50 == 0 or step == steps: log(f"  llm step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval(); model.config.use_cache = True
    return LLMPolicy(model, fmt, device), model, fmt


# ------------------------------------------------------------------ from scratch, the goal as the start token

def step_tokens(acts, obss, length, goal, H):
    n = len(acts); t = torch.zeros(n, H + 1, dtype=torch.long)
    code = 1 + 2 * acts + obss; mask = torch.arange(H)[None] < length[:, None]
    t[:, 0] = 7 + goal; t[:, 1:] = torch.where(mask, code, torch.zeros_like(code)); return t


class SmallPolicy:
    def __init__(self, model, H, device): self.model, self.H, self.dev = model, H, device

    @torch.no_grad()
    def logits(self, hist):
        acts = torch.zeros(len(hist), self.H, dtype=torch.long); obss = torch.zeros_like(acts); ln = torch.tensor([len(a) for a, _, _ in hist]); gl = torch.tensor([g for _, _, g in hist])
        for i, (a, o, _) in enumerate(hist): acts[i, :len(a)] = torch.tensor(a); obss[i, :len(o)] = torch.tensor(o)
        z = self.model(step_tokens(acts, obss, ln, gl, self.H).to(self.dev))[torch.arange(len(hist)), ln.to(self.dev)].double().cpu()
        return z - z.mean(1, keepdim=True)

    def act(self, hist): return self.logits(hist).argmax(1).tolist()


def train_small(worlds, data, n_train, epochs, device, kind, log=print, batch=256, lr=1e-3):
    H = worlds[0].H
    model = (StepTransformer(H) if kind == "tfm" else StepGRU(H)); model.tok = nn.Embedding(7 + K, model.tok.embedding_dim); model = model.to(device)
    toks = step_tokens(data["acts"][:n_train], data["obss"][:n_train], data["length"][:n_train], data["goal"][:n_train], H)
    labels = torch.full((n_train, H + 1), -100, dtype=torch.long); lab = data["labels"][:n_train]; labels[:, :H] = torch.where(lab >= 0, lab, torch.full_like(lab, -100))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01); steps = epochs * math.ceil(n_train / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(0); step, t0 = 0, time.time()
    for ep in range(epochs):
        order = torch.randperm(n_train, generator=g)
        for b in range(0, n_train, batch):
            idx = order[b:b + batch]; logits = model(toks[idx].to(device))
            loss = nn.functional.cross_entropy(logits.reshape(-1, 3), labels[idx].reshape(-1).to(device), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if step % 200 == 0 or step == steps: log(f"  {kind} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval(); return SmallPolicy(model, H, device), model


# ------------------------------------------------------------------ evaluation

def closed_loop(w, sol, gi, policy, starts):
    n = len(starts); e = BMZ.Env(w, n, torch.Generator().manual_seed(0), start=starts); ret = torch.zeros(n); L = torch.full((n,), w.H); hits = []
    for t in range(w.H):
        alive = torch.nonzero(~e.done).squeeze(1).tolist()
        if not alive: break
        ch = policy.act([([int(x[i]) for x in e.acts], [int(x[i]) for x in e.obss], gi) for i in alive])
        _, Q = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); opt_set = Q >= Q.max(1, keepdims=True) - 1e-9
        hits += [bool(opt_set[i][c]) for i, c in zip(alive, ch)]
        act = torch.zeros(n, dtype=torch.long); act[alive] = torch.tensor(ch)
        was = e.done.clone(); r, _ = e.step(act); ret += r; L[~was & e.done] = t + 1
    succ = (ret > 0).float(); boot = [succ[torch.randint(n, (n,))].mean().item() for _ in range(1000)]
    return dict(ret=float(ret.mean()), success=float(succ.mean()), success_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], length=float(L.float().mean()), own_decisions_optimal=float(np.mean(hits)), n=n)


def teacher_forced(policy, dec, gi):
    ch = policy.act([(d["acts"], d["obss"], gi) for d in dec])
    hit = np.array([d["opt"][c] for d, c in zip(dec, ch)]); size = np.array([d["mask"].sum() for d in dec])
    rng = np.random.default_rng(0); boot = [hit[rng.integers(len(hit), size=len(hit))].mean() for _ in range(1000)]
    return dict(optimal=float(hit.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], n=int(len(hit)),
                by_size={"1": float(hit[size == 1].mean()), "2-4": float(hit[(size >= 2) & (size <= 4)].mean()), "5-16": float(hit[(size >= 5) & (size <= 16)].mean()), ">16": float(hit[size > 16].mean())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llm", "tfm", "gru"], default="tfm"); ap.add_argument("--n_train", type=int, default=40_000); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--n_test", type=int, default=N_TEST); ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    name = f"{a.model}_{a.n_train}"; out = Path(a.out) if a.out else HERE / "runs" / name; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    torch.manual_seed(0); t0 = time.time()
    free, goals, worlds, sols = worlds_and_solutions()
    data = dataset(worlds, sols, max(a.n_train, 120_000) if a.n_train <= 120_000 else a.n_train); log(f"{name}: data {len(data['length'])} episodes {time.time() - t0:.0f}s")
    if a.n_train < len(data["length"]):                                   # a balanced subset: the first n/K episodes of every goal
        per = a.n_train // K; keep = torch.cat([torch.nonzero(data["goal"] == gi).squeeze(1)[:per] for gi in range(K)])
        data = {k: v[keep] for k, v in data.items()}
    if a.model == "llm": policy, model, fmt = train_llm(worlds, data, len(data["length"]), a.epochs, a.device, log=log)
    else: policy, model = train_small(worlds, data, len(data["length"]), a.epochs, a.device, a.model, log=log)
    log(f"{name}: trained {time.time() - t0:.0f}s")
    res = dict(model=a.model, n_train=a.n_train, epochs=a.epochs, per_goal=[])
    for gi, (w, sol) in enumerate(zip(worlds, sols)):
        cl = closed_loop(w, sol, gi, policy, test_starts(w, gi)[: a.n_test])
        dec_all = optimum_decisions(w, sol, 512, 2000 + gi); rng = np.random.default_rng(3 + gi); pick = rng.choice(len(dec_all), 2048, replace=False); dec = [dec_all[i] for i in pick]
        tf = teacher_forced(policy, dec, gi); res["per_goal"].append(dict(closed_loop=cl, teacher_forced=tf))
        log(f"{name} goal {gi}: closed loop return {cl['ret']:.3f} success {cl['success']:.3f} length {cl['length']:.1f}; teacher-forced optimality {tf['optimal']:.3f} by size " + " ".join(f"{k} {v:.2f}" for k, v in tf["by_size"].items()) + f" {time.time() - t0:.0f}s")
    res["mean"] = dict(success=float(np.mean([r["closed_loop"]["success"] for r in res["per_goal"]])), ret=float(np.mean([r["closed_loop"]["ret"] for r in res["per_goal"]])), optimal=float(np.mean([r["teacher_forced"]["optimal"] for r in res["per_goal"]])))
    res["seconds"] = time.time() - t0; log(f"{name}: mean success {res['mean']['success']:.3f} return {res['mean']['ret']:.3f} optimality {res['mean']['optimal']:.3f}")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    if a.save or a.model != "llm":
        if a.model == "llm": model.save_pretrained(out / "model"); fmt.tok.save_pretrained(out / "model")
        else: torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
