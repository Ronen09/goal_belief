"""Training on the bump maze (PLAN.md, part B): a language model (Qwen2.5-0.5B, full fine-tuning) and from-scratch
models (a small transformer, a GRU) by supervision from the exact optimum; episodes from the optimum with epsilon
random actions, the label at every step the exact optimal action for the exact belief. Evaluation: closed loop on
fixed test starts and teacher-forced decision optimality on the optimum's trajectories.

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_maze/train.py --model llm --n_train 100000
    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_maze/train.py --model tfm --n_train 10000
"""

from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zero_shot import CANDS, SEED, SIZE, LOOPS, optimum_decisions          # noqa: E402

EPS, N_TEST, DATA_SEED, TEST_SEED = 0.2, 2048, 10, 1000
LLM = "Qwen/Qwen2.5-0.5B"


# ------------------------------------------------------------------ data

def generate(w, sol, n_ep, seed, eps, batch=4096):
    """Episodes from the exact optimum with eps random actions; per episode the executed actions, observations and
    the optimal action at every step. Returns padded long tensors acts, obss, labels [n, H] and lengths [n]."""
    gen = torch.Generator().manual_seed(seed)
    A, O, Lb, Ln = [], [], [], []
    for b in range(0, n_ep, batch):
        n = min(batch, n_ep - b); e = BMZ.Env(w, n, gen)
        acts, obss, labs = torch.zeros(n, w.H, dtype=torch.long), torch.zeros(n, w.H, dtype=torch.long), torch.full((n, w.H), -1, dtype=torch.long)
        for t in range(w.H):
            a_opt, _ = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); a_opt = torch.as_tensor(a_opt)
            rnd = torch.rand(n, generator=gen) < eps; a = torch.where(rnd, torch.randint(3, (n,), generator=gen), a_opt)
            alive = ~e.done; labs[alive, t] = a_opt[alive]
            _, o = e.step(a); acts[:, t] = a; obss[:, t] = o
            if e.done.all(): break
        A.append(acts); O.append(obss); Lb.append(labs); Ln.append((labs >= 0).sum(1))
    return dict(acts=torch.cat(A), obss=torch.cat(O), labels=torch.cat(Lb), length=torch.cat(Ln))


def dataset(w, sol, n_train):
    p = HERE / "data" / f"train_{n_train}.pt"
    if p.exists(): return torch.load(p)
    p.parent.mkdir(exist_ok=True); d = generate(w, sol, n_train, DATA_SEED, EPS); torch.save(d, p); return d


def test_starts(w):
    return torch.as_tensor(w.start_states)[torch.randint(len(w.start_states), (N_TEST,), generator=torch.Generator().manual_seed(TEST_SEED))]


# ------------------------------------------------------------------ the language model: segments and sequences

class LLMFormat:
    """The prefix (rules and map), then one line per step 'i. <action>[ -> moved|bumped]\\n'. Tokenised by segments,
    consistently in training and at inference; the action words are single tokens."""
    def __init__(self, tok, w):
        self.tok = tok
        self.prefix = tok(BMZ.describe(w) + "\nTranscript:\n", add_special_tokens=False)["input_ids"]
        self.act_ids = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in CANDS]
        assert all(len(x) == 1 for x in self.act_ids), self.act_ids
        self.act_tok = torch.tensor([x[0] for x in self.act_ids])
        self.obs_ids = {0: tok(" -> moved", add_special_tokens=False)["input_ids"], 1: tok(" -> bumped", add_special_tokens=False)["input_ids"]}
        self.step_ids = [tok(f"{i + 1}.", add_special_tokens=False)["input_ids"] for i in range(w.H)]
        self.nl = tok("\n", add_special_tokens=False)["input_ids"]

    def sequence(self, acts, obss, labels=None):
        """ids and, if labels are given, the per-position target (-100 except at action positions)."""
        ids = list(self.prefix); tgt = [-100] * len(ids)
        for i, (a, o) in enumerate(zip(acts, obss)):
            ids += self.step_ids[i]; tgt += [-100] * len(self.step_ids[i])
            ids += self.act_ids[a]; tgt += [int(self.act_tok[labels[i]]) if labels is not None else -100]     # predicted after 'i.': the optimal action, whatever was executed
            if a == 2: ids += self.obs_ids[o]; tgt += [-100] * len(self.obs_ids[o])
            ids += self.nl; tgt += [-100]
        return ids, tgt

    def prompt(self, acts, obss):
        ids, _ = self.sequence(acts, obss); return ids + self.step_ids[len(acts)]


class LLMPolicy:
    def __init__(self, model, fmt, device, batch=64):
        self.model, self.fmt, self.dev, self.batch = model, fmt, device, batch

    @torch.no_grad()
    def act(self, hist):
        """hist: list of (acts, obss); returns the argmax action over the three candidate tokens at the next position."""
        out = []
        for b in range(0, len(hist), self.batch):
            seqs = [self.fmt.prompt(a, o) for a, o in hist[b:b + self.batch]]; L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), self.fmt.tok.pad_token_id); att = torch.zeros(len(seqs), L, dtype=torch.long)
            for r, s in enumerate(seqs): ids[r, L - len(s):] = torch.tensor(s); att[r, L - len(s):] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.model.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state[:, -1]
                logits = self.model.lm_head(h).float()
            out += logits[:, self.fmt.act_tok.to(self.dev)].argmax(1).tolist()
        return out


def train_llm(w, data, n_train, epochs, device, lr=2e-5, batch=64, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(LLM, dtype=torch.float32).to(device); model.gradient_checkpointing_enable(); model.config.use_cache = False
    fmt = LLMFormat(tok, w)
    seqs = [fmt.sequence(data["acts"][i, : data["length"][i]].tolist(), data["obss"][i, : data["length"][i]].tolist(), data["labels"][i, : data["length"][i]].tolist()) for i in range(n_train)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    steps = epochs * math.ceil(n_train / batch); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 50) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(0); step, t0 = 0, time.time(); model.train()
    for ep in range(epochs):
        order = torch.randperm(n_train, generator=g).tolist()
        for b in range(0, n_train, batch):
            chunk = [seqs[i] for i in order[b:b + batch]]; L = max(len(s[0]) for s in chunk)
            ids = torch.full((len(chunk), L), tok.pad_token_id); tgt = torch.full((len(chunk), L), -100); att = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, (s, t_) in enumerate(chunk): ids[r, :len(s)] = torch.tensor(s); tgt[r, :len(s)] = torch.tensor(t_); att[r, :len(s)] = 1
            tgt = tgt.to(device); sel = (tgt[:, 1:] != -100)                                   # the action positions, predicted from the token before
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.model(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state
                logits = model.lm_head(h[:, :-1][sel])                                      # the full vocabulary only where the loss is
            loss = nn.functional.cross_entropy(logits.float(), tgt[:, 1:][sel])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
            if step % 50 == 0 or step == steps: log(f"  llm step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval(); model.config.use_cache = True
    return LLMPolicy(model, fmt, device), model, fmt


# ------------------------------------------------------------------ from scratch: a small transformer and a GRU on (action, observation) tokens

class StepTransformer(nn.Module):
    """Token t = (action, observation) of step t, with a start token at position 0; the logits at position t are the
    next action. Pre-LN causal transformer as in the project."""
    def __init__(self, H, d=128, layers=4, heads=4):
        super().__init__()
        from goalgeo.navmodel import Block
        self.d, self.nl = d, layers
        self.tok = nn.Embedding(7, d); self.pos = nn.Embedding(H + 1, d)     # 0 start, 1 + a * 2 + o
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)]); self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, 3)

    def forward(self, tokens, record=False):
        x = self.tok(tokens) + self.pos.weight[: tokens.shape[1]]; rec = [x]
        for b in self.blocks:
            a, _, _, _ = b.attn(x); mid = x + a; x = mid + b.mlp(b.ln2(mid)); rec.append(x)
        return (self.head(self.ln(x)), rec) if record else self.head(self.ln(x))


class StepGRU(nn.Module):
    def __init__(self, H, d=128):
        super().__init__()
        self.tok = nn.Embedding(7, d); self.gru = nn.GRU(d, d, batch_first=True); self.head = nn.Linear(d, 3)

    def forward(self, tokens, record=False):
        h, _ = self.gru(self.tok(tokens)); return (self.head(h), [h]) if record else self.head(h)


def step_tokens(acts, obss, length, H):
    """[n, H + 1] tokens: start, then 1 + 2a + o for each executed step (padded with 0 beyond the length)."""
    n = len(acts); t = torch.zeros(n, H + 1, dtype=torch.long)
    code = 1 + 2 * acts + obss; mask = torch.arange(H)[None] < length[:, None]
    t[:, 1:] = torch.where(mask, code, torch.zeros_like(code)); return t


class SmallPolicy:
    def __init__(self, model, H, device): self.model, self.H, self.dev = model, H, device

    @torch.no_grad()
    def act(self, hist):
        acts = torch.zeros(len(hist), self.H, dtype=torch.long); obss = torch.zeros_like(acts); ln = torch.tensor([len(a) for a, _ in hist])
        for i, (a, o) in enumerate(hist): acts[i, :len(a)] = torch.tensor(a); obss[i, :len(o)] = torch.tensor(o)
        logits = self.model(step_tokens(acts, obss, ln, self.H).to(self.dev))
        return logits[torch.arange(len(hist)), ln.to(self.dev)].argmax(1).tolist()


def train_small(w, data, n_train, epochs, device, kind, log=print, batch=256, lr=1e-3):
    model = (StepTransformer(w.H) if kind == "tfm" else StepGRU(w.H)).to(device)
    toks = step_tokens(data["acts"][:n_train], data["obss"][:n_train], data["length"][:n_train], w.H)
    labels = torch.full((n_train, w.H + 1), -100, dtype=torch.long)
    lab = data["labels"][:n_train]; labels[:, :w.H] = torch.where(lab >= 0, lab, torch.full_like(lab, -100))          # position t predicts the optimal action of step t (after seeing t tokens)
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
    model.eval(); return SmallPolicy(model, w.H, device), model


# ------------------------------------------------------------------ evaluation

def closed_loop(w, sol, policy, starts, log=print):
    n = len(starts); e = BMZ.Env(w, n, torch.Generator().manual_seed(0), start=starts); ret = torch.zeros(n); L = torch.full((n,), w.H); hits = []
    for t in range(w.H):
        alive = torch.nonzero(~e.done).squeeze(1).tolist()
        if not alive: break
        ch = policy.act([([int(x[i]) for x in e.acts], [int(x[i]) for x in e.obss]) for i in alive])
        _, Q = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); opt_set = Q >= Q.max(1, keepdims=True) - 1e-9
        hits += [bool(opt_set[i][c]) for i, c in zip(alive, ch)]
        act = torch.zeros(n, dtype=torch.long); act[alive] = torch.tensor(ch)
        was = e.done.clone(); r, _ = e.step(act); ret += r; L[~was & e.done] = t + 1
    succ = (ret > 0).float(); boot = [succ[torch.randint(n, (n,))].mean().item() for _ in range(1000)]
    return dict(ret=float(ret.mean()), success=float(succ.mean()), success_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], length=float(L.float().mean()), own_decisions_optimal=float(np.mean(hits)), n=n)


def teacher_forced(policy, dec):
    ch = policy.act([(d["acts"], d["obss"]) for d in dec])
    hit = np.array([d["opt"][c] for d, c in zip(dec, ch)]); size = np.array([d["mask"].sum() for d in dec])
    rng = np.random.default_rng(0); boot = [hit[rng.integers(len(hit), size=len(hit))].mean() for _ in range(1000)]
    return dict(optimal=float(hit.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], n=int(len(hit)),
                by_size={"1": float(hit[size == 1].mean()), "2-4": float(hit[(size >= 2) & (size <= 4)].mean()), "5-16": float(hit[(size >= 5) & (size <= 16)].mean()), ">16": float(hit[size > 16].mean())},
                marginal=[float(np.mean(np.array(ch) == k)) for k in range(3)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["llm", "tfm", "gru"], default="tfm"); ap.add_argument("--n_train", type=int, default=100_000); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default=None); ap.add_argument("--n_test", type=int, default=N_TEST); ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    name = f"{a.model}_{a.n_train}"; out = Path(a.out) if a.out else HERE / "runs" / name; out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")
    def log(s): print(s, flush=True); logf.write(s + "\n"); logf.flush()
    torch.manual_seed(0)
    w = BMZ.make(SEED, SIZE, LOOPS); sol = BMZ.solve(w)
    t0 = time.time(); data = dataset(w, sol, max(a.n_train, 100_000) if a.n_train <= 100_000 else a.n_train); log(f"{name}: data {len(data['length'])} episodes (mean length {data['length'].float().mean():.1f}) {time.time() - t0:.0f}s")
    if a.model == "llm": policy, model, fmt = train_llm(w, data, a.n_train, a.epochs, a.device, log=log)
    else: policy, model = train_small(w, data, a.n_train, a.epochs, a.device, a.model, log=log)
    log(f"{name}: trained {time.time() - t0:.0f}s")
    starts = test_starts(w)[: a.n_test]; cl = closed_loop(w, sol, policy, starts, log)
    log(f"{name}: closed loop return {cl['ret']:.3f} success {cl['success']:.3f} length {cl['length']:.1f} (own decisions optimal {cl['own_decisions_optimal']:.3f})")
    dec_all = optimum_decisions(w, sol, 512, 2000); rng = np.random.default_rng(3); pick = rng.choice(len(dec_all), 2048, replace=False); dec = [dec_all[i] for i in pick]
    tf = teacher_forced(policy, dec)
    log(f"{name}: teacher-forced decision optimality {tf['optimal']:.3f} by set size " + " ".join(f"{k} {v:.2f}" for k, v in tf["by_size"].items()) + f" {time.time() - t0:.0f}s")
    res = dict(model=a.model, n_train=a.n_train, epochs=a.epochs, closed_loop=cl, teacher_forced=tf, seconds=time.time() - t0)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    if a.save or a.model != "llm" or a.n_train >= 100_000:
        if a.model == "llm": model.save_pretrained(out / "model"); fmt.tok.save_pretrained(out / "model")
        else: torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
