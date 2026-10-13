"""Zero-shot: instruction-tuned language models on the bump maze (PLAN.md, part A). Closed loop on fixed starts,
teacher-forced decision optimality on the exact optimum's trajectories, and belief by questioning (could you be in
row r, column c?). Actions are chosen by scoring the three candidate replies; the think variant generates reasoning.

    PYTHONPATH=. .venv/bin/python studies/8_language_models/bump_maze/zero_shot.py --models Qwen/Qwen2.5-0.5B-Instruct [--think]
"""

from __future__ import annotations

import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
SEED, SIZE, LOOPS = 0, 9, 4
CANDS = ["left", "right", "forward"]


def transcript(acts, obss):
    if not acts: return "Transcript so far: (no actions yet)\n"
    lines = [f"{i + 1}. {BMZ.ACTIONS[a]}" + (f" -> {'bumped' if o else 'moved'}" if a == 2 else "") for i, (a, o) in enumerate(zip(acts, obss))]
    return "Transcript so far:\n" + "\n".join(lines) + "\n"


class Scorer:
    def __init__(self, name, device="cuda", batch=48):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name); self.tok.padding_side = "left"
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device).eval()
        self.dev, self.batch = device, batch

    def prompt(self, system, user):
        return self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)

    @torch.no_grad()
    def score(self, prompts, cands):
        """Sum log-probability of each candidate reply after each prompt: [n, k]."""
        cand_ids = [self.tok(c, add_special_tokens=False)["input_ids"] for c in cands]
        out = torch.zeros(len(prompts), len(cands))
        pairs = [(i, j) for i in range(len(prompts)) for j in range(len(cands))]
        for b in range(0, len(pairs), self.batch):
            chunk = pairs[b:b + self.batch]
            seqs = [self.tok(prompts[i], add_special_tokens=False)["input_ids"] + cand_ids[j] for i, j in chunk]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), self.tok.pad_token_id); att = torch.zeros(len(seqs), L, dtype=torch.long)
            for r, s in enumerate(seqs): ids[r, L - len(s):] = torch.tensor(s); att[r, L - len(s):] = 1
            logits = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1)
            for r, (i, j) in enumerate(chunk):
                n = len(cand_ids[j]); tgt = ids[r, L - n:].to(self.dev)
                out[i, j] = lp[r, L - n - 1:L - 1].gather(1, tgt[:, None]).sum().item()
        return out

    @torch.no_grad()
    def generate(self, prompts, max_new=256):
        outs = []
        for b in range(0, len(prompts), self.batch // 2):
            enc = self.tok(prompts[b:b + self.batch // 2], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
            gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.pad_token_id)
            outs += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs


def parse_action(text):
    found = re.findall(r"\b(left|right|forward)\b", text.lower())
    return CANDS.index(found[-1]) if found else 2


def optimum_decisions(w, sol, n_ep, seed):
    """Decisions along the exact optimum's trajectories: (acts, obss, mask, optimal set, step) per decision."""
    gen = torch.Generator().manual_seed(seed); e = BMZ.Env(w, n_ep, gen); dec = []
    for t in range(w.H):
        a_opt, Q = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); alive = ~e.done
        opt_set = Q >= Q.max(1, keepdims=True) - 1e-9
        for i in torch.nonzero(alive).squeeze(1).tolist():
            dec.append(dict(acts=[int(x[i]) for x in e.acts], obss=[int(x[i]) for x in e.obss], mask=e.filter.mask[i].numpy().copy(), opt=opt_set[i].copy(), step=t))
        e.step(torch.as_tensor(a_opt))
        if e.done.all(): break
    return dec


def auc(score, pos):
    """AUC of score for the boolean pos; None if one class is missing."""
    pos = np.asarray(pos, dtype=bool)
    if pos.all() or (~pos).all(): return None
    s = np.asarray(score); P, Nn = s[pos], s[~pos]
    return float(((P[:, None] > Nn[None]).mean() + 0.5 * (P[:, None] == Nn[None]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-0.5B-Instruct"])
    ap.add_argument("--think", action="store_true"); ap.add_argument("--n_episodes", type=int, default=128); ap.add_argument("--n_dec", type=int, default=2048)
    ap.add_argument("--n_belief", type=int, default=256); ap.add_argument("--out", default=str(HERE / "zero_shot.json")); ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    w = BMZ.make(SEED, SIZE, LOOPS); sol = BMZ.solve(w); system = BMZ.describe(w)
    starts = torch.as_tensor(w.start_states)[torch.randint(len(w.start_states), (a.n_episodes,), generator=torch.Generator().manual_seed(1000))]
    dec_all = optimum_decisions(w, sol, 512, 2000)
    rng = np.random.default_rng(3); pick = rng.choice(len(dec_all), min(a.n_dec, len(dec_all)), replace=False); dec = [dec_all[i] for i in pick]
    bel = dec[: a.n_belief]
    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    cell_of = lambda m: np.array([m[4 * c: 4 * c + 4].any() for c in range(w.n_cells)])
    for name in a.models:
        t0 = time.time(); S = Scorer(name, batch=a.batch); key = name.split("/")[-1] + ("-think" if a.think else "")
        row = dict(model=name, think=a.think)
        # ---- teacher-forced decision optimality
        prompts = [S.prompt(system, transcript(d["acts"], d["obss"]) + "Next action?") for d in dec]
        if a.think:
            replies = S.generate([p for p in prompts[:512]], 256); acts = np.array([parse_action(r) for r in replies]); dsub = dec[:512]; row["example_reply"] = replies[0][:600]
        else:
            sc = S.score(prompts, CANDS); acts = sc.argmax(1).numpy(); dsub = dec
        hit = np.array([d["opt"][x] for d, x in zip(dsub, acts)]); known = np.array([d["mask"].sum() == 1 for d in dsub]); size = np.array([d["mask"].sum() for d in dsub])
        boot = [hit[rng.integers(len(hit), size=len(hit))].mean() for _ in range(1000)]
        row["decision"] = dict(optimal=float(hit.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], n=int(len(hit)), known=float(hit[known].mean()), unknown=float(hit[~known].mean()),
                               by_size={"1": float(hit[size == 1].mean()), "2-4": float(hit[(size >= 2) & (size <= 4)].mean()), "5-16": float(hit[(size >= 5) & (size <= 16)].mean()), ">16": float(hit[size > 16].mean())},
                               marginal=[float((acts == k).mean()) for k in range(3)], chance=float(np.mean([d["opt"].sum() / 3 for d in dsub])))
        print(f"{key}: decision optimality {row['decision']['optimal']:.3f} (known {row['decision']['known']:.2f}, unknown {row['decision']['unknown']:.2f}; forward share {row['decision']['marginal'][2]:.2f}) {time.time() - t0:.0f}s", flush=True)
        # ---- closed loop
        n_ep = a.n_episodes if not a.think else min(a.n_episodes, 64)
        e = BMZ.Env(w, n_ep, torch.Generator().manual_seed(0), start=starts[:n_ep]); ret = torch.zeros(n_ep); L = torch.full((n_ep,), w.H); marg = np.zeros(3); n_act = 0; hits = []
        for t in range(w.H):
            alive = torch.nonzero(~e.done).squeeze(1).tolist()
            if not alive: break
            pr = [S.prompt(system, transcript([int(x[i]) for x in e.acts], [int(x[i]) for x in e.obss]) + "Next action?") for i in alive]
            if a.think: ch = [parse_action(r) for r in S.generate(pr, 256)]
            else: ch = S.score(pr, CANDS).argmax(1).tolist()
            a_opt, Q = BMZ.optimal_action(w, sol, e.filter.mask.numpy()); opt_set = Q >= Q.max(1, keepdims=True) - 1e-9
            hits += [bool(opt_set[i][c]) for i, c in zip(alive, ch)]
            act = torch.zeros(n_ep, dtype=torch.long); act[alive] = torch.tensor(ch)
            for c in ch: marg[c] += 1
            n_act += len(ch); was = e.done.clone(); r, _ = e.step(act); ret += r; L[~was & e.done] = t + 1
        row["closed_loop"] = dict(ret=float(ret.mean()), success=float((ret > 0).float().mean()), length=float(L.float().mean()), n=n_ep, marginal=(marg / max(n_act, 1)).tolist(), own_decisions_optimal=float(np.mean(hits)),
                                  success_ci=[float(np.percentile(b_, q)) for b_ in [[(ret > 0).float()[torch.randint(n_ep, (n_ep,))].mean().item() for _ in range(1000)]] for q in (2.5, 97.5)])
        print(f"{key}: closed loop return {row['closed_loop']['ret']:.3f} success {row['closed_loop']['success']:.3f} length {row['closed_loop']['length']:.1f} {time.time() - t0:.0f}s", flush=True)
        # ---- belief by questioning (scored models only)
        if not a.think:
            qs, meta = [], []
            for di, d in enumerate(bel):
                tr = transcript(d["acts"], d["obss"])
                for c, (x, y) in enumerate(w.cells):
                    qs.append(S.prompt(system, tr + f"Given this history, could you currently be in row {y}, column {x}? Answer yes or no.")); meta.append((di, "cell", c))
                for h in range(4):
                    qs.append(S.prompt(system, tr + f"Given this history, could you currently be facing {BMZ.DIR_NAMES[h]}? Answer yes or no.")); meta.append((di, "head", h))
            sc = S.score(qs, ["yes", "no"]); yes = (sc[:, 0] - sc[:, 1]).numpy()
            cell_auc, head_auc, mism_auc = [], [], []
            per = {}
            for (di, kind, k), v in zip(meta, yes): per.setdefault((di, kind), {})[k] = v
            for di, d in enumerate(bel):
                cm = cell_of(d["mask"]); hm = np.array([d["mask"][h::4].any() for h in range(4)])
                sc_c = np.array([per[(di, "cell")][c] for c in range(w.n_cells)]); sc_h = np.array([per[(di, "head")][h] for h in range(4)])
                v = auc(sc_c, cm); cell_auc.append(v); head_auc.append(auc(sc_h, hm))
                # mismatched-history control: another decision's answers against this decision's set
                dj = (di + 1) % len(bel); sc_j = np.array([per[(dj, "cell")][c] for c in range(w.n_cells)]); mism_auc.append(auc(sc_j, cm))
            f = lambda xs: (float(np.mean([x for x in xs if x is not None])), int(sum(x is not None for x in xs)))
            row["belief"] = dict(cell_auc=f(cell_auc), head_auc=f(head_auc), mismatched_cell_auc=f(mism_auc), yes_share=float((yes > 0).mean()))
            print(f"{key}: belief AUC cell {row['belief']['cell_auc'][0]:.3f} (mismatched history {row['belief']['mismatched_cell_auc'][0]:.3f}) heading {row['belief']['head_auc'][0]:.3f}; yes share {row['belief']['yes_share']:.2f} {time.time() - t0:.0f}s", flush=True)
        row["seconds"] = time.time() - t0
        out[key] = row; Path(a.out).write_text(json.dumps(out, indent=1))
        del S; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
