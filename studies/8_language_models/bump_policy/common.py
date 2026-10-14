"""Shared pieces of the policy study: the maze and solver, the reachable sets (`sets.pt`), the fine-tuned language
model with hooked forward passes (logits at the decision token, residuals at chosen layers and positions, patches)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from goalgeo import bumpmaze as BMZ

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bump_maze"))
from zero_shot import SEED, SIZE, LOOPS                                   # noqa: E402
from train import LLM, LLMFormat, StepTransformer, step_tokens            # noqa: E402

RUNS = HERE.parent / "bump_maze" / "runs"
key = lambda m: np.packbits(np.asarray(m, dtype=bool)).tobytes()
SIZES = (("1", 1, 1), ("2-4", 2, 4), ("5-16", 5, 16), (">16", 17, 10 ** 9))


def world():
    w = BMZ.make(SEED, SIZE, LOOPS); sol = BMZ.solve(w); return w, sol


def load_sets():
    return torch.load(HERE / "sets.pt")


def size_bins(size):
    size = np.asarray(size); return {b: (size >= lo) & (size <= hi) for b, lo, hi in SIZES}


def boot_mean(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float); rng = np.random.default_rng(seed)
    if len(x) == 0: return [float("nan"), float("nan")]
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)]; return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


class Runner:
    """The fine-tuned model. run(prompts) -> centred action logits [n, 3] at the last position; optionally the residual
    entering layer l (0 = after the embedding, L = after the last layer) at given positions (from the end: -1 is the
    decision token), and a patch: (layer, position-from-end, tensor [n, d]) replacing that residual."""
    def __init__(self, run_dir, w, device="cuda", batch=32):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(run_dir); tok.padding_side = "left"
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        self.tok, self.dev, self.batch = tok, device, batch
        self.model = AutoModelForCausalLM.from_pretrained(run_dir, dtype=torch.bfloat16, device_map=device).eval(); self.fmt = LLMFormat(tok, w); self.w = w
        self.n_layers = len(self.model.model.layers); self._rec = None; self._patch = None; self._pos = None
        self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: self._hook(0, o))
        for l, layer in enumerate(self.model.model.layers): layer.register_forward_hook(lambda m, i, o, l=l: self._hook(l + 1, o))

    def _hook(self, l, out):
        x = out[0] if isinstance(out, tuple) else out
        if self._rec is not None and l in self._rec:
            if self._pos is None: self._rec[l] = x.float().clone()
            elif torch.is_tensor(self._pos):                                  # per-row positions from the end [n, k] (negative; -10**6 = none)
                T = x.shape[1]; p = self._pos.to(x.device); valid = p > -T; idx = torch.where(valid, T + p, torch.zeros_like(p)).clamp(0, T - 1)
                g = torch.gather(x.float(), 1, idx[..., None].expand(-1, -1, x.shape[-1])); g[~valid] = 0; self._rec[l] = g.clone()
            else: self._rec[l] = x[:, self._pos].float().clone()
        if self._patch is not None:
            for (pl, pp, val) in self._patch:
                if pl == l:
                    x = x.clone(); x[:, pp] = val.to(x.dtype).to(x.device)
            if any(pl == l for pl, _, _ in self._patch): return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x
        return None

    def positions(self, acts):
        """Token index (from the end of the prompt, negative) of the newline ending each step i, and of the decision token (-1)."""
        f = self.fmt; n = len(acts); L = len(f.prefix); ends = []
        for i, a in enumerate(acts):
            L += len(f.step_ids[i]) + 1 + (len(f.obs_ids[0]) if a == 2 else 0) + 1; ends.append(L - 1)
        total = L + len(f.step_ids[n]); return [e - total for e in ends]

    @torch.no_grad()
    def run(self, hists, record=None, pos=None, patch=None):
        """hists: list of (acts, obss). record: layers to record; pos: positions from the end (list of negative ints)
        or None for every position; patch: list of (layer, pos-from-end, [n, d]). Returns logits [n, 3] and the dict
        layer -> [n, len(pos), d] (or [n, T, d])."""
        Z, R = [], {l: [] for l in (record or [])}
        for b in range(0, len(hists), self.batch):
            chunk = hists[b:b + self.batch]; seqs = [self.fmt.prompt(a, o) for a, o in chunk]; T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), self.tok.pad_token_id); att = torch.zeros(len(seqs), T, dtype=torch.long)
            for r, s in enumerate(seqs): ids[r, T - len(s):] = torch.tensor(s); att[r, T - len(s):] = 1
            self._rec = {l: None for l in (record or [])} if record else None; self._pos = pos[b:b + self.batch] if torch.is_tensor(pos) else pos
            self._patch = [(pl, pp, val[b:b + self.batch]) for pl, pp, val in patch] if patch else None
            h = self.model.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state[:, -1]
            z = self.model.lm_head(h).float()[:, self.fmt.act_tok.to(self.dev)]; Z.append((z - z.mean(1, keepdim=True)).cpu())
            if record:
                for l in record: R[l].append(self._rec[l].cpu())
            self._rec = self._patch = self._pos = None
        return torch.cat(Z), ({l: torch.cat(v) for l, v in R.items()} if record else None)


class TfmRunner:
    def __init__(self, path, w, device="cuda"):
        self.model = StepTransformer(w.H).to(device); self.model.load_state_dict(torch.load(path, map_location=device)); self.model.eval(); self.w, self.dev = w, device

    @torch.no_grad()
    def run(self, hists, batch=1024):
        Z = []
        for b in range(0, len(hists), batch):
            chunk = hists[b:b + batch]; n = len(chunk); acts = torch.zeros(n, self.w.H, dtype=torch.long); obss = torch.zeros_like(acts); ln = torch.tensor([len(a) for a, _ in chunk])
            for i, (a, o) in enumerate(chunk): acts[i, :len(a)] = torch.tensor(a); obss[i, :len(o)] = torch.tensor(o)
            z = self.model(step_tokens(acts, obss, ln, self.w.H).to(self.dev))[torch.arange(n), ln.to(self.dev)].float().cpu(); Z.append(z - z.mean(1, keepdim=True))
        return torch.cat(Z), None
