"""Train many copies of `tfm.CausalTransformer` simultaneously as one stacked model.

The unbatched runner trains one tiny transformer per process; at d=64 that is bound by kernel
launches, not arithmetic, so the GPU sits at a few percent of its bandwidth. Here the M models
of a grid are stacked along a leading model axis and stepped together: every op becomes one
batched einsum over M, the loss is the sum of the per-model losses (so gradients never mix,
and Adam on stacked tensors is elementwise identical to M separate Adams), and one optimiser
step advances the whole grid.

Weights are initialised by stacking the state dicts of ordinary CausalTransformers, one per
seed, and unstacked back into them afterwards, so the trained models are the same objects the
measurement code in `cuts.py` / `factorize.py` already handles, and `test_tfm_batched.py`
checks the batched forward and a batched training run against the per-model ones.

Per-model degrees of freedom: the final-norm variant (none / frozen / learned), the fixed
readout gain, the two attention masks (a route restriction is a per-model mask), the two
attention penalty coefficients, and the training data (so δ may differ across models)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import hmm4 as H4
from . import tfm as Tf

NORM_ID = {"none": 0, "frozen": 1, "learn": 2}

# TF32 makes the stacked training step ~1.6x faster on an A100 and perturbs the trajectory at
# the 1e-3 level, like a change of seed. It is switched on *only* inside train_batched, never
# during measurement: the identities the experiments test (complete-cut patching, C = g D cos)
# are checked in plain fp32 on the unstacked models.


@dataclass
class Spec:
    """One model of the grid."""
    seed: int = 0
    delta: float = 0.4
    r: float = 1.0
    gain: float | None = None
    norm: str = "learn"
    attn_diag: tuple[int, ...] = ()
    mu_l1: float = 0.0            # penalty on block-1 off-diagonal attention
    mu_l2: float = 0.0            # penalty on block-2 attention to the previous position
    label: str = ""
    extra: dict = field(default_factory=dict)


def _ln(x, g, b, eps=1e-5):
    """LayerNorm over the last axis with per-model affine [M, 1, 1, d]."""
    mu = x.mean(-1, keepdim=True); var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * g + b


class BatchedTransformer(torch.nn.Module):
    """M stacked 2-layer pre-LN causal transformers, numerically identical to tfm.CausalTransformer."""

    def __init__(self, specs: list[Spec], d=64, n_heads=4, n_layers=2, mlp=128, T=48, n_vocab=H4.V, device="cuda"):
        super().__init__()
        self.specs = specs; self.M = len(specs); self.d = d; self.H = n_heads; self.L = n_layers; self.T = T
        nets = [Tf.CausalTransformer(n_vocab=n_vocab, d=d, n_heads=n_heads, n_layers=n_layers, mlp=mlp, T=T,
                                     seed=s.seed, gain=s.gain, final_norm=s.norm, attn_diag=s.attn_diag)
                for s in specs]
        self.proto_kw = dict(n_vocab=n_vocab, d=d, n_heads=n_heads, n_layers=n_layers, mlp=mlp, T=T)
        sd = [n.state_dict() for n in nets]
        stack = lambda k: torch.nn.Parameter(torch.stack([s[k] for s in sd]).to(device))   # noqa: E731
        self.P = torch.nn.ParameterDict()
        self.P["emb"] = stack("emb.weight"); self.P["pos"] = stack("pos")
        for i in range(n_layers):
            for src, dst in ((f"blocks.{i}.ln1.weight", f"ln1g{i}"), (f"blocks.{i}.ln1.bias", f"ln1b{i}"),
                             (f"blocks.{i}.attn.in_proj_weight", f"qkvw{i}"), (f"blocks.{i}.attn.in_proj_bias", f"qkvb{i}"),
                             (f"blocks.{i}.attn.out_proj.weight", f"ow{i}"), (f"blocks.{i}.attn.out_proj.bias", f"ob{i}"),
                             (f"blocks.{i}.ln2.weight", f"ln2g{i}"), (f"blocks.{i}.ln2.bias", f"ln2b{i}"),
                             (f"blocks.{i}.mlp.0.weight", f"m0w{i}"), (f"blocks.{i}.mlp.0.bias", f"m0b{i}"),
                             (f"blocks.{i}.mlp.2.weight", f"m1w{i}"), (f"blocks.{i}.mlp.2.bias", f"m1b{i}")):
                self.P[dst] = stack(src)
        # final norm: keep affine parameters for every model, use / freeze them per model
        have = [k for k in sd[0] if k.startswith("ln_f.")]
        ones = torch.ones(self.M, d); zeros = torch.zeros(self.M, d)
        for j, (name, default) in enumerate((("ln_fg", ones), ("ln_fb", zeros))):
            key = "ln_f.weight" if j == 0 else "ln_f.bias"
            vals = torch.stack([s[key] if key in s else default[i] for i, s in enumerate(sd)])
            self.P[name] = torch.nn.Parameter(vals.to(device))
        self.register_buffer("norm_mode", torch.tensor([NORM_ID[s.norm] for s in specs], device=device))
        self.register_buffer("use_ln", (self.norm_mode > 0).view(-1, 1, 1, 1).float())
        self.fixed_gain = specs[0].gain is not None
        assert all((s.gain is None) == (not self.fixed_gain) for s in specs), "mix of fixed and free readouts"
        if self.fixed_gain:
            self.P["outv"] = stack("out.v"); self.P["outb"] = stack("out.bias")
            self.register_buffer("gain", torch.tensor([s.gain for s in specs], device=device).view(-1, 1, 1))
        else:
            self.P["outw"] = stack("out.weight"); self.P["outb"] = stack("out.bias")
        masks = []
        for i in range(n_layers):
            masks.append(torch.stack([nets[j]._mask(T, "cpu", i) for j in range(self.M)]).to(device))
        self.register_buffer("mask0", masks[0]); self.register_buffer("mask1", masks[1] if n_layers > 1 else masks[0])
        self.to(device)
        self.nets_cpu = nets

    # ---- forward -------------------------------------------------------
    def out_weight(self):
        if not self.fixed_gain:
            return self.P["outw"]
        v = self.P["outv"]
        return self.gain * v / v.norm(dim=-1, keepdim=True)

    def _attn(self, a, i, need_weights, mask_override=None):
        M, B, T, d = a.shape; H = self.H; dh = d // H
        qkv = torch.einsum("mbtd,mfd->mbtf", a, self.P[f"qkvw{i}"]) + self.P[f"qkvb{i}"][:, None, None]
        q, k, v = qkv.chunk(3, dim=-1)
        sh = lambda z: z.reshape(M, B, T, H, dh).permute(0, 1, 3, 2, 4)        # noqa: E731  [M,B,H,T,dh]
        q, k, v = sh(q), sh(k), sh(v)
        mask = (self.mask0 if i == 0 else self.mask1)[:, None, None]           # [M,1,1,T,T]
        if mask_override is not None:                                          # per-sample masks [M,B,T,T] (TASK15)
            mask = mask_override[:, :, None]
        if need_weights:                       # explicit scores: the penalties read them
            w = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(dh) + mask, dim=-1)
            a_out = w @ v
        else:                                  # fused kernel: no [T, T] tensor is materialised
            w = None
            a_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.expand(M, B, H, T, T))
        o = a_out.permute(0, 1, 3, 2, 4).reshape(M, B, T, d)
        o = torch.einsum("mbtd,med->mbte", o, self.P[f"ow{i}"]) + self.P[f"ob{i}"][:, None, None]
        return o, (w.mean(2) if need_weights else None)                        # head-averaged [M,B,T,T]

    def residuals(self, X, need_weights=False, mask=None):
        """X long [M, B, T] -> list of residual streams [M, B, T, d] (+ per-block attention)."""
        M, B, T = X.shape
        e = self.P["emb"][torch.arange(M, device=X.device)[:, None, None], X]
        x = e + self.P["pos"][:, :T][:, None]
        res = [x]; attn = []
        for i in range(self.L):
            g1, b1 = self.P[f"ln1g{i}"][:, None, None], self.P[f"ln1b{i}"][:, None, None]
            o, w = self._attn(_ln(x, g1, b1), i, need_weights, mask)
            x = x + o
            g2, b2 = self.P[f"ln2g{i}"][:, None, None], self.P[f"ln2b{i}"][:, None, None]
            h = _ln(x, g2, b2)
            h = torch.einsum("mbtd,mfd->mbtf", h, self.P[f"m0w{i}"]) + self.P[f"m0b{i}"][:, None, None]
            h = torch.einsum("mbtf,mdf->mbtd", F.gelu(h), self.P[f"m1w{i}"]) + self.P[f"m1b{i}"][:, None, None]
            x = x + h
            res.append(x); attn.append(w)
        return (res, attn) if need_weights else res

    def readout_input(self, x):
        normed = _ln(x, self.P["ln_fg"][:, None, None], self.P["ln_fb"][:, None, None])
        return self.use_ln * normed + (1 - self.use_ln) * x

    def logits(self, X, need_weights=False, mask=None):
        out = self.residuals(X, need_weights, mask)
        res, attn = out if need_weights else (out, None)
        h = self.readout_input(res[-1])
        z = torch.einsum("mbtd,mvd->mbtv", h, self.out_weight()) + self.P["outb"][:, None, None]
        return (z, attn) if need_weights else z

    # ---- unstacking ----------------------------------------------------
    def to_nets(self):
        """Copy the trained stacked weights back into ordinary CausalTransformers (on CPU)."""
        for j, net in enumerate(self.nets_cpu):
            sd = net.state_dict()
            put = lambda k, t: sd[k].copy_(t[j].detach().cpu())                # noqa: E731
            put("emb.weight", self.P["emb"]); put("pos", self.P["pos"])
            for i in range(self.L):
                for src, key in ((f"ln1g{i}", f"blocks.{i}.ln1.weight"), (f"ln1b{i}", f"blocks.{i}.ln1.bias"),
                                 (f"qkvw{i}", f"blocks.{i}.attn.in_proj_weight"), (f"qkvb{i}", f"blocks.{i}.attn.in_proj_bias"),
                                 (f"ow{i}", f"blocks.{i}.attn.out_proj.weight"), (f"ob{i}", f"blocks.{i}.attn.out_proj.bias"),
                                 (f"ln2g{i}", f"blocks.{i}.ln2.weight"), (f"ln2b{i}", f"blocks.{i}.ln2.bias"),
                                 (f"m0w{i}", f"blocks.{i}.mlp.0.weight"), (f"m0b{i}", f"blocks.{i}.mlp.0.bias"),
                                 (f"m1w{i}", f"blocks.{i}.mlp.2.weight"), (f"m1b{i}", f"blocks.{i}.mlp.2.bias")):
                    put(key, self.P[src])
            if "ln_f.weight" in sd:
                put("ln_f.weight", self.P["ln_fg"]); put("ln_f.bias", self.P["ln_fb"])
            if self.fixed_gain:
                put("out.v", self.P["outv"])
            else:
                put("out.weight", self.P["outw"])
            put("out.bias", self.P["outb"])
            net.load_state_dict(sd)
        return self.nets_cpu


def make_data(specs: list[Spec], pool: int = 4000, T: int = 48, device="cuda"):
    """Per-model training pool and exact next-token targets, identical to `cuts.train_routed`."""
    Xs, Ys = [], []
    for s in specs:
        m = H4.make_hmm4(s.r, s.delta, 1)
        X, _ = H4.sample(m, pool, T, seed=s.seed)
        Y = H4.next_token(m, H4.beliefs_seq(m, X))[:, :-1]
        Xs.append(torch.as_tensor(X)); Ys.append(torch.as_tensor(Y, dtype=torch.float32))
    return torch.stack(Xs).to(device), torch.stack(Ys).to(device)


def train_batched(model: BatchedTransformer, Xp, Yp, steps: int = 10000, batch: int = 128, lr: float = 1e-3,
                  log_every: int = 0, progress=None, tf32: bool = True):
    """One Adam over the stacked parameters; the loss is the sum of the per-model losses, so the
    gradient of each model's parameters is exactly the gradient it would get on its own."""
    M, pool, T = Xp.shape
    dev = Xp.device
    rngs = [np.random.default_rng(s.seed) for s in model.specs]
    mu1 = torch.tensor([s.mu_l1 for s in model.specs], device=dev)
    mu2 = torch.tensor([s.mu_l2 for s in model.specs], device=dev)
    need_w = bool((mu1.abs() + mu2.abs()).max() > 0)
    eye = torch.eye(T, device=dev, dtype=torch.bool)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = []
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    for step in range(steps):
        idx = torch.as_tensor(np.stack([r.integers(0, pool, batch) for r in rngs]), device=dev)
        X = torch.gather(Xp, 1, idx[:, :, None].expand(-1, -1, T))
        Y = torch.gather(Yp, 1, idx[:, :, None, None].expand(-1, -1, T - 1, Yp.shape[-1]))
        out = model.logits(X, need_weights=need_w)
        z, attn = out if need_w else (out, None)
        per_model = -(Y * F.log_softmax(z[:, :, :-1], -1)).sum(-1).mean((1, 2))      # [M]
        loss = per_model.sum()
        if need_w:
            pen1 = attn[0].masked_fill(eye, 0.0).sum(-1).mean((1, 2))
            pen2 = torch.diagonal(attn[1], offset=-1, dim1=2, dim2=3).mean((1, 2))
            loss = loss + (mu1 * pen1).sum() + (mu2 * pen2).sum()
        opt.zero_grad(set_to_none=True); loss.backward()
        # models whose final norm is frozen (or absent) do not train gamma / beta
        with torch.no_grad():
            free = (model.norm_mode == NORM_ID["learn"]).float()[:, None]
            for k in ("ln_fg", "ln_fb"):
                if model.P[k].grad is not None:
                    model.P[k].grad.mul_(free)
        opt.step()
        hist.append(per_model.detach().cpu().numpy())
        if log_every and (step + 1) % log_every == 0 and progress is not None:
            progress(step + 1, hist[-1])
    torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return np.stack(hist)          # [steps, M] per-model task loss
