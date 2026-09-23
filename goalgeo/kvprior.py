"""TASK14 (round 15): prior or recomputation? Belief transplants through the K/V cache of a
next-token transformer (docs/task14_prior_theory.md).

Position t+1 of a pre-LN causal transformer reads positions s <= t only through keys and values
computed from their residual streams res_i(s), one per block i. `forward_query` computes position
t+1 from an arbitrary set of such exported residuals (per layer, positions 0..t), optionally hiding
positions from its attention, so every intervention of the round is one call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import beliefcausal as BC
from . import filterstate as FS
from . import latentgoal as LG
from . import tfm as Tf
from . import tfm_batched as Tb

NEG = -1e9


# ---- training: plain next-token prediction on sampled tokens ------------------------------------
@dataclass
class LMSpec:
    L: int
    T: int          # observations per training sequence
    seed: int

    @property
    def name(self):
        return f"lm_channel_L{self.L}_T{self.T}_s{self.seed}"


def lm_data(env, n, T, seed):
    """Inputs [BOS, o_1..o_T] and next-token targets o_1..o_{T+1} (as vocabulary indices)."""
    Xf, _, _ = LG.sample(env, n, T=T + 1, seed=seed)
    return Xf[:, :-1], Xf[:, 1:]


def drop_visible(p, n, T, gen=None, keep_prev=True, device="cpu"):
    """Causal visibility [n, T, T] (query, key) with every key s < u - 1 (or s < u without keep_prev)
    dropped independently with probability p; the query itself (and u - 1) always visible. TASK15."""
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    keep = torch.rand(n, T, T, generator=gen, device=device) >= p
    guaranteed = torch.eye(T, dtype=torch.bool, device=device)
    if keep_prev:
        guaranteed = guaranteed | torch.diag(torch.ones(T - 1, dtype=torch.bool, device=device), -1)
    return causal & (keep | guaranteed)


def train_lm_stack(specs: list[LMSpec], steps: int, n_pool: int = 100000, batch: int = 128, lr: float = 1e-3,
                   device="cuda", log=None, drop: list[float] | None = None):
    """drop: per-model probability of dropping each historical K/V entry older than the previous
    position, resampled every step (TASK15); None trains with full attention."""
    L, T = specs[0].L, specs[0].T; assert all(s.L == L and s.T == T for s in specs)
    env = LG.make_env("channel", 4); V = env.V
    model = Tb.BatchedTransformer([Tb.Spec(seed=s.seed, norm="learn") for s in specs], n_vocab=V, T=T + 1,
                                  n_layers=L, d=64, n_heads=4, mlp=128, device=device)
    data = [lm_data(env, n_pool, T, s.seed + 1000) for s in specs]
    Xp = torch.stack([torch.as_tensor(x) for x, _ in data]).to(device)
    Yp = torch.stack([torch.as_tensor(y) for _, y in data]).to(device)
    mask = torch.zeros(V, device=device); mask[0] = NEG                         # BOS is never a target
    rngs = [np.random.default_rng(s.seed + 7) for s in specs]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prev = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = True
    hist = []
    for step in range(steps):
        idx = torch.as_tensor(np.stack([r.integers(0, n_pool, batch) for r in rngs]), device=device)
        X = torch.gather(Xp, 1, idx[:, :, None].expand(-1, -1, T + 1)); Y = torch.gather(Yp, 1, idx[:, :, None].expand(-1, -1, T + 1))
        am = None
        if drop is not None:
            vis = torch.stack([drop_visible(p, batch, T + 1, device=device) for p in drop])       # [M, B, T, T]
            am = torch.zeros(vis.shape, device=device).masked_fill(~vis, float("-inf"))
        z = model.logits(X, mask=am) + mask
        per = F.cross_entropy(z.reshape(-1, V), Y.reshape(-1), reduction="none").view(len(specs), -1).mean(1)
        opt.zero_grad(set_to_none=True); per.sum().backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            hist.append(per.detach().cpu().numpy())
            if log and step % 5000 == 0:
                log(step, hist[-1])
    torch.backends.cuda.matmul.allow_tf32 = prev
    return model.to_nets(), np.stack(hist)


def make_net(spec: LMSpec):
    env = LG.make_env("channel", 4)
    return Tf.CausalTransformer(n_vocab=env.V, d=64, n_heads=4, n_layers=spec.L, mlp=128, T=spec.T + 1, seed=spec.seed,
                                final_norm="learn")


# ---- the model's predictive and the exact one ----------------------------------------------------
@torch.no_grad()
def predictive(net, X, batch=2000):
    """P(o_{t+1} | o_<=t) over the M observation tokens, at every position: [n, T+1, M]."""
    out = []
    for i in range(0, len(X), batch):
        z = net.forward_all(torch.as_tensor(X[i:i + batch], device=net.pos.device))
        out.append(torch.softmax(z[..., 1:], -1).double().cpu().numpy())
    return np.concatenate(out)


# ---- computing position t+1 from exported residuals ---------------------------------------------
@torch.no_grad()
def residuals(net, X):
    """[res_0, ..., res_L] for full sequences X, each [n, T, d] (float64 on CPU)."""
    return [r.double().cpu().numpy() for r in net.residuals(torch.as_tensor(X, device=net.pos.device))]


@torch.no_grad()
def forward_query(net, src, tok, pos, visible=None):
    """Position `pos` (= t+1) with token `tok` [n], reading exported residuals src[i] [n, S, d]
    (block i's input at positions 0..S-1 = 0..t). visible: bool [S] of source positions the query
    may attend to (default all). Returns (residuals [res_1..res_L] of the query, u, predictive
    over the M observation tokens, head-averaged attention on the sources per block)."""
    dev = net.pos.device
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)        # noqa: E731
    tok = torch.as_tensor(tok, device=dev); n = len(tok)
    x = net.emb(tok) + net.pos[pos]
    S = src[0].shape[1]
    allow = torch.ones(n, S + 1, dtype=torch.bool, device=dev)
    if visible is not None:                                                        # [S] or per sample [n, S]
        allow[:, :S] = torch.as_tensor(visible, device=dev)
    res, attn = [], []
    for i, blk in enumerate(net.blocks):
        a = blk.ln1(x); kv = torch.cat([blk.ln1(to(src[i])), a[:, None]], 1)       # [n, S+1, d]
        W, b = blk.attn.in_proj_weight, blk.attn.in_proj_bias; d = x.shape[-1]; H = blk.attn.num_heads; dh = d // H
        q = a @ W[:d].T + b[:d]; k = kv @ W[d:2 * d].T + b[d:2 * d]; v = kv @ W[2 * d:].T + b[2 * d:]
        q = q.view(n, H, dh); k = k.view(n, S + 1, H, dh).transpose(1, 2); v = v.view(n, S + 1, H, dh).transpose(1, 2)
        s = (k @ q[..., None]).squeeze(-1) / dh ** 0.5                              # [n, H, S+1]
        s = s.masked_fill(~allow[:, None, :], float("-inf"))
        w = torch.softmax(s, -1)
        o = (w[..., None] * v).sum(2).reshape(n, d)
        x = x + blk.attn.out_proj(o)
        x = x + blk.mlp(blk.ln2(x))
        res.append(x.double().cpu().numpy()); attn.append(w.mean(1).double().cpu().numpy())
    u = net.ln_f(x)
    p = torch.softmax(net.out(u)[:, 1:], -1).double().cpu().numpy()
    return res, u.double().cpu().numpy(), p, attn


@torch.no_grad()
def predictive_masked(net, X, vis, batch=1000):
    """Next-token predictive at every position with per-sample visibility vis [n, T, T] (query, key)."""
    dev = net.pos.device; out = []
    for b0 in range(0, len(X), batch):
        Xb = torch.as_tensor(X[b0:b0 + batch], device=dev); V = torch.as_tensor(vis[b0:b0 + batch], device=dev)
        n, T = Xb.shape
        x = net.emb(Xb) + net.pos[:T]
        for blk in net.blocks:
            a = blk.ln1(x); W, bb = blk.attn.in_proj_weight, blk.attn.in_proj_bias; d = x.shape[-1]; H = blk.attn.num_heads; dh = d // H
            q, k, v = (a @ W.T + bb).chunk(3, -1)
            sh = lambda z: z.view(n, T, H, dh).transpose(1, 2)                         # noqa: E731
            sc = (sh(q) @ sh(k).transpose(-1, -2)) / dh ** 0.5
            w = torch.softmax(sc.masked_fill(~V[:, None], float("-inf")), -1)
            x = x + blk.attn.out_proj((w @ sh(v)).transpose(1, 2).reshape(n, T, d))
            x = x + blk.mlp(blk.ln2(x))
        out.append(torch.softmax(net.out(net.ln_f(x))[..., 1:], -1).double().cpu().numpy())
    return np.concatenate(out)


# ---- read-out -----------------------------------------------------------------------------------
def along(z, zA, zB):
    """Pooled position along A -> B and the perpendicular residual as a fraction of |B - A|."""
    D = zB - zA; r = z - zA
    lam = float((r * D).sum() / (D * D).sum())
    perp = r - np.einsum("n,nd->nd", (r * D).sum(1) / np.clip((D * D).sum(1), 1e-12, None), D)
    return lam, float(np.sqrt((perp ** 2).sum() / (D * D).sum()))


def centred_log(p):
    lp = np.log(np.clip(p, 1e-12, None)); return lp - lp.mean(-1, keepdims=True)


def update(env, J, x):
    """One exact filter step from joint states J [n, K, C] with observation tokens x [n]."""
    return BC.filter_continue(env, J, x[:, None])[:, 1]
