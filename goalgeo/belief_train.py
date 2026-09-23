"""TASK11: training for the hidden-goal round.

Transformers: the round-11 stacked trainer's model (`tfm_batched.BatchedTransformer`), with a
per-model output mask so that models with different objectives (output sizes K, K+1, K+2) share
one stack; the unused logits are removed from the softmax, not trained down.
GRUs: `seqmodels.SeqNet`, one CPU process per model."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import latentgoal as LG
from . import seqmodels as Sm
from . import tfm_batched as Tb

NEG = -1e9


@dataclass
class Job:
    arch: str                    # 'tfm' or 'gru'
    kind: str                    # 'iid' or 'channel'
    K: int
    objective: str
    seed: int
    netho: bool = False          # training pool avoids the conflict region
    extra: dict = field(default_factory=dict)

    @property
    def name(self):
        n = f"_n{self.extra['hidden']}" if "hidden" in self.extra else ""
        return f"{self.arch}_{self.kind}_K{self.K}_{self.objective}{'_netho' if self.netho else ''}{n}_s{self.seed}"


def pool(job: Job, n: int):
    env = LG.make_env(job.kind, job.K)
    if job.netho:
        X, _ = LG.sample_excluding(env, n, seed=job.seed + 101)
    else:
        X, _, _ = LG.sample(env, n, seed=job.seed + 101)
    return X, LG.targets(env, X, job.objective)


def _pad(Y, V):
    out = np.zeros(Y.shape[:-1] + (V,), np.float32); out[..., :Y.shape[-1]] = Y
    return out


# ---- transformers ------------------------------------------------------------------------
TFM_KW = dict(d=64, n_heads=4, n_layers=2, mlp=128)


def train_tfm_stack(jobs: list[Job], steps: int, n_pool: int = 20000, batch: int = 128, lr: float = 1e-3,
                    device="cuda", log=None):
    """All jobs must share K (one vocabulary). Returns (unstacked CausalTransformers, loss history)."""
    K = jobs[0].K; assert all(j.K == K for j in jobs)
    V = K + 3; T1 = LG.T_OBS + 1
    specs = [Tb.Spec(seed=j.seed, norm="learn", label=j.name) for j in jobs]
    model = Tb.BatchedTransformer(specs, n_vocab=V, T=T1, device=device, **TFM_KW)
    env0 = LG.make_env(jobs[0].kind, K)
    Xs, Ys, masks = [], [], []
    for j in jobs:
        X, Y = pool(j, n_pool)
        Xs.append(torch.as_tensor(X)); Ys.append(torch.as_tensor(_pad(Y, V)))
        m = torch.full((V,), NEG); m[:LG.out_dim(env0, j.objective)] = 0.0; masks.append(m)
    Xp = torch.stack(Xs).to(device); Yp = torch.stack(Ys).to(device); mask = torch.stack(masks).to(device)
    rngs = [np.random.default_rng(j.seed + 7) for j in jobs]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prev = torch.backends.cuda.matmul.allow_tf32; torch.backends.cuda.matmul.allow_tf32 = True
    hist = []
    for step in range(steps):
        idx = torch.as_tensor(np.stack([r.integers(0, n_pool, batch) for r in rngs]), device=device)
        X = torch.gather(Xp, 1, idx[:, :, None].expand(-1, -1, T1))
        Y = torch.gather(Yp, 1, idx[:, :, None, None].expand(-1, -1, T1, V))
        z = model.logits(X) + mask[:, None, None]
        per_model = -(Y * F.log_softmax(z, -1)).sum(-1).mean((1, 2))
        opt.zero_grad(set_to_none=True); per_model.sum().backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append(per_model.detach().cpu().numpy())
            if log and step % 2000 == 0:
                log(step, hist[-1])
    torch.backends.cuda.matmul.allow_tf32 = prev
    nets = model.to_nets()
    for net, m in zip(nets, mask.cpu()):
        net.out_mask = m.clone()
    return nets, np.stack(hist)


# ---- GRUs --------------------------------------------------------------------------------
def train_gru(job: Job, steps: int, n_pool: int = 20000, batch: int = 128, lr: float = 3e-3):
    torch.set_num_threads(1)
    env = LG.make_env(job.kind, job.K)
    X, Y = pool(job, n_pool)
    net = Sm.SeqNet(n_vocab=env.V, hidden=job.extra.get("hidden", 64), emb=16, out_dim=LG.out_dim(env, job.objective), seed=job.seed)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y, dtype=torch.float32)
    rng = np.random.default_rng(job.seed + 7)
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []
    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, n_pool, batch))
        loss = -(Yt[idx] * F.log_softmax(net.forward_all(Xt[idx]), -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps - 1:
            hist.append(loss.item())
    return {k: v.detach().clone() for k, v in net.state_dict().items()}, np.array(hist)


def make_gru(job: Job, state=None):
    env = LG.make_env(job.kind, job.K)
    net = Sm.SeqNet(n_vocab=env.V, hidden=job.extra.get("hidden", 64), emb=16, out_dim=LG.out_dim(env, job.objective), seed=job.seed)
    if state is not None:
        net.load_state_dict(state)
    return net.eval()


def load_net(job: Job, path=None):
    """A trained network from its checkpoint, or (path None) the untrained network of that seed.
    Transformers carry `out_mask`, the per-objective output mask used in training."""
    from . import tfm as Tf
    env = LG.make_env(job.kind, job.K)
    if job.arch == "gru":
        return make_gru(job, None if path is None else torch.load(path)["state"])
    net = Tf.CausalTransformer(n_vocab=env.V, T=LG.T_OBS + 1, seed=job.seed, final_norm="learn", **TFM_KW)
    mask = torch.full((env.V,), NEG); mask[:LG.out_dim(env, job.objective)] = 0.0
    if path is not None:
        ck = torch.load(path); net.load_state_dict(ck["state"]); mask = ck["out_mask"]
    net.out_mask = mask
    return net.eval()
