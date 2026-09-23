"""TASK10 experiment 1: complete cuts vs single nodes under forced routing.

Claim A (rounds/r11_implementation_freedom/THEORY.md): an interchange effect measured on a *complete causal cut*
between a flipped input token and a read node equals the behavioural divergence between the
two inputs, so it descends to functional equivalence; a single-node effect measures only the
share of the computation routed through that node, which the architecture and the optimiser
are free to move. Here the routing is moved on purpose, by hard attention masks or by an
attention penalty in the loss, and both kinds of effect are measured on the same models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import hmm4 as H4
from . import prominence as Pr

# patch sets as (layer, position offsets from the cue); read node is the output at t+1
CUT_SETS: dict[str, tuple[int, tuple[int, ...]]] = {
    "L0{t}":            (0, (0,)),          # complete: the token enters only through its embedding
    "L1{t}":            (1, (0,)),          # incomplete: block 1 may already have copied the cue to t+1
    "L1{t+1}":          (1, (1,)),          # incomplete: block 2 at t+1 can still read t
    "L1{t,t+1}":        (1, (0, 1)),        # complete (minimal)
    "L1{t,t+1,t+2}":    (1, (0, 1, 2)),     # complete (superset)
    "L1{t,t+2}":        (1, (0, 2)),        # incomplete: leaves the t+1 route open
}
COMPLETE = ("L0{t}", "L1{t,t+1}", "L1{t,t+1,t+2}")


def cue_anchors(m: H4.HMM4, Z: np.ndarray, n_anchor: int, rng, margin: int = 3) -> np.ndarray:
    """(sequence, position) pairs whose state is a cue (P or Q) with room for the read node."""
    P, _ = Pr.positions(m, Z)
    mask = P["cueA"] | P["cueB"]
    mask[:, Z.shape[1] - margin:] = False
    idx = np.argwhere(mask)
    return idx[rng.choice(len(idx), min(n_anchor, len(idx)), replace=False)]


def minimal_pairs(X: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Anchored sequences and their minimal flips: the cue token p<->q at the anchor position,
    everything else identical. Both members have positive probability under the chain."""
    i, t = anchors[:, 0], anchors[:, 1]
    Xa = X[i].copy(); Xf = Xa.copy()
    tok = Xf[np.arange(len(i)), t]
    assert set(np.unique(tok)) <= {H4.TOK["p"], H4.TOK["q"]}, "anchors must sit on cue tokens"
    Xf[np.arange(len(i)), t] = np.where(tok == H4.TOK["p"], H4.TOK["q"], H4.TOK["p"])
    return Xa, Xf


def _js(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl = lambda a: (a * (torch.log(a + 1e-12) - torch.log(m + 1e-12))).sum(-1)   # noqa: E731
    return 0.5 * (kl(p) + kl(q))


@torch.no_grad()
def cut_effects(net, Xa: np.ndarray, Xf: np.ndarray, t: np.ndarray, sets=CUT_SETS) -> dict[str, float]:
    """Interchange effects at the read node out(t+1): patch each set of residual sites with the
    values they take on the flipped sequence, and compare the prediction to the clean one.
    Also returns the behavioural divergence, which Theorem A.3 says every complete cut must hit."""
    dev = next(net.parameters()).device
    Xa_t, Xf_t = torch.as_tensor(Xa, device=dev), torch.as_tensor(Xf, device=dev)
    ar = torch.arange(len(t), device=dev); tt = torch.as_tensor(t, device=dev)
    res_a, res_f = net.residuals(Xa_t), net.residuals(Xf_t)
    p_clean = torch.softmax(net.out(net.ln_f(res_a[-1]))[ar, tt + 1], -1)
    p_flip = torch.softmax(net.out(net.ln_f(res_f[-1]))[ar, tt + 1], -1)
    out = {"phi_behav": float(_js(p_flip, p_clean).mean())}
    for name, (layer, offs) in sets.items():
        R = res_a[layer].clone()
        for off in offs:
            R[ar, tt + off] = res_f[layer][ar, tt + off]
        p = torch.softmax(net.run_from(R, layer)[ar, tt + 1], -1)
        out[f"phi_{name}"] = float(_js(p, p_clean).mean())
        # Theorem A.3 is an identity of distributions, not of averages: for a complete cut the
        # patched prediction must BE the prediction on the flipped input, anchor by anchor.
        out[f"dev_{name}"] = float((p - p_flip).abs().max())
    return out


@torch.no_grad()
def route_weights(net, Xa: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """The two routes by which the cue at t can reach the prediction at t+1: block-2 attention
    from t+1 back to t, and block-1 attention from t+1 back to t (which copies it forward)."""
    dev = next(net.parameters()).device
    _, attn = net.residuals(torch.as_tensor(Xa, device=dev), need_weights=True)
    ar = torch.arange(len(t), device=dev); tt = torch.as_tensor(t, device=dev)
    return {"attn1_t1_to_t": float(attn[0][ar, tt + 1, tt].mean()),
            "attn2_t1_to_t": float(attn[1][ar, tt + 1, tt].mean()),
            "attn2_t1_to_t1": float(attn[1][ar, tt + 1, tt + 1].mean())}


@torch.no_grad()
def kl_to_floor(net, ev: dict, m: H4.HMM4, n: int = 1000) -> dict[str, float]:
    """Mean KL(exact posterior predictive || model), i.e. the excess over the Bayes floor."""
    dev = next(net.parameters()).device
    X, Y, Z = ev["X"][:n], ev["Y"][:n, :-1], ev["Z"][:n]
    logp = torch.log_softmax(net.forward_all(torch.as_tensor(X, device=dev)), -1)[:, :-1]
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    kl = (Yt * (torch.log(Yt + 1e-12) - logp)).sum(-1).cpu().numpy()
    rel = m.relevant[Z[:, 1:]]
    return {"kl": float(kl.mean()), "kl_relevant": float(kl[rel].mean()) if rel.any() else 0.0}


def attention_penalty(attn, kind: str) -> torch.Tensor:
    """Soft route restrictions, the gradient version of the hard masks.
    'pen_l2': block-2 attention to the previous position (discourages the fetch route);
    'pen_l1': block-1 off-diagonal attention mass (discourages the copy-forward route)."""
    if kind == "pen_l2":
        return torch.diagonal(attn[1], offset=-1, dim1=1, dim2=2).mean()
    if kind == "pen_l1":
        T = attn[0].shape[-1]
        eye = torch.eye(T, device=attn[0].device, dtype=torch.bool)
        return attn[0].masked_fill(eye, 0.0).sum(-1).mean()
    raise ValueError(kind)


def train_routed(net, m: H4.HMM4, steps: int = 4000, batch: int = 128, seed: int = 0, T: int = 48,
                 lr: float = 1e-3, pool: int = 4000, penalty: str | None = None, mu: float = 1.0):
    """Sequential next-token training with exact targets (as in TASK9) plus an optional
    attention penalty. Hard route restrictions live in the model (`attn_diag`), not here."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    dev = next(net.parameters()).device
    X, Z = H4.sample(m, pool, T, seed=seed)
    Y = H4.next_token(m, H4.beliefs_seq(m, X))[:, :-1]
    Xt = torch.as_tensor(X, device=dev); Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, pool, batch), device=dev)
        need = penalty is not None
        res, attn = net.residuals(Xt[idx], need_weights=True) if need else (net.residuals(Xt[idx]), None)
        logits = net.out(net.ln_f(res[-1]))[:, :-1]
        loss = -(Yt[idx] * F.log_softmax(logits, -1)).sum(-1).mean()
        task = loss.detach().item()
        if need:
            loss = loss + mu * attention_penalty(attn, penalty)
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(task)
    return hist


def measure(net, m: H4.HMM4, ev: dict, n_anchor: int = 200, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    anchors = cue_anchors(m, ev["Z"], n_anchor, rng)
    Xa, Xf = minimal_pairs(ev["X"], anchors)
    r = cut_effects(net, Xa, Xf, anchors[:, 1])
    r.update(route_weights(net, Xa, anchors[:, 1]))
    r.update(kl_to_floor(net, ev, m))
    r["n_anchor"] = len(anchors)
    return r
