"""Round 20: edits of prefix-token states in a probe-defined subspace, and their effect on decisions.

The interface is the residual stream entering block l at the prefix tokens (positions before the goal token).
An edit replaces, inside a chosen subspace, the recipient's state by the donor's at the same position, and keeps the
recipient's state outside it. The subspace of the `belief` edit is the row space of an affine decoder of the exact
posterior; whether that is the network's own interface is what the round tests.
"""

from __future__ import annotations

import torch

from goalgeo import mazemodel as MM
from goalgeo.navprobe import Affine


def projector(W):
    """Orthogonal projector on the column space of W [d, k] (float64)."""
    U, S, _ = torch.linalg.svd(W.double(), full_matrices=False)
    U = U[:, S > S.max() * 1e-8]
    return U @ U.T


def decoder_subspaces(X, Y, rank=7, seed=0):
    """From prefix states X [N, d] and posteriors Y [N, n]: the decoder, the projector on its row space, the projector
    on the `rank` directions that carry most of the decoded posterior's variance, and a random projector of the
    full rank."""
    dec = Affine(X, Y)
    W = dec.W                                                        # [d, n], acts on raw states
    P = projector(W)
    Z = (X.double() - X.double().mean(0)) @ W                        # decoded, centred
    _, _, Vt = torch.linalg.svd(Z, full_matrices=False)
    Pk = projector(W @ Vt[:rank].T)
    k = int(round(torch.trace(P).item()))
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(X.shape[1], k, generator=g, dtype=torch.float64))
    Q = Q.to(X.device)
    return dec, dict(belief=P, belief_rank=Pk, random_subspace=Q @ Q.T), k


def make_edit(kind, xA, xB, subs, gen):
    """New states for the masked positions. xA, xB [M, d] float32."""
    d = (xB - xA).double()
    if kind == "whole":
        return xB
    if kind in ("belief", "belief_rank", "random_subspace"):
        return (xA.double() + d @ subs[kind]).float()
    if kind == "complement":
        return (xB.double() - d @ subs["belief"]).float()
    if kind == "random_norm":
        r = torch.randn(d.shape, device=d.device, generator=gen, dtype=torch.float64)
        return (xA.double() + r / r.norm(dim=1, keepdim=True) * (d @ subs["belief"]).norm(dim=1, keepdim=True)).float()
    raise ValueError(kind)


EDITS = ("whole", "belief", "belief_rank", "complement", "random_subspace", "random_norm")


@torch.no_grad()
def states(net, tok, l):
    """Residual stream entering block l. [N, L, d]"""
    return net(tok, record=True)[2]["resid"][l]


def prefix_mask(tok, pos):
    return torch.arange(tok.shape[1], device=tok.device)[None] < pos[:, None]


@torch.no_grad()
def run(net, tok, at, l=None, new=None, mask=None, ablate=None, goal_pos=None):
    """Action distribution and final residual at positions `at`. If `new` is given, the states entering block l at
    `mask` are set to it. ablate [N, d]: the first attention layer's output at goal_pos is set to it."""
    n = torch.arange(len(tok), device=tok.device)
    patch = {}
    if new is not None:
        def f(x):
            x = x.clone(); x[mask] = new
            return x
        patch[("resid", l)] = f
    if ablate is not None:
        def g(a):
            a = a.clone(); a[n, goal_pos] = ablate
            return a
        patch[("attn", 0)] = g
    lg, _, rec = net(tok, record=True, patch=patch)
    return lg[n, at].softmax(-1), rec["resid"][net.nl][n, at], rec
