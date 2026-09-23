"""TASK10 experiment 2: how the required contrast C = g * D * cos(theta) is factorised.

Claim B (rounds/r11_implementation_freedom/THEORY.md): C is fixed by the function (it is a difference of output
log-odds), while the architecture decides which of the three factors can absorb a change of
budget. A frozen final LayerNorm pins ||h~|| = sqrt(d), which caps D at 2*sqrt(d) and hence
caps C at 4*c*sqrt(d) for a fixed-gain readout with row norm c -- a hard failure boundary at
c_crit = C*(delta) / (4*sqrt(d)) that no amount of training can cross."""

from __future__ import annotations

import numpy as np
import torch

from . import hmm4 as H4
from . import prominence as Pr
from .cuts import kl_to_floor


def required_gap(delta: float) -> float:
    """Log-odds gap between the two branches at the decision position, at the Bayes floor."""
    return float(2 * np.log((0.5 + delta) / (0.5 - delta)))


def ceiling(gain: float | None, norm: str, d: int) -> float:
    """Proposition B.3: the largest contrast the interface can express, or inf if unbounded."""
    if norm != "frozen" or gain is None:
        return float("inf")
    return 4.0 * gain * np.sqrt(d)


@torch.no_grad()
def interface(net, m: H4.HMM4, ev: dict, seed: int = 0) -> dict[str, float]:
    """C, g, D, cos(theta) at the post-final-norm interface, at the decision position (the last
    filler of each branch, whose next token is the one the cue disambiguates)."""
    dev = next(net.parameters()).device
    X, Z = ev["X"], ev["Z"]
    P, _ = Pr.positions(m, Z)
    # a decision position whose cue fell before the start of the sequence carries no branch
    # information at all (its log-odds are 0 by symmetry); it is not a state of the pair.
    for key in ("preA", "preB"):
        P[key] = P[key].copy(); P[key][:, :m.k] = False
    res = net.residuals(torch.as_tensor(X, device=dev))
    Rf = net.ln_f(res[-1]).cpu().numpy().astype(np.float64)
    Rpre = res[-1].cpu().numpy().astype(np.float64)
    W = net.out.weight.detach().cpu().numpy().astype(np.float64)
    wd = W[H4.TOK["x"]] - W[H4.TOK["y"]]
    dh = Rf[P["preA"]].mean(0) - Rf[P["preB"]].mean(0)
    g, D = float(np.linalg.norm(wd)), float(np.linalg.norm(dh))
    C = float(wd @ dh)
    r = {"C": C, "g": g, "D": D, "cos_theta": C / (g * D + 1e-12),
         "C_required": required_gap(m.delta),
         "postnorm_norm": float(np.linalg.norm(Rf.reshape(-1, Rf.shape[-1]), axis=1).mean()),
         "prenorm_norm": float(np.linalg.norm(Rpre.reshape(-1, Rpre.shape[-1]), axis=1).mean()),
         "sep_prenorm": float(np.linalg.norm(Rpre[P["preA"]].mean(0) - Rpre[P["preB"]].mean(0))),
         "w_row_norm": float(np.linalg.norm(W, axis=1).mean())}
    r["gamma_norm"] = (float(net.ln_f.weight.norm()) if net.final_norm == "learn"
                       else float(np.sqrt(net.hidden_size)) if net.final_norm == "frozen" else float("nan"))
    r["C_over_required"] = r["C"] / r["C_required"]
    r.update(kl_to_floor(net, ev, m))
    return r


def measure(net, m: H4.HMM4, ev: dict, gain: float | None, seed: int = 0) -> dict[str, float]:
    r = interface(net, m, ev, seed=seed)
    r["ceiling"] = ceiling(gain, net.final_norm, net.hidden_size)
    r["C_over_ceiling"] = r["C"] / r["ceiling"] if np.isfinite(r["ceiling"]) else float("nan")
    r["converged"] = float(r["kl_relevant"] < 0.01)
    return r
