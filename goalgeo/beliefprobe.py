"""TASK11: affine recoverability of the posterior log-odds, with held-out splits.

All probes are unregularised least squares on [h, 1], so every number here is exactly invariant
to invertible affine maps of h (docs/task11_theory.md §2.1)."""

from __future__ import annotations

import numpy as np
import torch

from . import latentgoal as LG

EXT_FIT, EXT_TEST = 2.0, 3.0
T_SPLIT = 12


# ---- the evaluation set --------------------------------------------------------------------
class EvalSet:
    """Shared evaluation sequences for one (environment, K), flattened to states at t = 1..T."""

    def __init__(self, kind: str, K: int, n: int = 8000, seed: int = 12345):
        self.env = env = LG.make_env(kind, K)
        self.X, self.G, _ = LG.sample(env, n, seed=seed)
        J = LG.filter_joint(env, self.X)
        b = J.sum(-1)
        self.b_full = b                                           # [n, T+1, K] incl. BOS
        self.b = b[:, 1:].reshape(-1, K)
        self.y = LG.log_odds(self.b)
        self.counts = LG.counts(self.X, env.M)[:, 1:].reshape(-1, env.M)
        T = self.X.shape[1] - 1
        self.t = np.tile(np.arange(1, T + 1), n)
        self.seq = np.repeat(np.arange(n), T)
        self.conflict = LG.in_conflict(self.b, LG.CONFLICT[kind])
        ymax = np.abs(self.y).max(1)
        self.ext_fit, self.ext_test = ymax < EXT_FIT, ymax >= EXT_TEST
        self.action = LG.act_hard(self.b).argmax(-1)               # ties are measure-zero off t=0
        self.targets = {o: LG.targets(env, self.X, o) for o in LG.OBJECTIVES}   # [n, T+1, dim]
        rng = np.random.default_rng(seed)
        self.fold = rng.permutation(n)[self.seq] % 5              # sequence-level folds

    def splits(self):
        """name -> (fit mask, test mask). IID is the 5-fold split and handled separately."""
        return {"EXT": (self.ext_fit, self.ext_test),
                "CONF": (~self.conflict, self.conflict),
                "TIME": (self.t <= T_SPLIT, self.t > T_SPLIT)}


# ---- probes --------------------------------------------------------------------------------
def _fit(H, Y):
    H1 = np.c_[H, np.ones(len(H))]
    W, *_ = np.linalg.lstsq(H1, Y, rcond=None)
    return W


def _pred(H, W):
    return np.c_[H, np.ones(len(H))] @ W


def probe_pred(H, Y, fit, test):
    return _pred(H[test], _fit(H[fit], Y[fit]))


def cv_pred(H, Y, fold):
    P = np.zeros_like(Y, dtype=float)
    for f in range(5):
        te = fold == f
        P[te] = probe_pred(H, Y, ~te, te)
    return P


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def rmse(Y, P):
    return float(np.sqrt(((Y - P) ** 2).sum(1).mean()))


def kl_from_logodds(b, yhat):
    z = np.c_[yhat, np.zeros(len(yhat))]
    z = z - z.max(1, keepdims=True); lq = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float((b * (np.log(np.clip(b, 1e-300, None)) - lq)).sum(1).mean())


def measure_site(H, E: EvalSet) -> dict:
    """Every probe number for one representation site: H [states, d]."""
    H = np.asarray(H, np.float64)
    out = {}
    Py = cv_pred(H, E.y, E.fold)
    out["IID_r2y"] = r2(E.y, Py); out["IID_rmse"] = rmse(E.y, Py); out["IID_kl"] = kl_from_logodds(E.b, Py)
    out["IID_r2b"] = r2(E.b, cv_pred(H, E.b, E.fold))
    Pc = cv_pred(E.counts, E.y, E.fold)
    out["G_count"] = float(1 - ((E.y - Py) ** 2).sum() / ((E.y - Pc) ** 2).sum()) if ((E.y - Pc) ** 2).sum() > 1e-9 else float("nan")
    onehot = np.eye(E.env.K + 1)[E.action]
    out["action_acc"] = float((cv_pred(H, onehot, E.fold).argmax(1) == E.action).mean())
    for name, (fit, test) in E.splits().items():
        P = probe_pred(H, E.y, fit, test)
        out[f"{name}_r2y"] = r2(E.y[test], P); out[f"{name}_rmse"] = rmse(E.y[test], P)
        out[f"{name}_kl"] = kl_from_logodds(E.b[test], P)
        if name == "EXT":
            out["EXT_r2b"] = r2(E.b[test], probe_pred(H, E.b, fit, test))
    return out


def baselines(E: EvalSet) -> dict:
    tf = E.t[:, None].astype(float)
    return {"counts": measure_site(E.counts, E), "mean_t": measure_site(np.c_[E.counts / tf, tf], E)}


# ---- extracting sites ----------------------------------------------------------------------
@torch.no_grad()
def tfm_sites(net, X, batch: int = 2000, device="cuda") -> dict:
    """res0, res1, res2, u at t = 1..T, flattened, plus the model's output distribution."""
    net = net.to(device).eval(); mask = net.out_mask.to(device)
    acc = {k: [] for k in ("res0", "res1", "res2", "u", "p")}
    for i in range(0, len(X), batch):
        Xb = torch.as_tensor(X[i:i + batch], device=device)
        res = net.residuals(Xb); u = net.ln_f(res[-1])
        z = net.out(u) + mask
        for k, v in (("res0", res[0]), ("res1", res[1]), ("res2", res[2]), ("u", u), ("p", torch.softmax(z, -1))):
            acc[k].append(v[:, 1:].reshape(-1, v.shape[-1]).double().cpu().numpy())
    out = {k: np.concatenate(v) for k, v in acc.items()}
    dim = int((mask == 0).sum()); out["p"] = out["p"][:, :dim]
    net.cpu()
    return out


@torch.no_grad()
def gru_sites(net, X, batch: int = 2000) -> dict:
    hs, ps = [], []
    for i in range(0, len(X), batch):
        h = net.states(torch.as_tensor(X[i:i + batch]))
        hs.append(h[:, 1:].reshape(-1, h.shape[-1]).double().numpy())
        ps.append(torch.softmax(net.out(h), -1)[:, 1:].reshape(-1, net.out.out_features).double().numpy())
    return {"h": np.concatenate(hs), "p": np.concatenate(ps)}


def behaviour(p, E: EvalSet, objective: str) -> dict:
    """Output quality against the exact target, overall and in / out of the conflict region."""
    Y = E.targets[objective][:, 1:].reshape(-1, p.shape[-1])
    kl = (Y * (np.log(np.clip(Y, 1e-300, None)) - np.log(np.clip(p, 1e-300, None)))).sum(1)
    out = {"kl": float(kl.mean()), "kl_in": float(kl[E.conflict].mean()), "kl_out": float(kl[~E.conflict].mean())}
    if objective.startswith("act"):
        q = LG.q_values(E.b); best = q >= q.max(1, keepdims=True) - 1e-9
        untied = best.sum(1) == 1
        agree = best[np.arange(len(p)), p.argmax(1)]
        out["act_agree"] = float(agree[untied].mean())
    return out
