"""Round 17: activations at decision tokens, held-out decoders of the exact belief, and the policy on fixed histories.

Sites, in the order the computation visits them (9 for 4 layers):
    L0.pre  L0.mid  L1.pre  L1.mid  ...  L3.mid  L3.post
'pre' is the residual stream entering a block (before attention), 'mid' is after its attention and before
its MLP, and block l's 'post' is block l + 1's 'pre'.

Targets (7 columns): the four goal probabilities b(g), the marginals P(right) and P(bottom), and the
interaction (P(right) - 1/2)(P(bottom) - 1/2). With a uniform prior and independent reports
    b(g) = 1/4 + s_lr (P(right) - 1/2) / 2 + s_tb (P(bottom) - 1/2) / 2 + s_lr s_tb * interaction,
s = +-1 by the corner's side, so the posterior is the two marginals plus their product: a 2-parameter
surface in the 3-dimensional simplex. An affine decoder of b(g) needs the product to be represented.
"""

from __future__ import annotations

import torch

from goalgeo import navmodel as NM

TARGETS = ("b_TL", "b_TR", "b_BL", "b_BR", "p_right", "p_bottom", "interaction")


def site_names(layers):
    return [f"L{l}.{s}" for l in range(layers) for s in ("pre", "mid")] + [f"L{layers - 1}.post"]


def site_stack(rec, layers):
    out = []
    for l in range(layers):
        out += [rec["resid"][l], rec["mid"][l]]
    return out + [rec["resid"][layers]]


def targets(dec):
    inter = (dec["p_right"] - 0.5) * (dec["p_bottom"] - 0.5)
    return torch.cat([dec["belief"], dec["p_right"][:, None], dec["p_bottom"][:, None], inter[:, None]], 1)


@torch.no_grad()
def collect(net, batch, dec, chunk=4096, heads=False):
    """Activations at the decisions of `dec`. Returns acts [S, N, d] and logits [N, 6] (illegal masked).
    With heads=True also attn [layers, N, d], mlp [layers, N, d] and head outputs [layers, h, N, d]."""
    S = 2 * net.nl + 1
    N = len(dec["hist"])
    acts = torch.zeros(S, N, net.d, device=batch.tok.device)
    logits = torch.zeros(N, 6, device=batch.tok.device)
    extra = dict(attn=torch.zeros(net.nl, N, net.d, device=acts.device), mlp=torch.zeros(net.nl, N, net.d, device=acts.device),
                 heads=torch.zeros(net.nl, net.nh, N, net.d, device=acts.device)) if heads else None
    nh = batch.tok.shape[0]
    for s in range(0, nh, chunk):
        sel = (dec["hist"] >= s) & (dec["hist"] < s + chunk)
        if not sel.any():
            continue
        lg, _, rec = net(batch.tok[s:s + chunk], batch.q[s:s + chunk], record=True)
        hi, p = dec["hist"][sel] - s, dec["pos"][sel]
        for i, a in enumerate(site_stack(rec, net.nl)):
            acts[i, sel] = a[hi, p]
        logits[sel] = lg[hi, p].masked_fill(~dec["legal"][sel], -1e9)
        if heads:
            for l in range(net.nl):
                extra["attn"][l, sel], extra["mlp"][l, sel] = rec["attn"][l][hi, p], rec["mlp"][l][hi, p]
                extra["heads"][l][:, sel] = rec["heads"][l][hi, :, p].transpose(0, 1)
    return (acts, logits, extra) if heads else (acts, logits)


class Affine:
    """Ridge regression with standardised inputs and an unpenalised intercept, in float64. x -> x @ W + b."""

    def __init__(self, X, Y, lam=1e-3):
        X, Y = X.double(), Y.double()
        self.mu, self.sd = X.mean(0), X.std(0).clamp(min=1e-8)
        Z = (X - self.mu) / self.sd
        ym = Y.mean(0)
        G = Z.T @ Z / len(Z) + lam * torch.eye(Z.shape[1], dtype=Z.dtype, device=Z.device)
        Wz = torch.linalg.solve(G, Z.T @ (Y - ym) / len(Z))
        self.W = Wz / self.sd[:, None]
        self.b = ym - self.mu @ self.W

    def __call__(self, X):
        return X.double() @ self.W + self.b

    def delta(self, D):
        """The decoded displacement produced by adding D to the input."""
        return D.double() @ self.W


def scores(pred, Y):
    """Held-out R^2 and RMSE per target, and the validity of the four decoded goal probabilities."""
    Y = Y.double()
    err = ((pred - Y) ** 2).mean(0)
    r2 = 1 - err / Y.var(0).clamp(min=1e-12)
    b = pred[:, :4]
    return dict(r2=r2.tolist(), rmse=err.sqrt().tolist(),
                invalid=((b < 0) | (b > 1)).any(1).double().mean().item(),
                sum_error=(b.sum(1) - 1).abs().mean().item(),
                belief_l1=(b - Y[:, :4]).abs().sum(1).mean().item())


def fit_adam(Xf, Yf, Xt, Yt, hidden=0, steps=400, lr=1e-2, seed=0):
    """Decoders for every site at once, the same optimiser and number of steps whatever the decoder.
    Xf [S, N, d]. hidden = 0: affine; otherwise one hidden layer of that width. Returns held-out predictions [S, Nt, 7]."""
    S, _, d = Xf.shape
    g = torch.Generator(device=Xf.device); g.manual_seed(seed)
    mu, sd = Xf.mean(1, keepdim=True), Xf.std(1, keepdim=True).clamp(min=1e-8)
    Zf, Zt = (Xf - mu) / sd, (Xt - mu) / sd
    ym = Yf.mean(0)
    k = Yf.shape[1]
    init = lambda *s: (torch.randn(*s, device=Xf.device, generator=g) / s[-2] ** 0.5).requires_grad_()
    if hidden:
        P = [init(S, d, hidden), torch.zeros(S, 1, hidden, device=Xf.device, requires_grad=True),
             init(S, hidden, k), torch.zeros(S, 1, k, device=Xf.device, requires_grad=True)]
        f = lambda Z: torch.relu(Z @ P[0] + P[1]) @ P[2] + P[3] + ym
    else:
        P = [init(S, d, k), torch.zeros(S, 1, k, device=Xf.device, requires_grad=True)]
        f = lambda Z: Z @ P[0] + P[1] + ym
    opt = torch.optim.Adam(P, lr=lr)
    for i in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = lr * 0.5 * (1 + torch.cos(torch.tensor(i / steps * 3.14159)).item())
        loss = ((f(Zf) - Yf) ** 2).mean() * 100
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return f(Zt).double()


def raw_features(batch, dec, tab):
    """What the decision token and the two configuration tokens state, one-hot: own cell, remaining actions,
    used flags, station cells, reliabilities. No report is in these tokens."""
    oh = lambda x, n: torch.nn.functional.one_hot(x, n).float()
    nc = tab.n * tab.n
    return torch.cat([oh(dec["cell"], nc), oh(dec["k"], tab.H + 1), (dec["sh"] > 0).float()[:, None], (dec["sv"] > 0).float()[:, None],
                      oh(tab.hcell[dec["cfg"]], nc), oh(tab.vcell[dec["cfg"]], nc), dec["qh"][:, None], dec["qv"][:, None]], 1)


def policy_on_bank(logits, dec):
    """The model's action distribution on fixed decisions: expected local regret and mass on the optimal set,
    overall and by the number of clues seen."""
    pi = logits.softmax(-1)
    reg = (dec["q_star"].max(-1, keepdim=True).values - dec["q_star"]).masked_fill(~dec["legal"], 0)
    r = (pi * reg).sum(-1); m = (pi * dec["optimal"]).sum(-1)
    clues = (dec["sh"] > 0).long() + (dec["sv"] > 0).long()
    out = dict(regret=r.mean().item(), optimal_mass=m.mean().item())
    for c in range(3):
        s = clues == c
        out[f"regret_{c}clue"], out[f"optimal_mass_{c}clue"] = r[s].mean().item(), m[s].mean().item()
    return out
