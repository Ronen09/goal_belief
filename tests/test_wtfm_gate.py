"""The gated carry transformer: forced-open equals the ungated computation with the same weights,
forced-closed equals a window-1 model with the same weights, the training gate carries gradient,
and a price closes the reads."""
import torch
import torch.nn.functional as F

from goalgeo import readgate as RG, wtfm as W

KW = dict(d=16, n_heads=2, n_layers=2, mlp=32)


def _copy_into(dst, src):
    """Load src's weights into dst where names and shapes match (rel is truncated to dst's window)."""
    sd = dst.state_dict()
    for k, v in src.state_dict().items():
        if k in sd:
            sd[k].copy_(v[..., :sd[k].shape[-1]] if k.endswith(".rel") else v)
    dst.load_state_dict(sd)


def test_gated_open_equals_ungated_with_same_weights():
    g = W.WindowTransformer(6, 3, 8, True, seed=0, gated=True).eval()
    u = W.WindowTransformer(6, 3, 8, True, seed=1, gated=False).eval(); _copy_into(u, g)
    X = torch.randint(0, 6, (4, 12))
    with torch.no_grad():
        zg, p = g.forward_gated(X, gate="open"); zu = u.forward_all(X)
    assert torch.allclose(zg, zu, atol=1e-6) and torch.all(p == 1)


def test_gated_own_at_init_is_open_and_closed_equals_window_one():
    g = W.WindowTransformer(6, 3, 8, True, seed=0, gated=True).eval()
    X = torch.randint(0, 6, (4, 12))
    with torch.no_grad():
        z_own, p = g.forward_gated(X, gate="own"); z_open, _ = g.forward_gated(X, gate="open")
    assert torch.allclose(z_own, z_open, atol=1e-6) and torch.all(p[:, 1:] > 0.9)      # b = 3 opens every gate
    for blk in g.blocks:                                                               # force closed
        torch.nn.init.constant_(blk.gate.bias, -10.0)
    w1 = W.WindowTransformer(6, 3, 1, True, seed=2, gated=False).eval(); _copy_into(w1, g)
    with torch.no_grad():
        z_closed, p = g.forward_gated(X, gate="own"); z_w1 = w1.forward_all(X)
    assert torch.allclose(z_closed, z_w1, atol=1e-5) and torch.all(p[:, 1:] < 0.1)


def test_train_gate_has_gradient_to_gate_parameters():
    g = W.WindowTransformer(6, 3, 8, True, seed=0, gated=True)
    X = torch.randint(0, 6, (4, 12))
    z, p = g.forward_gated(X, gate="train", gen=torch.Generator().manual_seed(0))
    (z.pow(2).mean() + RG.penalty(p)).backward()
    assert all(blk.gate.weight.grad.abs().sum() > 0 for blk in g.blocks)


def test_price_closes_reads_and_zero_price_keeps_them():
    """On real channel-environment tokens reading the history helps, so a free gate stays open;
    a price of 1 nat per read (20x the accuracy it buys) closes it. Adam moves the bias ~lr per
    step, so 300 steps at 2e-2 can carry it from +3 past 0."""
    from goalgeo import kvprior as KP, latentgoal as LG
    env = LG.make_env("channel", 4)
    X, Y = KP.lm_data(env, 512, 11, seed=3); X, Y = torch.as_tensor(X), torch.as_tensor(Y - 1)
    rates = {}
    for c in (0.0, 1.0):
        g = W.WindowTransformer(env.V, env.M, 12, True, seed=0, gated=True, **KW); opt = torch.optim.Adam(g.parameters(), lr=2e-2)
        gen = torch.Generator().manual_seed(0); rng = torch.Generator().manual_seed(1)
        for _ in range(300):
            idx = torch.randint(0, 512, (64,), generator=rng)
            z, p = g.forward_gated(X[idx], gate="train", gen=gen)
            loss = F.cross_entropy(z.reshape(-1, env.M), Y[idx].reshape(-1)) + c * RG.penalty(p)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            rates[c] = g.eval().forward_gated(X[:256], gate="own")[1][:, 2:].gt(0.5).float().mean().item()
    assert rates[1.0] < 0.1 and rates[0.0] > 0.5, rates
