"""The gated plain transformer: open equals ungated, closed equals the round-16 mask (self and the
previous position visible), forward_query's own gate agrees with the full forward."""
import numpy as np
import torch

from goalgeo import kvprior as KP, latentgoal as LG, tfm as Tf

KW = dict(n_vocab=7, d=16, n_heads=2, n_layers=3, mlp=32, T=12, final_norm="learn")


def _copy_into(dst, src):
    sd = dst.state_dict()
    for k, v in src.state_dict().items():
        if k in sd:
            sd[k].copy_(v)
    dst.load_state_dict(sd)


def test_gated_open_equals_ungated_and_init_is_open():
    g = Tf.CausalTransformer(seed=0, gated=True, **KW).eval()
    u = Tf.CausalTransformer(seed=1, gated=False, **KW).eval(); _copy_into(u, g)
    X = torch.randint(0, 7, (4, 12))
    with torch.no_grad():
        zo = g.forward_all(X, gate="open"); zu = u.forward_all(X); zw = g.forward_all(X, gate="own")
        _, p = g.residuals_g(X, gate="own")
    assert torch.allclose(zo, zu, atol=1e-6) and torch.allclose(zw, zu, atol=1e-6) and torch.all(p[:, 2:] > 0.9)


def test_gated_closed_equals_masked_forward():
    """Closed everywhere == predictive_masked with only u and u-1 visible (drop p = 1, keep_prev)."""
    g = Tf.CausalTransformer(seed=0, gated=True, **KW).eval()
    for blk in g.blocks:
        torch.nn.init.constant_(blk.gate.bias, -10.0)
    X = np.random.default_rng(0).integers(1, 7, (5, 12)); X[:, 0] = 0
    vis = KP.drop_visible(1.0, 5, 12, keep_prev=True).numpy()
    with torch.no_grad():
        own = KP.predictive(g, X, gate="own"); masked = KP.predictive_masked(g, X, vis)
    assert np.allclose(own, masked, atol=1e-6)


def test_forward_query_own_gate_agrees_with_full_forward():
    g = Tf.CausalTransformer(seed=0, gated=True, **KW).eval()
    for blk in g.blocks:                                             # a gate that actually varies with the input
        torch.nn.init.normal_(blk.gate.weight, std=2.0); torch.nn.init.zeros_(blk.gate.bias)
    X = np.random.default_rng(1).integers(1, 7, (40, 12)); X[:, 0] = 0
    t = 6
    with torch.no_grad():
        full = KP.predictive(g, X, gate="own")[:, t + 1]
        _, pfull = g.residuals_g(torch.as_tensor(X), gate="own")
        src = [r[:, :t + 1] for r in KP.residuals(g, X[:, :t + 1], gate="own")]     # the model's own sources
        _, _, p, _, gates = KP.forward_query_g(g, src, X[:, t + 1], t + 1, gate="own")
    assert np.allclose(p, full, atol=1e-5)
    assert np.array_equal(gates, (pfull[:, t + 1] > 0.5).numpy().astype(float))
    assert 0.1 < gates.mean() < 0.9                                  # both decisions occur in the sample


def test_gate_refuses_route_restricted_blocks():
    import pytest
    with pytest.raises(AssertionError):
        Tf.CausalTransformer(seed=0, gated=True, attn_diag=(0,), **KW)
