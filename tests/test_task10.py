"""TASK10: the cut-substitution identity, the route restrictions, and the interface
factorisation identity / LayerNorm ceiling (rounds/r11_implementation_freedom/THEORY.md)."""

import numpy as np
import torch

from goalgeo import cuts as Ct, factorize as Fz, hmm4 as H4, prominence as Pr, tfm as Tf

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _net(**kw):
    return Tf.CausalTransformer(n_vocab=H4.V, d=16, n_heads=2, n_layers=2, mlp=32, T=16, seed=0, **kw).to(DEV).eval()


def _pairs(n=8, T=16, seed=0):
    m = H4.make_hmm4(1.0, 0.4, 1)
    X, Z = H4.sample(m, n, T, seed=seed)
    anchors = Ct.cue_anchors(m, Z, 6, np.random.default_rng(0))
    Xa, Xf = Ct.minimal_pairs(X, anchors)
    return m, Xa, Xf, anchors[:, 1]


def test_minimal_pairs_flip_exactly_one_cue_token():
    _, Xa, Xf, t = _pairs()
    assert (Xa != Xf).sum(1).tolist() == [1] * len(Xa)
    a, b = Xa[np.arange(len(t)), t], Xf[np.arange(len(t)), t]
    assert set(np.unique(np.stack([a, b]))) <= {H4.TOK["p"], H4.TOK["q"]} and (a != b).all()


def test_complete_cuts_reproduce_the_flipped_prediction_exactly():
    """Theorem A.3: patching a complete cut with the flipped run's values IS the flipped run."""
    net = _net(); _, Xa, Xf, t = _pairs()
    r = Ct.cut_effects(net, Xa, Xf, t)
    for k in Ct.COMPLETE:
        assert r[f"dev_{k}"] < 1e-6, (k, r[f"dev_{k}"])
        assert abs(r[f"phi_{k}"] - r["phi_behav"]) < 1e-6
    # an untrained model still mixes positions, so the single-position patches are not the flip
    assert max(r[f"dev_{k}"] for k in Ct.CUT_SETS if k not in Ct.COMPLETE) > 1e-4


def test_diagonal_attention_isolates_positions_and_kills_one_route():
    """block-1 diagonal: no information crosses positions in block 1.
    block-2 diagonal: out(t+1) cannot read the residual at t, so that single node has no effect."""
    _, Xa, Xf, t = _pairs()
    net1 = _net(attn_diag=(0,))
    with torch.no_grad():
        R = net1.residuals(torch.as_tensor(Xa, device=DEV))[1]
        Xb = Xa.copy(); Xb[:, 0] = (Xb[:, 0] + 1) % H4.V
        Rb = net1.residuals(torch.as_tensor(Xb, device=DEV))[1]
    assert torch.allclose(R[:, 3:], Rb[:, 3:], atol=1e-6)
    net2 = _net(attn_diag=(1,))
    r = Ct.cut_effects(net2, Xa, Xf, t)
    assert r["phi_L1{t}"] < 1e-9 and r["dev_L1{t+1}"] < 1e-6      # the t+1 residual is the only route
    for k in Ct.COMPLETE:
        assert r[f"dev_{k}"] < 1e-6


def test_attention_penalty_is_zero_for_a_diagonal_block():
    _, Xa, _, _ = _pairs()
    net = _net(attn_diag=(0, 1))
    _, attn = net.residuals(torch.as_tensor(Xa, device=DEV), need_weights=True)
    assert float(Ct.attention_penalty(attn, "pen_l1")) < 1e-6
    assert float(Ct.attention_penalty(attn, "pen_l2")) < 1e-6


def test_frozen_layernorm_pins_the_interface_norm_and_caps_the_contrast():
    m = H4.make_hmm4(1.0, 0.4, 1); ev = Pr.make_eval(m, n=60, T=16, seed=0)
    for c in (0.2, 1.0):
        net = _net(final_norm="frozen", gain=c)
        r = Fz.measure(net, m, ev, c)
        d = net.hidden_size
        assert abs(r["postnorm_norm"] - np.sqrt(d)) < 1e-4       # ||h~|| = sqrt(d) exactly
        assert r["g"] <= 2 * c + 1e-6                            # fixed-gain rows
        assert r["D"] <= 2 * np.sqrt(d) + 1e-6                   # Proposition B.3
        assert abs(r["C"]) <= r["ceiling"] + 1e-6
        assert not np.isfinite(Fz.ceiling(c, "none", d))


def test_C_is_the_difference_of_output_log_odds():
    """Proposition B.1: C = <log p(x)/p(y)>_A - <log p(x)/p(y)>_B, computable from outputs alone."""
    m = H4.make_hmm4(1.0, 0.4, 1); ev = Pr.make_eval(m, n=80, T=16, seed=1)
    net = _net(final_norm="learn")
    r = Fz.interface(net, m, ev)
    P, _ = Pr.positions(m, ev["Z"])
    for key in ("preA", "preB"):                       # cue must be inside the sequence
        P[key][:, :m.k] = False
    with torch.no_grad():
        z = net.forward_all(torch.as_tensor(ev["X"], device=DEV)).cpu().numpy().astype(np.float64)
    odds = z[..., H4.TOK["x"]] - z[..., H4.TOK["y"]]
    assert abs(r["C"] - (odds[P["preA"]].mean() - odds[P["preB"]].mean())) < 1e-6
    assert abs(r["C"] - r["g"] * r["D"] * r["cos_theta"]) < 1e-6
    assert abs(r["C_required"] - 2 * np.log(0.9 / 0.1)) < 1e-9


def test_train_routed_reduces_loss_with_and_without_a_penalty():
    m = H4.make_hmm4(1.0, 0.4, 1)
    for pen in (None, "pen_l1", "pen_l2"):
        net = Tf.CausalTransformer(n_vocab=H4.V, d=16, n_heads=2, n_layers=2, mlp=32, T=16, seed=0).to(DEV)
        hist = Ct.train_routed(net, m, steps=40, batch=32, seed=0, T=16, pool=100, lr=1e-3, penalty=pen, mu=1.0)
        assert np.mean(hist[-5:]) < np.mean(hist[:5])
