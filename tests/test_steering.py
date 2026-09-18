import numpy as np
import torch

from goalgeo import hmm4 as H4, prominence as Pr, seqmodels as Sm, steering as St


def _setup():
    m = H4.make_hmm4(1.0, 0.4, 2)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=12, emb=6, out_dim=H4.V, seed=1)
    ev = Pr.make_eval(m, n=40, T=20, seed=3)
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(ev["X"])).numpy().astype(np.float64)
    P, anchors = Pr.positions(m, ev["Z"]); anchors = anchors[:8]
    h0 = Hs[anchors[:, 0], anchors[:, 1]]; win = np.stack([ev["X"][i, t + 1:t + 4] for i, t in anchors])
    return net, h0, win


def test_effect_curve_zero_at_alpha_zero_and_linear_prediction_matches_small_scale():
    net, h0, win = _setup(); v = np.ones(12) * 0.1
    js, P, p0 = St.effect_curve(net, h0, win, v, [0.0, 0.5], step=1)
    assert js[0] < 1e-12 and js[1] > js[0] and P.shape == (2, 8, H4.V)
    s = 0.01; dz = St.directional_logit_derivative(net, h0, win, s * v)[:, 1]
    pred = St.js_linear_prediction(dz, p0).mean(); fin = St.effect_curve(net, h0, win, v, [s], step=1)[0][0]
    assert abs(pred - fin) / fin < 0.05


def test_depth_curves_shapes_and_zero_for_identical_states():
    net, h0, win = _setup()
    d = St.depth_curves(net, h0, h0.copy(), win)
    assert d["js"].shape == (4,) and d["raw"].shape == (4,) and d["lin"].shape == (4,) and np.all(d["js"] < 1e-12) and np.all(d["raw"] < 1e-12)
    d2 = St.depth_curves(net, h0, h0 + 0.5, win)
    assert d2["js"][0] > 0 and d2["raw"][0] > 0 and d2["lin"][0] > 0


def test_align_states_recovers_linear_map_and_match_alpha():
    rng = np.random.default_rng(0); H1 = rng.standard_normal((200, 5)); A = rng.standard_normal((5, 5)); H2 = H1 @ A + 1
    B, r2 = St.align_states(H1, H2); assert r2 > 0.999 and np.allclose(B, A, atol=1e-8)
    assert np.isclose(St.match_alpha(0.5, [0, 1, 2], np.array([0, 0.25, 1.0])), 1.0 + (0.5 - 0.25) / 0.75)
