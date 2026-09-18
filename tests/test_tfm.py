import numpy as np
import torch

from goalgeo import hmm4 as H4, seqmodels as Sm, tfm as Tf


def test_causal_and_patching_consistency():
    net = Tf.CausalTransformer(n_vocab=H4.V, d=16, n_heads=2, n_layers=2, mlp=32, T=12, seed=0).eval()
    X = torch.randint(0, H4.V, (4, 12)); X2 = X.clone(); X2[:, 6:] = (X2[:, 6:] + 1) % H4.V
    with torch.no_grad():
        z, z2 = net.forward_all(X), net.forward_all(X2)
    assert torch.allclose(z[:, :6], z2[:, :6], atol=1e-6) and not torch.allclose(z[:, 6:], z2[:, 6:])
    with torch.no_grad():
        R1 = net.states(X); zp = net.patched(X, np.arange(4), np.full(4, 3), R1[:, 3])
    assert torch.allclose(zp, z, atol=1e-5)
    with torch.no_grad():
        zq = net.patched(X, np.arange(4), np.full(4, 3), R1[:, 3] + torch.randn(4, 16))
    assert torch.allclose(zq[:, :3], z[:, :3], atol=1e-5) and not torch.allclose(zq[:, 3:], z[:, 3:])


def test_transformer_trains_with_weighted_objective_and_variants():
    m = H4.make_hmm4(1.0, 0.4, 1)
    for kw in ({}, {"final_ln": False}, {"gain": 2.0}, {"out_scale": 0.1}):
        net = Tf.CausalTransformer(n_vocab=H4.V, d=16, n_heads=2, n_layers=2, mlp=32, T=16, seed=0, **kw)
        hist, ckpt = Sm.train_weighted(net, m, steps=40, batch=32, seed=0, T=16, pool=100, checkpoints=(0, 40), lr=1e-3, lr_out=1e-3)
        assert np.mean(hist[-5:]) < np.mean(hist[:5]) and sorted(ckpt) == [0, 40]
    assert isinstance(net.out, torch.nn.Linear)


def test_measure_tfm_returns_finite_keys():
    from goalgeo import prominence as Pr, tfm_measure as Tm
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Tf.CausalTransformer(n_vocab=H4.V, d=16, n_heads=2, n_layers=2, mlp=32, T=24, seed=0).eval()
    ev = Pr.make_eval(m, n=120, T=24, seed=0)
    r = Tm.measure_tfm(net, m, ev, n_dec=300, n_anchor=30, K=3)
    for k in ("P_metric", "P_metric_pre_l1", "branch_acc_pre_l1", "branch_acc_pre_postnorm", "belief_r2_l1", "g_eff", "achieved_gap_postnorm", "cos_theta_postnorm",
              "kl", "JS_future", "D_future", "D_F", "ln_gain_norm"):
        assert k in r and np.isfinite(r[k]), k
    assert len(r["depth_js"]) == 4 and r["depth_js"][0] >= 0
