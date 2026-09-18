import numpy as np
import torch

from goalgeo import hmm4 as H4, invariants as Iv, prominence as Pr, seqmodels as Sm


def _fake_ex(rng, n=300, d=8, V=7):
    H = rng.standard_normal((n, d)); M = rng.standard_normal((d, 3))
    ex = {"H_sub": H, "B_sub": np.abs(H @ M) + 0.1, "H_dec": H, "B_dec": H @ M + 0.05 * rng.standard_normal((n, 3)), "Y_dec": np.abs(H[:, :2]),
          "H_pre": H, "y_pre": H[:, 0] > 0, "H_cueA": H[:100] + 1, "H_cueB": H[100:200], "H_preA": H[:100] + 0.5, "H_preB": H[100:200],
          "H_U": H[:100] + 2, "H_V": H[100:200], "W": rng.standard_normal((V, d)), "b": np.zeros(V), "J": rng.standard_normal((20, V, d)),
          "p_star": np.full((20, V), 1 / V), "p_U": np.full(V, 1 / V), "JS_future": 0.1, "JS_control": 0.2, "required_gap": 1.0, "logp": None}
    return ex


def test_exact_invariants_under_full_transform_and_non_invariants_under_diagonal():
    rng = np.random.default_rng(0); ex = _fake_ex(rng)
    A = Iv.random_transform("full", 8, rng, cond=30)
    m0, a0 = Iv.metrics_from_arrays(ex); m1, a1 = Iv.metrics_from_arrays(Iv.transform(ex, A))
    for key in ("r2_belief", "r2_target", "acc_branch_pre", "rank", "rsa_whitened", "D_logit_pre", "I_contrast_pre", "D_logit_control", "D_future", "D_F", "D_F_control"):
        assert np.isclose(m0[key], m1[key], rtol=1e-5, atol=1e-8), (key, m0[key], m1[key])
    assert np.linalg.norm(Iv.projection(a0["Q"]) - Iv.projection(a1["Q"])) < 1e-8
    D = Iv.random_transform("diagonal", 8, rng)
    m2, _ = Iv.metrics_from_arrays(Iv.transform(ex, D))
    for key in ("sep_cue", "PR", "rsa_euclid", "hidden_norm"):
        assert not np.isclose(m0[key], m2[key], rtol=1e-2), key
    Q = Iv.random_transform("orthogonal", 8, rng)
    m3, _ = Iv.metrics_from_arrays(Iv.transform(ex, Q))
    assert np.isclose(m0["sep_cue"], m3["sep_cue"]) and np.isclose(m0["rsa_euclid"], m3["rsa_euclid"])


def test_future_jacobian_matches_finite_differences():
    m = H4.make_hmm4(1.0, 0.4, 2)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=12, emb=6, out_dim=H4.V, seed=1)
    ev = Pr.make_eval(m, n=30, T=20, seed=3)
    with torch.no_grad():
        Hs = net.states(torch.as_tensor(ev["X"])).numpy()
    P, anchors = Pr.positions(m, ev["Z"]); anchors = anchors[:5]
    J, p = Iv.future_jacobian(net, ev["X"], Hs, anchors, m.k)
    assert J.shape == (5, H4.V, 12) and np.allclose(p.sum(1), 1)
    v = np.random.default_rng(0).standard_normal(12); eps = 1e-3
    def z(h, i, t):
        win = torch.as_tensor(ev["X"][i, t + 1:t + m.k + 1])[None]
        with torch.no_grad():
            return net.out(net.gru(net.emb(win), torch.as_tensor(h, dtype=torch.float32)[None, None])[0][0, -1]).numpy()
    i, t = anchors[0]; h = Hs[i, t].astype(np.float64)
    fd = (z(h + eps * v, i, t) - z(h - eps * v, i, t)) / (2 * eps)
    assert np.allclose(fd, J[0] @ v, atol=1e-3)


def test_js_and_subspace_helpers():
    p = np.array([[0.2, 0.8]]); assert Iv.js_divergence(p, p)[0] == 0 and Iv.js_divergence(p, p[:, ::-1])[0] > 0
    rng = np.random.default_rng(0); X = rng.standard_normal((50, 4)) @ rng.standard_normal((4, 10))
    assert Iv.exact_rank(X) == 4 and Iv.subspace_basis(X).shape == (50, 4)
    assert np.isclose(Iv.subspace_overlap(Iv.subspace_basis(X), Iv.subspace_basis(X @ rng.standard_normal((10, 10)))), 1.0)


def test_extract_and_metrics_on_small_model():
    m = H4.make_hmm4(1.0, 0.4, 1)
    net = Sm.SeqNet(n_vocab=H4.V, hidden=16, emb=8, out_dim=H4.V, seed=0)
    ev = Pr.make_eval(m, n=120, T=24, seed=0)
    rng = np.random.default_rng(0); n = 120 * 24
    sub = rng.choice(n, 150, replace=False); dec = rng.choice(n, 400, replace=False)
    ex = Iv.extract(net, m, ev, sub, dec, n_group=50, n_anchor=30)
    r, arrays = Iv.metrics_from_arrays(ex)
    for k in Iv.ALL:
        assert k in r and np.isfinite(r[k]), k
    assert arrays["Q"].shape[0] == 150 and ex["logp"].shape == (120, 23, H4.V)
