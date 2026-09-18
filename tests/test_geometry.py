import numpy as np

from goalgeo import geometry as G


def test_rdm_is_symmetric_zero_diagonal():
    X = np.random.default_rng(0).normal(size=(10, 5))
    R = G.rdm(X)
    assert R.shape == (10, 10)
    assert np.allclose(R, R.T) and np.allclose(np.diag(R), 0)
    assert np.isclose(R[0, 1], np.linalg.norm(X[0] - X[1]))


def test_rsa_of_identical_geometry_is_one():
    X = np.random.default_rng(0).normal(size=(20, 5))
    Y = 3.0 * X @ np.linalg.qr(np.random.default_rng(1).normal(size=(5, 5)))[0]  # rotate+scale
    assert np.isclose(G.rsa(G.rdm(X), G.rdm(Y)), 1.0)


def test_partial_rsa_removes_shared_component():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(30, 4))
    B = A + 0.01 * rng.normal(size=(30, 4))  # B ~ A
    C = rng.normal(size=(30, 4))
    rA, rB, rC = G.rdm(A), G.rdm(B), G.rdm(C)
    # C explains nothing about A once B is controlled, and B still explains A given C
    assert abs(G.partial_rsa(rA, rC, [rB])) < 0.15
    assert G.partial_rsa(rA, rB, [rC]) > 0.9


def test_effective_rank_of_isotropic_and_rank_one():
    rng = np.random.default_rng(0)
    iso = rng.normal(size=(5000, 8))
    pr = G.participation_ratio(iso)
    assert 7.5 < pr <= 8.0
    rank1 = np.outer(rng.normal(size=100), rng.normal(size=8))
    assert np.isclose(G.participation_ratio(rank1), 1.0)


def test_linear_cka_invariant_to_rotation_and_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert np.isclose(G.linear_cka(X, 2.0 * X @ Q), 1.0)
    assert G.linear_cka(X, rng.normal(size=(40, 6))) < 0.3


def test_ridge_cv_r2_recovers_linear_map_and_fails_on_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    W = rng.normal(size=(10, 3))
    Y = X @ W
    assert G.ridge_cv_r2(X, Y, n_folds=5, alpha=1e-3, rng=rng) > 0.99
    assert G.ridge_cv_r2(X, rng.normal(size=(200, 3)), n_folds=5, alpha=1e-3, rng=rng) < 0.1


def test_subspace_overlap_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    assert np.isclose(G.subspace_overlap(X, X, k=3), 1.0)
    Y = rng.normal(size=(50, 6))
    assert 0.0 <= G.subspace_overlap(X, Y, k=3) <= 1.0


def test_two_way_decomposition_variance_fractions_sum_to_one():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(7, 5, 4))  # [state, goal, dim]
    parts = G.two_way_decomposition(H)
    assert np.isclose(parts["state"] + parts["goal"] + parts["interaction"], 1.0)
    # purely additive tensor has no interaction
    H_add = rng.normal(size=(7, 1, 4)) + rng.normal(size=(1, 5, 4))
    assert G.two_way_decomposition(H_add)["interaction"] < 1e-9


def test_procrustes_distance_zero_under_rotation_and_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    Q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    d = G.procrustes_rowwise_distance(X, 2.0 * X @ Q + 1.0)
    assert d.shape == (40,) and np.allclose(d, 0, atol=1e-8)
    d2 = G.procrustes_rowwise_distance(X, rng.normal(size=(40, 6)))
    assert d2.mean() > 0.1


def test_two_nn_intrinsic_dimension_recovers_manifold_dim():
    rng = np.random.default_rng(0)
    X2 = rng.uniform(size=(3000, 2)) @ rng.normal(size=(2, 10))  # 2-d plane in 10-d
    assert 1.6 < G.two_nn_dimension(X2) < 2.4
    X8 = rng.uniform(size=(3000, 8))
    assert 6.5 < G.two_nn_dimension(X8) < 9.5
