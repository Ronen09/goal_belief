"""Representational-geometry toolkit: RDMs, RSA, CKA, dimensionality,
clustering, cross-validated linear decoding, two-way decomposition."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr


# ---- distances ----------------------------------------------------------
def rdm(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    return squareform(pdist(np.asarray(X, dtype=float), metric=metric))


def cosine_similarity(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def upper(R: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(R, k=1)
    return R[iu]


# ---- RSA ------------------------------------------------------------------
def rsa(R1: np.ndarray, R2: np.ndarray) -> float:
    """Spearman correlation between the upper triangles of two RDMs."""
    a, b = upper(R1), upper(R2)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def partial_rsa(R_target: np.ndarray, R_model: np.ndarray, R_controls: list[np.ndarray]) -> float:
    """Spearman partial correlation of target with model, controlling for the
    control RDMs (rank-transform, regress out controls, correlate residuals)."""
    y = rankdata(upper(R_target)); x = rankdata(upper(R_model))
    Z = np.column_stack([np.ones_like(y)] + [rankdata(upper(c)) for c in R_controls])
    beta_y, *_ = np.linalg.lstsq(Z, y, rcond=None)
    beta_x, *_ = np.linalg.lstsq(Z, x, rcond=None)
    ry, rx = y - Z @ beta_y, x - Z @ beta_x
    if np.std(ry) == 0 or np.std(rx) == 0:
        return 0.0
    return float(np.corrcoef(ry, rx)[0, 1])


def rdm_regression(R_target: np.ndarray, R_models: dict[str, np.ndarray]) -> dict[str, float]:
    """Standardised multiple regression of the target RDM on model RDMs
    (rank-transformed). Returns beta per model plus R^2."""
    y = rankdata(upper(R_target)); y = (y - y.mean()) / y.std()
    cols, names = [], []
    for k, R in R_models.items():
        x = rankdata(upper(R)); sd = x.std()
        if sd == 0:
            continue
        cols.append((x - x.mean()) / sd); names.append(k)
    Z = np.column_stack([np.ones_like(y)] + cols)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    out = {f"beta_{n}": float(b) for n, b in zip(names, beta[1:])}
    out["r2"] = float(1 - resid.var() / y.var())
    return out


# ---- CKA ------------------------------------------------------------------
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    nx = np.linalg.norm(X.T @ X, "fro"); ny = np.linalg.norm(Y.T @ Y, "fro")
    if nx == 0 or ny == 0:
        return 0.0
    return float(hsic / (nx * ny))


# ---- dimensionality -----------------------------------------------------------
def pca_spectrum(X: np.ndarray) -> np.ndarray:
    """Eigenvalues of the covariance, descending."""
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    return s ** 2 / max(len(X) - 1, 1)


def participation_ratio(X: np.ndarray) -> float:
    lam = pca_spectrum(X)
    if lam.sum() == 0:
        return 0.0
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def entropy_rank(X: np.ndarray) -> float:
    lam = pca_spectrum(X)
    p = lam / lam.sum() if lam.sum() > 0 else lam
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def principal_axes(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T  # [d, k]


def subspace_overlap(X: np.ndarray, Y: np.ndarray, k: int) -> float:
    """Mean squared cosine between top-k principal *score* subspaces of X and Y
    (computed in sample space, so X and Y may have different feature dims)."""
    def scores(M):
        Mc = M - M.mean(0)
        U, _, _ = np.linalg.svd(Mc, full_matrices=False)
        return U[:, :k]
    Ux, Uy = scores(X), scores(Y)
    return float(np.linalg.norm(Ux.T @ Uy, "fro") ** 2 / k)


# ---- clustering ----------------------------------------------------------------
def hierarchical_clusters(X: np.ndarray, n_clusters: int, method: str = "ward") -> tuple[np.ndarray, np.ndarray]:
    L = linkage(X, method=method)
    return fcluster(L, n_clusters, criterion="maxclust"), L


# ---- decoding ----------------------------------------------------------------
def ridge_cv_r2(X: np.ndarray, Y: np.ndarray, n_folds: int = 5, alpha: float = 1.0,
                rng: np.random.Generator | None = None, groups: np.ndarray | None = None) -> float:
    """Cross-validated R^2 of a ridge map X -> Y. If ``groups`` is given, folds
    are made over unique groups (e.g. states) so that held-out rows never share
    a group with training rows."""
    rng = rng or np.random.default_rng(0)
    n = len(X)
    keys = np.arange(n) if groups is None else np.unique(groups)
    perm = rng.permutation(len(keys))
    folds = np.array_split(keys[perm], n_folds)
    grp = np.arange(n) if groups is None else groups
    sse, sst = 0.0, 0.0
    for held in folds:
        te = np.isin(grp, held); tr = ~te
        Xtr, Ytr = X[tr], Y[tr]
        mx, my = Xtr.mean(0), Ytr.mean(0)
        Xc, Yc = Xtr - mx, Ytr - my
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
        pred = (X[te] - mx) @ W + my
        sse += ((Y[te] - pred) ** 2).sum()
        sst += ((Y[te] - my) ** 2).sum()
    return float(1 - sse / sst) if sst > 0 else 0.0


# ---- two-way decomposition --------------------------------------------------------
def two_way_decomposition(H: np.ndarray) -> dict[str, float]:
    """H[s, g, :] = mu + a_s + b_g + r_sg. Returns variance fractions."""
    mu = H.mean((0, 1), keepdims=True)
    a = H.mean(1, keepdims=True) - mu
    b = H.mean(0, keepdims=True) - mu
    r = H - mu - a - b
    tot = ((H - mu) ** 2).sum()
    S, Gn, _ = H.shape
    return {
        "state": float((a ** 2).sum() * Gn / tot),
        "goal": float((b ** 2).sum() * S / tot),
        "interaction": float((r ** 2).sum() / tot),
        "components": (mu[0, 0], a[:, 0, :], b[0, :, :], r),
    }


# ---- alignment across networks ----------------------------------------------------
def procrustes_rowwise_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-row distance between X and Y after centring, scaling to unit Frobenius
    norm and the optimal orthogonal alignment of X onto Y (X and Y: [n, d])."""
    Xc = X - X.mean(0); Yc = Y - Y.mean(0)
    Xc /= np.linalg.norm(Xc) + 1e-12; Yc /= np.linalg.norm(Yc) + 1e-12
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    return np.linalg.norm(Xc @ R - Yc, axis=1) * np.sqrt(len(X))  # ~ per-row scale


def two_nn_dimension(X: np.ndarray, discard_frac: float = 0.1) -> float:
    """Two-nearest-neighbour intrinsic dimension (Facco et al. 2017)."""
    from scipy.spatial import cKDTree
    X = np.asarray(X, dtype=float)
    d, _ = cKDTree(X).query(X, k=3)
    r1, r2 = d[:, 1], d[:, 2]
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    if n < 10:
        return float("nan")
    F = np.arange(1, n + 1) / n
    keep = slice(0, int(n * (1 - discard_frac)))
    x = np.log(mu[keep]); y = -np.log(1 - F[keep])
    return float((x @ y) / (x @ x))
