"""Randomized clustering utilities."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

_SUPPORTED = {"kmeans"}


def cluster_samples(
    X: np.ndarray,
    method: str,
    n_clusters: int,
    random_state: int | None,
) -> np.ndarray:
    """Return integer label vector of shape (n_samples,) in [0, n_clusters)."""
    labels, _ = cluster_samples_with_info(X, method, n_clusters, random_state)
    return labels


def cluster_samples_with_info(
    X: np.ndarray,
    method: str,
    n_clusters: int,
    random_state: int | None,
) -> tuple[np.ndarray, dict]:
    """Cluster and return (labels, info) with solver-specific diagnostics.

    Stage 7 supports only ``method='kmeans'``; for kmeans ``info`` contains
    ``{"inertia": float}``.
    """
    if method not in _SUPPORTED:
        raise ValueError(f"unsupported clustering method {method!r}; supported: {_SUPPORTED}")
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(X).astype(np.int64)
        return labels, {"inertia": float(km.inertia_)}
    raise AssertionError("unreachable")


def cluster_means(X: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Per-cluster mean of rows of X. Empty clusters get a zero row."""
    out = np.zeros((n_clusters, X.shape[1]), dtype=X.dtype)
    np.add.at(out, labels, X)
    counts = np.bincount(labels, minlength=n_clusters).astype(out.dtype)
    counts_safe = np.where(counts > 0, counts, 1.0)
    return out / counts_safe[:, None]


def cluster_sizes(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.bincount(labels, minlength=n_clusters).astype(np.int64)
