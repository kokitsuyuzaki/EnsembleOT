"""Randomized clustering utilities."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

_SUPPORTED = {"kmeans", "random_voronoi"}


# ---------------------------------------------------------------------------
# random_voronoi helpers
# ---------------------------------------------------------------------------

def _random_voronoi_seed_indices(
    n_samples: int,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select *n_clusters* distinct sample indices as Voronoi seeds."""
    return rng.choice(n_samples, size=n_clusters, replace=False)


def _assign_to_nearest_seed(
    X: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Assign each row of *X* to the nearest seed (squared-Euclidean).

    Complexity: O(n_samples × n_clusters × d).  Acceptable when
    *n_clusters* is moderate (typical for EnsembleOT).
    """
    # seeds: (K, d),  X: (N, d) → diffs: (N, K, d)
    diff = X[:, None, :] - seeds[None, :, :]
    sqdist = np.sum(diff * diff, axis=2)       # (N, K)
    return np.argmin(sqdist, axis=1).astype(np.int64)


def _random_voronoi_labels(
    X: np.ndarray,
    n_clusters: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, seed_indices) for a Voronoi partition.

    Labels are in ``[0, n_clusters)``.  Each seed point is guaranteed to
    belong to its own cluster, so no cluster is empty.
    """
    rng = np.random.default_rng(random_state)
    seed_idx = _random_voronoi_seed_indices(X.shape[0], n_clusters, rng)
    seeds = X[seed_idx]                        # (K, d)
    labels = _assign_to_nearest_seed(X, seeds)
    return labels, seed_idx


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------

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

    Supported methods:

    * ``"kmeans"`` — scikit-learn k-means.  ``info`` contains
      ``{"inertia": float}``.
    * ``"random_voronoi"`` — pick *n_clusters* random sample points as
      seeds and assign every point to its nearest seed (squared-Euclidean,
      no iterative refinement).  ``info`` contains
      ``{"method": "random_voronoi", "seed_indices": np.ndarray}``.
    """
    if method not in _SUPPORTED:
        raise ValueError(f"unsupported clustering method {method!r}; supported: {_SUPPORTED}")
    if method == "kmeans":
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = km.fit_predict(X).astype(np.int64)
        return labels, {"inertia": float(km.inertia_)}
    if method == "random_voronoi":
        labels, seed_idx = _random_voronoi_labels(X, n_clusters, random_state)
        return labels, {"method": "random_voronoi", "seed_indices": seed_idx}
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
