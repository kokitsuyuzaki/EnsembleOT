"""Randomized clustering utilities (skeleton)."""

from __future__ import annotations

import numpy as np

from .config import ClusteringConfig


def cluster_samples(
    X: np.ndarray,
    config: ClusteringConfig,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    """Return an integer label vector of shape (n_samples,).

    Stage 1: skeleton only. Real implementation lands in Stage 2.
    """
    raise NotImplementedError("clustering will be implemented in Stage 2")


def cluster_sizes(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Count samples per cluster id in [0, n_clusters)."""
    counts = np.bincount(labels, minlength=n_clusters)
    return counts.astype(np.int64)
