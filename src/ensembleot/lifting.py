"""Sample <-> cluster aggregation/lifting primitives.

All operations are expressed via label vectors — never materialize
the dense sample x cluster assignment matrix A.
"""

from __future__ import annotations

import numpy as np


def aggregate_by_label(
    Y: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    weights: np.ndarray | None = None,
    reduce: str = "mean",
) -> np.ndarray:
    """Aggregate rows of Y by cluster label.

    Returns an array of shape (n_clusters, n_features).
    `reduce` is one of {"sum", "mean"}.
    """
    if Y.ndim == 1:
        Y = Y[:, None]
    n_features = Y.shape[1]
    out = np.zeros((n_clusters, n_features), dtype=Y.dtype)
    if weights is None:
        np.add.at(out, labels, Y)
    else:
        np.add.at(out, labels, Y * weights[:, None])
    if reduce == "sum":
        return out
    if reduce == "mean":
        counts = np.bincount(labels, minlength=n_clusters).astype(out.dtype)
        counts[counts == 0] = 1.0
        return out / counts[:, None]
    raise ValueError(f"unknown reduce={reduce!r}")


def broadcast_to_samples(
    cluster_values: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Expand per-cluster values back to per-sample values via indexing."""
    return cluster_values[labels]
