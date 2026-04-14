"""Per-run quality metrics for cluster-level OT operators.

All values returned by these helpers are plain Python scalars so that
they roundtrip cleanly through JSON inside ``operator.meta``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-16


def cluster_shape_metrics(
    labels_x: np.ndarray,
    labels_y: np.ndarray,
    cluster_mass_x: np.ndarray,
    cluster_mass_y: np.ndarray,
    T_cluster: np.ndarray,
) -> dict[str, float | int]:
    """Structural metrics that don't depend on the solver family."""
    sizes_x = np.bincount(labels_x, minlength=T_cluster.shape[0])
    sizes_y = np.bincount(labels_y, minlength=T_cluster.shape[1])
    return {
        "n_clusters_x": int(T_cluster.shape[0]),
        "n_clusters_y": int(T_cluster.shape[1]),
        "cluster_mass_sum_x": float(np.asarray(cluster_mass_x).sum()),
        "cluster_mass_sum_y": float(np.asarray(cluster_mass_y).sum()),
        "T_cluster_sum": float(np.asarray(T_cluster).sum()),
        "cluster_size_min_x": int(sizes_x.min()),
        "cluster_size_max_x": int(sizes_x.max()),
        "cluster_size_min_y": int(sizes_y.min()),
        "cluster_size_max_y": int(sizes_y.max()),
    }


def transport_metrics(
    T_cluster: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> dict[str, float]:
    """Marginal errors and entropy of the cluster-level coupling.

    - ``marginal_error_row``: max_i |Σ_j T_ij - a_i|
    - ``marginal_error_col``: max_j |Σ_i T_ij - b_j|
    - ``transport_entropy``:  -Σ T_ij log(T_ij + eps)   (log(0)-safe)
    """
    T = np.asarray(T_cluster, dtype=np.float64)
    row = T.sum(axis=1)
    col = T.sum(axis=0)
    marginal_row = float(np.max(np.abs(row - np.asarray(a, dtype=np.float64))))
    marginal_col = float(np.max(np.abs(col - np.asarray(b, dtype=np.float64))))
    entropy = float(-np.sum(T * np.log(T + _EPS)))
    return {
        "marginal_error_row": marginal_row,
        "marginal_error_col": marginal_col,
        "transport_entropy": entropy,
    }
