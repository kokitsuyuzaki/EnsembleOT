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
    - ``transported_mass``:   Σ_ij T_ij

    For balanced solvers the marginal errors measure solver accuracy and
    ``transported_mass`` is 1. For partial / unbalanced solvers the coupling
    is *meant* to deviate from ``a`` / ``b``: there the "errors" report how
    much mass was dropped or created, not a defect, so they should not be
    used as a quality score for weighting runs.
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
        "transported_mass": float(T.sum()),
    }


def cost_metrics(T_cluster: np.ndarray, M: np.ndarray) -> dict[str, float]:
    """Linear transport cost ``<T, M>`` of the cluster-level coupling.

    ``M`` is the cluster-level cost matrix actually handed to the solver
    (i.e. after any max-normalization), so values are comparable across
    runs and across a sweep of ``m`` / ``reg_m``.
    """
    T = np.asarray(T_cluster, dtype=np.float64)
    return {"transport_cost": float(np.sum(T * np.asarray(M, dtype=np.float64)))}


def partial_mass_metrics(T_cluster: np.ndarray, m: float) -> dict[str, float]:
    """How closely a partial coupling hits its requested mass ``m``.

    - ``mass_error``: |Σ_ij T_ij - m|
    """
    T = np.asarray(T_cluster, dtype=np.float64)
    return {"mass_error": float(abs(T.sum() - float(m)))}
