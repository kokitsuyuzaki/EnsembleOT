"""Shared plumbing for the per-run pipeline of the ensemble entry points.

Every entry point does the same three things around its POT call:

  1. cluster X and Y independently (seed shifted per run / per side),
  2. build small cluster-level cost matrices from the cluster means,
  3. wrap the cluster-level coupling in an ``ImplicitTransportOperator``.

The helpers here keep that identical across solver families. Nothing in this
module ever allocates a sample × sample array.

Argument forwarding convention
------------------------------
Solver arguments are forwarded to POT *verbatim* and keep POT's names. An
argument left at ``None`` is simply not forwarded, so POT's own default for
the selected solver applies (POT's defaults differ between solvers, e.g.
``stopThr`` is ``1e-6`` for ``sinkhorn_unbalanced`` and ``1e-15`` for
``mm_unbalanced``). Passing an argument the selected solver does not accept
raises ``ValueError`` instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import ot

from .clustering import cluster_means, cluster_samples_with_info, cluster_sizes
from .metrics import cluster_shape_metrics, transport_metrics
from .operator import ImplicitTransportOperator


@dataclass
class ClusterProblem:
    """Cluster-level view of one run: labels, centers and marginals."""

    labels_x: np.ndarray
    labels_y: np.ndarray
    centers_x: np.ndarray          # (K_x, d_x) cluster means of X
    centers_y: np.ndarray          # (K_y, d_y) cluster means of Y
    sizes_x: np.ndarray            # (K_x,) cluster cardinalities (float)
    sizes_y: np.ndarray            # (K_y,)
    a: np.ndarray                  # (K_x,) sizes_x / n_x, sums to 1
    b: np.ndarray                  # (K_y,) sizes_y / n_y, sums to 1
    info_x: dict
    info_y: dict
    seed: int


def make_seeds(random_state: int | None, n_runs: int) -> list[int]:
    """Per-run seeds; identical to what the Sinkhorn / GW / FGW entry points draw."""
    rng = np.random.default_rng(random_state)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_runs)]


def validate_inputs(
    X: np.ndarray, Y: np.ndarray, n_runs: int, same_dim: bool, who: str
) -> None:
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D")
    if same_dim and X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"{who} needs X and Y in the same feature space to build the "
            f"cross-domain cost; got X.shape[1]={X.shape[1]} vs "
            f"Y.shape[1]={Y.shape[1]}"
        )


def build_cluster_problem(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    clustering_method: str,
    seed: int,
    coords_x: np.ndarray | None = None,
    coords_y: np.ndarray | None = None,
) -> ClusterProblem:
    """Cluster both sides (seed for X, seed + 1 for Y) and collect marginals.

    When ``coords_x`` / ``coords_y`` are given, clustering runs on them while
    the centers are still the feature-space means of X / Y — the same
    convention as ``run_ensemble_gw``.
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    labels_x, info_x = cluster_samples_with_info(
        coords_x if coords_x is not None else X, clustering_method, n_clusters_x, random_state=seed
    )
    labels_y, info_y = cluster_samples_with_info(
        coords_y if coords_y is not None else Y, clustering_method, n_clusters_y, random_state=seed + 1
    )
    sizes_x = cluster_sizes(labels_x, n_clusters_x).astype(float)
    sizes_y = cluster_sizes(labels_y, n_clusters_y).astype(float)
    return ClusterProblem(
        labels_x=labels_x,
        labels_y=labels_y,
        centers_x=cluster_means(X, labels_x, n_clusters_x),
        centers_y=cluster_means(Y, labels_y, n_clusters_y),
        sizes_x=sizes_x,
        sizes_y=sizes_y,
        a=sizes_x / n_x,
        b=sizes_y / n_y,
        info_x=info_x,
        info_y=info_y,
        seed=int(seed),
    )


def max_normalize(C: np.ndarray, normalize_cost: bool) -> np.ndarray:
    """Divide a cost matrix by its maximum (no-op when ``normalize_cost`` is False).

    This is what the Sinkhorn / GW / FGW entry points have always done. It
    makes ``reg`` / ``reg_m`` / ``epsilon`` comparable across runs and
    datasets, at the price of not being the raw ``ot.dist`` cost a direct
    POT call would see — hence the switch.
    """
    C = np.asarray(C, dtype=float)
    if normalize_cost and C.size and C.max() > 0:
        return C / C.max()
    return C


def cross_cost(
    prob: ClusterProblem,
    X: np.ndarray,
    Y: np.ndarray,
    metric: str,
    normalize_cost: bool,
    cross_feature_cost_fn: Callable[..., np.ndarray] | None = None,
    cross_feature_cost_kwargs: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, str]:
    """Cluster-level cross-domain cost ``M`` (K_x, K_y) and how it was made.

    Default: ``ot.dist(centers_x, centers_y, metric=metric)``. With
    ``cross_feature_cost_fn`` the callable supplies ``M`` — same contract as
    ``run_ensemble_fgw`` — which is what allows X and Y of different
    dimension.
    """
    # imported here: fgw imports nothing from this module, but keep it lazy
    from .fgw import _validate_custom_cross_feature_cost

    K_x, K_y = prob.centers_x.shape[0], prob.centers_y.shape[0]
    if cross_feature_cost_fn is None:
        M = ot.dist(prob.centers_x, prob.centers_y, metric=metric)
        mode = "default"
    else:
        M_raw = cross_feature_cost_fn(
            X=X, Y=Y,
            centers_x=prob.centers_x, centers_y=prob.centers_y,
            labels_x=prob.labels_x, labels_y=prob.labels_y,
            n_clusters_x=K_x, n_clusters_y=K_y,
            seed=prob.seed,
            metric=metric,
            **dict(cross_feature_cost_kwargs or {}),
        )
        M = _validate_custom_cross_feature_cost(M_raw, K_x, K_y)
        mode = "custom_fn"
    return max_normalize(M, normalize_cost), mode


def structure_costs(
    prob: ClusterProblem, metric: str, normalize_cost: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Intra-domain cluster-level structure matrices (K_x, K_x) and (K_y, K_y)."""
    C1 = ot.dist(prob.centers_x, prob.centers_x, metric=metric)
    C2 = ot.dist(prob.centers_y, prob.centers_y, metric=metric)
    return max_normalize(C1, normalize_cost), max_normalize(C2, normalize_cost)


def forwarded(**kwargs: Any) -> dict[str, Any]:
    """Keep only the arguments the caller actually set (``None`` = POT default)."""
    return {k: v for k, v in kwargs.items() if v is not None}


def reject_unused(solver_method: str, **kwargs: Any) -> None:
    """Raise if the caller set an argument ``solver_method`` does not accept."""
    bad = sorted(k for k, v in kwargs.items() if v is not None)
    if bad:
        raise ValueError(
            f"solver_method={solver_method!r} does not accept: {', '.join(bad)}"
        )


def require(solver_method: str, **kwargs: Any) -> None:
    """Raise if an argument that ``solver_method`` needs was left unset."""
    missing = sorted(k for k, v in kwargs.items() if v is None)
    if missing:
        raise ValueError(
            f"solver_method={solver_method!r} requires: {', '.join(missing)}"
        )


def jsonable(value: Any) -> Any:
    """Turn solver arguments into plain JSON-friendly Python values."""
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in np.asarray(value).tolist()]
    return str(value)


def make_operator(
    prob: ClusterProblem,
    T_cluster: np.ndarray,
    solver_family: str,
    solver_name: str,
    clustering_method: str,
    solver_params: Mapping[str, Any],
    extra_metrics: Mapping[str, Any] | None = None,
) -> ImplicitTransportOperator:
    """Attach metrics / meta and wrap the coupling in an implicit operator."""
    T_cluster = np.asarray(T_cluster, dtype=float)
    metrics = cluster_shape_metrics(
        prob.labels_x, prob.labels_y, prob.sizes_x, prob.sizes_y, T_cluster
    )
    metrics.update(transport_metrics(T_cluster, prob.a, prob.b))
    metrics.update(extra_metrics or {})
    if "inertia" in prob.info_x:
        metrics["clustering_inertia_x"] = float(prob.info_x["inertia"])
    if "inertia" in prob.info_y:
        metrics["clustering_inertia_y"] = float(prob.info_y["inertia"])
    meta = {
        "solver_family": solver_family,
        "solver_name": solver_name,
        "clustering_method": clustering_method,
        "seed": int(prob.seed),
        "solver_params": {k: jsonable(v) for k, v in solver_params.items()},
        "metrics": metrics,
    }
    return ImplicitTransportOperator(
        labels_x=prob.labels_x,
        labels_y=prob.labels_y,
        T_cluster=T_cluster,
        cluster_mass_x=prob.sizes_x,
        cluster_mass_y=prob.sizes_y,
        meta=meta,
    )
