"""Entropic / EMD ensemble OT entry point.

This Stage 3 implementation:

  * clusters X and Y independently with k-means
  * builds a cluster-level squared-euclidean cost matrix on the cluster means
  * solves a cluster-level OT with POT (sinkhorn or emd)
  * wraps each result in an ImplicitTransportOperator (uniform lifting)

Full sample × sample cost / transport matrices are *never* materialized.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import ot

from .clustering import cluster_means, cluster_samples_with_info, cluster_sizes
from .metrics import cluster_shape_metrics, transport_metrics
from .operator import ImplicitTransportOperator

SolverMethod = Literal["sinkhorn", "emd"]


def _solve_cluster_ot(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    solver_method: SolverMethod,
    reg: float,
    numItermax: int,
    stopThr: float,
) -> np.ndarray:
    if solver_method == "sinkhorn":
        return ot.sinkhorn(a, b, C, reg=reg, numItermax=numItermax, stopThr=stopThr)
    if solver_method == "emd":
        return ot.emd(a, b, C, numItermax=numItermax)
    raise ValueError(f"unknown solver_method {solver_method!r}")


def _single_run(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    clustering_method: str,
    solver_method: SolverMethod,
    reg: float,
    numItermax: int,
    stopThr: float,
    seed: int,
) -> ImplicitTransportOperator:
    n_x, n_y = X.shape[0], Y.shape[0]
    labels_x, info_x = cluster_samples_with_info(X, clustering_method, n_clusters_x, random_state=seed)
    labels_y, info_y = cluster_samples_with_info(Y, clustering_method, n_clusters_y, random_state=seed + 1)

    centers_x = cluster_means(X, labels_x, n_clusters_x)
    centers_y = cluster_means(Y, labels_y, n_clusters_y)

    sizes_x = cluster_sizes(labels_x, n_clusters_x).astype(float)
    sizes_y = cluster_sizes(labels_y, n_clusters_y).astype(float)

    # POT marginals: sample-mass normalized, sums to 1
    a = sizes_x / n_x
    b = sizes_y / n_y

    # cluster-level squared-euclidean cost (small: K_x × K_y)
    C = ot.dist(centers_x, centers_y, metric="sqeuclidean")
    if C.max() > 0:
        C = C / C.max()

    T_cluster = _solve_cluster_ot(
        a, b, C, solver_method, reg=reg, numItermax=numItermax, stopThr=stopThr
    )

    T_cluster = np.asarray(T_cluster)

    metrics = cluster_shape_metrics(labels_x, labels_y, sizes_x, sizes_y, T_cluster)
    metrics.update(transport_metrics(T_cluster, a, b))
    if "inertia" in info_x:
        metrics["clustering_inertia_x"] = float(info_x["inertia"])
    if "inertia" in info_y:
        metrics["clustering_inertia_y"] = float(info_y["inertia"])

    meta = {
        "solver_family": "sinkhorn",
        "solver_name": solver_method,
        "clustering_method": clustering_method,
        "seed": int(seed),
        "solver_params": {
            "reg": float(reg),
            "numItermax": int(numItermax),
            "stopThr": float(stopThr),
        },
        "metrics": metrics,
    }

    return ImplicitTransportOperator(
        labels_x=labels_x,
        labels_y=labels_y,
        T_cluster=T_cluster,
        cluster_mass_x=sizes_x,
        cluster_mass_y=sizes_y,
        meta=meta,
    )


def run_ensemble_sinkhorn(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    clustering_method: str = "kmeans",
    solver_method: SolverMethod = "sinkhorn",
    random_state: int | None = None,
    reg: float = 0.1,
    numItermax: int = 1000,
    stopThr: float = 1e-6,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level Sinkhorn / EMD OT trials.

    Returns
    -------
    list[ImplicitTransportOperator]
        One operator per run. Aggregation across runs lands in a later stage —
        for now the caller gets the raw per-run operators back.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must share feature dimension for sinkhorn (sqeuclidean cost)")

    rng = np.random.default_rng(random_state)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_runs)]

    return [
        _single_run(
            X, Y,
            n_clusters_x=n_clusters_x,
            n_clusters_y=n_clusters_y,
            clustering_method=clustering_method,
            solver_method=solver_method,
            reg=reg,
            numItermax=numItermax,
            stopThr=stopThr,
            seed=seed,
        )
        for seed in seeds
    ]
