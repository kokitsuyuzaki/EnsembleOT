"""Gromov-Wasserstein ensemble OT entry point.

This Stage 4 implementation:

  * clusters X and Y independently with k-means
  * builds per-domain *cluster-level* intra-domain distance matrices
    Cx (K_x × K_x), Cy (K_y × K_y) on the cluster centroids
  * solves a cluster-level (entropic or not) Gromov-Wasserstein coupling
    with POT
  * wraps the result in an ImplicitTransportOperator with uniform lifting

X and Y may live in *different* feature dimensions (d_x ≠ d_y). Full
sample × sample distance / transport matrices are never materialized.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import ot
import ot.gromov

from .clustering import cluster_means, cluster_samples_with_info, cluster_sizes
from .metrics import cluster_shape_metrics, transport_metrics
from .operator import ImplicitTransportOperator

GWSolverMethod = Literal["gw", "entropic_gw"]


def _solve_cluster_gw(
    Cx: np.ndarray,
    Cy: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    solver_method: GWSolverMethod,
    loss_fun: str,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    if solver_method == "gw":
        T = ot.gromov.gromov_wasserstein(
            Cx, Cy, a, b, loss_fun=loss_fun, max_iter=max_iter, tol=tol,
        )
    elif solver_method == "entropic_gw":
        T = ot.gromov.entropic_gromov_wasserstein(
            Cx, Cy, a, b,
            loss_fun=loss_fun, epsilon=epsilon, max_iter=max_iter, tol=tol,
        )
    else:
        raise ValueError(f"unknown solver_method {solver_method!r}")
    return np.asarray(T)


def _single_run(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    clustering_method: str,
    solver_method: GWSolverMethod,
    loss_fun: str,
    epsilon: float,
    max_iter: int,
    tol: float,
    seed: int,
) -> ImplicitTransportOperator:
    n_x, n_y = X.shape[0], Y.shape[0]
    labels_x, info_x = cluster_samples_with_info(X, clustering_method, n_clusters_x, random_state=seed)
    labels_y, info_y = cluster_samples_with_info(Y, clustering_method, n_clusters_y, random_state=seed + 1)

    centers_x = cluster_means(X, labels_x, n_clusters_x)
    centers_y = cluster_means(Y, labels_y, n_clusters_y)

    # Intra-domain cluster-level distance matrices (small: K_x×K_x, K_y×K_y)
    Cx = ot.dist(centers_x, centers_x, metric="sqeuclidean")
    Cy = ot.dist(centers_y, centers_y, metric="sqeuclidean")
    if Cx.max() > 0:
        Cx = Cx / Cx.max()
    if Cy.max() > 0:
        Cy = Cy / Cy.max()

    sizes_x = cluster_sizes(labels_x, n_clusters_x).astype(float)
    sizes_y = cluster_sizes(labels_y, n_clusters_y).astype(float)

    a = sizes_x / n_x
    b = sizes_y / n_y

    T_cluster = _solve_cluster_gw(
        Cx, Cy, a, b,
        solver_method=solver_method,
        loss_fun=loss_fun,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    metrics = cluster_shape_metrics(labels_x, labels_y, sizes_x, sizes_y, T_cluster)
    metrics.update(transport_metrics(T_cluster, a, b))
    if "inertia" in info_x:
        metrics["clustering_inertia_x"] = float(info_x["inertia"])
    if "inertia" in info_y:
        metrics["clustering_inertia_y"] = float(info_y["inertia"])

    meta = {
        "solver_family": "gw",
        "solver_name": solver_method,
        "clustering_method": clustering_method,
        "seed": int(seed),
        "solver_params": {
            "loss_fun": str(loss_fun),
            "epsilon": float(epsilon),
            "max_iter": int(max_iter),
            "tol": float(tol),
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


def run_ensemble_gw(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    clustering_method: str = "kmeans",
    solver_method: GWSolverMethod = "entropic_gw",
    random_state: int | None = None,
    loss_fun: str = "square_loss",
    epsilon: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level Gromov-Wasserstein OT trials.

    X and Y may live in different feature spaces. Each run solves a
    K_x × K_y cluster-level GW coupling and returns an implicit
    sample-level transport operator via uniform lifting. Aggregation and
    storage land in a later stage.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D")

    rng = np.random.default_rng(random_state)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_runs)]

    return [
        _single_run(
            X, Y,
            n_clusters_x=n_clusters_x,
            n_clusters_y=n_clusters_y,
            clustering_method=clustering_method,
            solver_method=solver_method,
            loss_fun=loss_fun,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        for seed in seeds
    ]
