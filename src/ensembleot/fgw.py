"""Fused Gromov-Wasserstein ensemble OT entry point (Stage 12a).

This minimal FGW implementation:

  * clusters X and Y independently with k-means
  * builds intra-domain cluster-level structure matrices C1, C2
  * builds a cross-domain cluster-level feature cost matrix M
  * solves a cluster-level (entropic or not) Fused Gromov-Wasserstein
    coupling with POT, forwarding ``alpha`` as-is
  * wraps the result in an ImplicitTransportOperator with uniform lifting

Stage 12a default behavior: X and Y must share the same feature
dimension (``X.shape[1] == Y.shape[1]``) and ``M`` is built internally
from cluster centers.

Stage 12b extension: pass ``cross_feature_cost_fn=...`` to supply a
per-run cluster-level ``M`` externally. This enables cross-modal FGW
where ``X`` and ``Y`` can have different feature dimensions. Full sample × sample cost / transport
matrices are never materialized.

On ``alpha``
------------
``alpha`` is forwarded verbatim to POT's FGW solvers. Its semantics are
therefore **exactly** what POT defines:

    T* ∈ argmin_T  (1 - alpha) <T, M>_F
                 + alpha · Σ L(C1_ik, C2_jl) T_ij T_kl

(see ``ot.gromov.fused_gromov_wasserstein`` /
``ot.gromov.entropic_fused_gromov_wasserstein``). EnsembleOT does not
rename, invert, or reinterpret this parameter.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping

import numpy as np
import ot
import ot.gromov

from .clustering import cluster_means, cluster_samples_with_info, cluster_sizes
from .metrics import cluster_shape_metrics, transport_metrics
from .operator import ImplicitTransportOperator

FGWSolverMethod = Literal["fgw", "entropic_fgw"]

CrossFeatureCostFn = Callable[..., np.ndarray]


def _default_cross_feature_cost(
    centers_x: np.ndarray,
    centers_y: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Stage 12a default: ``ot.dist(centers_x, centers_y, metric=metric)``."""
    return ot.dist(centers_x, centers_y, metric=metric)


def _validate_custom_cross_feature_cost(
    M: Any,
    n_clusters_x: int,
    n_clusters_y: int,
) -> np.ndarray:
    """Validate and cast a user-supplied cluster-level cross feature cost."""
    M_arr = np.asarray(M, dtype=float)
    if M_arr.ndim != 2:
        raise ValueError(
            f"cross_feature_cost_fn must return a 2-D array, got ndim={M_arr.ndim}"
        )
    if M_arr.shape != (n_clusters_x, n_clusters_y):
        raise ValueError(
            "cross_feature_cost_fn returned shape "
            f"{M_arr.shape}, expected ({n_clusters_x}, {n_clusters_y})"
        )
    if not np.all(np.isfinite(M_arr)):
        raise ValueError("cross_feature_cost_fn returned non-finite values (NaN/inf)")
    return M_arr


def _solve_cluster_fgw(
    M: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    solver_method: FGWSolverMethod,
    loss_fun: str,
    alpha: float,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    if solver_method == "fgw":
        T = ot.gromov.fused_gromov_wasserstein(
            M, C1, C2, p, q,
            loss_fun=loss_fun,
            alpha=alpha,
            max_iter=max_iter,
            tol_rel=tol,
            tol_abs=tol,
        )
    elif solver_method == "entropic_fgw":
        T = ot.gromov.entropic_fused_gromov_wasserstein(
            M, C1, C2, p, q,
            loss_fun=loss_fun,
            epsilon=epsilon,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
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
    solver_method: FGWSolverMethod,
    alpha: float,
    metric: str,
    loss_fun: str,
    epsilon: float,
    max_iter: int,
    tol: float,
    seed: int,
    cross_feature_cost_fn: CrossFeatureCostFn | None,
    cross_feature_cost_kwargs: Mapping[str, Any] | None,
) -> ImplicitTransportOperator:
    n_x, n_y = X.shape[0], Y.shape[0]
    labels_x, info_x = cluster_samples_with_info(X, clustering_method, n_clusters_x, random_state=seed)
    labels_y, info_y = cluster_samples_with_info(Y, clustering_method, n_clusters_y, random_state=seed + 1)

    centers_x = cluster_means(X, labels_x, n_clusters_x)
    centers_y = cluster_means(Y, labels_y, n_clusters_y)

    # intra-domain structure costs (K_x×K_x, K_y×K_y). These always come
    # from the per-domain cluster centers regardless of the feature-cost
    # mode — Stage 12b only generalizes the *cross-domain* cost.
    C1 = ot.dist(centers_x, centers_x, metric=metric)
    C2 = ot.dist(centers_y, centers_y, metric=metric)

    # cross-domain feature cost (K_x×K_y)
    if cross_feature_cost_fn is None:
        M = _default_cross_feature_cost(centers_x, centers_y, metric=metric)
        cross_feature_cost_mode = "default"
    else:
        extra = dict(cross_feature_cost_kwargs or {})
        M_raw = cross_feature_cost_fn(
            X=X, Y=Y,
            centers_x=centers_x, centers_y=centers_y,
            labels_x=labels_x, labels_y=labels_y,
            n_clusters_x=n_clusters_x, n_clusters_y=n_clusters_y,
            seed=int(seed),
            metric=metric,
            **extra,
        )
        M = _validate_custom_cross_feature_cost(M_raw, n_clusters_x, n_clusters_y)
        cross_feature_cost_mode = "custom_fn"

    # Numerical stabilization: each matrix is max-normalized independently.
    # This is a common preprocessing for FGW so that structure and feature
    # costs live on comparable scales before `alpha` trades them off.
    # Custom M returned from cross_feature_cost_fn is normalized the same
    # way for consistency with the default mode.
    if C1.max() > 0:
        C1 = C1 / C1.max()
    if C2.max() > 0:
        C2 = C2 / C2.max()
    if M.max() > 0:
        M = M / M.max()

    sizes_x = cluster_sizes(labels_x, n_clusters_x).astype(float)
    sizes_y = cluster_sizes(labels_y, n_clusters_y).astype(float)

    p = sizes_x / n_x
    q = sizes_y / n_y

    T_cluster = _solve_cluster_fgw(
        M, C1, C2, p, q,
        solver_method=solver_method,
        loss_fun=loss_fun,
        alpha=alpha,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    metrics = cluster_shape_metrics(labels_x, labels_y, sizes_x, sizes_y, T_cluster)
    metrics.update(transport_metrics(T_cluster, p, q))
    if "inertia" in info_x:
        metrics["clustering_inertia_x"] = float(info_x["inertia"])
    if "inertia" in info_y:
        metrics["clustering_inertia_y"] = float(info_y["inertia"])

    meta = {
        "solver_family": "fgw",
        "solver_name": solver_method,
        "clustering_method": clustering_method,
        "seed": int(seed),
        "solver_params": {
            "alpha": float(alpha),
            "metric": str(metric),
            "loss_fun": str(loss_fun),
            "epsilon": float(epsilon),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "cross_feature_cost_mode": cross_feature_cost_mode,
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


def run_ensemble_fgw(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    clustering_method: str = "kmeans",
    solver_method: FGWSolverMethod = "fgw",
    alpha: float = 0.5,
    random_state: int | None = None,
    metric: str = "sqeuclidean",
    loss_fun: str = "square_loss",
    epsilon: float = 0.05,
    max_iter: int = 1000,
    tol: float = 1e-6,
    cross_feature_cost_fn: CrossFeatureCostFn | None = None,
    cross_feature_cost_kwargs: Mapping[str, Any] | None = None,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level Fused Gromov-Wasserstein OT trials.

    Each run clusters X and Y, builds cluster-level structure matrices
    C1, C2 and a cross-domain feature cost M, then solves a K_x × K_y
    FGW coupling via POT. The sample-level transport is held implicitly
    through an ``ImplicitTransportOperator`` (uniform lifting).

    Parameters
    ----------
    alpha : float, default 0.5
        Trade-off between feature and structure costs. **The semantics are
        those of POT's FGW solvers** (``ot.gromov.fused_gromov_wasserstein``
        / ``ot.gromov.entropic_fused_gromov_wasserstein``): the objective
        is ``(1 - alpha) <T, M> + alpha · <L(C1,C2) ⊗ T, T>``. EnsembleOT
        forwards this parameter verbatim and does not reinterpret it.
    cross_feature_cost_fn : callable, optional
        **Stage 12b extension.** Per-run callable that returns the
        cluster-level cross-domain feature cost ``M`` of shape
        ``(n_clusters_x, n_clusters_y)``. Called as::

            cross_feature_cost_fn(
                X=X, Y=Y,
                centers_x=centers_x, centers_y=centers_y,
                labels_x=labels_x, labels_y=labels_y,
                n_clusters_x=..., n_clusters_y=...,
                seed=..., metric=...,
                **cross_feature_cost_kwargs,
            )

        When ``None`` (default), Stage 12a behavior is preserved:
        ``M = ot.dist(centers_x, centers_y, metric=metric)``, which
        requires ``X.shape[1] == Y.shape[1]``. When provided, X and Y
        may live in *different* feature dimensions — this is the
        cross-modal FGW path.
    cross_feature_cost_kwargs : dict, optional
        Extra keyword arguments forwarded to ``cross_feature_cost_fn``.

    Notes
    -----
    - Stage 12a: same feature dimension only (``cross_feature_cost_fn=None``).
    - Stage 12b: pass ``cross_feature_cost_fn`` to supply a custom
      cluster-level ``M``; this lifts the same-dimension restriction.
    - In both modes ``alpha`` is forwarded verbatim to POT's FGW solvers;
      its meaning is defined by POT and is not reinterpreted here.
    - Both the default and custom ``M`` are max-normalized before being
      passed to the solver, matching how C1/C2 are preprocessed.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2-D")
    if cross_feature_cost_fn is None and X.shape[1] != Y.shape[1]:
        raise ValueError(
            "run_ensemble_fgw default mode requires X and Y to share feature "
            f"dimension; got X.shape[1]={X.shape[1]} vs Y.shape[1]={Y.shape[1]}. "
            "Pass cross_feature_cost_fn=... to supply a custom cluster-level M "
            "for cross-modal FGW."
        )

    rng = np.random.default_rng(random_state)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_runs)]

    return [
        _single_run(
            X, Y,
            n_clusters_x=n_clusters_x,
            n_clusters_y=n_clusters_y,
            clustering_method=clustering_method,
            solver_method=solver_method,
            alpha=alpha,
            metric=metric,
            loss_fun=loss_fun,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
            cross_feature_cost_fn=cross_feature_cost_fn,
            cross_feature_cost_kwargs=cross_feature_cost_kwargs,
        )
        for seed in seeds
    ]
