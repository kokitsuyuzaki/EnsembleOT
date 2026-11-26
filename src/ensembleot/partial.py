"""Partial ensemble OT entry point (feature cost).

Partial OT transports only a prescribed amount of mass ``m``: row sums may
fall below the source masses and column sums below the target masses, with
``Σ T = m``. Mass is never created, and the solver itself decides which mass
to leave behind (the most expensive to move). ``partial_wasserstein_lagrange``
is the same problem with a cost threshold ``reg_m`` in place of ``m``.

Cluster marginals sum to 1, so ``m`` is the *fraction* of the total mass to
transport — the same meaning it has in POT for normalized histograms.

All solver arguments are POT's own and are forwarded verbatim; see
``ensembleot._runs`` for the forwarding convention.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import ot
import ot.partial

from ._runs import (
    build_cluster_problem,
    clip_mass,
    cross_cost,
    forwarded,
    make_operator,
    make_seeds,
    reject_unused,
    require,
    validate_inputs,
)
from .metrics import cost_metrics, partial_mass_metrics
from .operator import ImplicitTransportOperator

PartialSolverMethod = Literal[
    "partial_wasserstein",
    "entropic_partial_wasserstein",
    "partial_wasserstein_lagrange",
]


def _solver_call(
    solver_method: str,
    *,
    m: Any,
    nb_dummies: Any,
    reg: Any,
    reg_m: Any,
    numItermax: Any,
    stopThr: Any,
):
    """Validate the argument combination and return ``(fn, kwargs)`` for POT."""
    if solver_method == "partial_wasserstein":
        reject_unused(solver_method, reg=reg, reg_m=reg_m, stopThr=stopThr)
        return ot.partial.partial_wasserstein, forwarded(
            m=m, nb_dummies=nb_dummies, numItermax=numItermax,
        )
    if solver_method == "entropic_partial_wasserstein":
        require(solver_method, reg=reg)
        reject_unused(solver_method, nb_dummies=nb_dummies, reg_m=reg_m)
        return ot.partial.entropic_partial_wasserstein, forwarded(
            reg=reg, m=m, numItermax=numItermax, stopThr=stopThr,
        )
    if solver_method == "partial_wasserstein_lagrange":
        reject_unused(solver_method, m=m, reg=reg, stopThr=stopThr)
        return ot.partial.partial_wasserstein_lagrange, forwarded(
            reg_m=reg_m, nb_dummies=nb_dummies, numItermax=numItermax,
        )
    raise ValueError(f"unknown solver_method {solver_method!r}")


def _mass_metrics(solver_method: str, T_cluster: np.ndarray, m: Any, prob) -> dict:
    if solver_method == "partial_wasserstein_lagrange":
        return {}                      # no target mass: it is an outcome here
    target = min(prob.a.sum(), prob.b.sum()) if m is None else m   # POT's default
    return partial_mass_metrics(T_cluster, target)


def run_ensemble_partial(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    *,
    m: float | None = None,
    clustering_method: str = "kmeans",
    solver_method: PartialSolverMethod = "partial_wasserstein",
    random_state: int | None = None,
    metric: str = "sqeuclidean",
    normalize_cost: bool = True,
    nb_dummies: int | None = None,
    reg: float | None = None,
    reg_m: float | None = None,
    numItermax: int | None = None,
    stopThr: float | None = None,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level partial OT trials.

    Parameters
    ----------
    X, Y : (n_x, d), (n_y, d) arrays in the same feature space.
    n_clusters_x, n_clusters_y, n_runs, clustering_method, random_state :
        As in ``run_ensemble_sinkhorn``.
    solver_method : which POT function solves the cluster-level problem.

        - ``"partial_wasserstein"`` — ``ot.partial.partial_wasserstein``
          (exact; accepts ``m``, ``nb_dummies``, ``numItermax``).
        - ``"entropic_partial_wasserstein"`` —
          ``ot.partial.entropic_partial_wasserstein`` (needs ``reg``;
          accepts ``m``, ``numItermax``, ``stopThr``).
        - ``"partial_wasserstein_lagrange"`` —
          ``ot.partial.partial_wasserstein_lagrange`` (threshold form;
          accepts ``reg_m``, ``nb_dummies``, ``numItermax`` and no ``m``).
    m : float in (0, 1], optional
        Amount of mass to transport, POT semantics. Since the cluster
        marginals sum to 1 this is the transported fraction. ``None`` is
        POT's default (``min(|a|, |b|)`` = 1, i.e. balanced OT).
    nb_dummies, reg, reg_m, numItermax, stopThr :
        POT's arguments of the same name, forwarded verbatim. ``None``
        means "do not pass", so POT's default for the chosen solver applies.
    metric : metric of ``ot.dist`` for the cluster-level cost.
    normalize_cost : bool, default True
        Divide the cluster-level cost by its maximum, as the other
        EnsembleOT entry points do. Irrelevant for which mass ``m`` selects,
        but it sets the scale of ``reg`` and of the Lagrangian ``reg_m``.

    Returns
    -------
    list[ImplicitTransportOperator]
        One per run. Mass is dropped per *cluster* and spread uniformly
        inside it, so a single run says "every sample of this cluster keeps
        60 % of its mass" rather than singling samples out; sample-level
        resolution comes from averaging runs with different clusterings
        (``make_mean_operator(runs).transported_mass_x(relative=True)``).
        Samples with zero transported mass get NaN from
        ``apply_to_features``.
    """
    validate_inputs(X, Y, n_runs, same_dim=True, who="run_ensemble_partial")
    fn, kwargs = _solver_call(
        solver_method, m=m, nb_dummies=nb_dummies, reg=reg, reg_m=reg_m,
        numItermax=numItermax, stopThr=stopThr,
    )

    operators = []
    for seed in make_seeds(random_state, n_runs):
        prob = build_cluster_problem(
            X, Y, n_clusters_x, n_clusters_y, clustering_method, seed
        )
        M, _ = cross_cost(prob, X, Y, metric, normalize_cost)
        T_cluster = np.asarray(fn(prob.a, prob.b, M, **clip_mass(kwargs, prob)))
        extra = cost_metrics(T_cluster, M)
        extra.update(_mass_metrics(solver_method, T_cluster, m, prob))
        operators.append(
            make_operator(
                prob, T_cluster,
                solver_family="partial",
                solver_name=solver_method,
                clustering_method=clustering_method,
                solver_params={**kwargs, "metric": metric, "normalize_cost": normalize_cost},
                extra_metrics=extra,
            )
        )
    return operators


def scan_partial_mass(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    m_values: Sequence[float],
    *,
    clustering_method: str = "kmeans",
    solver_method: PartialSolverMethod = "partial_wasserstein",
    random_state: int | None = None,
    metric: str = "sqeuclidean",
    normalize_cost: bool = True,
    nb_dummies: int | None = None,
    reg: float | None = None,
    numItermax: int | None = None,
    stopThr: float | None = None,
) -> dict[str, np.ndarray]:
    """Sweep the transported mass ``m`` to help choose it.

    For every run the clustering is computed once and the cluster-level
    partial OT is solved for each value in ``m_values``, which is cheap
    because the problems are only ``K_x × K_y``. With the same
    ``random_state`` the runs use the same clusterings as
    ``run_ensemble_partial``.

    The usual reading: the cost grows slowly while well-matched mass is being
    transported and bends upwards once mass without a counterpart is forced
    across. ``marginal_cost`` — the cost of the last unit of mass — makes
    that bend easier to see than the cost itself.

    Returns
    -------
    dict with
        ``m``                : (n_m,) the scanned values, ascending
        ``transport_cost``   : (n_m, n_runs) ``<T, M>`` per run
        ``transported_mass`` : (n_m, n_runs) ``Σ T`` per run
        ``marginal_cost``    : (n_m - 1,) run-averaged finite difference
                               ``Δcost / Δm`` between consecutive ``m``
        ``m_mid``            : (n_m - 1,) midpoints the differences refer to
    """
    validate_inputs(X, Y, n_runs, same_dim=True, who="scan_partial_mass")
    if solver_method == "partial_wasserstein_lagrange":
        raise ValueError("scan_partial_mass needs a solver that takes m")
    m_arr = np.sort(np.asarray(list(m_values), dtype=float))
    if m_arr.ndim != 1 or m_arr.size < 1:
        raise ValueError("m_values must be a non-empty 1-D sequence")
    if np.any(m_arr <= 0) or np.any(m_arr > 1):
        raise ValueError("m_values must lie in (0, 1]")
    calls = [
        _solver_call(
            solver_method, m=float(m), nb_dummies=nb_dummies, reg=reg, reg_m=None,
            numItermax=numItermax, stopThr=stopThr,
        )
        for m in m_arr
    ]

    cost = np.empty((m_arr.size, n_runs), dtype=float)
    mass = np.empty((m_arr.size, n_runs), dtype=float)
    for r, seed in enumerate(make_seeds(random_state, n_runs)):
        prob = build_cluster_problem(
            X, Y, n_clusters_x, n_clusters_y, clustering_method, seed
        )
        M, _ = cross_cost(prob, X, Y, metric, normalize_cost)
        for k, (fn, kwargs) in enumerate(calls):
            T_cluster = np.asarray(fn(prob.a, prob.b, M, **clip_mass(kwargs, prob)))
            cost[k, r] = float(np.sum(T_cluster * M))
            mass[k, r] = float(T_cluster.sum())

    mean_cost = cost.mean(axis=1)
    dm = np.diff(m_arr)
    marginal = np.divide(
        np.diff(mean_cost), dm, out=np.full(dm.shape, np.nan), where=dm > 0
    )
    return {
        "m": m_arr,
        "transport_cost": cost,
        "transported_mass": mass,
        "marginal_cost": marginal,
        "m_mid": (m_arr[:-1] + m_arr[1:]) / 2.0,
    }
