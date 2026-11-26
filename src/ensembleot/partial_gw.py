"""Partial Gromov–Wasserstein ensemble entry point (structure cost only).

The partial counterpart of ``run_ensemble_gw``: X and Y may live in
different feature spaces, only intra-domain structure matrices are compared,
and only a fraction ``m`` of the mass is transported.

Solvers are POT's ``ot.gromov.partial_gromov_wasserstein`` and
``ot.gromov.entropic_partial_gromov_wasserstein`` (POT >= 0.9.6). Their
arguments are forwarded verbatim; see ``ensembleot._runs`` for the
forwarding convention.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import ot
import ot.gromov

from ._runs import (
    build_cluster_problem,
    forwarded,
    make_operator,
    make_seeds,
    reject_unused,
    structure_costs,
    validate_inputs,
)
from .metrics import partial_mass_metrics
from .operator import ImplicitTransportOperator

PartialGWSolverMethod = Literal[
    "partial_gromov_wasserstein", "entropic_partial_gromov_wasserstein"
]


def _solver_call(
    solver_method: str,
    *,
    m: Any,
    loss_fun: Any,
    nb_dummies: Any,
    thres: Any,
    reg: Any,
    numItermax: Any,
    tol: Any,
    symmetric: Any,
):
    """Validate the argument combination and return ``(fn, kwargs)`` for POT."""
    if solver_method == "partial_gromov_wasserstein":
        reject_unused(solver_method, reg=reg)
        return ot.gromov.partial_gromov_wasserstein, forwarded(
            m=m, loss_fun=loss_fun, nb_dummies=nb_dummies, thres=thres,
            numItermax=numItermax, tol=tol, symmetric=symmetric,
        )
    if solver_method == "entropic_partial_gromov_wasserstein":
        reject_unused(solver_method, nb_dummies=nb_dummies, thres=thres)
        return ot.gromov.entropic_partial_gromov_wasserstein, forwarded(
            reg=reg, m=m, loss_fun=loss_fun,
            numItermax=numItermax, tol=tol, symmetric=symmetric,
        )
    raise ValueError(f"unknown solver_method {solver_method!r}")


def run_ensemble_partial_gw(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    *,
    m: float | None = None,
    clustering_method: str = "kmeans",
    solver_method: PartialGWSolverMethod = "partial_gromov_wasserstein",
    random_state: int | None = None,
    metric: str = "sqeuclidean",
    normalize_cost: bool = True,
    loss_fun: str | None = None,
    nb_dummies: int | None = None,
    thres: float | None = None,
    reg: float | None = None,
    numItermax: int | None = None,
    tol: float | None = None,
    symmetric: bool | None = None,
    coords_x: np.ndarray | None = None,
    coords_y: np.ndarray | None = None,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level partial Gromov–Wasserstein trials.

    Parameters
    ----------
    X, Y : (n_x, d_x), (n_y, d_y) arrays; the feature dimensions may differ.
    n_clusters_x, n_clusters_y, n_runs, clustering_method, random_state,
    coords_x, coords_y :
        As in ``run_ensemble_gw``.
    solver_method : which POT function solves the cluster-level problem.

        - ``"partial_gromov_wasserstein"`` —
          ``ot.gromov.partial_gromov_wasserstein`` (accepts ``m``,
          ``loss_fun``, ``nb_dummies``, ``thres``, ``numItermax``, ``tol``,
          ``symmetric``).
        - ``"entropic_partial_gromov_wasserstein"`` —
          ``ot.gromov.entropic_partial_gromov_wasserstein`` (accepts ``reg``,
          ``m``, ``loss_fun``, ``numItermax``, ``tol``, ``symmetric``).
    m : float in (0, 1], optional
        Amount of mass to transport, POT semantics; with cluster marginals
        summing to 1 it is the transported fraction. ``None`` is POT's
        default ``min(|p|, |q|)`` = 1.
    loss_fun, nb_dummies, thres, reg, numItermax, tol, symmetric :
        POT's arguments of the same name, forwarded verbatim. ``None``
        means "do not pass", so POT's default applies (note POT's default
        ``reg=1.0`` for the entropic solver is large for max-normalized
        structure matrices).
    metric : metric of ``ot.dist`` for the intra-domain structure matrices.
    normalize_cost : bool, default True
        Divide each structure matrix by its maximum, as ``run_ensemble_gw``
        does. Set to False to hand POT the raw ``ot.dist`` matrices.

    Returns
    -------
    list[ImplicitTransportOperator]
        One per run; see ``run_ensemble_partial`` for how partially
        transported clusters lift to samples.
    """
    validate_inputs(X, Y, n_runs, same_dim=False, who="run_ensemble_partial_gw")
    fn, kwargs = _solver_call(
        solver_method, m=m, loss_fun=loss_fun, nb_dummies=nb_dummies, thres=thres,
        reg=reg, numItermax=numItermax, tol=tol, symmetric=symmetric,
    )

    operators = []
    for seed in make_seeds(random_state, n_runs):
        prob = build_cluster_problem(
            X, Y, n_clusters_x, n_clusters_y, clustering_method, seed,
            coords_x=coords_x, coords_y=coords_y,
        )
        C1, C2 = structure_costs(prob, metric, normalize_cost)
        T_cluster = np.asarray(fn(C1, C2, prob.a, prob.b, **kwargs))
        target = min(prob.a.sum(), prob.b.sum()) if m is None else m
        operators.append(
            make_operator(
                prob, T_cluster,
                solver_family="partial_gw",
                solver_name=solver_method,
                clustering_method=clustering_method,
                solver_params={**kwargs, "metric": metric, "normalize_cost": normalize_cost},
                extra_metrics=partial_mass_metrics(T_cluster, target),
            )
        )
    return operators
