"""Partial Fused Gromov–Wasserstein ensemble entry point.

The partial counterpart of ``run_ensemble_fgw``: a cross-domain feature cost
``M`` and intra-domain structure matrices are combined through ``alpha``,
and only a fraction ``m`` of the mass is transported.

Solvers are POT's ``ot.gromov.partial_fused_gromov_wasserstein`` and
``ot.gromov.entropic_partial_fused_gromov_wasserstein`` (POT >= 0.9.6).
Their arguments are forwarded verbatim; see ``ensembleot._runs`` for the
forwarding convention.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping

import numpy as np
import ot
import ot.gromov

from ._runs import (
    build_cluster_problem,
    clip_mass,
    cross_cost,
    forwarded,
    make_operator,
    make_seeds,
    reject_unused,
    structure_costs,
    validate_inputs,
)
from .metrics import cost_metrics, partial_mass_metrics
from .operator import ImplicitTransportOperator

PartialFGWSolverMethod = Literal[
    "partial_fused_gromov_wasserstein", "entropic_partial_fused_gromov_wasserstein"
]


def _solver_call(
    solver_method: str,
    *,
    m: Any,
    loss_fun: Any,
    alpha: Any,
    nb_dummies: Any,
    thres: Any,
    reg: Any,
    numItermax: Any,
    tol: Any,
    symmetric: Any,
):
    """Validate the argument combination and return ``(fn, kwargs)`` for POT."""
    if solver_method == "partial_fused_gromov_wasserstein":
        reject_unused(solver_method, reg=reg)
        return ot.gromov.partial_fused_gromov_wasserstein, forwarded(
            m=m, loss_fun=loss_fun, alpha=alpha, nb_dummies=nb_dummies, thres=thres,
            numItermax=numItermax, tol=tol, symmetric=symmetric,
        )
    if solver_method == "entropic_partial_fused_gromov_wasserstein":
        reject_unused(solver_method, nb_dummies=nb_dummies, thres=thres)
        return ot.gromov.entropic_partial_fused_gromov_wasserstein, forwarded(
            reg=reg, m=m, loss_fun=loss_fun, alpha=alpha,
            numItermax=numItermax, tol=tol, symmetric=symmetric,
        )
    raise ValueError(f"unknown solver_method {solver_method!r}")


def run_ensemble_partial_fgw(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    *,
    m: float | None = None,
    alpha: float | None = None,
    clustering_method: str = "kmeans",
    solver_method: PartialFGWSolverMethod = "partial_fused_gromov_wasserstein",
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
    cross_feature_cost_fn: Callable[..., np.ndarray] | None = None,
    cross_feature_cost_kwargs: Mapping[str, Any] | None = None,
    coords_x: np.ndarray | None = None,
    coords_y: np.ndarray | None = None,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level partial Fused Gromov–Wasserstein trials.

    Parameters
    ----------
    X, Y : (n_x, d_x), (n_y, d_y) arrays. They must share the feature
        dimension unless ``cross_feature_cost_fn`` supplies ``M``.
    n_clusters_x, n_clusters_y, n_runs, clustering_method, random_state,
    cross_feature_cost_fn, cross_feature_cost_kwargs, coords_x, coords_y :
        As in ``run_ensemble_fgw``.
    solver_method : which POT function solves the cluster-level problem.

        - ``"partial_fused_gromov_wasserstein"`` —
          ``ot.gromov.partial_fused_gromov_wasserstein`` (accepts ``m``,
          ``loss_fun``, ``alpha``, ``nb_dummies``, ``thres``,
          ``numItermax``, ``tol``, ``symmetric``).
        - ``"entropic_partial_fused_gromov_wasserstein"`` —
          ``ot.gromov.entropic_partial_fused_gromov_wasserstein`` (accepts
          ``reg``, ``m``, ``loss_fun``, ``alpha``, ``numItermax``, ``tol``,
          ``symmetric``).
    m : float in (0, 1], optional
        Amount of mass to transport, POT semantics (the transported
        fraction, since cluster marginals sum to 1). ``None`` is POT's
        default ``min(|p|, |q|)`` = 1.
    alpha : float, optional
        Trade-off between feature and structure costs with **POT's
        semantics for these solvers** — the same as ``run_ensemble_fgw``:
        ``(1 - alpha) <T, M> + alpha · <L(C1, C2) ⊗ T, T>``. ``None`` is
        POT's default 0.5.
    loss_fun, nb_dummies, thres, reg, numItermax, tol, symmetric :
        POT's arguments of the same name, forwarded verbatim. ``None``
        means "do not pass", so POT's default applies.
    metric : metric of ``ot.dist`` for ``M`` and the structure matrices.
    normalize_cost : bool, default True
        Divide ``M``, ``C1`` and ``C2`` each by its own maximum, as
        ``run_ensemble_fgw`` does, so that ``alpha`` trades off quantities
        on comparable scales. Set to False for the raw matrices.

    Returns
    -------
    list[ImplicitTransportOperator]
        One per run; see ``run_ensemble_partial`` for how partially
        transported clusters lift to samples.
    """
    validate_inputs(
        X, Y, n_runs, same_dim=cross_feature_cost_fn is None,
        who="run_ensemble_partial_fgw (without cross_feature_cost_fn)",
    )
    fn, kwargs = _solver_call(
        solver_method, m=m, loss_fun=loss_fun, alpha=alpha, nb_dummies=nb_dummies,
        thres=thres, reg=reg, numItermax=numItermax, tol=tol, symmetric=symmetric,
    )

    operators = []
    for seed in make_seeds(random_state, n_runs):
        prob = build_cluster_problem(
            X, Y, n_clusters_x, n_clusters_y, clustering_method, seed,
            coords_x=coords_x, coords_y=coords_y,
        )
        C1, C2 = structure_costs(prob, metric, normalize_cost)
        M, cost_mode = cross_cost(
            prob, X, Y, metric, normalize_cost,
            cross_feature_cost_fn, cross_feature_cost_kwargs,
        )
        T_cluster = np.asarray(fn(M, C1, C2, prob.a, prob.b, **clip_mass(kwargs, prob)))
        target = min(prob.a.sum(), prob.b.sum()) if m is None else m
        extra = cost_metrics(T_cluster, M)
        extra.update(partial_mass_metrics(T_cluster, target))
        operators.append(
            make_operator(
                prob, T_cluster,
                solver_family="partial_fgw",
                solver_name=solver_method,
                clustering_method=clustering_method,
                solver_params={
                    **kwargs, "metric": metric, "normalize_cost": normalize_cost,
                    "cross_feature_cost_mode": cost_mode,
                },
                extra_metrics=extra,
            )
        )
    return operators
