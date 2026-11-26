"""Unbalanced ensemble OT entry point (feature cost).

Unbalanced OT replaces the hard marginal constraints of balanced OT by a
penalty ``reg_m · Div(T1 | a) + reg_m · Div(Tᵀ1 | b)``: mass may be
destroyed *or created* wherever paying the penalty is cheaper than paying
the transport cost. The total transported mass is an outcome of the
optimization, not an input.

Each run clusters X and Y, builds the cluster-level cost on the cluster
means and calls one of POT's ``ot.unbalanced`` solvers. All solver arguments
are POT's own and are forwarded verbatim; see ``ensembleot._runs`` for the
forwarding convention (``None`` = POT's default for the selected solver).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import ot
import ot.unbalanced

from ._runs import (
    build_cluster_problem,
    cross_cost,
    forwarded,
    make_operator,
    make_seeds,
    reject_unused,
    require,
    validate_inputs,
)
from .metrics import cost_metrics
from .operator import ImplicitTransportOperator

UnbalancedSolverMethod = Literal[
    "sinkhorn_unbalanced", "mm_unbalanced", "lbfgsb_unbalanced"
]


def _solver_call(
    solver_method: str,
    *,
    reg: Any,
    reg_m: Any,
    method: Any,
    reg_type: Any,
    div: Any,
    reg_div: Any,
    regm_div: Any,
    numItermax: Any,
    stopThr: Any,
):
    """Validate the argument combination and return ``(fn, kwargs)`` for POT."""
    if solver_method == "sinkhorn_unbalanced":
        require(solver_method, reg=reg)
        reject_unused(solver_method, div=div, reg_div=reg_div, regm_div=regm_div)
        return ot.unbalanced.sinkhorn_unbalanced, forwarded(
            reg=reg, reg_m=reg_m, method=method, reg_type=reg_type,
            numItermax=numItermax, stopThr=stopThr,
        )
    if solver_method in ("mm_unbalanced", "lbfgsb_unbalanced") and not np.all(
        np.isfinite(np.asarray(reg_m, dtype=float))
    ):
        # POT only handles an infinite reg_m (exact marginal) in the Sinkhorn
        # solver; mm returns NaN and lbfgsb silently returns a wrong plan.
        raise ValueError(
            f"solver_method={solver_method!r} needs a finite reg_m; use "
            "'sinkhorn_unbalanced' to keep a marginal exact with reg_m=inf"
        )
    if solver_method == "mm_unbalanced":
        reject_unused(
            solver_method, method=method, reg_type=reg_type,
            reg_div=reg_div, regm_div=regm_div,
        )
        return ot.unbalanced.mm_unbalanced, forwarded(
            reg_m=reg_m, reg=reg, div=div, numItermax=numItermax, stopThr=stopThr,
        )
    if solver_method == "lbfgsb_unbalanced":
        require(solver_method, reg=reg)
        reject_unused(solver_method, reg_type=reg_type, div=div)
        return ot.unbalanced.lbfgsb_unbalanced, forwarded(
            reg=reg, reg_m=reg_m, reg_div=reg_div, regm_div=regm_div,
            method=method, numItermax=numItermax, stopThr=stopThr,
        )
    raise ValueError(f"unknown solver_method {solver_method!r}")


def run_ensemble_unbalanced(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    *,
    reg_m: float | tuple[float, float],
    reg: float | None = None,
    clustering_method: str = "kmeans",
    solver_method: UnbalancedSolverMethod = "sinkhorn_unbalanced",
    random_state: int | None = None,
    metric: str = "sqeuclidean",
    normalize_cost: bool = True,
    method: str | None = None,
    reg_type: str | None = None,
    div: str | None = None,
    reg_div: str | None = None,
    regm_div: str | None = None,
    numItermax: int | None = None,
    stopThr: float | None = None,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level unbalanced OT trials.

    Parameters
    ----------
    X, Y : (n_x, d), (n_y, d) arrays in the same feature space.
    n_clusters_x, n_clusters_y, n_runs, clustering_method, random_state :
        As in ``run_ensemble_sinkhorn``.
    solver_method : which POT function solves the cluster-level problem.

        - ``"sinkhorn_unbalanced"`` — ``ot.unbalanced.sinkhorn_unbalanced``
          (entropic; needs ``reg``; accepts ``method``, ``reg_type``).
        - ``"mm_unbalanced"`` — ``ot.unbalanced.mm_unbalanced``
          (majorization-minimization; ``reg`` optional, POT default 0;
          accepts ``div``).
        - ``"lbfgsb_unbalanced"`` — ``ot.unbalanced.lbfgsb_unbalanced``
          (needs ``reg``; accepts ``reg_div``, ``regm_div`` and ``method``,
          which here is the scipy optimizer name).
    reg_m : float or pair
        Marginal relaxation strength, POT semantics. A pair
        ``(reg_m1, reg_m2)`` relaxes the X and Y marginals differently.
        Larger values approach balanced OT. ``float("inf")`` on a side keeps
        that marginal exact, which only ``"sinkhorn_unbalanced"`` supports;
        the other solvers need finite values.
    reg, method, reg_type, div, reg_div, regm_div, numItermax, stopThr :
        POT's arguments of the same name, forwarded verbatim. ``None``
        means "do not pass", so POT's default for the chosen solver applies.
    metric : metric of ``ot.dist`` for the cluster-level cost.
    normalize_cost : bool, default True
        Divide the cluster-level cost by its maximum, as the other
        EnsembleOT entry points do. **The effect of ``reg_m`` depends on the
        cost scale**; set this to False to hand POT the raw ``ot.dist`` cost
        exactly as a direct POT call would.

    Returns
    -------
    list[ImplicitTransportOperator]
        One per run. Row / column sums of the coupling no longer equal the
        cluster masses; use ``op.transported_mass_x()`` /
        ``transported_mass_y()`` to see how much mass each sample keeps.
        ``op.meta["metrics"]`` carries ``transported_mass`` and
        ``transport_cost``; the ``marginal_error_*`` entries measure the
        intended relaxation here, not solver error.
    """
    validate_inputs(X, Y, n_runs, same_dim=True, who="run_ensemble_unbalanced")
    if reg_m is None:
        raise ValueError("reg_m is required")
    fn, kwargs = _solver_call(
        solver_method, reg=reg, reg_m=reg_m, method=method, reg_type=reg_type,
        div=div, reg_div=reg_div, regm_div=regm_div,
        numItermax=numItermax, stopThr=stopThr,
    )

    operators = []
    for seed in make_seeds(random_state, n_runs):
        prob = build_cluster_problem(
            X, Y, n_clusters_x, n_clusters_y, clustering_method, seed
        )
        M, _ = cross_cost(prob, X, Y, metric, normalize_cost)
        T_cluster = np.asarray(fn(prob.a, prob.b, M, **kwargs))
        operators.append(
            make_operator(
                prob, T_cluster,
                solver_family="unbalanced",
                solver_name=solver_method,
                clustering_method=clustering_method,
                solver_params={**kwargs, "metric": metric, "normalize_cost": normalize_cost},
                extra_metrics=cost_metrics(T_cluster, M),
            )
        )
    return operators
