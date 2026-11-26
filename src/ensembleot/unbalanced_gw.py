"""Fused Unbalanced Gromov–Wasserstein ensemble entry point.

POT exposes unbalanced GW and its fused variant through a single function,
``ot.gromov.fused_unbalanced_gromov_wasserstein`` (POT >= 0.9.5), and this
entry point mirrors that: ``alpha=0`` (POT's default) is plain unbalanced
GW on structure matrices only, ``alpha > 0`` adds the linear feature term.

Two things differ from the balanced ``run_ensemble_gw`` / ``run_ensemble_fgw``
and both follow POT:

* Argument names are those of the POT function (``reg_marginals``,
  ``epsilon``, ``divergence``, ``unbalanced_solver``, ``max_iter``, ``tol``,
  ``max_iter_ot``, ``tol_ot``), not the ``reg`` / ``reg_m`` names of
  ``ot.unbalanced``.
* ``alpha`` is **not** the convex weight of balanced FGW. Here the objective
  is ``GW(P, Q) + (alpha / 2) · <P + Q, M> + marginal and entropic
  penalties``, so ``alpha`` is a free non-negative weight on the feature
  term and ``alpha=0`` switches it off.

The solver optimizes two couplings ``(pi_samp, pi_feat)`` of equal mass by
block coordinate descent; the first one is used as the cluster-level plan.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import ot
import ot.gromov

from ._runs import (
    build_cluster_problem,
    cross_cost,
    forwarded,
    make_operator,
    make_seeds,
    structure_costs,
    validate_inputs,
)
from .metrics import cost_metrics
from .operator import ImplicitTransportOperator


def run_ensemble_fused_unbalanced_gw(
    X: np.ndarray,
    Y: np.ndarray,
    n_clusters_x: int,
    n_clusters_y: int,
    n_runs: int,
    *,
    reg_marginals: float | tuple[float, float] | None = None,
    epsilon: float | None = None,
    divergence: str | None = None,
    unbalanced_solver: str | None = None,
    alpha: float | None = None,
    clustering_method: str = "kmeans",
    random_state: int | None = None,
    metric: str = "sqeuclidean",
    normalize_cost: bool = True,
    max_iter: int | None = None,
    tol: float | None = None,
    max_iter_ot: int | None = None,
    tol_ot: float | None = None,
    cross_feature_cost_fn: Callable[..., np.ndarray] | None = None,
    cross_feature_cost_kwargs: Mapping[str, Any] | None = None,
    coords_x: np.ndarray | None = None,
    coords_y: np.ndarray | None = None,
) -> list[ImplicitTransportOperator]:
    """Run an ensemble of cluster-level (fused) unbalanced GW trials.

    Parameters
    ----------
    X, Y : (n_x, d_x), (n_y, d_y) arrays. With ``alpha`` unset or 0 the
        feature dimensions may differ (structure only). With ``alpha > 0``
        they must match unless ``cross_feature_cost_fn`` supplies ``M``.
    n_clusters_x, n_clusters_y, n_runs, clustering_method, random_state,
    coords_x, coords_y :
        As in ``run_ensemble_gw``.
    reg_marginals : float or pair, optional
        Marginal relaxation strength, POT semantics (POT default 10). A
        pair relaxes the two marginals differently. Larger values approach
        balanced GW.
    epsilon : float, optional
        Entropic regularization (POT default 0). POT requires
        ``unbalanced_solver`` to be ``"mm"`` or ``"lbfgsb"`` when it is 0.
    divergence : ``"kl"`` or ``"l2"``, optional (POT default ``"kl"``).
    unbalanced_solver : ``"mm"``, ``"lbfgsb"`` or ``"sinkhorn"``, optional
        Inner unbalanced OT solver (POT default ``"mm"``).
    alpha : float >= 0, optional
        Weight of the linear feature term — see the module docstring; this
        is POT's meaning for this solver and differs from balanced FGW.
        ``None`` or 0 runs plain unbalanced GW and no ``M`` is built.
    max_iter, tol, max_iter_ot, tol_ot :
        POT's arguments of the same name (outer BCD and inner OT loops).
    cross_feature_cost_fn, cross_feature_cost_kwargs :
        Custom cluster-level ``M`` for ``alpha > 0``, same contract as in
        ``run_ensemble_fgw``.
    metric : metric of ``ot.dist`` for the structure matrices and ``M``.
    normalize_cost : bool, default True
        Divide each cost matrix by its maximum, as the balanced GW / FGW
        entry points do. **The effect of ``reg_marginals`` depends on the
        cost scale**; set to False for the raw ``ot.dist`` matrices.

    Returns
    -------
    list[ImplicitTransportOperator]
        One per run, built from POT's first coupling ``pi_samp``. The
        total mass is an outcome of the optimization; see
        ``op.transported_mass_x()`` and ``op.meta["metrics"]``.
    """
    use_feature_term = alpha is not None and float(alpha) != 0.0
    if alpha is not None and float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    if not use_feature_term and cross_feature_cost_fn is not None:
        raise ValueError("cross_feature_cost_fn is only used when alpha > 0")
    validate_inputs(
        X, Y, n_runs,
        same_dim=use_feature_term and cross_feature_cost_fn is None,
        who="run_ensemble_fused_unbalanced_gw with alpha > 0 (without cross_feature_cost_fn)",
    )
    kwargs = forwarded(
        reg_marginals=reg_marginals, epsilon=epsilon, divergence=divergence,
        unbalanced_solver=unbalanced_solver, alpha=alpha,
        max_iter=max_iter, tol=tol, max_iter_ot=max_iter_ot, tol_ot=tol_ot,
    )

    operators = []
    for seed in make_seeds(random_state, n_runs):
        prob = build_cluster_problem(
            X, Y, n_clusters_x, n_clusters_y, clustering_method, seed,
            coords_x=coords_x, coords_y=coords_y,
        )
        C1, C2 = structure_costs(prob, metric, normalize_cost)
        M, cost_mode, extra = None, "none", {}
        if use_feature_term:
            M, cost_mode = cross_cost(
                prob, X, Y, metric, normalize_cost,
                cross_feature_cost_fn, cross_feature_cost_kwargs,
            )
        pi_samp, _pi_feat = ot.gromov.fused_unbalanced_gromov_wasserstein(
            C1, C2, wx=prob.a, wy=prob.b, M=M, **kwargs
        )
        T_cluster = np.asarray(pi_samp)
        if M is not None:
            extra = cost_metrics(T_cluster, M)
        operators.append(
            make_operator(
                prob, T_cluster,
                solver_family="fused_unbalanced_gw",
                solver_name="fused_unbalanced_gromov_wasserstein",
                clustering_method=clustering_method,
                solver_params={
                    **kwargs, "metric": metric, "normalize_cost": normalize_cost,
                    "cross_feature_cost_mode": cost_mode,
                },
                extra_metrics=extra,
            )
        )
    return operators
