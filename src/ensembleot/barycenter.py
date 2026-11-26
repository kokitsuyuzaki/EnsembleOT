"""Ensemble Wasserstein barycenters of K >= 2 datasets.

Every run follows the usual EnsembleOT recipe — randomized clustering, a
small cluster-level problem solved by POT — and yields a discrete measure
(``support``, ``weights``) wrapped in a :class:`BarycenterResult`.

Two separate entry points, one per solver family:

* :func:`run_ensemble_free_support_barycenter` — the support points are
  optimized (``ot.lp.free_support_barycenter`` /
  ``ot.bregman.free_support_sinkhorn_barycenter``).
* :func:`run_ensemble_barycenter` — the support is fixed to cluster centers
  shared by all datasets and only the weights are optimized
  (``ot.lp.barycenter`` / ``ot.bregman.barycenter``).

POT conventions. ``solver_method`` values name the POT function that is
called. Arguments forwarded to POT keep POT's names and meanings and default
to ``None``; ``None`` means "do not forward", so POT's own default applies.
Passing an argument the chosen solver does not accept raises ``ValueError``.

Aggregation. Support points of different runs have no correspondence, so
coordinates are never averaged. :func:`pool_barycenters` /
:func:`weighted_pool_barycenters` instead pool the atoms of all runs into
one measure (the run-weighted mixture of the per-run barycenters).

Only cluster-level arrays are ever allocated; no sample × sample matrix is
built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import ot

from .aggregate import _normalize_weights
from .clustering import cluster_means, cluster_samples_with_info, cluster_sizes

FreeSupportSolverMethod = Literal[
    "free_support_barycenter", "free_support_sinkhorn_barycenter"
]

# solver_method -> (POT function, forwarded optional arguments it accepts)
_FREE_SUPPORT_SOLVERS: dict[str, tuple[str, frozenset[str]]] = {
    "free_support_barycenter": (
        "ot.lp.free_support_barycenter",
        frozenset({"numItermax", "stopThr"}),
    ),
    "free_support_sinkhorn_barycenter": (
        "ot.bregman.free_support_sinkhorn_barycenter",
        frozenset({"reg", "numItermax", "numInnerItermax", "stopThr"}),
    ),
}


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------

@dataclass
class BarycenterResult:
    """A discrete barycenter: ``weights[i]`` mass at ``support[i]``.

    ``meta`` is JSON-serializable (same convention as
    ``ImplicitTransportOperator.meta``); per-run results carry
    ``solver_family`` / ``solver_name`` / ``seed`` / ``solver_params`` /
    ``metrics``. ``run_index`` is set only on pooled results and gives, for
    each atom, the index of the run it came from.
    """

    support: np.ndarray                      # (n_atoms, d)
    weights: np.ndarray                      # (n_atoms,), sums to 1
    meta: dict[str, Any] = field(default_factory=dict)
    run_index: np.ndarray | None = None      # (n_atoms,) int, pooled results only

    def __post_init__(self) -> None:
        if self.support.ndim != 2:
            raise ValueError("support must be 2-D (n_atoms, d)")
        if self.weights.shape != (self.support.shape[0],):
            raise ValueError("weights shape mismatch")
        if self.run_index is not None and self.run_index.shape != self.weights.shape:
            raise ValueError("run_index shape mismatch")

    @property
    def n_atoms(self) -> int:
        return int(self.support.shape[0])

    @property
    def dim(self) -> int:
        return int(self.support.shape[1])

    def mean(self) -> np.ndarray:
        """Weighted mean of the atoms, shape (d,)."""
        return self.weights @ self.support

    def covariance(self) -> np.ndarray:
        """Weighted covariance of the atoms, shape (d, d)."""
        centered = self.support - self.mean()
        return (centered * self.weights[:, None]).T @ centered


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _validate_datasets(Xs: Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(Xs, np.ndarray) and Xs.ndim != 3:
        raise ValueError("Xs must be a sequence of 2-D arrays, one per dataset")
    Xs = [np.asarray(X) for X in Xs]
    if len(Xs) < 2:
        raise ValueError("a barycenter needs at least 2 datasets")
    for X in Xs:
        if X.ndim != 2:
            raise ValueError("every dataset must be 2-D (n_samples, d)")
        if X.shape[0] == 0:
            raise ValueError("every dataset must be non-empty")
        if X.shape[1] != Xs[0].shape[1]:
            raise ValueError("all datasets must share the feature dimension")
    return Xs


def _validate_simplex(v: np.ndarray | Sequence[float], size: int, name: str) -> np.ndarray:
    """POT expects ``weights`` / ``b`` on the simplex and does not renormalize."""
    out = np.asarray(v, dtype=np.float64)
    if out.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {out.shape}")
    if not np.all(np.isfinite(out)) or np.any(out < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    if abs(float(out.sum()) - 1.0) > 1e-8:
        raise ValueError(f"{name} must sum to 1")
    return out


def _forwarded_kwargs(
    solver_method: str,
    solvers: dict[str, tuple[str, frozenset[str]]],
    **given: Any,
) -> dict[str, Any]:
    """Keep the arguments the caller set; reject those the solver lacks."""
    if solver_method not in solvers:
        raise ValueError(
            f"unknown solver_method {solver_method!r}; supported: {sorted(solvers)}"
        )
    pot_name, accepted = solvers[solver_method]
    out: dict[str, Any] = {}
    for name, value in given.items():
        if value is None:
            continue
        if name not in accepted:
            raise ValueError(
                f"{name!r} is not an argument of {pot_name} "
                f"(solver_method={solver_method!r})"
            )
        out[name] = value
    if "reg" in accepted and "reg" not in out:
        raise ValueError(f"{pot_name} requires `reg` (POT defines no default)")
    return out


def _json_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}


def _run_seeds(random_state: int | None, n_runs: int) -> list[int]:
    rng = np.random.default_rng(random_state)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_runs)]


def _couple(
    a: np.ndarray,
    b: np.ndarray,
    M: np.ndarray,
    entropic: bool,
    **kwargs: Any,
) -> np.ndarray:
    """Cluster-level plan between two small measures, shape (len(a), len(b)).

    Solved with ``ot.emd`` (or ``ot.sinkhorn`` when ``entropic``) on the
    strictly positive entries of ``a`` and ``b`` only; rows / columns of
    zero-mass atoms are exactly zero in the returned plan.
    """
    pa, pb = a > 0, b > 0
    a_pos = a[pa] / a[pa].sum()
    b_pos = b[pb] / b[pb].sum()
    M_pos = np.ascontiguousarray(M[np.ix_(pa, pb)])
    if entropic:
        P = ot.sinkhorn(a_pos, b_pos, M_pos, **kwargs)
    else:
        P = ot.emd(a_pos, b_pos, M_pos)
    out = np.zeros(M.shape, dtype=np.float64)
    out[np.ix_(pa, pb)] = np.asarray(P, dtype=np.float64)
    return out


def _coupling_metrics(
    plans: list[np.ndarray],
    costs: list[np.ndarray],
    bary_weights: np.ndarray,
    masses: list[np.ndarray],
    weights: np.ndarray,
) -> dict[str, Any]:
    """Transport cost of each barycenter -> dataset plan and marginal errors."""
    per_dataset = [float(np.sum(P * M)) for P, M in zip(plans, costs)]
    return {
        "transport_cost": per_dataset,
        "barycenter_cost": float(np.dot(weights, per_dataset)),
        "marginal_error_row": float(
            max(np.max(np.abs(P.sum(axis=1) - bary_weights)) for P in plans)
        ),
        "marginal_error_col": float(
            max(np.max(np.abs(P.sum(axis=0) - m)) for P, m in zip(plans, masses))
        ),
    }


# ---------------------------------------------------------------------------
# pooling across runs
# ---------------------------------------------------------------------------

def weighted_pool_barycenters(
    results: Sequence[BarycenterResult],
    run_weights: np.ndarray | Sequence[float],
) -> BarycenterResult:
    """Pool the atoms of several runs into one measure.

    The pooled measure is the mixture ``sum_r w_r * barycenter_r``: all atoms
    are kept (``support`` is the row-wise concatenation, in run order) and
    the atom weights of run ``r`` are scaled by ``w_r``. Coordinates are
    *not* averaged, because atoms of different runs are not in
    correspondence. ``run_weights`` must be non-negative with a positive
    total and is normalized to sum to 1, like the weights of
    ``make_weighted_mean_operator``; the output of
    :func:`ensembleot.compute_run_weights` (called on ``results``) fits.
    """
    results = list(results)
    if len(results) == 0:
        raise ValueError("results must be non-empty")
    w = _normalize_weights(run_weights, len(results))
    dim = results[0].dim
    if any(r.dim != dim for r in results):
        raise ValueError("all barycenters must share the feature dimension")

    first = results[0].meta
    meta: dict[str, Any] = {
        "pooled": True,
        "n_runs": len(results),
        "run_weights": [float(v) for v in w],
        "n_atoms_per_run": [r.n_atoms for r in results],
    }
    for key in ("solver_family", "solver_name"):
        if key in first and all(r.meta.get(key) == first[key] for r in results):
            meta[key] = first[key]

    return BarycenterResult(
        support=np.concatenate([r.support for r in results], axis=0),
        weights=np.concatenate([wr * r.weights for wr, r in zip(w, results)]),
        meta=meta,
        run_index=np.repeat(
            np.arange(len(results), dtype=np.int64), [r.n_atoms for r in results]
        ),
    )


def pool_barycenters(results: Sequence[BarycenterResult]) -> BarycenterResult:
    """Uniform pooling: :func:`weighted_pool_barycenters` with weights 1/R."""
    results = list(results)
    if len(results) == 0:
        raise ValueError("results must be non-empty")
    return weighted_pool_barycenters(results, np.full(len(results), 1.0 / len(results)))


# ---------------------------------------------------------------------------
# free-support barycenter
# ---------------------------------------------------------------------------

def _draw_from_pooled(Xs: list[np.ndarray], n_draw: int, seed: int) -> np.ndarray:
    """Draw ``n_draw`` distinct rows from the pooled datasets (no stacking)."""
    sizes = np.array([X.shape[0] for X in Xs])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    rng = np.random.default_rng(seed)
    idx = rng.choice(int(offsets[-1]), size=n_draw, replace=False)
    which = np.searchsorted(offsets, idx, side="right") - 1
    return np.stack(
        [Xs[k][i - offsets[k]] for k, i in zip(which, idx)]
    ).astype(np.float64)


def _single_run_free_support(
    Xs: list[np.ndarray],
    n_clusters: list[int],
    n_support: int,
    weights: np.ndarray,
    b: np.ndarray,
    clustering_method: str,
    solver_method: str,
    pot_kwargs: dict[str, Any],
    seed: int,
) -> BarycenterResult:
    K = len(Xs)
    centers, masses, inertias = [], [], []
    for k, X in enumerate(Xs):
        labels, info = cluster_samples_with_info(
            X, clustering_method, n_clusters[k], random_state=seed + k
        )
        sizes = cluster_sizes(labels, n_clusters[k]).astype(float)
        centers.append(cluster_means(X, labels, n_clusters[k]).astype(np.float64))
        masses.append(sizes / X.shape[0])
        if "inertia" in info:
            inertias.append(float(info["inertia"]))

    # POT gets the non-empty clusters only (an empty cluster has no mean).
    keep = [m > 0 for m in masses]
    measures_locations = [c[kp] for c, kp in zip(centers, keep)]
    measures_weights = [m[kp] for m, kp in zip(masses, keep)]
    X_init = _draw_from_pooled(Xs, n_support, seed + K)

    entropic = solver_method == "free_support_sinkhorn_barycenter"
    if entropic:
        kwargs = dict(pot_kwargs)
        support, log = ot.bregman.free_support_sinkhorn_barycenter(
            measures_locations, measures_weights, X_init, kwargs.pop("reg"),
            b=b, weights=weights, log=True, **kwargs,
        )
    else:
        support, log = ot.lp.free_support_barycenter(
            measures_locations, measures_weights, X_init,
            b=b, weights=weights, log=True, **pot_kwargs,
        )
    support = np.asarray(support, dtype=np.float64)
    if not np.all(np.isfinite(support)):
        raise RuntimeError(
            f"{_FREE_SUPPORT_SOLVERS[solver_method][0]} returned non-finite support "
            "points; for the Sinkhorn solver try a larger `reg` (it applies to the "
            "raw squared Euclidean cost)"
        )

    # Barycenter -> dataset plans, with the same inner solver POT iterates on.
    couple_kwargs: dict[str, Any] = {}
    if entropic:
        couple_kwargs["reg"] = pot_kwargs["reg"]
        if "numInnerItermax" in pot_kwargs:
            couple_kwargs["numItermax"] = pot_kwargs["numInnerItermax"]
    costs = [ot.dist(support, c, metric="sqeuclidean") for c in centers]
    plans = [_couple(b, m, M, entropic, **couple_kwargs) for m, M in zip(masses, costs)]

    displacement = log["displacement_square_norms"]
    metrics = _coupling_metrics(plans, costs, b, masses, weights)
    metrics["n_iter"] = int(len(displacement))
    if len(displacement):
        metrics["displacement_square_norm"] = float(displacement[-1])
    if inertias:
        metrics["clustering_inertia"] = inertias

    meta = {
        "solver_family": "free_support_barycenter",
        "solver_name": solver_method,
        "clustering_method": clustering_method,
        "seed": int(seed),
        "n_datasets": K,
        "n_clusters": [int(n) for n in n_clusters],
        "barycentric_weights": [float(v) for v in weights],
        "solver_params": _json_params(pot_kwargs),
        "metrics": metrics,
    }
    return BarycenterResult(support=support, weights=b.copy(), meta=meta)


def run_ensemble_free_support_barycenter(
    Xs: Sequence[np.ndarray],
    n_clusters: int | Sequence[int],
    n_support: int,
    n_runs: int,
    weights: np.ndarray | Sequence[float] | None = None,
    b: np.ndarray | Sequence[float] | None = None,
    clustering_method: str = "kmeans",
    solver_method: FreeSupportSolverMethod = "free_support_barycenter",
    reg: float | None = None,
    numItermax: int | None = None,
    numInnerItermax: int | None = None,
    stopThr: float | None = None,
    random_state: int | None = None,
) -> list[BarycenterResult]:
    """Run an ensemble of cluster-level free-support Wasserstein barycenters.

    Per run: every dataset is clustered independently (seed shifted per
    dataset and per run), cluster means / relative cluster sizes form the
    input measures, ``X_init`` is drawn from the pooled samples with a
    per-run seed, and POT optimizes the ``n_support`` barycenter locations.

    Parameters
    ----------
    Xs : sequence of K >= 2 arrays ``(n_k, d)`` sharing ``d``.
    n_clusters : clusters per dataset; an int, or one int per dataset.
    n_support : number of barycenter atoms (rows of POT's ``X_init``).
    n_runs : number of randomized runs.
    weights : ``(K,)`` barycentric weights of the datasets (POT ``weights``);
        uniform when ``None``.
    b : ``(n_support,)`` weights of the barycenter atoms (POT ``b``);
        uniform when ``None``. Must be strictly positive.
    solver_method : ``"free_support_barycenter"`` calls
        ``ot.lp.free_support_barycenter``; ``"free_support_sinkhorn_barycenter"``
        calls ``ot.bregman.free_support_sinkhorn_barycenter``.
    reg, numItermax, numInnerItermax, stopThr : forwarded to the POT function
        under the same name when not ``None`` (otherwise POT's default
        applies). ``reg`` and ``numInnerItermax`` exist only for the Sinkhorn
        solver, where ``reg`` is required and applies to the raw squared
        Euclidean cost between support points and cluster means, exactly as
        in POT. An argument the chosen solver lacks raises ``ValueError``.

    Returns
    -------
    list[BarycenterResult]
        One barycenter per run (``support`` of shape ``(n_support, d)``,
        ``weights == b``). Combine them with :func:`pool_barycenters`.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    Xs = _validate_datasets(Xs)
    K = len(Xs)
    pot_kwargs = _forwarded_kwargs(
        solver_method, _FREE_SUPPORT_SOLVERS,
        reg=reg, numItermax=numItermax, numInnerItermax=numInnerItermax, stopThr=stopThr,
    )

    if np.ndim(n_clusters) == 0:
        n_clusters_list = [int(n_clusters)] * K
    else:
        n_clusters_list = [int(n) for n in n_clusters]
        if len(n_clusters_list) != K:
            raise ValueError("n_clusters must be an int or one int per dataset")
    for n, X in zip(n_clusters_list, Xs):
        if not 1 <= n <= X.shape[0]:
            raise ValueError("n_clusters must be in [1, n_samples] for every dataset")
    if not 1 <= n_support <= sum(X.shape[0] for X in Xs):
        raise ValueError("n_support must be in [1, total number of samples]")

    weights_arr = (
        np.full(K, 1.0 / K) if weights is None else _validate_simplex(weights, K, "weights")
    )
    if b is None:
        b_arr = np.full(n_support, 1.0 / n_support)
    else:
        b_arr = _validate_simplex(b, n_support, "b")
        if np.any(b_arr <= 0):
            raise ValueError("b must be strictly positive")

    return [
        _single_run_free_support(
            Xs,
            n_clusters=n_clusters_list,
            n_support=n_support,
            weights=weights_arr,
            b=b_arr,
            clustering_method=clustering_method,
            solver_method=solver_method,
            pot_kwargs=pot_kwargs,
            seed=seed,
        )
        for seed in _run_seeds(random_state, n_runs)
    ]
