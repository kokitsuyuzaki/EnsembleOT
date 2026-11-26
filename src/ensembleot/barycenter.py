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

import warnings
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
    zero-mass atoms are exactly zero in the returned plan. POT's convergence
    warning is silenced for this auxiliary solve; callers record the
    marginal errors of the plan in ``meta["metrics"]`` instead.
    """
    pa, pb = a > 0, b > 0
    a_pos = a[pa] / a[pa].sum()
    b_pos = b[pb] / b[pb].sum()
    M_pos = np.ascontiguousarray(M[np.ix_(pa, pb)])
    if entropic:
        # log-domain iterations: same optimum as method="sinkhorn", but they
        # do not break down when exp(-M / reg) underflows.
        P = ot.sinkhorn(
            a_pos, b_pos, M_pos, method="sinkhorn_log", warn=False, **kwargs
        )
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

    # Barycenter -> dataset plans for the same inner problem POT iterates on.
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


# ---------------------------------------------------------------------------
# fixed-support barycenter
# ---------------------------------------------------------------------------

FixedSupportSolverMethod = Literal["lp_barycenter", "bregman_barycenter"]

# ``histogram_eps`` is EnsembleOT's own (not forwarded); it only exists for
# the entropic solver, so it is validated through the same table.
_FIXED_SUPPORT_SOLVERS: dict[str, tuple[str, frozenset[str]]] = {
    "lp_barycenter": ("ot.lp.barycenter", frozenset({"solver"})),
    "bregman_barycenter": (
        "ot.bregman.barycenter",
        frozenset({"reg", "method", "numItermax", "stopThr", "histogram_eps"}),
    ),
}

_DEFAULT_HISTOGRAM_EPS = 1e-12


def _smooth_histograms(A: np.ndarray, eps: float) -> np.ndarray:
    """Mix every column with the uniform histogram: ``(1 - eps) A + eps / n``.

    Columns stay exactly normalized and become strictly positive.
    """
    if eps == 0.0:
        return A
    return (1.0 - eps) * A + eps / A.shape[0]


def _single_run_fixed_support(
    Xs: list[np.ndarray],
    pooled: np.ndarray,
    n_clusters: int,
    weights: np.ndarray,
    clustering_method: str,
    solver_method: str,
    pot_kwargs: dict[str, Any],
    histogram_eps: float,
    normalize_cost: bool,
    seed: int,
) -> BarycenterResult:
    K = len(Xs)
    labels, info = cluster_samples_with_info(
        pooled, clustering_method, n_clusters, random_state=seed
    )
    centers = cluster_means(pooled, labels, n_clusters).astype(np.float64)

    # A[:, k]: fraction of dataset k's samples in each shared cluster.
    bounds = np.concatenate([[0], np.cumsum([X.shape[0] for X in Xs])])
    A = np.stack(
        [
            cluster_sizes(labels[bounds[k]:bounds[k + 1]], n_clusters) / Xs[k].shape[0]
            for k in range(K)
        ],
        axis=1,
    )

    M = ot.dist(centers, centers, metric="sqeuclidean")
    cost_scale = float(M.max()) if normalize_cost and M.max() > 0 else 1.0
    M = M / cost_scale

    pot_name = _FIXED_SUPPORT_SOLVERS[solver_method][0]
    entropic = solver_method == "bregman_barycenter"
    metrics: dict[str, Any] = {}
    if entropic:
        kwargs = dict(pot_kwargs)
        bary, log = ot.bregman.barycenter(
            _smooth_histograms(A, histogram_eps), M, kwargs.pop("reg"),
            weights=weights, log=True, **kwargs,
        )
        metrics["n_iter"] = int(log["niter"])
        if len(log["err"]):
            metrics["solver_err"] = float(log["err"][-1])
    else:
        bary, log = ot.lp.barycenter(A, M, weights=weights, log=True, **pot_kwargs)
        metrics["lp_objective"] = float(log["fun"])
        metrics["lp_status"] = int(log["status"])

    bary = np.asarray(bary, dtype=np.float64)
    if not np.all(np.isfinite(bary)):
        raise RuntimeError(
            f"{pot_name} returned non-finite barycenter weights; for the entropic "
            "solver try a larger `reg` or method='sinkhorn_log'"
        )
    bary = np.maximum(bary, 0.0)  # LP solvers may return -0.0 / tiny negatives
    total = float(bary.sum())
    metrics["weights_sum_raw"] = total
    if total <= 0:
        raise RuntimeError(f"{pot_name} returned an all-zero barycenter")
    if abs(total - 1.0) > 1e-3:
        # happens when the entropic iterations stop early or break down
        warnings.warn(
            f"{pot_name} returned weights summing to {total:.6g} instead of 1; "
            "renormalizing, but the solver has probably not converged "
            "(see meta['metrics']['weights_sum_raw'])",
            RuntimeWarning,
            stacklevel=3,
        )
    bary = bary / total

    couple_kwargs: dict[str, Any] = {}
    if entropic:
        couple_kwargs["reg"] = pot_kwargs["reg"]
    masses = [A[:, k] for k in range(K)]
    plans = [_couple(bary, m, M, entropic, **couple_kwargs) for m in masses]
    metrics.update(_coupling_metrics(plans, [M] * K, bary, masses, weights))
    if "inertia" in info:
        metrics["clustering_inertia"] = float(info["inertia"])

    solver_params = _json_params(pot_kwargs)
    if entropic:
        solver_params["histogram_eps"] = float(histogram_eps)
    meta = {
        "solver_family": "barycenter",
        "solver_name": solver_method,
        "clustering_method": clustering_method,
        "seed": int(seed),
        "n_datasets": K,
        "n_clusters": int(n_clusters),
        "barycentric_weights": [float(v) for v in weights],
        "normalize_cost": bool(normalize_cost),
        "cost_scale": cost_scale,
        "solver_params": solver_params,
        "metrics": metrics,
    }
    return BarycenterResult(support=centers, weights=bary, meta=meta)


def run_ensemble_barycenter(
    Xs: Sequence[np.ndarray],
    n_clusters: int,
    n_runs: int,
    weights: np.ndarray | Sequence[float] | None = None,
    clustering_method: str = "kmeans",
    solver_method: FixedSupportSolverMethod = "lp_barycenter",
    reg: float | None = None,
    method: str | None = None,
    numItermax: int | None = None,
    stopThr: float | None = None,
    solver: str | None = None,
    histogram_eps: float | None = None,
    normalize_cost: bool = True,
    random_state: int | None = None,
) -> list[BarycenterResult]:
    """Run an ensemble of fixed-support Wasserstein barycenters.

    Per run: the pooled samples of all datasets are clustered once (per-run
    seed) into ``n_clusters`` shared clusters whose means are the fixed
    support. Dataset ``k`` becomes the histogram ``A[:, k]`` (its fraction of
    samples in each shared cluster), the cost is the squared Euclidean
    distance between the shared centers, and POT optimizes the barycenter
    weights on those centers. The pooled clustering sees every sample once,
    so larger datasets have more influence on where the centers are placed.

    Parameters
    ----------
    Xs : sequence of K >= 2 arrays ``(n_k, d)`` sharing ``d``.
    n_clusters : number of shared clusters (= barycenter atoms).
    n_runs : number of randomized runs.
    weights : ``(K,)`` barycentric weights of the datasets (POT ``weights``);
        uniform when ``None``.
    solver_method : ``"lp_barycenter"`` calls ``ot.lp.barycenter`` (exact LP);
        ``"bregman_barycenter"`` calls ``ot.bregman.barycenter`` (entropic).
    reg, method, numItermax, stopThr : forwarded to ``ot.bregman.barycenter``
        under the same name when not ``None`` (otherwise POT's default
        applies); ``reg`` is required there. ``reg`` is relative to the cost
        matrix actually passed, i.e. to the max-normalized one by default.
    solver : forwarded to ``ot.lp.barycenter`` when not ``None``.
        An argument the chosen solver lacks raises ``ValueError``.
    histogram_eps : entropic solver only. Datasets usually occupy only some
        of the shared clusters, so ``A`` contains exact zeros. Each column is
        replaced by ``(1 - eps) * A[:, k] + eps / n_clusters`` (a mixture
        with the uniform histogram, so columns still sum to exactly 1)
        before calling ``ot.bregman.barycenter``. ``None`` means ``1e-12``,
        which changes converged results only marginally but keeps
        ``method="sinkhorn"`` finite when the Gibbs kernel underflows at
        small ``reg``; ``0`` passes ``A`` unchanged. The LP solver handles
        zeros exactly and always receives the unmodified ``A``.
    normalize_cost : divide the cost matrix by its maximum (default, like the
        other EnsembleOT entry points). The LP solution is unaffected; for
        the entropic solver it fixes the scale ``reg`` refers to.

    Returns
    -------
    list[BarycenterResult]
        One barycenter per run: ``support`` holds the ``(n_clusters, d)``
        shared centers and ``weights`` the barycenter mass on them (clipped
        at 0 and renormalized to sum to 1). Combine runs with
        :func:`pool_barycenters`.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    Xs = _validate_datasets(Xs)
    K = len(Xs)
    pot_kwargs = _forwarded_kwargs(
        solver_method, _FIXED_SUPPORT_SOLVERS,
        reg=reg, method=method, numItermax=numItermax, stopThr=stopThr,
        solver=solver, histogram_eps=histogram_eps,
    )
    eps = float(pot_kwargs.pop("histogram_eps", _DEFAULT_HISTOGRAM_EPS))
    if not 0.0 <= eps < 1.0:
        raise ValueError("histogram_eps must be in [0, 1)")

    pooled = np.concatenate(Xs, axis=0)
    if not 1 <= n_clusters <= pooled.shape[0]:
        raise ValueError("n_clusters must be in [1, total number of samples]")
    weights_arr = (
        np.full(K, 1.0 / K) if weights is None else _validate_simplex(weights, K, "weights")
    )

    return [
        _single_run_fixed_support(
            Xs,
            pooled,
            n_clusters=int(n_clusters),
            weights=weights_arr,
            clustering_method=clustering_method,
            solver_method=solver_method,
            pot_kwargs=pot_kwargs,
            histogram_eps=eps,
            normalize_cost=normalize_cost,
            seed=seed,
        )
        for seed in _run_seeds(random_state, n_runs)
    ]
