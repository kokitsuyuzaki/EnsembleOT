"""Stage 25: result container, pooling, free-support barycenter ensemble."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ensembleot import (
    BarycenterResult,
    compute_run_weights,
    pool_barycenters,
    run_ensemble_free_support_barycenter,
    weighted_pool_barycenters,
)

LP = "free_support_barycenter"
SINKHORN = "free_support_sinkhorn_barycenter"


def _toy(K: int = 3, n: int = 60, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)][:K]
    return [rng.standard_normal((n + 5 * k, 2)) + s for k, s in enumerate(shifts)]


def _solver_kwargs(solver_method: str) -> dict:
    return {"reg": 1.0} if solver_method == SINKHORN else {}


@pytest.mark.parametrize("solver_method", [LP, SINKHORN])
@pytest.mark.parametrize("K", [2, 3])
def test_shapes_weights_and_meta(solver_method, K):
    Xs = _toy(K)
    runs = run_ensemble_free_support_barycenter(
        Xs, n_clusters=6, n_support=8, n_runs=2,
        solver_method=solver_method, random_state=0, **_solver_kwargs(solver_method),
    )
    assert len(runs) == 2
    for res in runs:
        assert isinstance(res, BarycenterResult)
        assert res.support.shape == (8, 2)
        assert res.weights.shape == (8,)
        assert res.weights.sum() == pytest.approx(1.0)
        assert np.all(np.isfinite(res.support))
        assert res.run_index is None
        assert res.meta["solver_family"] == "free_support_barycenter"
        assert res.meta["solver_name"] == solver_method
        assert res.meta["n_datasets"] == K
        assert len(res.meta["metrics"]["transport_cost"]) == K
        assert res.meta["metrics"]["barycenter_cost"] > 0
        assert res.meta["metrics"]["marginal_error_row"] < 1e-6
        assert res.meta["metrics"]["marginal_error_col"] < 1e-6
        json.dumps(res.meta)  # JSON-serializable


def test_runs_differ_and_random_state_is_deterministic():
    Xs = _toy()
    kw = dict(n_clusters=6, n_support=8, n_runs=3)
    r1 = run_ensemble_free_support_barycenter(Xs, random_state=42, **kw)
    r2 = run_ensemble_free_support_barycenter(Xs, random_state=42, **kw)
    r3 = run_ensemble_free_support_barycenter(Xs, random_state=43, **kw)
    for a, b in zip(r1, r2):
        np.testing.assert_array_equal(a.support, b.support)
        assert a.meta["seed"] == b.meta["seed"]
        # (k-means inertia may differ in the last bit between calls, so the
        # float metrics are compared approximately)
        assert a.meta["metrics"]["barycenter_cost"] == pytest.approx(
            b.meta["metrics"]["barycenter_cost"]
        )
    assert len({res.meta["seed"] for res in r1}) == 3
    assert not np.allclose(r1[0].support, r1[1].support)
    assert not np.allclose(r1[0].support, r3[0].support)


def test_only_set_arguments_are_forwarded():
    Xs = _toy()
    kw = dict(n_clusters=6, n_support=8, n_runs=1, random_state=0)
    default = run_ensemble_free_support_barycenter(Xs, **kw)[0]
    assert default.meta["solver_params"] == {}  # POT defaults apply
    assert default.meta["metrics"]["n_iter"] > 1

    capped = run_ensemble_free_support_barycenter(Xs, numItermax=1, stopThr=1e-3, **kw)[0]
    assert capped.meta["solver_params"] == {"numItermax": 1, "stopThr": 1e-3}
    assert capped.meta["metrics"]["n_iter"] == 1

    sk = run_ensemble_free_support_barycenter(
        Xs, solver_method=SINKHORN, reg=1.0, numInnerItermax=500, **kw
    )[0]
    assert sk.meta["solver_params"] == {"reg": 1.0, "numInnerItermax": 500}


def test_per_dataset_n_clusters_weights_and_b():
    Xs = _toy()
    b = np.array([0.4, 0.3, 0.2, 0.1])
    res = run_ensemble_free_support_barycenter(
        Xs, n_clusters=[4, 5, 6], n_support=4, n_runs=1,
        weights=[0.5, 0.25, 0.25], b=b, random_state=0,
    )[0]
    np.testing.assert_allclose(res.weights, b)
    assert res.meta["n_clusters"] == [4, 5, 6]
    assert res.meta["barycentric_weights"] == [0.5, 0.25, 0.25]


def test_barycentric_weights_move_the_barycenter():
    Xs = _toy(K=2)
    kw = dict(n_clusters=6, n_support=8, n_runs=2, random_state=0)
    near_first = pool_barycenters(
        run_ensemble_free_support_barycenter(Xs, weights=[0.9, 0.1], **kw)
    )
    near_second = pool_barycenters(
        run_ensemble_free_support_barycenter(Xs, weights=[0.1, 0.9], **kw)
    )
    m0, m1 = Xs[0].mean(axis=0), Xs[1].mean(axis=0)
    # the mean of a W2 barycenter is the weighted mean of the input means
    np.testing.assert_allclose(near_first.mean(), 0.9 * m0 + 0.1 * m1, atol=1e-6)
    np.testing.assert_allclose(near_second.mean(), 0.1 * m0 + 0.9 * m1, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_runs=0), "n_runs"),
        (dict(solver_method="lp"), "unknown solver_method"),
        (dict(reg=0.5), "not an argument of ot.lp.free_support_barycenter"),
        (dict(numInnerItermax=10), "not an argument of ot.lp.free_support_barycenter"),
        (dict(solver_method=SINKHORN), "requires `reg`"),
        (dict(weights=[0.5, 0.5]), "weights must have shape"),
        (dict(weights=[0.5, 0.3, 0.3]), "weights must sum to 1"),
        (dict(weights=[1.5, -0.25, -0.25]), "non-negative"),
        (dict(b=np.full(4, 0.25)), "b must have shape"),
        (dict(b=[1.0, 0, 0, 0, 0, 0, 0, 0]), "strictly positive"),
        (dict(n_clusters=[3, 3]), "one int per dataset"),
        (dict(n_clusters=1000), "n_clusters"),
        (dict(n_support=0), "n_support"),
        (dict(n_support=10_000), "n_support"),
        (dict(clustering_method="nope"), "unsupported clustering method"),
    ],
)
def test_input_validation(kwargs, match):
    base = dict(n_clusters=6, n_support=8, n_runs=1, random_state=0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        run_ensemble_free_support_barycenter(_toy(), **base)


def test_dataset_validation():
    X = _toy()[0]
    kw = dict(n_clusters=3, n_support=4, n_runs=1)
    with pytest.raises(ValueError, match="at least 2 datasets"):
        run_ensemble_free_support_barycenter([X], **kw)
    with pytest.raises(ValueError, match="feature dimension"):
        run_ensemble_free_support_barycenter([X, np.zeros((10, 3))], **kw)
    with pytest.raises(ValueError, match="2-D"):
        run_ensemble_free_support_barycenter([X, np.zeros(10)], **kw)


def test_random_voronoi_clustering_backend():
    res = run_ensemble_free_support_barycenter(
        _toy(), n_clusters=6, n_support=8, n_runs=1,
        clustering_method="random_voronoi", random_state=0,
    )[0]
    assert res.meta["clustering_method"] == "random_voronoi"
    assert "clustering_inertia" not in res.meta["metrics"]
    json.dumps(res.meta)


# ----------------------------- pooling -----------------------------

def test_pool_keeps_every_atom_and_does_not_average_coordinates():
    runs = run_ensemble_free_support_barycenter(
        _toy(), n_clusters=6, n_support=8, n_runs=3, random_state=0
    )
    pooled = pool_barycenters(runs)
    assert pooled.support.shape == (24, 2)
    assert pooled.weights.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(pooled.run_index, np.repeat(np.arange(3), 8))
    for r, res in enumerate(runs):
        np.testing.assert_array_equal(pooled.support[pooled.run_index == r], res.support)
        np.testing.assert_allclose(pooled.weights[pooled.run_index == r], res.weights / 3)
    # mixture: pooled mean is the mean of the per-run means
    np.testing.assert_allclose(
        pooled.mean(), np.mean([res.mean() for res in runs], axis=0), atol=1e-12
    )
    assert pooled.meta["pooled"] is True
    assert pooled.meta["n_runs"] == 3
    assert pooled.meta["solver_name"] == LP
    json.dumps(pooled.meta)


def test_weighted_pool_normalizes_and_validates():
    runs = run_ensemble_free_support_barycenter(
        _toy(), n_clusters=6, n_support=8, n_runs=3, random_state=0
    )
    pooled = weighted_pool_barycenters(runs, [2.0, 1.0, 1.0])
    assert pooled.weights.sum() == pytest.approx(1.0)
    assert pooled.weights[pooled.run_index == 0].sum() == pytest.approx(0.5)
    assert pooled.meta["run_weights"] == pytest.approx([0.5, 0.25, 0.25])

    only_last = weighted_pool_barycenters(runs, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(only_last.mean(), runs[2].mean(), atol=1e-12)

    with pytest.raises(ValueError):
        weighted_pool_barycenters(runs, [1.0, 1.0])
    with pytest.raises(ValueError):
        weighted_pool_barycenters(runs, [1.0, -1.0, 1.0])
    with pytest.raises(ValueError):
        weighted_pool_barycenters(runs, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        pool_barycenters([])


def test_pool_accepts_compute_run_weights():
    runs = run_ensemble_free_support_barycenter(
        _toy(), n_clusters=6, n_support=8, n_runs=4, random_state=0
    )
    w = compute_run_weights(runs, policy="inverse", key="metrics.barycenter_cost")
    assert w.shape == (4,)
    assert w.sum() == pytest.approx(1.0)
    pooled = weighted_pool_barycenters(runs, w)
    assert pooled.meta["run_weights"] == pytest.approx(list(w))
    uniform = compute_run_weights(runs, policy="uniform")
    np.testing.assert_allclose(
        weighted_pool_barycenters(runs, uniform).weights, pool_barycenters(runs).weights
    )


def test_result_container_validation_and_moments():
    res = BarycenterResult(
        support=np.array([[0.0, 0.0], [2.0, 0.0]]), weights=np.array([0.5, 0.5])
    )
    assert (res.n_atoms, res.dim) == (2, 2)
    np.testing.assert_allclose(res.mean(), [1.0, 0.0])
    np.testing.assert_allclose(res.covariance(), [[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        BarycenterResult(support=np.zeros((2, 2)), weights=np.ones(3) / 3)
    with pytest.raises(ValueError):
        BarycenterResult(support=np.zeros(2), weights=np.ones(2) / 2)


# ----------------------------- numerical sanity -----------------------------

def test_three_gaussians_reference_mean():
    """W2 barycenter of N((0,0),I), N((6,0),I), N((3,5),I) has mean (3, 5/3)."""
    rng = np.random.default_rng(0)
    Xs = [rng.standard_normal((3000, 2)) + c for c in [(0, 0), (6, 0), (3, 5)]]
    runs = run_ensemble_free_support_barycenter(
        Xs, n_clusters=40, n_support=60, n_runs=8, random_state=0
    )
    pooled = pool_barycenters(runs)
    assert pooled.support.shape == (480, 2)
    np.testing.assert_allclose(pooled.mean(), [3.0, 5.0 / 3.0], atol=0.08)
    # Cluster means drop the within-cluster spread, so the pooled variance is
    # below the true value of 1; only a loose band is asserted.
    var = np.diag(pooled.covariance())
    assert np.all(var > 0.5) and np.all(var < 1.1)
