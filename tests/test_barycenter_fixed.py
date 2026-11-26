"""Stage 26: fixed-support barycenter ensemble."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ensembleot import (
    BarycenterResult,
    compute_run_weights,
    pool_barycenters,
    run_ensemble_barycenter,
    weighted_pool_barycenters,
)

LP = "lp_barycenter"
BREGMAN = "bregman_barycenter"


def _toy(K: int = 3, n: int = 60, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)][:K]
    return [rng.standard_normal((n + 5 * k, 2)) + s for k, s in enumerate(shifts)]


def _solver_kwargs(solver_method: str) -> dict:
    return {"reg": 1e-2} if solver_method == BREGMAN else {}


@pytest.mark.parametrize("solver_method", [LP, BREGMAN])
@pytest.mark.parametrize("K", [2, 3])
def test_shapes_weights_and_meta(solver_method, K):
    Xs = _toy(K)
    runs = run_ensemble_barycenter(
        Xs, n_clusters=10, n_runs=2,
        solver_method=solver_method, random_state=0, **_solver_kwargs(solver_method),
    )
    assert len(runs) == 2
    for res in runs:
        assert isinstance(res, BarycenterResult)
        assert res.support.shape == (10, 2)
        assert res.weights.shape == (10,)
        assert res.weights.sum() == pytest.approx(1.0)
        assert np.all(res.weights >= 0)
        assert res.meta["solver_family"] == "barycenter"
        assert res.meta["solver_name"] == solver_method
        assert res.meta["n_datasets"] == K
        assert res.meta["normalize_cost"] is True
        assert res.meta["metrics"]["weights_sum_raw"] == pytest.approx(1.0, abs=1e-3)
        assert len(res.meta["metrics"]["transport_cost"]) == K
        json.dumps(res.meta)


def test_support_is_the_pooled_cluster_means():
    from ensembleot.clustering import cluster_means, cluster_samples

    Xs = _toy()
    res = run_ensemble_barycenter(Xs, n_clusters=10, n_runs=1, random_state=3)[0]
    pooled = np.concatenate(Xs)
    labels = cluster_samples(pooled, "kmeans", 10, random_state=res.meta["seed"])
    np.testing.assert_allclose(res.support, cluster_means(pooled, labels, 10), atol=1e-10)


def test_runs_differ_and_random_state_is_deterministic():
    Xs = _toy()
    r1 = run_ensemble_barycenter(Xs, n_clusters=10, n_runs=3, random_state=42)
    r2 = run_ensemble_barycenter(Xs, n_clusters=10, n_runs=3, random_state=42)
    for a, b in zip(r1, r2):
        np.testing.assert_allclose(a.support, b.support, atol=1e-12)
        np.testing.assert_allclose(a.weights, b.weights, atol=1e-9)
        assert a.meta["seed"] == b.meta["seed"]
    assert len({res.meta["seed"] for res in r1}) == 3
    assert not np.allclose(r1[0].support, r1[1].support)


def test_only_set_arguments_are_forwarded():
    Xs = _toy()
    kw = dict(n_clusters=10, n_runs=1, random_state=0)
    lp = run_ensemble_barycenter(Xs, **kw)[0]
    assert lp.meta["solver_params"] == {}  # POT default solver applies
    lp2 = run_ensemble_barycenter(Xs, solver="highs", **kw)[0]
    assert lp2.meta["solver_params"] == {"solver": "highs"}
    assert lp2.meta["metrics"]["lp_objective"] == pytest.approx(
        lp.meta["metrics"]["lp_objective"], rel=1e-6
    )

    sk = run_ensemble_barycenter(Xs, solver_method=BREGMAN, reg=1e-2, **kw)[0]
    assert sk.meta["solver_params"] == {"reg": 1e-2, "histogram_eps": 1e-12}
    assert sk.meta["metrics"]["n_iter"] > 1
    with pytest.warns(Warning):  # one iteration: POT and EnsembleOT both warn
        capped = run_ensemble_barycenter(
            Xs, solver_method=BREGMAN, reg=1e-2, numItermax=1, stopThr=1e-3,
            method="sinkhorn_log", **kw,
        )[0]
    assert capped.weights.sum() == pytest.approx(1.0)
    assert capped.meta["solver_params"] == {
        "reg": 1e-2, "method": "sinkhorn_log", "numItermax": 1, "stopThr": 1e-3,
        "histogram_eps": 1e-12,
    }
    assert capped.meta["metrics"]["n_iter"] <= 1


def test_lp_barycenter_matches_direct_pot_call():
    import ot
    from ensembleot.clustering import cluster_samples, cluster_sizes

    Xs = _toy()
    res = run_ensemble_barycenter(Xs, n_clusters=10, n_runs=1, random_state=1)[0]
    pooled = np.concatenate(Xs)
    labels = cluster_samples(pooled, "kmeans", 10, random_state=res.meta["seed"])
    bounds = np.cumsum([0] + [len(X) for X in Xs])
    A = np.stack(
        [cluster_sizes(labels[bounds[k]:bounds[k + 1]], 10) / len(Xs[k]) for k in range(3)],
        axis=1,
    )
    np.testing.assert_allclose(A.sum(axis=0), 1.0)
    assert (A == 0).any()  # datasets occupy only part of the shared clusters
    M = ot.dist(res.support, res.support)
    M /= M.max()
    _, log = ot.lp.barycenter(A, M, log=True)
    assert res.meta["metrics"]["lp_objective"] == pytest.approx(log["fun"], rel=1e-6)
    # the LP optimum equals the weighted cost of the barycenter -> dataset plans
    assert res.meta["metrics"]["barycenter_cost"] == pytest.approx(log["fun"], rel=1e-6)


def test_normalize_cost_does_not_change_the_lp_solution():
    Xs = _toy()
    kw = dict(n_clusters=10, n_runs=1, random_state=0)
    scaled = run_ensemble_barycenter(Xs, normalize_cost=True, **kw)[0]
    raw = run_ensemble_barycenter(Xs, normalize_cost=False, **kw)[0]
    assert raw.meta["cost_scale"] == 1.0
    assert scaled.meta["cost_scale"] > 1.0
    assert raw.meta["metrics"]["lp_objective"] == pytest.approx(
        scaled.meta["metrics"]["lp_objective"] * scaled.meta["cost_scale"], rel=1e-6
    )


def test_barycentric_weights_move_the_barycenter():
    Xs = _toy(K=2)
    kw = dict(n_clusters=12, n_runs=2, random_state=0)
    near_first = pool_barycenters(run_ensemble_barycenter(Xs, weights=[0.9, 0.1], **kw))
    near_second = pool_barycenters(run_ensemble_barycenter(Xs, weights=[0.1, 0.9], **kw))
    m0, m1 = Xs[0].mean(axis=0), Xs[1].mean(axis=0)
    assert np.linalg.norm(near_first.mean() - m0) < np.linalg.norm(near_first.mean() - m1)
    assert np.linalg.norm(near_second.mean() - m1) < np.linalg.norm(near_second.mean() - m0)


# ----------------------------- zeros in A -----------------------------

def test_entropic_histogram_smoothing_is_negligible_when_zeros_are_harmless():
    Xs = _toy()
    kw = dict(n_clusters=10, n_runs=1, random_state=0, solver_method=BREGMAN, reg=1e-2)
    default = run_ensemble_barycenter(Xs, **kw)[0]
    exact = run_ensemble_barycenter(Xs, histogram_eps=0.0, **kw)[0]
    assert exact.meta["solver_params"]["histogram_eps"] == 0.0
    np.testing.assert_allclose(default.weights, exact.weights, atol=1e-6)


def test_entropic_histogram_smoothing_rescues_kernel_underflow():
    """With a tiny ``reg`` the Gibbs kernel underflows; method='sinkhorn' then
    yields NaN on histograms with exact zeros but stays finite once smoothed."""
    rng = np.random.default_rng(0)
    Xs = [rng.standard_normal((300, 2)) + c for c in [(0, 0), (6, 0), (3, 5)]]
    kw = dict(
        n_clusters=30, n_runs=1, random_state=0, solver_method=BREGMAN,
        reg=5e-4, numItermax=300,
    )
    with pytest.warns(Warning):
        with pytest.raises(RuntimeError, match="non-finite"):
            run_ensemble_barycenter(Xs, histogram_eps=0.0, **kw)
    with pytest.warns(Warning):  # 300 iterations are not enough to converge
        res = run_ensemble_barycenter(Xs, **kw)[0]
    assert np.all(np.isfinite(res.weights))
    assert res.weights.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(res.mean(), [3.0, 5.0 / 3.0], atol=0.5)


def test_smoothed_histograms_stay_normalized():
    from ensembleot.barycenter import _smooth_histograms

    A = np.array([[0.5, 0.0], [0.5, 0.0], [0.0, 1.0]])
    S = _smooth_histograms(A, 1e-3)
    np.testing.assert_allclose(S.sum(axis=0), 1.0, atol=1e-15)
    assert np.all(S > 0)
    assert _smooth_histograms(A, 0.0) is A


# ----------------------------- validation -----------------------------

@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(n_runs=0), "n_runs"),
        (dict(solver_method="sinkhorn"), "unknown solver_method"),
        (dict(reg=0.1), "not an argument of ot.lp.barycenter"),
        (dict(method="sinkhorn_log"), "not an argument of ot.lp.barycenter"),
        (dict(numItermax=10), "not an argument of ot.lp.barycenter"),
        (dict(stopThr=1e-3), "not an argument of ot.lp.barycenter"),
        (dict(histogram_eps=1e-9), "not an argument of ot.lp.barycenter"),
        (dict(solver_method=BREGMAN), "requires `reg`"),
        (dict(solver_method=BREGMAN, reg=0.1, solver="highs"),
         "not an argument of ot.bregman.barycenter"),
        (dict(solver_method=BREGMAN, reg=0.1, histogram_eps=1.0), "histogram_eps"),
        (dict(weights=[0.5, 0.5]), "weights must have shape"),
        (dict(weights=[0.5, 0.3, 0.3]), "weights must sum to 1"),
        (dict(n_clusters=0), "n_clusters"),
        (dict(n_clusters=100_000), "n_clusters"),
        (dict(clustering_method="nope"), "unsupported clustering method"),
    ],
)
def test_input_validation(kwargs, match):
    base = dict(n_clusters=10, n_runs=1, random_state=0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        run_ensemble_barycenter(_toy(), **base)


def test_dataset_validation():
    X = _toy()[0]
    with pytest.raises(ValueError, match="at least 2 datasets"):
        run_ensemble_barycenter([X], n_clusters=3, n_runs=1)
    with pytest.raises(ValueError, match="feature dimension"):
        run_ensemble_barycenter([X, np.zeros((10, 3))], n_clusters=3, n_runs=1)


# ----------------------------- pooling / sanity -----------------------------

def test_pooling_and_run_weights():
    runs = run_ensemble_barycenter(_toy(), n_clusters=10, n_runs=3, random_state=0)
    pooled = pool_barycenters(runs)
    assert pooled.support.shape == (30, 2)
    assert pooled.weights.sum() == pytest.approx(1.0)
    assert pooled.meta["solver_family"] == "barycenter"
    w = compute_run_weights(runs, policy="softmax_negative", key="metrics.barycenter_cost")
    weighted = weighted_pool_barycenters(runs, w)
    assert weighted.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "solver_method, kwargs",
    [(LP, {}), (BREGMAN, {"reg": 5e-3})],
)
def test_three_gaussians_reference_mean(solver_method, kwargs):
    """W2 barycenter of N((0,0),I), N((6,0),I), N((3,5),I) has mean (3, 5/3)."""
    rng = np.random.default_rng(0)
    Xs = [rng.standard_normal((3000, 2)) + c for c in [(0, 0), (6, 0), (3, 5)]]
    runs = run_ensemble_barycenter(
        Xs, n_clusters=60, n_runs=8, solver_method=solver_method, random_state=0, **kwargs
    )
    pooled = pool_barycenters(runs)
    np.testing.assert_allclose(pooled.mean(), [3.0, 5.0 / 3.0], atol=0.08)
    # Pooled-data centers are sparse between the datasets, which inflates the
    # variance above the true value of 1; only a loose band is asserted.
    var = np.diag(pooled.covariance())
    assert np.all(var > 0.9) and np.all(var < 2.5)
