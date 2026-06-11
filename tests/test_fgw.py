"""Stage 12a: Fused Gromov-Wasserstein ensemble OT."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot import (
    run_ensemble_fgw,
    make_mean_operator,
    make_weighted_mean_operator,
    consensus_edges,
)
from ensembleot.operator import ImplicitTransportOperator


def _toy_xy(d: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((10, d))
    Y = rng.standard_normal((9, d)) + 0.3
    return X, Y


@pytest.mark.parametrize("solver_method", ["fgw", "entropic_fgw"])
def test_single_run_returns_operator_fgw(solver_method):
    X, Y = _toy_xy()
    runs = run_ensemble_fgw(
        X, Y,
        n_clusters_x=3,
        n_clusters_y=2,
        n_runs=1,
        solver_method=solver_method,
        alpha=0.5,
        random_state=0,
        epsilon=0.05,
    )
    assert isinstance(runs, list) and len(runs) == 1
    op = runs[0]
    assert isinstance(op, ImplicitTransportOperator)
    assert op.shape == (10, 9)
    assert op.T_cluster.shape == (3, 2)
    assert op.T_cluster.sum() == pytest.approx(1.0, rel=1e-3)


def test_fgw_requires_same_feature_dimension():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 4))
    Y = rng.standard_normal((9, 5))
    with pytest.raises(ValueError):
        run_ensemble_fgw(
            X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
            solver_method="fgw", random_state=0,
        )


def test_materialize_dense_matches_uniform_lifting_formula():
    X, Y = _toy_xy()
    op = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="entropic_fgw", random_state=1,
    )[0]
    dense = op.materialize_dense()
    for i in range(op.n_x):
        for j in range(op.n_y):
            a, b = int(op.labels_x[i]), int(op.labels_y[j])
            expected = op.T_cluster[a, b] / (op.cluster_mass_x[a] * op.cluster_mass_y[b])
            assert dense[i, j] == pytest.approx(expected)


def test_apply_to_features_matches_dense_lifted():
    X, Y = _toy_xy()
    op = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="entropic_fgw", random_state=2,
    )[0]
    rng = np.random.default_rng(123)
    F = rng.standard_normal((op.n_y, 4))
    implicit = op.apply_to_features(F, normalize=False)
    dense = op.materialize_dense() @ F
    np.testing.assert_allclose(implicit, dense, atol=1e-10, rtol=1e-10)


def test_multiple_runs_return_list():
    X, Y = _toy_xy()
    runs = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=3,
        solver_method="fgw", random_state=7, alpha=0.3,
    )
    assert len(runs) == 3
    for op in runs:
        assert isinstance(op, ImplicitTransportOperator)
        assert op.shape == (10, 9)


def test_random_state_reproducible():
    X, Y = _toy_xy()
    kwargs = dict(
        n_clusters_x=3, n_clusters_y=2, n_runs=2,
        solver_method="entropic_fgw", random_state=42,
        alpha=0.4, epsilon=0.05,
    )
    a = run_ensemble_fgw(X, Y, **kwargs)
    b = run_ensemble_fgw(X, Y, **kwargs)
    for opa, opb in zip(a, b):
        np.testing.assert_array_equal(opa.labels_x, opb.labels_x)
        np.testing.assert_array_equal(opa.labels_y, opb.labels_y)
        np.testing.assert_allclose(opa.T_cluster, opb.T_cluster, atol=1e-10)


def test_fgw_operator_contains_metrics():
    X, Y = _toy_xy()
    op = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="fgw", random_state=3, alpha=0.5,
    )[0]
    assert op.meta["solver_family"] == "fgw"
    assert op.meta["solver_name"] == "fgw"
    assert op.meta["solver_params"]["alpha"] == 0.5
    m = op.meta["metrics"]
    for key in (
        "n_clusters_x", "n_clusters_y",
        "cluster_mass_sum_x", "cluster_mass_sum_y",
        "T_cluster_sum",
        "marginal_error_row", "marginal_error_col",
        "transport_entropy",
    ):
        assert key in m, f"missing metric {key}"
        assert not np.isnan(float(m[key]))


def test_uniform_weights_pipeline_works_after_fgw():
    X, Y = _toy_xy()
    runs = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=3,
        solver_method="entropic_fgw", random_state=11, alpha=0.5,
    )
    mean_op = make_mean_operator(runs)
    F = np.random.default_rng(0).standard_normal((runs[0].n_y, 2))
    out = mean_op.apply_to_features(F)
    assert out.shape == (runs[0].n_x, 2)

    weights = np.full(len(runs), 1.0 / len(runs))
    wmean = make_weighted_mean_operator(runs, weights)
    out_w = wmean.apply_to_features(F)
    assert out_w.shape == (runs[0].n_x, 2)

    edges = consensus_edges(runs, threshold=1e-6, min_frequency=0.0)
    assert isinstance(edges, list)
