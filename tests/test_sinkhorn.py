"""Stage 3: Sinkhorn / EMD ensemble OT."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot import run_ensemble_sinkhorn
from ensembleot.operator import ImplicitTransportOperator


def _toy_xy(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((8, 3))
    Y = rng.standard_normal((7, 3)) + 0.5
    return X, Y


@pytest.mark.parametrize("solver_method", ["sinkhorn", "emd"])
def test_single_run_returns_operator(solver_method):
    X, Y = _toy_xy()
    runs = run_ensemble_sinkhorn(
        X, Y,
        n_clusters_x=3,
        n_clusters_y=2,
        n_runs=1,
        solver_method=solver_method,
        random_state=0,
        reg=0.1,
    )
    assert isinstance(runs, list)
    assert len(runs) == 1
    op = runs[0]
    assert isinstance(op, ImplicitTransportOperator)
    assert op.shape == (8, 7)
    assert op.T_cluster.shape == (3, 2)
    # cluster-level coupling should approximately preserve mass (sinkhorn/emd: sums to 1)
    assert op.T_cluster.sum() == pytest.approx(1.0, rel=1e-3)


def test_materialize_dense_matches_uniform_lifting_formula():
    X, Y = _toy_xy()
    op = run_ensemble_sinkhorn(
        X, Y,
        n_clusters_x=3, n_clusters_y=2,
        n_runs=1, solver_method="emd", random_state=1,
    )[0]
    dense = op.materialize_dense()
    for i in range(op.n_x):
        for j in range(op.n_y):
            a, b = int(op.labels_x[i]), int(op.labels_y[j])
            expected = op.T_cluster[a, b] / (op.cluster_mass_x[a] * op.cluster_mass_y[b])
            assert dense[i, j] == pytest.approx(expected)


def test_apply_to_features_matches_dense_lifted():
    X, Y = _toy_xy()
    op = run_ensemble_sinkhorn(
        X, Y,
        n_clusters_x=3, n_clusters_y=2,
        n_runs=1, solver_method="sinkhorn", random_state=2, reg=0.05,
    )[0]
    rng = np.random.default_rng(123)
    F = rng.standard_normal((op.n_y, 4))
    implicit = op.apply_to_features(F)
    dense = op.materialize_dense() @ F
    np.testing.assert_allclose(implicit, dense, atol=1e-10, rtol=1e-10)


def test_multiple_runs_return_list():
    X, Y = _toy_xy()
    runs = run_ensemble_sinkhorn(
        X, Y,
        n_clusters_x=3, n_clusters_y=2,
        n_runs=3, solver_method="sinkhorn", random_state=7,
    )
    assert len(runs) == 3
    for op in runs:
        assert isinstance(op, ImplicitTransportOperator)
        assert op.shape == (8, 7)


def test_random_state_reproducible():
    X, Y = _toy_xy()
    kwargs = dict(n_clusters_x=3, n_clusters_y=2, n_runs=2,
                  solver_method="sinkhorn", random_state=42, reg=0.1)
    a = run_ensemble_sinkhorn(X, Y, **kwargs)
    b = run_ensemble_sinkhorn(X, Y, **kwargs)
    for opa, opb in zip(a, b):
        np.testing.assert_array_equal(opa.labels_x, opb.labels_x)
        np.testing.assert_array_equal(opa.labels_y, opb.labels_y)
        np.testing.assert_allclose(opa.T_cluster, opb.T_cluster, atol=1e-12)


def test_random_state_different_seeds_differ():
    X, Y = _toy_xy()
    a = run_ensemble_sinkhorn(X, Y, n_clusters_x=3, n_clusters_y=2,
                              n_runs=1, solver_method="sinkhorn", random_state=1)[0]
    b = run_ensemble_sinkhorn(X, Y, n_clusters_x=3, n_clusters_y=2,
                              n_runs=1, solver_method="sinkhorn", random_state=999)[0]
    # Either labels or T_cluster should differ
    differ = (
        not np.array_equal(a.labels_x, b.labels_x)
        or not np.array_equal(a.labels_y, b.labels_y)
        or not np.allclose(a.T_cluster, b.T_cluster)
    )
    assert differ
