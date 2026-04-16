"""Stage 13: random_voronoi clustering backend."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot.clustering import cluster_samples, cluster_samples_with_info
from ensembleot import (
    run_ensemble_sinkhorn,
    run_ensemble_gw,
    run_ensemble_fgw,
)
from ensembleot.operator import ImplicitTransportOperator


def _coords_2d(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2))


# ---------------------------------------------------------------
# clustering unit tests
# ---------------------------------------------------------------

def test_random_voronoi_returns_valid_labels():
    X = _coords_2d(20)
    K = 5
    labels, info = cluster_samples_with_info(X, "random_voronoi", K, random_state=0)
    assert labels.shape == (20,)
    assert labels.min() >= 0
    assert labels.max() < K
    assert len(np.unique(labels)) == K
    # every cluster has >= 1 point (seeds guarantee this)
    for k in range(K):
        assert np.sum(labels == k) >= 1


def test_random_voronoi_reproducible_with_same_seed():
    X = _coords_2d(30)
    a, _ = cluster_samples_with_info(X, "random_voronoi", 5, random_state=42)
    b, _ = cluster_samples_with_info(X, "random_voronoi", 5, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_random_voronoi_changes_with_different_seed():
    # spread out points to ensure different seeds produce different partitions
    rng = np.random.default_rng(0)
    X = rng.uniform(-100, 100, size=(50, 2))
    a, _ = cluster_samples_with_info(X, "random_voronoi", 5, random_state=0)
    b, _ = cluster_samples_with_info(X, "random_voronoi", 5, random_state=999)
    assert not np.array_equal(a, b)


def test_random_voronoi_cluster_samples_wrapper_works():
    X = _coords_2d(20)
    labels = cluster_samples(X, "random_voronoi", 4, random_state=7)
    assert labels.shape == (20,)
    assert len(np.unique(labels)) == 4


def test_random_voronoi_info_contains_seed_metadata():
    X = _coords_2d(20)
    K = 5
    _, info = cluster_samples_with_info(X, "random_voronoi", K, random_state=0)
    assert info["method"] == "random_voronoi"
    assert "seed_indices" in info
    idx = info["seed_indices"]
    assert idx.shape == (K,)
    assert len(np.unique(idx)) == K  # no duplicates
    assert idx.min() >= 0 and idx.max() < 20


# ---------------------------------------------------------------
# solver integration tests
# ---------------------------------------------------------------

def _toy_xy(n_x: int = 12, n_y: int = 10, d: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_x, d))
    Y = rng.standard_normal((n_y, d)) + 0.3
    return X, Y


def test_sinkhorn_accepts_random_voronoi_clustering():
    X, Y = _toy_xy()
    runs = run_ensemble_sinkhorn(
        X, Y, n_clusters_x=3, n_clusters_y=3, n_runs=2,
        clustering_method="random_voronoi", random_state=0,
    )
    assert len(runs) == 2
    for op in runs:
        assert isinstance(op, ImplicitTransportOperator)
        assert op.shape == (12, 10)
        assert op.T_cluster.sum() == pytest.approx(1.0, rel=1e-3)
        assert op.meta["clustering_method"] == "random_voronoi"


def test_gw_accepts_random_voronoi_clustering():
    X, Y = _toy_xy()
    runs = run_ensemble_gw(
        X, Y, n_clusters_x=3, n_clusters_y=3, n_runs=2,
        clustering_method="random_voronoi",
        solver_method="entropic_gw", random_state=0,
    )
    assert len(runs) == 2
    for op in runs:
        assert isinstance(op, ImplicitTransportOperator)
        assert op.shape == (12, 10)
        assert op.T_cluster.sum() == pytest.approx(1.0, rel=1e-3)


def test_fgw_accepts_random_voronoi_clustering():
    X, Y = _toy_xy()
    runs = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=3, n_runs=2,
        clustering_method="random_voronoi",
        solver_method="entropic_fgw", alpha=0.5, random_state=0,
    )
    assert len(runs) == 2
    for op in runs:
        assert isinstance(op, ImplicitTransportOperator)
        assert op.shape == (12, 10)
        assert op.T_cluster.sum() == pytest.approx(1.0, rel=1e-3)
