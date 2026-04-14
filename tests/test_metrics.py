"""Stage 7: per-run metrics on Sinkhorn / GW operators."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ensembleot import run_ensemble_gw, run_ensemble_sinkhorn, save_operators, load_operators

_COMMON_KEYS = {
    "n_clusters_x", "n_clusters_y",
    "cluster_mass_sum_x", "cluster_mass_sum_y",
    "T_cluster_sum",
    "cluster_size_min_x", "cluster_size_max_x",
    "cluster_size_min_y", "cluster_size_max_y",
    "marginal_error_row", "marginal_error_col",
    "transport_entropy",
    "clustering_inertia_x", "clustering_inertia_y",
}


def _toy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3))
    Y = rng.standard_normal((7, 3)) + 0.2
    return X, Y


def _toy_gw():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 3))
    Y = rng.standard_normal((7, 5)) + 0.2
    return X, Y


def test_sinkhorn_operator_contains_metrics():
    X, Y = _toy()
    op = run_ensemble_sinkhorn(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="sinkhorn", random_state=0, reg=0.1,
    )[0]
    assert op.meta["solver_family"] == "sinkhorn"
    assert op.meta["solver_name"] == "sinkhorn"
    assert op.meta["clustering_method"] == "kmeans"
    assert isinstance(op.meta["seed"], int)
    assert "metrics" in op.meta


def test_gw_operator_contains_metrics():
    X, Y = _toy_gw()
    op = run_ensemble_gw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="entropic_gw", random_state=0,
    )[0]
    assert op.meta["solver_family"] == "gw"
    assert op.meta["solver_name"] == "entropic_gw"
    assert "metrics" in op.meta


@pytest.mark.parametrize(
    "runner",
    [
        lambda: run_ensemble_sinkhorn(*_toy(), n_clusters_x=3, n_clusters_y=2,
                                      n_runs=1, solver_method="sinkhorn",
                                      random_state=1, reg=0.1)[0],
        lambda: run_ensemble_sinkhorn(*_toy(), n_clusters_x=3, n_clusters_y=2,
                                      n_runs=1, solver_method="emd",
                                      random_state=1)[0],
        lambda: run_ensemble_gw(*_toy_gw(), n_clusters_x=3, n_clusters_y=2,
                                n_runs=1, solver_method="entropic_gw",
                                random_state=1)[0],
    ],
    ids=["sinkhorn", "emd", "entropic_gw"],
)
def test_metrics_have_expected_keys(runner):
    op = runner()
    m = op.meta["metrics"]
    missing = _COMMON_KEYS - set(m.keys())
    assert not missing, f"missing metrics keys: {missing}"
    for k, v in m.items():
        assert not (isinstance(v, float) and math.isnan(v)), f"{k} is NaN"


@pytest.mark.parametrize("solver_method", ["sinkhorn", "emd"])
def test_marginal_errors_are_small_for_sinkhorn_or_emd(solver_method):
    X, Y = _toy()
    op = run_ensemble_sinkhorn(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method=solver_method, random_state=2, reg=0.05,
    )[0]
    m = op.meta["metrics"]
    assert m["marginal_error_row"] < 1e-3
    assert m["marginal_error_col"] < 1e-3


def test_entropy_is_finite_and_nonnegative():
    X, Y = _toy()
    op = run_ensemble_sinkhorn(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="sinkhorn", random_state=3, reg=0.1,
    )[0]
    H = op.meta["metrics"]["transport_entropy"]
    assert math.isfinite(H)
    # For sub-probability-ish T with entries in [0,1], -Σ T log T >= 0
    assert H >= 0.0


def test_storage_roundtrip_preserves_metrics(tmp_path):
    X, Y = _toy()
    ops = run_ensemble_sinkhorn(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=2,
        solver_method="sinkhorn", random_state=4, reg=0.1,
    )
    path = tmp_path / "ops.npz"
    save_operators(path, ops)
    loaded = load_operators(path)
    assert len(loaded) == len(ops)
    for orig, lo in zip(ops, loaded):
        assert lo.meta["solver_name"] == orig.meta["solver_name"]
        assert lo.meta["seed"] == orig.meta["seed"]
        assert lo.meta["clustering_method"] == orig.meta["clustering_method"]
        for k, v in orig.meta["metrics"].items():
            lv = lo.meta["metrics"][k]
            if isinstance(v, float):
                assert lv == pytest.approx(v)
            else:
                assert lv == v
