"""Stage 12b: FGW with external cross-domain feature cost M."""

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


def _toy_cross_modal(d_x: int = 4, d_y: int = 7, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((10, d_x))
    Y = rng.standard_normal((9, d_y)) + 0.3
    return X, Y


def _random_M(**kw) -> np.ndarray:
    """A trivial custom cost: random but reproducible via `seed`."""
    kx = kw["n_clusters_x"]
    ky = kw["n_clusters_y"]
    rng = np.random.default_rng(int(kw["seed"]) + 99)
    return rng.uniform(0.1, 1.0, size=(kx, ky))


def test_fgw_default_mode_still_requires_same_feature_dimension():
    X, Y = _toy_cross_modal(d_x=4, d_y=5)
    with pytest.raises(ValueError):
        run_ensemble_fgw(
            X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
            solver_method="fgw", random_state=0,
        )


def test_fgw_custom_cross_feature_cost_allows_different_feature_dimensions():
    X, Y = _toy_cross_modal(d_x=4, d_y=7)
    assert X.shape[1] != Y.shape[1]

    def cost_fn(*, centers_x, centers_y, **_):
        # L2 on mean-centered, truncated features (toy cross-modal cost)
        d = min(centers_x.shape[1], centers_y.shape[1])
        diff = centers_x[:, None, :d] - centers_y[None, :, :d]
        return np.linalg.norm(diff, axis=-1)

    runs = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="entropic_fgw", alpha=0.5, random_state=0,
        cross_feature_cost_fn=cost_fn,
    )
    assert len(runs) == 1
    op = runs[0]
    assert isinstance(op, ImplicitTransportOperator)
    assert op.shape == (10, 9)
    assert op.T_cluster.shape == (3, 2)
    assert op.T_cluster.sum() == pytest.approx(1.0, rel=1e-3)


def test_custom_cross_feature_cost_fn_is_called_with_expected_information():
    X, Y = _toy_cross_modal(d_x=4, d_y=7)
    captured: dict = {}

    def cost_fn(**kwargs):
        captured.update(kwargs)
        kx = kwargs["n_clusters_x"]
        ky = kwargs["n_clusters_y"]
        return np.full((kx, ky), 0.5)

    run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="fgw", random_state=0,
        cross_feature_cost_fn=cost_fn,
        cross_feature_cost_kwargs={"extra_flag": "hello"},
    )
    for key in (
        "X", "Y", "centers_x", "centers_y", "labels_x", "labels_y",
        "n_clusters_x", "n_clusters_y", "seed", "metric", "extra_flag",
    ):
        assert key in captured, f"cost_fn did not receive {key}"
    assert captured["centers_x"].shape == (3, 4)
    assert captured["centers_y"].shape == (2, 7)
    assert captured["labels_x"].shape == (10,)
    assert captured["labels_y"].shape == (9,)
    assert captured["extra_flag"] == "hello"
    assert isinstance(captured["seed"], int)


def test_invalid_custom_M_shape_raises_value_error():
    X, Y = _toy_cross_modal()

    def bad_shape(*, n_clusters_x, n_clusters_y, **_):
        return np.zeros((n_clusters_x + 1, n_clusters_y))

    with pytest.raises(ValueError, match="shape"):
        run_ensemble_fgw(
            X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
            solver_method="fgw", random_state=0,
            cross_feature_cost_fn=bad_shape,
        )


def test_invalid_custom_M_nan_or_inf_raises_value_error():
    X, Y = _toy_cross_modal()

    def nan_fn(*, n_clusters_x, n_clusters_y, **_):
        M = np.ones((n_clusters_x, n_clusters_y))
        M[0, 0] = np.nan
        return M

    def inf_fn(*, n_clusters_x, n_clusters_y, **_):
        M = np.ones((n_clusters_x, n_clusters_y))
        M[0, 0] = np.inf
        return M

    for fn in (nan_fn, inf_fn):
        with pytest.raises(ValueError, match="finite"):
            run_ensemble_fgw(
                X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
                solver_method="fgw", random_state=0,
                cross_feature_cost_fn=fn,
            )


def test_fgw_custom_mode_preserves_reproducibility():
    X, Y = _toy_cross_modal()
    kwargs = dict(
        n_clusters_x=3, n_clusters_y=2, n_runs=2,
        solver_method="entropic_fgw", alpha=0.4, random_state=42,
        epsilon=0.05,
        cross_feature_cost_fn=_random_M,
    )
    a = run_ensemble_fgw(X, Y, **kwargs)
    b = run_ensemble_fgw(X, Y, **kwargs)
    for opa, opb in zip(a, b):
        np.testing.assert_array_equal(opa.labels_x, opb.labels_x)
        np.testing.assert_array_equal(opa.labels_y, opb.labels_y)
        np.testing.assert_allclose(opa.T_cluster, opb.T_cluster, atol=1e-10)


def test_fgw_custom_mode_pipeline_works_with_aggregation():
    X, Y = _toy_cross_modal()
    runs = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=3,
        solver_method="entropic_fgw", alpha=0.5, random_state=11,
        cross_feature_cost_fn=_random_M,
    )
    mean_op = make_mean_operator(runs)
    F = np.random.default_rng(0).standard_normal((runs[0].n_y, 2))
    out = mean_op.apply_to_features(F)
    assert out.shape == (runs[0].n_x, 2)

    weights = np.full(len(runs), 1.0 / len(runs))
    wmean = make_weighted_mean_operator(runs, weights)
    assert wmean.apply_to_features(F).shape == (runs[0].n_x, 2)

    edges = consensus_edges(runs, threshold=1e-6, min_frequency=0.0)
    assert isinstance(edges, list)


def test_custom_M_roundtrip_in_meta():
    X, Y = _toy_cross_modal()
    op_default = run_ensemble_fgw(
        X[:, :4], Y[:, :4],  # same dim for default mode
        n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="fgw", random_state=0,
    )[0]
    assert op_default.meta["solver_params"]["cross_feature_cost_mode"] == "default"

    op_custom = run_ensemble_fgw(
        X, Y, n_clusters_x=3, n_clusters_y=2, n_runs=1,
        solver_method="fgw", random_state=0,
        cross_feature_cost_fn=_random_M,
    )[0]
    assert op_custom.meta["solver_params"]["cross_feature_cost_mode"] == "custom_fn"
