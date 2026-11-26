"""Stage 21 / 22: run_ensemble_partial_gw and run_ensemble_partial_fgw."""

import json

import numpy as np
import ot
import ot.gromov
import pytest

from ensembleot import (
    load_operators,
    make_mean_operator,
    run_ensemble_partial_fgw,
    run_ensemble_partial_gw,
    save_operators,
)
from ensembleot._runs import (
    build_cluster_problem,
    cross_cost,
    make_seeds,
    structure_costs,
)


def _data(seed=0, n_x=120, n_y=100, d_x=4, d_y=4):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_x, d_x)), rng.standard_normal((n_y, d_y)) + 0.5


COMMON = dict(n_clusters_x=8, n_clusters_y=6, n_runs=3, random_state=0)


def _check_partial(runs, m, family, name):
    assert len(runs) == 3
    for op in runs:
        T = op.T_cluster
        a = op.cluster_mass_x / op.n_x
        b = op.cluster_mass_y / op.n_y
        assert np.isfinite(T).all()
        assert T.sum() == pytest.approx(m, abs=1e-5)
        assert (T.sum(axis=1) <= a + 1e-5).all() and (T.sum(axis=0) <= b + 1e-5).all()
        assert op.meta["solver_family"] == family and op.meta["solver_name"] == name
        assert op.meta["metrics"]["mass_error"] < 1e-5
        json.dumps(op.meta)


# ---------------------------------------------------------------- partial GW

@pytest.mark.parametrize(
    "solver_method, extra",
    [
        ("partial_gromov_wasserstein", dict()),
        ("partial_gromov_wasserstein", dict(nb_dummies=2, loss_fun="square_loss")),
        ("entropic_partial_gromov_wasserstein", dict(reg=0.5, numItermax=100)),
    ],
)
def test_partial_gw_transports_m(solver_method, extra):
    X, Y = _data(d_y=7)                       # different feature dimensions are fine
    runs = run_ensemble_partial_gw(X, Y, m=0.6, solver_method=solver_method, **COMMON, **extra)
    _check_partial(runs, 0.6, "partial_gw", solver_method)
    assert runs[0].shape == (120, 100)


def test_partial_gw_matches_direct_pot_call():
    X, Y = _data(1, d_y=6)
    runs = run_ensemble_partial_gw(X, Y, m=0.7, **COMMON)
    for op, seed in zip(runs, make_seeds(0, 3)):
        prob = build_cluster_problem(X, Y, 8, 6, "kmeans", seed)
        C1, C2 = structure_costs(prob, "sqeuclidean", True)
        ref = ot.gromov.partial_gromov_wasserstein(C1, C2, prob.a, prob.b, m=0.7)
        np.testing.assert_allclose(op.T_cluster, ref)


def test_partial_gw_default_m_is_full_mass():
    X, Y = _data(2)
    for op in run_ensemble_partial_gw(X, Y, **COMMON):
        assert op.T_cluster.sum() == pytest.approx(1.0, abs=1e-6)


def test_partial_gw_deterministic_coords_and_pipeline(tmp_path):
    X, Y = _data(3)
    rng = np.random.default_rng(0)
    cx, cy = rng.random((120, 2)), rng.random((100, 2))
    kw = dict(m=0.5, clustering_method="random_voronoi", coords_x=cx, coords_y=cy, **COMMON)
    a = run_ensemble_partial_gw(X, Y, **kw)
    b = run_ensemble_partial_gw(X, Y, **kw)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
    mean_op = make_mean_operator(a)
    out = mean_op.apply_to_features(rng.standard_normal((100, 2)))
    assert out.shape == (120, 2)
    kept = mean_op.transported_mass_x(relative=True)
    assert kept.mean() == pytest.approx(0.5, abs=1e-5)
    np.testing.assert_array_equal(np.isnan(out[:, 0]), kept <= 1e-9)
    path = tmp_path / "ops.npz"
    save_operators(path, a)
    for u, v in zip(a, load_operators(path)):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
        assert u.meta == v.meta


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(reg=0.1), "does not accept: reg"),
        (dict(solver_method="entropic_partial_gromov_wasserstein", nb_dummies=2, thres=1.0), "does not accept: nb_dummies, thres"),
        (dict(solver_method="nope"), "unknown solver_method"),
    ],
)
def test_partial_gw_argument_validation(kwargs, match):
    X, Y = _data()
    with pytest.raises(ValueError, match=match):
        run_ensemble_partial_gw(X, Y, m=0.5, **COMMON, **kwargs)


# --------------------------------------------------------------- partial FGW

@pytest.mark.parametrize(
    "solver_method, extra",
    [
        ("partial_fused_gromov_wasserstein", dict()),
        ("partial_fused_gromov_wasserstein", dict(alpha=0.2)),
        ("entropic_partial_fused_gromov_wasserstein", dict(reg=0.5, alpha=0.8, numItermax=100)),
    ],
)
def test_partial_fgw_transports_m(solver_method, extra):
    X, Y = _data(4)
    runs = run_ensemble_partial_fgw(X, Y, m=0.6, solver_method=solver_method, **COMMON, **extra)
    _check_partial(runs, 0.6, "partial_fgw", solver_method)
    for op in runs:
        assert op.meta["solver_params"]["cross_feature_cost_mode"] == "default"
        assert op.meta["metrics"]["transport_cost"] >= 0


def test_partial_fgw_matches_direct_pot_call_and_alpha_is_forwarded():
    X, Y = _data(5)
    runs = run_ensemble_partial_fgw(X, Y, m=0.7, alpha=0.3, **COMMON)
    for op, seed in zip(runs, make_seeds(0, 3)):
        prob = build_cluster_problem(X, Y, 8, 6, "kmeans", seed)
        C1, C2 = structure_costs(prob, "sqeuclidean", True)
        M, _ = cross_cost(prob, X, Y, "sqeuclidean", True)
        ref = ot.gromov.partial_fused_gromov_wasserstein(M, C1, C2, prob.a, prob.b, m=0.7, alpha=0.3)
        np.testing.assert_allclose(op.T_cluster, ref)
        assert op.meta["solver_params"]["alpha"] == 0.3


def test_partial_fgw_cross_modal_needs_and_uses_custom_cost():
    X, Y = _data(6, d_y=9)
    with pytest.raises(ValueError, match="same feature space"):
        run_ensemble_partial_fgw(X, Y, m=0.6, **COMMON)

    seen = []

    def cost_fn(*, centers_x, centers_y, n_clusters_x, n_clusters_y, seed, scale, **_):
        seen.append(seed)
        return scale * ot.dist(centers_x[:, :2], centers_y[:, :2])

    runs = run_ensemble_partial_fgw(
        X, Y, m=0.6, cross_feature_cost_fn=cost_fn,
        cross_feature_cost_kwargs={"scale": 2.0}, **COMMON,
    )
    _check_partial(runs, 0.6, "partial_fgw", "partial_fused_gromov_wasserstein")
    assert seen == make_seeds(0, 3)
    assert runs[0].meta["solver_params"]["cross_feature_cost_mode"] == "custom_fn"

    with pytest.raises(ValueError, match="expected"):
        run_ensemble_partial_fgw(
            X, Y, m=0.6, cross_feature_cost_fn=lambda **_: np.zeros((2, 2)), **COMMON
        )


def test_partial_fgw_argument_validation():
    X, Y = _data()
    with pytest.raises(ValueError, match="does not accept: reg"):
        run_ensemble_partial_fgw(X, Y, m=0.5, reg=0.1, **COMMON)
    with pytest.raises(ValueError, match="does not accept: nb_dummies"):
        run_ensemble_partial_fgw(
            X, Y, m=0.5, nb_dummies=2,
            solver_method="entropic_partial_fused_gromov_wasserstein", **COMMON,
        )
