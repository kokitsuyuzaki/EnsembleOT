"""Stage 21: run_ensemble_partial_gw."""

import json

import numpy as np
import ot
import ot.gromov
import pytest

from ensembleot import (
    load_operators,
    make_mean_operator,
    run_ensemble_partial_gw,
    save_operators,
)
from ensembleot._runs import (
    build_cluster_problem,
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
