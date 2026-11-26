"""Stage 20: run_ensemble_partial and scan_partial_mass."""

import json

import numpy as np
import ot
import pytest

from ensembleot import (
    load_operators,
    make_mean_operator,
    run_ensemble_partial,
    save_operators,
    scan_partial_mass,
)
from ensembleot._runs import build_cluster_problem, cross_cost, make_seeds


def _data(seed=0, n_x=120, n_y=100, d=4):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_x, d)), rng.standard_normal((n_y, d)) + 0.5


COMMON = dict(n_clusters_x=8, n_clusters_y=6, n_runs=3, random_state=0)


@pytest.mark.parametrize(
    "solver_method, extra",
    [
        ("partial_wasserstein", dict(m=0.6)),
        ("partial_wasserstein", dict(m=0.6, nb_dummies=2)),
        ("entropic_partial_wasserstein", dict(m=0.6, reg=0.05)),
    ],
)
def test_transports_exactly_m_without_creating_mass(solver_method, extra):
    X, Y = _data()
    runs = run_ensemble_partial(X, Y, solver_method=solver_method, **COMMON, **extra)
    assert len(runs) == 3
    for op in runs:
        T = op.T_cluster
        a = op.cluster_mass_x / op.n_x
        b = op.cluster_mass_y / op.n_y
        assert T.sum() == pytest.approx(0.6, abs=1e-6)
        assert (T.sum(axis=1) <= a + 1e-6).all() and (T.sum(axis=0) <= b + 1e-6).all()
        assert op.meta["solver_family"] == "partial"
        assert op.meta["solver_name"] == solver_method
        assert op.meta["solver_params"]["m"] == 0.6
        assert op.meta["metrics"]["mass_error"] < 1e-6
        assert op.meta["metrics"]["transported_mass"] == pytest.approx(0.6, abs=1e-6)
        json.dumps(op.meta)
        rel = op.transported_mass_x(relative=True)
        assert rel.min() >= -1e-9 and rel.max() <= 1 + 1e-5


def test_matches_direct_pot_call_on_the_same_cluster_problem():
    X, Y = _data(1)
    runs = run_ensemble_partial(X, Y, m=0.7, **COMMON)
    for op, seed in zip(runs, make_seeds(0, 3)):
        prob = build_cluster_problem(X, Y, 8, 6, "kmeans", seed)
        M, _ = cross_cost(prob, X, Y, "sqeuclidean", True)
        np.testing.assert_allclose(op.T_cluster, ot.partial.partial_wasserstein(prob.a, prob.b, M, m=0.7))


def test_default_m_is_balanced_like_pot():
    X, Y = _data(2)
    for op in run_ensemble_partial(X, Y, **COMMON):
        assert op.T_cluster.sum() == pytest.approx(1.0)
        assert op.meta["metrics"]["marginal_error_row"] < 1e-9
        assert op.meta["metrics"]["mass_error"] < 1e-9
        assert "m" not in op.meta["solver_params"]


def test_lagrange_form_drops_more_mass_as_threshold_falls():
    X, Y = _data(3)
    kw = dict(solver_method="partial_wasserstein_lagrange", **COMMON)
    hi = run_ensemble_partial(X, Y, reg_m=10.0, **kw)
    lo = run_ensemble_partial(X, Y, reg_m=0.05, **kw)
    for h, l in zip(hi, lo):
        assert h.T_cluster.sum() == pytest.approx(1.0)
        assert l.T_cluster.sum() < 0.9
        assert "mass_error" not in l.meta["metrics"]


def test_deterministic():
    X, Y = _data(4)
    a = run_ensemble_partial(X, Y, m=0.5, **COMMON)
    b = run_ensemble_partial(X, Y, m=0.5, **COMMON)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)


def test_unmatched_population_is_dropped_across_the_ensemble():
    """X = shared blob + a blob with no counterpart in Y; m = shared fraction."""
    rng = np.random.default_rng(0)
    shared_x = rng.standard_normal((150, 2))
    extra_x = rng.standard_normal((150, 2)) + np.array([25.0, 0.0])
    X = np.vstack([shared_x, extra_x])
    Y = rng.standard_normal((200, 2))
    runs = run_ensemble_partial(X, Y, 12, 8, 6, m=0.5, random_state=0)
    mean_op = make_mean_operator(runs)
    kept = mean_op.transported_mass_x(relative=True)
    assert kept[:150].mean() > 0.95            # shared part keeps its mass
    assert kept[150:].mean() < 0.05            # the extra blob is left behind
    F_y = np.c_[Y, np.ones(len(Y))]
    out = mean_op.apply_to_features(F_y)
    assert np.isfinite(out[:150]).all()
    assert np.isnan(out[150:]).all()           # no image for unmatched samples


def test_downstream_storage_roundtrip(tmp_path):
    X, Y = _data(5)
    runs = run_ensemble_partial(X, Y, m=0.6, **COMMON)
    path = tmp_path / "ops.npz"
    save_operators(path, runs)
    for u, v in zip(runs, load_operators(path)):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
        assert u.meta == v.meta


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(m=0.5, reg=0.1), "does not accept: reg"),
        (dict(m=0.5, stopThr=1e-6), "does not accept: stopThr"),
        (dict(m=0.5, solver_method="entropic_partial_wasserstein"), "requires: reg"),
        (dict(m=0.5, reg=0.1, nb_dummies=2, solver_method="entropic_partial_wasserstein"), "does not accept: nb_dummies"),
        (dict(m=0.5, solver_method="partial_wasserstein_lagrange"), "does not accept: m"),
        (dict(m=0.5, solver_method="nope"), "unknown solver_method"),
    ],
)
def test_argument_validation(kwargs, match):
    X, Y = _data()
    with pytest.raises(ValueError, match=match):
        run_ensemble_partial(X, Y, **COMMON, **kwargs)


def test_scan_partial_mass_shapes_and_monotone_cost():
    X, Y = _data(6)
    ms = [0.9, 0.3, 0.6, 1.0]
    res = scan_partial_mass(X, Y, m_values=ms, **COMMON)
    np.testing.assert_allclose(res["m"], [0.3, 0.6, 0.9, 1.0])
    assert res["transport_cost"].shape == (4, 3) and res["transported_mass"].shape == (4, 3)
    np.testing.assert_allclose(res["transported_mass"], res["m"][:, None] * np.ones((1, 3)), atol=1e-6)
    assert (np.diff(res["transport_cost"], axis=0) >= -1e-10).all()      # more mass never costs less
    assert res["marginal_cost"].shape == (3,) and res["m_mid"].shape == (3,)
    assert (np.diff(res["marginal_cost"]) >= -1e-9).all()               # exact partial OT cost is convex in m
    # consistent with the entry point at the same random_state
    runs = run_ensemble_partial(X, Y, m=0.6, **COMMON)
    np.testing.assert_allclose(
        res["transport_cost"][1], [op.meta["metrics"]["transport_cost"] for op in runs]
    )


def test_scan_partial_mass_shows_the_bend_at_the_shared_fraction():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.standard_normal((150, 2)), rng.standard_normal((150, 2)) + [25.0, 0.0]])
    Y = rng.standard_normal((200, 2))
    res = scan_partial_mass(X, Y, 12, 8, 4, m_values=np.linspace(0.1, 1.0, 10), random_state=0)
    below = res["marginal_cost"][res["m_mid"] < 0.5].max()
    above = res["marginal_cost"][res["m_mid"] > 0.5].min()
    assert above > 10 * below


def test_scan_partial_mass_validation():
    X, Y = _data()
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        scan_partial_mass(X, Y, m_values=[0.0, 0.5], **COMMON)
    with pytest.raises(ValueError, match="takes m"):
        scan_partial_mass(X, Y, m_values=[0.5], solver_method="partial_wasserstein_lagrange", **COMMON)
