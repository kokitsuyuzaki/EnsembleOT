"""Stage 19: run_ensemble_unbalanced."""

import json

import numpy as np
import ot
import pytest

from ensembleot import (
    load_operators,
    make_mean_operator,
    run_ensemble_unbalanced,
    save_operators,
)
from ensembleot._runs import build_cluster_problem, cross_cost, make_seeds


def _data(seed=0, n_x=120, n_y=100, d=4):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_x, d)), rng.standard_normal((n_y, d)) + 0.5


COMMON = dict(n_clusters_x=8, n_clusters_y=6, n_runs=3, random_state=0)


@pytest.mark.parametrize(
    "solver_method, extra",
    [
        ("sinkhorn_unbalanced", dict(reg=0.05, reg_m=1.0)),
        ("sinkhorn_unbalanced", dict(reg=0.05, reg_m=(1.0, 5.0), method="sinkhorn_stabilized")),
        ("mm_unbalanced", dict(reg_m=1.0)),
        ("mm_unbalanced", dict(reg_m=1.0, div="l2")),
        ("lbfgsb_unbalanced", dict(reg=0.05, reg_m=1.0)),
    ],
)
def test_runs_and_meta(solver_method, extra):
    X, Y = _data()
    runs = run_ensemble_unbalanced(X, Y, solver_method=solver_method, **COMMON, **extra)
    assert len(runs) == 3
    for op in runs:
        assert op.shape == (120, 100)
        assert op.T_cluster.shape == (8, 6)
        assert np.isfinite(op.T_cluster).all() and (op.T_cluster >= -1e-12).all()
        assert op.meta["solver_family"] == "unbalanced"
        assert op.meta["solver_name"] == solver_method
        json.dumps(op.meta)                                    # JSON-serializable
        m = op.meta["metrics"]
        assert m["transported_mass"] == pytest.approx(op.T_cluster.sum())
        assert m["transport_cost"] >= 0
        np.testing.assert_allclose(op.transported_mass_x().sum(), op.T_cluster.sum())


def test_matches_direct_pot_call_on_the_same_cluster_problem():
    X, Y = _data(1)
    runs = run_ensemble_unbalanced(X, Y, reg=0.05, reg_m=(0.5, 2.0), **COMMON)
    for op, seed in zip(runs, make_seeds(0, 3)):
        prob = build_cluster_problem(X, Y, 8, 6, "kmeans", seed)
        np.testing.assert_array_equal(prob.labels_x, op.labels_x)
        M, _ = cross_cost(prob, X, Y, "sqeuclidean", True)
        ref = ot.unbalanced.sinkhorn_unbalanced(prob.a, prob.b, M, 0.05, (0.5, 2.0))
        np.testing.assert_allclose(op.T_cluster, ref)
        assert op.meta["solver_params"]["reg_m"] == [0.5, 2.0]


def test_normalize_cost_false_uses_raw_ot_dist():
    X, Y = _data(2)
    runs = run_ensemble_unbalanced(
        X, Y, solver_method="mm_unbalanced", reg_m=1.0, normalize_cost=False, **COMMON
    )
    seed = make_seeds(0, 3)[0]
    prob = build_cluster_problem(X, Y, 8, 6, "kmeans", seed)
    M = ot.dist(prob.centers_x, prob.centers_y)
    np.testing.assert_allclose(runs[0].T_cluster, ot.unbalanced.mm_unbalanced(prob.a, prob.b, M, 1.0))
    assert runs[0].meta["solver_params"]["normalize_cost"] is False


def test_deterministic_and_seed_dependent():
    X, Y = _data(3)
    kw = dict(reg=0.05, reg_m=1.0, **COMMON)
    a = run_ensemble_unbalanced(X, Y, **kw)
    b = run_ensemble_unbalanced(X, Y, **kw)
    c = run_ensemble_unbalanced(X, Y, **{**kw, "random_state": 1})
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
        np.testing.assert_array_equal(u.labels_x, v.labels_x)
    assert any(not np.array_equal(u.labels_x, v.labels_x) for u, v in zip(a, c))


def test_large_reg_m_approaches_balanced_and_small_drops_mass():
    X, Y = _data(4)
    kw = dict(solver_method="mm_unbalanced", **COMMON)
    tight = run_ensemble_unbalanced(X, Y, reg_m=1e4, **kw)
    loose = run_ensemble_unbalanced(X, Y, reg_m=1e-2, **kw)
    for op in tight:
        assert op.meta["metrics"]["transported_mass"] == pytest.approx(1.0, abs=1e-3)
        assert op.meta["metrics"]["marginal_error_row"] < 1e-3
    for t, l in zip(tight, loose):
        assert l.meta["metrics"]["transported_mass"] < t.meta["metrics"]["transported_mass"] - 0.05


def test_one_sided_relaxation_keeps_the_other_marginal():
    X, Y = _data(5)
    runs = run_ensemble_unbalanced(
        X, Y, reg=0.05, reg_m=(float("inf"), 0.05), **COMMON
    )
    for op in runs:
        assert op.meta["metrics"]["marginal_error_row"] < 1e-5      # X side exact
        assert op.meta["metrics"]["marginal_error_col"] > 1e-4      # Y side relaxed
        np.testing.assert_allclose(op.transported_mass_x(relative=True), 1.0, atol=1e-5)


def test_downstream_pipeline_and_storage(tmp_path):
    X, Y = _data(6)
    runs = run_ensemble_unbalanced(X, Y, reg=0.05, reg_m=0.5, **COMMON)
    mean_op = make_mean_operator(runs)
    F_y = np.random.default_rng(0).standard_normal((100, 3))
    out = mean_op.apply_to_features(F_y)
    assert out.shape == (120, 3) and np.isfinite(out).all()
    assert mean_op.transported_mass_x(relative=True).shape == (120,)
    path = tmp_path / "ops.npz"
    save_operators(path, runs)
    loaded = load_operators(path)
    for u, v in zip(runs, loaded):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
        assert u.meta == v.meta


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(reg_m=1.0), "requires: reg"),                                        # sinkhorn needs reg
        (dict(reg_m=1.0, reg=0.1, div="kl"), "does not accept: div"),
        (dict(reg_m=1.0, solver_method="mm_unbalanced", reg_type="kl"), "does not accept: reg_type"),
        (dict(reg_m=1.0, solver_method="mm_unbalanced", method="sinkhorn"), "does not accept: method"),
        (dict(reg_m=1.0, reg=0.1, solver_method="lbfgsb_unbalanced", div="kl"), "does not accept: div"),
        (dict(reg_m=1.0, reg=0.1, solver_method="nope"), "unknown solver_method"),
        (dict(reg_m=None, reg=0.1), "reg_m is required"),
        (dict(reg_m=(float("inf"), 1.0), solver_method="mm_unbalanced"), "finite reg_m"),
        (dict(reg_m=float("inf"), reg=0.1, solver_method="lbfgsb_unbalanced"), "finite reg_m"),
    ],
)
def test_argument_validation(kwargs, match):
    X, Y = _data()
    with pytest.raises(ValueError, match=match):
        run_ensemble_unbalanced(X, Y, **COMMON, **kwargs)


def test_input_validation():
    X, Y = _data()
    with pytest.raises(ValueError, match="same feature space"):
        run_ensemble_unbalanced(X, Y[:, :3], reg=0.1, reg_m=1.0, **COMMON)
    with pytest.raises(ValueError, match="n_runs"):
        run_ensemble_unbalanced(X, Y, reg=0.1, reg_m=1.0, **{**COMMON, "n_runs": 0})
    with pytest.raises(TypeError):
        run_ensemble_unbalanced(X, Y, 8, 6, 3, 1.0)            # solver args are keyword-only
