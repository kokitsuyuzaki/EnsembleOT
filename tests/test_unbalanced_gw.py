"""Stage 23: run_ensemble_fused_unbalanced_gw."""

import json

import numpy as np
import ot
import ot.gromov
import pytest

from ensembleot import (
    load_operators,
    make_mean_operator,
    run_ensemble_fused_unbalanced_gw,
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


@pytest.mark.parametrize(
    "extra",
    [
        dict(),                                                        # all POT defaults
        dict(reg_marginals=1.0),
        dict(reg_marginals=(1.0, 5.0), divergence="l2"),
        dict(reg_marginals=1.0, epsilon=0.01, unbalanced_solver="sinkhorn"),
    ],
)
def test_unbalanced_gw_structure_only(extra):
    X, Y = _data(d_y=7)                       # alpha unset: dimensions may differ
    runs = run_ensemble_fused_unbalanced_gw(X, Y, **COMMON, **extra)
    assert len(runs) == 3
    for op in runs:
        assert op.T_cluster.shape == (8, 6)
        assert np.isfinite(op.T_cluster).all() and (op.T_cluster >= -1e-12).all()
        assert op.T_cluster.sum() > 0.1
        assert op.meta["solver_family"] == "fused_unbalanced_gw"
        assert op.meta["solver_name"] == "fused_unbalanced_gromov_wasserstein"
        assert op.meta["solver_params"]["cross_feature_cost_mode"] == "none"
        assert "transport_cost" not in op.meta["metrics"]
        assert op.meta["metrics"]["transported_mass"] == pytest.approx(op.T_cluster.sum())
        json.dumps(op.meta)


def test_matches_direct_pot_call_and_uses_first_coupling():
    X, Y = _data(1)
    runs = run_ensemble_fused_unbalanced_gw(X, Y, reg_marginals=1.0, alpha=0.5, **COMMON)
    for op, seed in zip(runs, make_seeds(0, 3)):
        prob = build_cluster_problem(X, Y, 8, 6, "kmeans", seed)
        C1, C2 = structure_costs(prob, "sqeuclidean", True)
        M, _ = cross_cost(prob, X, Y, "sqeuclidean", True)
        pi_samp, _ = ot.gromov.fused_unbalanced_gromov_wasserstein(
            C1, C2, wx=prob.a, wy=prob.b, reg_marginals=1.0, alpha=0.5, M=M
        )
        np.testing.assert_allclose(op.T_cluster, pi_samp)
        assert op.meta["solver_params"]["alpha"] == 0.5
        assert op.meta["solver_params"]["cross_feature_cost_mode"] == "default"
        assert op.meta["metrics"]["transport_cost"] >= 0


def test_alpha_zero_is_plain_unbalanced_gw():
    X, Y = _data(2)
    kw = dict(reg_marginals=1.0, **COMMON)
    plain = run_ensemble_fused_unbalanced_gw(X, Y, **kw)
    zero = run_ensemble_fused_unbalanced_gw(X, Y, alpha=0.0, **kw)
    fused = run_ensemble_fused_unbalanced_gw(X, Y, alpha=5.0, **kw)
    for p, z in zip(plain, zero):
        np.testing.assert_allclose(p.T_cluster, z.T_cluster)
    assert any(not np.allclose(p.T_cluster, f.T_cluster, atol=1e-6) for p, f in zip(plain, fused))


def test_marginal_relaxation_strength():
    X, Y = _data(3)
    tight = run_ensemble_fused_unbalanced_gw(X, Y, reg_marginals=100.0, max_iter=30, **COMMON)
    loose = run_ensemble_fused_unbalanced_gw(X, Y, reg_marginals=0.01, max_iter=30, **COMMON)
    err = lambda runs: np.mean([op.meta["metrics"]["marginal_error_row"] for op in runs])
    assert err(tight) < err(loose)
    assert err(tight) < 5e-3


def test_cross_modal_feature_term_needs_custom_cost():
    X, Y = _data(4, d_y=9)
    with pytest.raises(ValueError, match="same feature space"):
        run_ensemble_fused_unbalanced_gw(X, Y, alpha=1.0, **COMMON)
    runs = run_ensemble_fused_unbalanced_gw(
        X, Y, alpha=1.0, reg_marginals=1.0,
        cross_feature_cost_fn=lambda *, centers_x, centers_y, **_: ot.dist(centers_x[:, :3], centers_y[:, :3]),
        **COMMON,
    )
    assert runs[0].meta["solver_params"]["cross_feature_cost_mode"] == "custom_fn"
    with pytest.raises(ValueError, match="only used when alpha > 0"):
        run_ensemble_fused_unbalanced_gw(X, Y, cross_feature_cost_fn=lambda **_: None, **COMMON)
    with pytest.raises(ValueError, match="alpha must be >= 0"):
        run_ensemble_fused_unbalanced_gw(X, Y, alpha=-1.0, **COMMON)


def test_deterministic_pipeline_and_storage(tmp_path):
    X, Y = _data(5)
    kw = dict(reg_marginals=1.0, **COMMON)
    a = run_ensemble_fused_unbalanced_gw(X, Y, **kw)
    b = run_ensemble_fused_unbalanced_gw(X, Y, **kw)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
    mean_op = make_mean_operator(a)
    out = mean_op.apply_to_features(np.random.default_rng(0).standard_normal((100, 2)))
    assert out.shape == (120, 2) and np.isfinite(out).all()
    path = tmp_path / "ops.npz"
    save_operators(path, a)
    for u, v in zip(a, load_operators(path)):
        np.testing.assert_array_equal(u.T_cluster, v.T_cluster)
        assert u.meta == v.meta
