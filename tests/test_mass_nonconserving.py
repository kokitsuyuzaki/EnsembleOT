"""Stage 18: couplings that do not conserve mass (partial / unbalanced).

These tests build operators by hand with zero-mass rows / columns so they do
not depend on any particular solver.
"""

import numpy as np
import pytest

from ensembleot import (
    ImplicitTransportOperator,
    make_mean_operator,
    make_weighted_mean_operator,
)
from ensembleot.metrics import cost_metrics, partial_mass_metrics, transport_metrics


def _op(T_cluster, labels_x, labels_y):
    labels_x = np.asarray(labels_x)
    labels_y = np.asarray(labels_y)
    T_cluster = np.asarray(T_cluster, dtype=float)
    return ImplicitTransportOperator(
        labels_x=labels_x,
        labels_y=labels_y,
        T_cluster=T_cluster,
        cluster_mass_x=np.bincount(labels_x, minlength=T_cluster.shape[0]).astype(float),
        cluster_mass_y=np.bincount(labels_y, minlength=T_cluster.shape[1]).astype(float),
    )


LX = np.array([0, 0, 1, 1, 2, 2, 2])       # 3 source clusters
LY = np.array([0, 1, 1, 2, 2, 2])          # 3 target clusters
# source cluster 1 and target cluster 2 are left completely unmatched
T_A = np.array([[0.20, 0.10, 0.0], [0.0, 0.0, 0.0], [0.05, 0.25, 0.0]])
# a second run in which source cluster 1 *is* matched, cluster 0 is not
T_B = np.array([[0.0, 0.0, 0.0], [0.15, 0.05, 0.0], [0.10, 0.10, 0.2]])


def _dense_bary(T, F):
    mass = T.sum(axis=1)
    out = np.full((T.shape[0], F.shape[1]), np.nan)
    ok = mass > 0
    out[ok] = (T[ok] @ F) / mass[ok][:, None]
    return out


def test_transported_mass_matches_dense_plan():
    op = _op(T_A, LX, LY)
    dense = op.materialize_dense()
    np.testing.assert_allclose(op.transported_mass_x(), dense.sum(axis=1))
    np.testing.assert_allclose(op.transported_mass_y(), dense.sum(axis=0))
    np.testing.assert_allclose(op.transported_mass_x(relative=True), dense.sum(axis=1) * op.n_x)
    assert np.all(op.transported_mass_x()[LX == 1] == 0)
    assert np.all(op.transported_mass_y()[LY == 2] == 0)


def test_barycentric_gives_nan_for_unmatched_samples():
    op = _op(T_A, LX, LY)
    rng = np.random.default_rng(0)
    F_y = rng.standard_normal((op.n_y, 3))
    F_x = rng.standard_normal((op.n_x, 2))
    dense = op.materialize_dense()

    out = op.apply_to_features(F_y)
    np.testing.assert_allclose(out, _dense_bary(dense, F_y), equal_nan=True)
    assert np.isnan(out[LX == 1]).all() and np.isfinite(out[LX != 1]).all()

    out_t = op.apply_transpose_to_features(F_x)
    np.testing.assert_allclose(out_t, _dense_bary(dense.T, F_x), equal_nan=True)
    assert np.isnan(out_t[LY == 2]).all() and np.isfinite(out_t[LY != 2]).all()

    # 1-D input keeps working
    np.testing.assert_allclose(op.apply_to_features(F_y[:, 0]), out[:, 0], equal_nan=True)
    # the raw operator has no division, hence no NaN: unmatched rows are 0
    raw = op.apply_to_features(F_y, normalize=False)
    np.testing.assert_allclose(raw, dense @ F_y)


def test_stochastic_and_matched_keep_unmatched_samples_nan():
    op = _op(T_A, LX, LY)
    rng = np.random.default_rng(1)
    F_y = rng.standard_normal((op.n_y, 2))

    sto = op.apply_to_features(F_y, mode="stochastic", random_state=0)
    assert np.isnan(sto[LX == 1]).all() and np.isfinite(sto[LX != 1]).all()
    # every drawn row is an actual row of F_y coming from a matched target cluster
    matched_rows = F_y[LY != 2]
    for row in sto[LX != 1]:
        assert (np.abs(matched_rows - row).sum(axis=1) < 1e-12).any()

    mat = op.apply_to_features(F_y, mode="matched")
    assert np.isnan(mat[LX == 1]).all() and np.isfinite(mat[LX != 1]).all()
    for k in range(F_y.shape[1]):
        assert set(mat[LX != 1, k]).issubset(set(F_y[:, k]))
    # rank order of the matched rows follows the barycentric output
    bary = op.apply_to_features(F_y)
    ok = LX != 1
    for k in range(F_y.shape[1]):
        order = np.argsort(bary[ok, k], kind="stable")
        assert np.all(np.diff(mat[ok, k][order]) >= 0)


@pytest.mark.parametrize("weights", [None, [0.3, 0.7]])
def test_mean_operator_over_runs_with_different_unmatched_sets(weights):
    ops = [_op(T_A, LX, LY), _op(T_B, LX, LY)]
    if weights is None:
        mean_op, w = make_mean_operator(ops), np.array([0.5, 0.5])
    else:
        mean_op, w = make_weighted_mean_operator(ops, weights), np.asarray(weights)
    dense = w[0] * ops[0].materialize_dense() + w[1] * ops[1].materialize_dense()
    rng = np.random.default_rng(2)
    F_y = rng.standard_normal((ops[0].n_y, 3))
    F_x = rng.standard_normal((ops[0].n_x, 3))

    np.testing.assert_allclose(mean_op.transported_mass_x(), dense.sum(axis=1))
    np.testing.assert_allclose(mean_op.transported_mass_y(), dense.sum(axis=0))

    out = mean_op.apply_to_features(F_y)
    # every source sample is matched in at least one run -> no NaN after averaging
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, _dense_bary(dense, F_y))
    np.testing.assert_allclose(
        mean_op.apply_transpose_to_features(F_x), _dense_bary(dense.T, F_x)
    )

    sto = mean_op.apply_to_features(F_y, mode="stochastic", random_state=3)
    assert np.isfinite(sto).all()
    np.testing.assert_array_equal(
        sto, mean_op.apply_to_features(F_y, mode="stochastic", random_state=3)
    )
    assert np.isfinite(mean_op.apply_to_features(F_y, mode="matched")).all()


def test_mean_operator_nan_when_unmatched_in_every_run():
    ops = [_op(T_A, LX, LY), _op(T_A * 0.5, LX, LY)]
    mean_op = make_mean_operator(ops)
    F_y = np.random.default_rng(4).standard_normal((ops[0].n_y, 2))
    for mode in ("barycentric", "stochastic", "matched"):
        out = mean_op.apply_to_features(F_y, mode=mode, random_state=0)
        assert np.isnan(out[LX == 1]).all(), mode
        assert np.isfinite(out[LX != 1]).all(), mode


def test_stochastic_mixture_draws_runs_in_proportion_to_mass():
    # Run A matches source cluster 0 only, run B matches it with 3x the mass;
    # targets are disjoint so the drawn feature reveals which run was used.
    lx = np.zeros(4000, dtype=int)
    ly = np.array([0, 1])
    op_a = _op(np.array([[0.1, 0.0]]), lx, ly)
    op_b = _op(np.array([[0.0, 0.3]]), lx, ly)
    F_y = np.array([0.0, 1.0])
    out = make_mean_operator([op_a, op_b]).apply_to_features(
        F_y, mode="stochastic", random_state=0
    )
    assert abs(out.mean() - 0.75) < 0.03        # P(run B) = 0.3 / (0.1 + 0.3)


def test_metrics_report_mass_and_cost():
    a = np.array([0.3, 0.3, 0.4])
    b = np.array([0.2, 0.4, 0.4])
    m = transport_metrics(T_A, a, b)
    assert m["transported_mass"] == pytest.approx(T_A.sum())
    M = np.arange(9, dtype=float).reshape(3, 3)
    assert cost_metrics(T_A, M)["transport_cost"] == pytest.approx(float((T_A * M).sum()))
    assert partial_mass_metrics(T_A, 0.6)["mass_error"] == pytest.approx(0.0, abs=1e-12)
    assert partial_mass_metrics(T_A, 0.5)["mass_error"] == pytest.approx(0.1)
