"""Stage 27: barycenter <-> dataset operators."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ensembleot import (
    ImplicitTransportOperator,
    compute_run_weights,
    make_mean_operator,
    pool_barycenters,
    run_ensemble_barycenter,
    run_ensemble_free_support_barycenter,
    transport_features_to_barycenter,
)


def _toy(K: int = 3, n: int = 60, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)][:K]
    return [rng.standard_normal((n + 5 * k, 2)) + s for k, s in enumerate(shifts)]


def _free(Xs, solver_method="free_support_barycenter", n_runs=1, **kw):
    if solver_method == "free_support_sinkhorn_barycenter":
        kw.setdefault("reg", 1.0)
    return run_ensemble_free_support_barycenter(
        Xs, n_clusters=6, n_support=8, n_runs=n_runs,
        solver_method=solver_method, random_state=0, **kw,
    )


def _fixed(Xs, solver_method="lp_barycenter", n_runs=1, **kw):
    if solver_method == "bregman_barycenter":
        kw.setdefault("reg", 1e-2)
    return run_ensemble_barycenter(
        Xs, n_clusters=10, n_runs=n_runs, solver_method=solver_method,
        random_state=0, **kw,
    )


CASES = [
    (_free, "free_support_barycenter"),
    (_free, "free_support_sinkhorn_barycenter"),
    (_fixed, "lp_barycenter"),
    (_fixed, "bregman_barycenter"),
]


@pytest.mark.parametrize("runner, solver_method", CASES)
@pytest.mark.parametrize("K", [2, 3])
def test_one_operator_per_dataset_with_expected_mass_convention(runner, solver_method, K):
    """Checked against materialize_dense: rows sum to the atom weights,
    columns to 1/n_k, and entries follow T_cluster / (m_x m_y)."""
    Xs = _toy(K)
    res = runner(Xs, solver_method)[0]
    assert len(res.operators) == K
    for k, op in enumerate(res.operators):
        assert isinstance(op, ImplicitTransportOperator)
        assert op.shape == (res.n_atoms, Xs[k].shape[0])
        np.testing.assert_array_equal(op.labels_x, np.arange(res.n_atoms))
        np.testing.assert_array_equal(op.cluster_mass_x, np.ones(res.n_atoms))
        np.testing.assert_array_equal(
            op.cluster_mass_y, np.bincount(op.labels_y, minlength=op.T_cluster.shape[1])
        )
        dense = op.materialize_dense()
        assert dense.shape == op.shape
        assert dense.sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(dense.sum(axis=1), res.weights, atol=1e-6)
        np.testing.assert_allclose(dense.sum(axis=0), 1.0 / Xs[k].shape[0], atol=1e-6)
        i, j = res.n_atoms - 1, 3
        expected = op.T_cluster[i, op.labels_y[j]] / op.cluster_mass_y[op.labels_y[j]]
        assert dense[i, j] == pytest.approx(expected)
        assert op.materialize_entry(i, j) == pytest.approx(expected)

        assert op.meta["role"] == "barycenter_to_dataset"
        assert op.meta["dataset_index"] == k
        assert op.meta["solver_name"] == solver_method
        assert op.meta["metrics"]["marginal_error_row"] < 1e-6
        assert op.meta["metrics"]["marginal_error_col"] < 1e-6
        json.dumps(op.meta)


@pytest.mark.parametrize("runner, solver_method", CASES)
def test_apply_to_features_matches_dense(runner, solver_method):
    Xs = _toy()
    res = runner(Xs, solver_method)[0]
    rng = np.random.default_rng(1)
    for k, op in enumerate(res.operators):
        F = rng.standard_normal((Xs[k].shape[0], 4))
        dense = op.materialize_dense()
        np.testing.assert_allclose(
            op.apply_to_features(F, normalize=False), dense @ F, atol=1e-12
        )
        positive = res.weights > 1e-12
        row_mass = dense.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            op.apply_to_features(F)[positive], (dense @ F)[positive] / row_mass[positive],
            atol=1e-10,
        )
        # a constant feature stays constant on every atom that carries mass
        np.testing.assert_allclose(
            op.apply_to_features(np.ones(Xs[k].shape[0]))[positive], 1.0, atol=1e-10
        )
        G = rng.standard_normal((res.n_atoms, 2))
        np.testing.assert_allclose(
            op.apply_transpose_to_features(G, normalize=False), dense.T @ G, atol=1e-12
        )


def test_free_support_atoms_are_the_average_of_their_projections():
    """Fixed point of the free-support iteration: every atom equals the
    barycentric-weighted average of its projections onto the datasets."""
    Xs = _toy()
    lam = np.array([0.5, 0.3, 0.2])
    res = _free(Xs, weights=lam)[0]
    assert res.meta["metrics"]["displacement_square_norm"] < 1e-7
    projected = sum(
        lam[k] * res.operators[k].apply_to_features(Xs[k]) for k in range(3)
    )
    np.testing.assert_allclose(projected, res.support, atol=1e-3)


def test_fixed_support_lp_zero_weight_atoms_have_zero_rows():
    Xs = _toy()
    res = _fixed(Xs)[0]
    zero = res.weights == 0
    assert zero.any()  # the LP barycenter is sparse on the shared centers
    for k, op in enumerate(res.operators):
        assert np.all(op.T_cluster[zero] == 0)
        out = op.apply_to_features(Xs[k])
        assert np.all(out[zero] == 0)
        assert np.all(np.isfinite(out))


def test_fixed_support_operator_uses_shared_cluster_labels():
    Xs = _toy()
    res = _fixed(Xs)[0]
    for k, op in enumerate(res.operators):
        assert op.T_cluster.shape == (10, 10)
        # every sample sits next to its own shared center
        d = ((Xs[k][:, None, :] - res.support[None, :, :]) ** 2).sum(axis=2)
        assert np.mean(d.argmin(axis=1) == op.labels_y) > 0.9
        # clusters without samples of dataset k receive no mass
        empty = op.cluster_mass_y == 0
        assert np.all(op.T_cluster[:, empty] == 0)


@pytest.mark.parametrize("runner", [_free, _fixed])
def test_transport_features_to_barycenter_aligns_with_pooling(runner):
    Xs = _toy()
    runs = runner(Xs, n_runs=3)
    pooled = pool_barycenters(runs)
    assert pooled.operators == []
    F = np.random.default_rng(2).standard_normal((Xs[1].shape[0], 3))
    out = transport_features_to_barycenter(runs, 1, F)
    assert out.shape == (pooled.n_atoms, 3)
    for r, res in enumerate(runs):
        np.testing.assert_allclose(
            out[pooled.run_index == r], res.operators[1].apply_to_features(F)
        )
    s1 = transport_features_to_barycenter(runs, 1, F, mode="stochastic", random_state=5)
    s2 = transport_features_to_barycenter(runs, 1, F, mode="stochastic", random_state=5)
    np.testing.assert_array_equal(s1, s2)

    with pytest.raises(ValueError, match="dataset_index"):
        transport_features_to_barycenter(runs, 3, F)
    with pytest.raises(ValueError, match="pooled results carry none"):
        transport_features_to_barycenter([pooled], 0, F)
    with pytest.raises(ValueError, match="non-empty"):
        transport_features_to_barycenter([], 0, F)


def test_operators_work_with_existing_weighting_and_fixed_support_aggregation():
    Xs = _toy()
    runs = _fixed(Xs, n_runs=3)
    ops0 = [res.operators[0] for res in runs]
    w = compute_run_weights(ops0, policy="inverse", key="metrics.transport_entropy")
    assert w.sum() == pytest.approx(1.0)
    # operators of different runs share the (n_atoms, n_k) shape, so the
    # existing mean operator accepts them
    mean_op = make_mean_operator(ops0)
    assert mean_op.apply_to_features(Xs[0]).shape == (10, 2)
