"""Stage 2: ImplicitTransportOperator correctness.

We compare the implicit apply / apply_transpose against a freshly
`materialize_dense()`-ed copy over random small problems.
"""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot.operator import ImplicitTransportOperator


def _random_operator(
    rng: np.random.Generator,
    n_x: int,
    n_y: int,
    K_x: int,
    K_y: int,
    mass_mode: str = "cardinality",
) -> ImplicitTransportOperator:
    labels_x = rng.integers(0, K_x, size=n_x)
    labels_y = rng.integers(0, K_y, size=n_y)
    # ensure every cluster is hit, so no zero-mass degeneracies
    for k in range(K_x):
        labels_x[k % n_x] = k
    for k in range(K_y):
        labels_y[k % n_y] = k

    if mass_mode == "cardinality":
        cluster_mass_x = np.bincount(labels_x, minlength=K_x).astype(float)
        cluster_mass_y = np.bincount(labels_y, minlength=K_y).astype(float)
    elif mass_mode == "fractional":
        cluster_mass_x = np.bincount(labels_x, minlength=K_x).astype(float) / n_x
        cluster_mass_y = np.bincount(labels_y, minlength=K_y).astype(float) / n_y
    else:
        raise ValueError(mass_mode)

    T_cluster = rng.random((K_x, K_y))
    # normalize to a coupling of the cluster-level marginals (cosmetic)
    T_cluster *= (cluster_mass_x.sum() + cluster_mass_y.sum()) / (2.0 * T_cluster.sum())

    return ImplicitTransportOperator(
        labels_x=labels_x,
        labels_y=labels_y,
        T_cluster=T_cluster,
        cluster_mass_x=cluster_mass_x,
        cluster_mass_y=cluster_mass_y,
    )


@pytest.mark.parametrize("mass_mode", ["cardinality", "fractional"])
@pytest.mark.parametrize(
    "n_x, n_y, K_x, K_y",
    [(12, 9, 3, 2), (20, 15, 5, 4), (7, 8, 4, 3)],
)
def test_apply_to_features_matches_dense(n_x, n_y, K_x, K_y, mass_mode):
    rng = np.random.default_rng(42)
    op = _random_operator(rng, n_x, n_y, K_x, K_y, mass_mode)
    F = 6
    Y = rng.standard_normal((n_y, F))

    implicit = op.apply_to_features(Y, normalize=False)
    dense = op.materialize_dense() @ Y
    np.testing.assert_allclose(implicit, dense, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("mass_mode", ["cardinality", "fractional"])
@pytest.mark.parametrize(
    "n_x, n_y, K_x, K_y",
    [(12, 9, 3, 2), (20, 15, 5, 4), (7, 8, 4, 3)],
)
def test_apply_to_features_normalized_matches_row_normalized_dense(
    n_x, n_y, K_x, K_y, mass_mode
):
    """Default (normalize=True) == barycentric projection of the dense plan."""
    rng = np.random.default_rng(42)
    op = _random_operator(rng, n_x, n_y, K_x, K_y, mass_mode)
    Y = rng.standard_normal((n_y, 6))

    T = op.materialize_dense()
    rowsum = T.sum(axis=1, keepdims=True)
    bary = (T @ Y) / np.where(rowsum > 0, rowsum, 1.0)

    np.testing.assert_allclose(
        op.apply_to_features(Y), bary, atol=1e-10, rtol=1e-10
    )
    # normalized rows of a barycentric map: each output is a convex
    # combination of target features, so it lies within their range.
    assert op.apply_to_features(Y).max() <= Y.max() + 1e-9
    assert op.apply_to_features(Y).min() >= Y.min() - 1e-9


@pytest.mark.parametrize(
    "n_x, n_y, K_x, K_y",
    [(12, 9, 3, 2), (20, 15, 5, 4)],
)
def test_apply_transpose_matches_dense(n_x, n_y, K_x, K_y):
    rng = np.random.default_rng(7)
    op = _random_operator(rng, n_x, n_y, K_x, K_y)
    X = rng.standard_normal((n_x, 4))
    implicit = op.apply_transpose_to_features(X, normalize=False)
    dense = op.materialize_dense().T @ X
    np.testing.assert_allclose(implicit, dense, atol=1e-10, rtol=1e-10)


def test_1d_feature_input_is_supported():
    rng = np.random.default_rng(0)
    op = _random_operator(rng, 10, 8, 3, 2)
    y = rng.standard_normal(op.n_y)
    out = op.apply_to_features(y, normalize=False)
    assert out.shape == (op.n_x,)
    np.testing.assert_allclose(out, op.materialize_dense() @ y, atol=1e-10)


def test_materialize_entry_matches_formula():
    labels_x = np.array([0, 0, 1, 1, 2])
    labels_y = np.array([0, 1, 1])
    T_cluster = np.array([[0.3, 0.2], [0.1, 0.4], [0.05, 0.15]])
    mx = np.array([2.0, 2.0, 1.0])
    my = np.array([1.0, 2.0])
    op = ImplicitTransportOperator(labels_x, labels_y, T_cluster, mx, my)
    # sample 0 (cluster 0) × sample 2 (cluster 1) = 0.2 / (2 * 2) = 0.05
    assert op.materialize_entry(0, 2) == pytest.approx(0.05)
    # sample 4 (cluster 2) × sample 0 (cluster 0) = 0.05 / (1 * 1) = 0.05
    assert op.materialize_entry(4, 0) == pytest.approx(0.05)
    # entry consistency with dense
    dense = op.materialize_dense()
    for i in range(op.n_x):
        for j in range(op.n_y):
            assert op.materialize_entry(i, j) == pytest.approx(dense[i, j])


def test_shape_property_and_validation():
    T = np.ones((2, 3))
    op = ImplicitTransportOperator(
        labels_x=np.array([0, 1, 1, 0]),
        labels_y=np.array([0, 1, 2]),
        T_cluster=T,
        cluster_mass_x=np.array([2.0, 2.0]),
        cluster_mass_y=np.array([1.0, 1.0, 1.0]),
    )
    assert op.shape == (4, 3)

    with pytest.raises(ValueError):
        ImplicitTransportOperator(
            labels_x=np.array([0, 5]),
            labels_y=np.array([0]),
            T_cluster=np.ones((2, 1)),
            cluster_mass_x=np.array([1.0, 1.0]),
            cluster_mass_y=np.array([1.0]),
        )


def test_does_not_materialize_dense_for_large_problem():
    # Sanity check: apply_to_features must run quickly without touching n_x * n_y memory.
    rng = np.random.default_rng(0)
    n_x, n_y, K_x, K_y = 5000, 4000, 20, 15
    op = _random_operator(rng, n_x, n_y, K_x, K_y)
    Y = rng.standard_normal((n_y, 3))
    out = op.apply_to_features(Y)
    assert out.shape == (n_x, 3)
    # spot-check a single entry against the formula
    i, j = 123, 456
    a, b = op.labels_x[i], op.labels_y[j]
    expected_entry = op.T_cluster[a, b] / (op.cluster_mass_x[a] * op.cluster_mass_y[b])
    assert op.materialize_entry(i, j) == pytest.approx(expected_entry)
