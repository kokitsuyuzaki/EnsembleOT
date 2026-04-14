"""Stage 8: weighted aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot.aggregate import (
    ConsensusEdge,
    make_mean_operator,
    make_weighted_mean_operator,
    weighted_consensus_edges,
)
from ensembleot.operator import ImplicitTransportOperator


def _random_operator(rng, n_x=10, n_y=8, K_x=3, K_y=2) -> ImplicitTransportOperator:
    labels_x = rng.integers(0, K_x, size=n_x)
    labels_y = rng.integers(0, K_y, size=n_y)
    for k in range(K_x):
        labels_x[k] = k
    for k in range(K_y):
        labels_y[k] = k
    sizes_x = np.bincount(labels_x, minlength=K_x).astype(float)
    sizes_y = np.bincount(labels_y, minlength=K_y).astype(float)
    T = rng.random((K_x, K_y))
    T /= T.sum()
    return ImplicitTransportOperator(
        labels_x=labels_x, labels_y=labels_y, T_cluster=T,
        cluster_mass_x=sizes_x, cluster_mass_y=sizes_y,
    )


def _suite(n_ops=4, seed=0):
    rng = np.random.default_rng(seed)
    return [_random_operator(rng) for _ in range(n_ops)]


# --------- WeightedMeanTransportOperator ---------

def test_weighted_mean_operator_apply_matches_weighted_dense_average():
    ops = _suite()
    w = np.array([0.1, 0.3, 0.4, 0.2])
    mop = make_weighted_mean_operator(ops, w)
    rng = np.random.default_rng(99)
    F = rng.standard_normal((mop.shape[1], 5))
    implicit = mop.apply_to_features(F)
    dense_mean = sum(wk * op.materialize_dense() for wk, op in zip(w / w.sum(), ops))
    expected = dense_mean @ F
    np.testing.assert_allclose(implicit, expected, atol=1e-10, rtol=1e-10)


def test_weighted_mean_operator_apply_transpose_matches_weighted_dense_average():
    ops = _suite(seed=3)
    w = np.array([2.0, 1.0, 4.0, 3.0])
    mop = make_weighted_mean_operator(ops, w)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((mop.shape[0], 3))
    implicit = mop.apply_transpose_to_features(X)
    wn = w / w.sum()
    dense_mean = sum(wk * op.materialize_dense() for wk, op in zip(wn, ops))
    expected = dense_mean.T @ X
    np.testing.assert_allclose(implicit, expected, atol=1e-10, rtol=1e-10)


def test_weighted_mean_operator_materialize_dense_matches_manual_weighted_average():
    ops = _suite(seed=5)
    w = np.array([1.0, 2.0, 3.0, 4.0])
    mop = make_weighted_mean_operator(ops, w)
    wn = w / w.sum()
    manual = sum(wk * op.materialize_dense() for wk, op in zip(wn, ops))
    np.testing.assert_allclose(mop.materialize_dense(), manual, atol=1e-12)


def test_uniform_weights_matches_unweighted_mean_operator():
    ops = _suite(seed=8)
    mean_op = make_mean_operator(ops)
    w = np.ones(len(ops))
    wmop = make_weighted_mean_operator(ops, w)
    np.testing.assert_allclose(
        wmop.materialize_dense(), mean_op.materialize_dense(), atol=1e-12
    )
    rng = np.random.default_rng(0)
    F = rng.standard_normal((mean_op.shape[1], 3))
    np.testing.assert_allclose(
        wmop.apply_to_features(F), mean_op.apply_to_features(F), atol=1e-12
    )


def test_make_weighted_mean_operator_rejects_invalid_weights():
    rng = np.random.default_rng(0)
    op_a = _random_operator(rng, n_x=10, n_y=8)
    op_b = _random_operator(rng, n_x=10, n_y=8)
    op_mismatch = _random_operator(rng, n_x=10, n_y=9)

    # empty operators
    with pytest.raises(ValueError):
        make_weighted_mean_operator([], [])

    # shape mismatch
    with pytest.raises(ValueError):
        make_weighted_mean_operator([op_a, op_mismatch], [0.5, 0.5])

    # length mismatch
    with pytest.raises(ValueError):
        make_weighted_mean_operator([op_a, op_b], [0.5, 0.3, 0.2])

    # negative weight
    with pytest.raises(ValueError):
        make_weighted_mean_operator([op_a, op_b], [-0.1, 1.0])

    # all zero
    with pytest.raises(ValueError):
        make_weighted_mean_operator([op_a, op_b], [0.0, 0.0])


# --------- weighted_consensus_edges ---------

def test_weighted_consensus_edges_returns_expected_weighted_values():
    ops = _suite(seed=1)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    stack = np.stack([op.materialize_dense() for op in ops], axis=0)
    wn = w / w.sum()
    mean_ref = np.tensordot(wn, stack, axes=1)
    freq_ref = np.tensordot(wn, (stack > 0.0).astype(float), axes=1)

    edges = weighted_consensus_edges(ops, w, threshold=0.0, min_frequency=1.0)
    assert len(edges) > 0
    for e in edges:
        assert isinstance(e, ConsensusEdge)
        assert e.mean_weight == pytest.approx(mean_ref[e.i, e.j])
        assert e.frequency == pytest.approx(freq_ref[e.i, e.j])
        assert e.frequency == pytest.approx(1.0)


def test_weighted_consensus_edges_min_frequency_filters_edges():
    ops = _suite(seed=2)
    w = np.array([0.25, 0.25, 0.25, 0.25])
    stack = np.stack([op.materialize_dense() for op in ops], axis=0)
    thr = float(np.median(stack))
    strict = weighted_consensus_edges(ops, w, threshold=thr, min_frequency=1.0)
    loose = weighted_consensus_edges(ops, w, threshold=thr, min_frequency=0.5)
    assert len(loose) >= len(strict)
    for e in strict:
        assert e.frequency == pytest.approx(1.0)


def test_weighted_consensus_edges_topk_per_source_works():
    ops = _suite(seed=6)
    w = np.array([0.4, 0.1, 0.3, 0.2])
    k = 2
    edges = weighted_consensus_edges(
        ops, w, threshold=-np.inf, min_frequency=0.0, topk_per_source=k
    )
    per_source: dict[int, list[ConsensusEdge]] = {}
    for e in edges:
        per_source.setdefault(e.i, []).append(e)
    stack = np.stack([op.materialize_dense() for op in ops], axis=0)
    wn = w / w.sum()
    mean_ref = np.tensordot(wn, stack, axes=1)
    for i, es in per_source.items():
        assert len(es) <= k
        top_js = set(np.argsort(-mean_ref[i])[:k].tolist())
        assert set(e.j for e in es).issubset(top_js)
