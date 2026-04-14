"""Stage 5: aggregation across runs."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot.aggregate import (
    ConsensusEdge,
    MeanTransportOperator,
    consensus_edges,
    make_mean_operator,
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
        labels_x=labels_x,
        labels_y=labels_y,
        T_cluster=T,
        cluster_mass_x=sizes_x,
        cluster_mass_y=sizes_y,
    )


def _suite(n_ops=4, seed=0):
    rng = np.random.default_rng(seed)
    return [_random_operator(rng) for _ in range(n_ops)]


# --------- MeanTransportOperator ---------

def test_mean_operator_apply_matches_dense_average():
    ops = _suite()
    mean_op = make_mean_operator(ops)
    rng = np.random.default_rng(99)
    F = rng.standard_normal((mean_op.shape[1], 5))
    implicit = mean_op.apply_to_features(F)
    dense_mean = np.mean([op.materialize_dense() for op in ops], axis=0)
    expected = dense_mean @ F
    np.testing.assert_allclose(implicit, expected, atol=1e-10, rtol=1e-10)


def test_mean_operator_apply_transpose_matches_dense_average():
    ops = _suite(seed=3)
    mean_op = make_mean_operator(ops)
    rng = np.random.default_rng(4)
    X = rng.standard_normal((mean_op.shape[0], 4))
    implicit = mean_op.apply_transpose_to_features(X)
    dense_mean = np.mean([op.materialize_dense() for op in ops], axis=0)
    expected = dense_mean.T @ X
    np.testing.assert_allclose(implicit, expected, atol=1e-10, rtol=1e-10)


def test_mean_operator_materialize_dense_matches_manual_average():
    ops = _suite(seed=5)
    mean_op = make_mean_operator(ops)
    manual = np.mean([op.materialize_dense() for op in ops], axis=0)
    np.testing.assert_allclose(mean_op.materialize_dense(), manual, atol=1e-12)


def test_make_mean_operator_rejects_empty_or_shape_mismatch():
    with pytest.raises(ValueError):
        make_mean_operator([])

    rng = np.random.default_rng(0)
    op_a = _random_operator(rng, n_x=10, n_y=8)
    op_b = _random_operator(rng, n_x=10, n_y=9)   # different n_y
    with pytest.raises(ValueError):
        make_mean_operator([op_a, op_b])


# --------- consensus_edges ---------

def test_consensus_edges_returns_expected_fields():
    ops = _suite(seed=1)
    edges = consensus_edges(ops, threshold=0.0, min_frequency=1.0)
    assert len(edges) > 0
    for e in edges:
        assert isinstance(e, ConsensusEdge)
        assert isinstance(e.i, int) and isinstance(e.j, int)
        assert isinstance(e.mean_weight, float)
        assert isinstance(e.frequency, float)
        assert e.frequency == pytest.approx(1.0)  # threshold=0 => all runs pass

    # cross-check mean_weight and frequency against manual computation
    stack = np.stack([op.materialize_dense() for op in ops], axis=0)
    mean = stack.mean(axis=0)
    freq = (stack > 0.0).mean(axis=0)
    for e in edges:
        assert e.mean_weight == pytest.approx(mean[e.i, e.j])
        assert e.frequency == pytest.approx(freq[e.i, e.j])


def test_consensus_edges_min_frequency_filters_edges():
    ops = _suite(seed=2)
    stack = np.stack([op.materialize_dense() for op in ops], axis=0)
    # choose a threshold that is exceeded only sometimes
    thr = float(np.median(stack))
    edges = consensus_edges(ops, threshold=thr, min_frequency=1.0)
    for e in edges:
        # every run must strictly exceed thr for this edge
        assert all(stack[r, e.i, e.j] > thr for r in range(stack.shape[0]))
    # Loosening min_frequency should never return fewer edges
    looser = consensus_edges(ops, threshold=thr, min_frequency=0.5)
    assert len(looser) >= len(edges)


def test_consensus_edges_topk_per_source_works():
    ops = _suite(seed=6)
    n_x, n_y = ops[0].shape
    k = 2
    edges = consensus_edges(
        ops, threshold=-np.inf, min_frequency=0.0, topk_per_source=k
    )
    # Per source at most k entries
    per_source: dict[int, list[ConsensusEdge]] = {}
    for e in edges:
        per_source.setdefault(e.i, []).append(e)
    for i, es in per_source.items():
        assert len(es) <= k
        # Must be the top-k by mean_weight
        mean_row = np.mean([op.materialize_dense()[i] for op in ops], axis=0)
        top_js = set(np.argsort(-mean_row)[:k].tolist())
        assert set(e.j for e in es).issubset(top_js)
