"""Stage 10: convenience wrappers that glue metrics → weights → aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot import (
    ConsensusEdge,
    compute_run_weights,
    make_metric_weighted_mean_operator,
    make_weighted_mean_operator,
    metric_weighted_consensus_edges,
    weighted_consensus_edges,
)
from ensembleot.operator import ImplicitTransportOperator


def _op(score: float, rng: np.random.Generator) -> ImplicitTransportOperator:
    labels_x = np.array([0, 0, 1, 1, 2])
    labels_y = np.array([0, 1, 1, 0])
    sizes_x = np.array([2.0, 2.0, 1.0])
    sizes_y = np.array([2.0, 2.0])
    T = rng.random((3, 2))
    T /= T.sum()
    op = ImplicitTransportOperator(
        labels_x=labels_x, labels_y=labels_y, T_cluster=T,
        cluster_mass_x=sizes_x, cluster_mass_y=sizes_y,
    )
    op.meta = {"metrics": {"score": float(score)}}
    return op


def _suite(scores):
    rng = np.random.default_rng(0)
    return [_op(s, rng) for s in scores]


def test_make_metric_weighted_mean_operator_matches_manual_pipeline():
    ops = _suite([0.1, 1.0, 10.0, 3.0])
    mop = make_metric_weighted_mean_operator(
        ops, policy="inverse", key="metrics.score",
    )
    # manual
    w = compute_run_weights(ops, policy="inverse", key="metrics.score")
    manual = make_weighted_mean_operator(ops, w)
    np.testing.assert_allclose(mop.weights, manual.weights, atol=1e-12)
    np.testing.assert_allclose(
        mop.materialize_dense(), manual.materialize_dense(), atol=1e-12
    )
    rng = np.random.default_rng(3)
    F = rng.standard_normal((mop.shape[1], 4))
    np.testing.assert_allclose(
        mop.apply_to_features(F), manual.apply_to_features(F), atol=1e-12
    )


def test_metric_weighted_consensus_edges_matches_manual_pipeline():
    ops = _suite([0.1, 1.0, 5.0, 0.5])
    kwargs = dict(policy="softmax_negative", key="metrics.score", temperature=0.5)
    edges = metric_weighted_consensus_edges(
        ops, threshold=0.0, min_frequency=1.0, **kwargs,
    )
    w = compute_run_weights(ops, **kwargs)
    manual = weighted_consensus_edges(ops, w, threshold=0.0, min_frequency=1.0)
    assert len(edges) == len(manual)
    for e, m in zip(edges, manual):
        assert isinstance(e, ConsensusEdge)
        assert (e.i, e.j) == (m.i, m.j)
        assert e.mean_weight == pytest.approx(m.mean_weight)
        assert e.frequency == pytest.approx(m.frequency)


def test_uniform_policy_matches_unweighted_behavior():
    ops = _suite([1.0, 2.0, 3.0])
    mop = make_metric_weighted_mean_operator(ops, policy="uniform")
    np.testing.assert_allclose(mop.weights, np.full(3, 1 / 3))

    # Same as passing a manual uniform weight vector
    manual = make_weighted_mean_operator(ops, np.ones(3))
    np.testing.assert_allclose(
        mop.materialize_dense(), manual.materialize_dense(), atol=1e-12
    )

    edges_conv = metric_weighted_consensus_edges(
        ops, threshold=0.0, policy="uniform", min_frequency=1.0
    )
    edges_manual = weighted_consensus_edges(
        ops, np.ones(3), threshold=0.0, min_frequency=1.0
    )
    assert [(e.i, e.j) for e in edges_conv] == [(e.i, e.j) for e in edges_manual]


def test_missing_key_or_invalid_policy_raises():
    ops = _suite([1.0, 2.0])
    with pytest.raises(ValueError):
        make_metric_weighted_mean_operator(ops, policy="inverse")  # no key
    with pytest.raises(ValueError):
        make_metric_weighted_mean_operator(ops, policy="not_a_policy")
    with pytest.raises(ValueError):
        make_metric_weighted_mean_operator(
            ops, policy="inverse", key="metrics.does_not_exist"
        )
    with pytest.raises(ValueError):
        metric_weighted_consensus_edges(
            ops, threshold=0.0, policy="softmax_negative",
            key="metrics.score", temperature=0.0,
        )
