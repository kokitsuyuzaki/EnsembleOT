"""Stage 11: streaming consensus-edge extraction vs dense reference."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot.aggregate import (
    ConsensusEdge,
    _stack_dense,
    consensus_edges,
    weighted_consensus_edges,
)
from ensembleot.operator import ImplicitTransportOperator


def _random_operator(rng, n_x=12, n_y=10, K_x=4, K_y=3) -> ImplicitTransportOperator:
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


def _suite(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [_random_operator(rng) for _ in range(n)]


def _dense_reference_unweighted(operators, threshold, min_frequency, topk_per_source):
    stack = _stack_dense(operators)
    mean = stack.mean(axis=0)
    freq = (stack > threshold).mean(axis=0)
    return _edges_from(mean, freq, min_frequency, topk_per_source)


def _dense_reference_weighted(operators, weights, threshold, min_frequency, topk_per_source):
    stack = _stack_dense(operators)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    mean = np.tensordot(w, stack, axes=1)
    freq = np.tensordot(w, (stack > threshold).astype(np.float64), axes=1)
    return _edges_from(mean, freq, min_frequency, topk_per_source)


def _edges_from(mean, freq, min_frequency, topk_per_source):
    mask = freq >= min_frequency
    out = []
    n_x = mean.shape[0]
    for i in range(n_x):
        js = np.flatnonzero(mask[i])
        if js.size == 0:
            continue
        if topk_per_source is not None and js.size > topk_per_source:
            order = np.argsort(-mean[i, js])[:topk_per_source]
            js = js[order]
        for j in js:
            out.append(ConsensusEdge(
                i=int(i), j=int(j),
                mean_weight=float(mean[i, j]),
                frequency=float(freq[i, j]),
            ))
    return out


def _assert_edges_equal(a, b):
    assert len(a) == len(b)
    key = lambda e: (e.i, e.j)
    for ea, eb in zip(sorted(a, key=key), sorted(b, key=key)):
        assert (ea.i, ea.j) == (eb.i, eb.j)
        assert ea.mean_weight == pytest.approx(eb.mean_weight)
        assert ea.frequency == pytest.approx(eb.frequency)


def test_streaming_consensus_matches_dense_reference():
    ops = _suite(seed=1)
    got = consensus_edges(ops, threshold=0.0, min_frequency=1.0, block_size=3)
    ref = _dense_reference_unweighted(ops, 0.0, 1.0, None)
    _assert_edges_equal(got, ref)


def test_streaming_weighted_consensus_matches_dense_reference():
    ops = _suite(seed=2)
    w = np.array([0.1, 0.4, 0.3, 0.2])
    got = weighted_consensus_edges(ops, w, threshold=0.0, min_frequency=1.0, block_size=4)
    ref = _dense_reference_weighted(ops, w, 0.0, 1.0, None)
    _assert_edges_equal(got, ref)


def test_streaming_consensus_topk_matches_dense_reference():
    ops = _suite(seed=3)
    got = consensus_edges(
        ops, threshold=-np.inf, min_frequency=0.0, topk_per_source=2, block_size=5,
    )
    ref = _dense_reference_unweighted(ops, -np.inf, 0.0, 2)
    _assert_edges_equal(got, ref)

    w = np.array([0.25, 0.25, 0.25, 0.25])
    got_w = weighted_consensus_edges(
        ops, w, threshold=-np.inf, min_frequency=0.0, topk_per_source=2, block_size=5,
    )
    ref_w = _dense_reference_weighted(ops, w, -np.inf, 0.0, 2)
    _assert_edges_equal(got_w, ref_w)


def test_streaming_consensus_min_frequency_matches_dense_reference():
    ops = _suite(seed=4)
    stack = _stack_dense(ops)
    thr = float(np.median(stack))
    for mf in (0.25, 0.5, 0.75, 1.0):
        got = consensus_edges(ops, threshold=thr, min_frequency=mf, block_size=7)
        ref = _dense_reference_unweighted(ops, thr, mf, None)
        _assert_edges_equal(got, ref)


def test_streaming_consensus_block_size_invariant():
    ops = _suite(seed=5)
    w = np.array([0.1, 0.2, 0.3, 0.4])
    base = weighted_consensus_edges(ops, w, threshold=0.0, min_frequency=0.5, block_size=1)
    for bs in (2, 3, 5, 12, None):  # None → default
        other = weighted_consensus_edges(
            ops, w, threshold=0.0, min_frequency=0.5, block_size=bs,
        )
        _assert_edges_equal(base, other)

    # unweighted flavor
    base_u = consensus_edges(ops, threshold=0.0, min_frequency=0.5, block_size=1)
    for bs in (2, 4, 11, None):
        other_u = consensus_edges(ops, threshold=0.0, min_frequency=0.5, block_size=bs)
        _assert_edges_equal(base_u, other_u)
