"""Stage 9: run-weight policies from per-run metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot import compute_run_weights, extract_metric
from ensembleot.operator import ImplicitTransportOperator


def _op_with_metric(value: float, extra: dict | None = None) -> ImplicitTransportOperator:
    op = ImplicitTransportOperator(
        labels_x=np.array([0, 0, 1]),
        labels_y=np.array([0, 1]),
        T_cluster=np.array([[0.5, 0.0], [0.0, 0.5]]),
        cluster_mass_x=np.array([2.0, 1.0]),
        cluster_mass_y=np.array([1.0, 1.0]),
    )
    meta = {"metrics": {"score": value}}
    if extra:
        meta.update(extra)
    op.meta = meta
    return op


def _suite(values):
    return [_op_with_metric(v) for v in values]


# --------- uniform ---------

def test_uniform_policy_returns_equal_weights():
    ops = _suite([1.0, 2.0, 3.0, 4.0])
    w = compute_run_weights(ops, policy="uniform")
    assert w.shape == (4,)
    np.testing.assert_allclose(w, np.full(4, 0.25))


def test_uniform_policy_matches_manual_vector():
    ops = _suite([0.0, 0.0, 0.0])
    w = compute_run_weights(ops, policy="uniform")
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])


# --------- inverse ---------

def test_inverse_policy_prefers_smaller_metric():
    ops = _suite([0.1, 1.0, 10.0])
    w = compute_run_weights(ops, policy="inverse", key="metrics.score")
    assert w[0] > w[1] > w[2]
    assert w.sum() == pytest.approx(1.0)
    # manual: raw = 1/(v+eps), normalized
    raw = 1.0 / np.array([0.1, 1.0, 10.0])
    np.testing.assert_allclose(w, raw / raw.sum(), atol=1e-6)


# --------- softmax_negative / positive ---------

def test_softmax_negative_prefers_smaller_metric():
    ops = _suite([0.0, 1.0, 2.0])
    w = compute_run_weights(ops, policy="softmax_negative", key="metrics.score",
                            temperature=1.0)
    assert w[0] > w[1] > w[2]
    # ratio matches exp(-Δ)
    assert w[0] / w[1] == pytest.approx(np.exp(1.0), rel=1e-6)


def test_softmax_positive_prefers_larger_metric():
    ops = _suite([0.0, 1.0, 2.0])
    w = compute_run_weights(ops, policy="softmax_positive", key="metrics.score",
                            temperature=1.0)
    assert w[2] > w[1] > w[0]
    assert w[2] / w[1] == pytest.approx(np.exp(1.0), rel=1e-6)


# --------- rank_inverse ---------

def test_rank_inverse_uses_metric_order():
    ops = _suite([5.0, 1.0, 9.0, 3.0])
    w = compute_run_weights(ops, policy="rank_inverse", key="metrics.score")
    # ranks ascending by metric: 1.0→1, 3.0→2, 5.0→3, 9.0→4
    # so inverse ranks per position = [1/3, 1/1, 1/4, 1/2]
    raw = np.array([1 / 3, 1 / 1, 1 / 4, 1 / 2])
    np.testing.assert_allclose(w, raw / raw.sum(), atol=1e-12)
    assert w[1] > w[3] > w[0] > w[2]


# --------- extract_metric ---------

def test_extract_metric_supports_dotted_keys():
    ops = [
        _op_with_metric(0.0, extra={"solver_name": "sinkhorn"}),
        _op_with_metric(1.5),
    ]
    ops[0].meta["metrics"]["nested"] = {"inner": 7.0}
    ops[1].meta["metrics"]["nested"] = {"inner": -3.0}
    vals = extract_metric(ops, "metrics.nested.inner")
    np.testing.assert_allclose(vals, [7.0, -3.0])

    vals2 = extract_metric(ops, "metrics.score")
    np.testing.assert_allclose(vals2, [0.0, 1.5])


# --------- error handling ---------

def test_missing_key_raises_value_error():
    ops = _suite([1.0, 2.0])
    with pytest.raises(ValueError):
        compute_run_weights(ops, policy="inverse", key="metrics.does_not_exist")
    with pytest.raises(ValueError):
        compute_run_weights(ops, policy="inverse")  # key missing
    with pytest.raises(ValueError):
        extract_metric(ops, "metrics.missing")


def test_invalid_temperature_raises():
    ops = _suite([1.0, 2.0])
    with pytest.raises(ValueError):
        compute_run_weights(ops, policy="softmax_negative", key="metrics.score",
                            temperature=0.0)
    with pytest.raises(ValueError):
        compute_run_weights(ops, policy="softmax_positive", key="metrics.score",
                            temperature=-1.0)


def test_empty_operators_raises():
    with pytest.raises(ValueError):
        compute_run_weights([], policy="uniform")


# --------- invariants ---------

@pytest.mark.parametrize(
    "policy,kwargs",
    [
        ("uniform", {}),
        ("inverse", {"key": "metrics.score"}),
        ("softmax_negative", {"key": "metrics.score", "temperature": 0.5}),
        ("softmax_positive", {"key": "metrics.score", "temperature": 2.0}),
        ("rank_inverse", {"key": "metrics.score"}),
    ],
)
def test_weights_sum_to_one_and_are_nonnegative(policy, kwargs):
    ops = _suite([0.1, 0.2, 0.3, 0.4, 0.5])
    w = compute_run_weights(ops, policy=policy, **kwargs)
    assert w.shape == (5,)
    assert np.all(w >= 0.0)
    assert w.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(w))
