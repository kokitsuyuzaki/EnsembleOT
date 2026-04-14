"""Run-level weight policies.

Consumes the per-run metrics attached to ``ImplicitTransportOperator.meta``
(see Stage 7) and produces a non-negative run-weight vector summing to 1.
The output is the expected input format of
:func:`ensembleot.make_weighted_mean_operator` /
:func:`ensembleot.weighted_consensus_edges`.

Supported policies
------------------
- ``"uniform"``              — equal weights
- ``"inverse"``              — w_r ∝ 1 / (metric_r + eps)
- ``"softmax_negative"``     — w_r ∝ exp(-metric_r / temperature)
- ``"softmax_positive"``     — w_r ∝ exp( metric_r / temperature)
- ``"rank_inverse"``         — w_r ∝ 1 / rank_r (rank by metric ascending)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .operator import ImplicitTransportOperator

_POLICIES_REQUIRING_KEY = {
    "inverse",
    "softmax_negative",
    "softmax_positive",
    "rank_inverse",
}
_ALL_POLICIES = {"uniform"} | _POLICIES_REQUIRING_KEY


def _follow_dotted(root: dict, path: str) -> Any:
    cursor: Any = root
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise ValueError(f"metric key not found: {path!r}")
        cursor = cursor[part]
    return cursor


def extract_metric(
    operators: Sequence[ImplicitTransportOperator],
    key: str,
) -> np.ndarray:
    """Return the per-run scalar metric identified by a dotted key path.

    ``key`` looks like ``"metrics.transport_entropy"`` and is resolved
    against each operator's ``meta`` dict. A missing key on any run is a
    ``ValueError``.
    """
    values = []
    for idx, op in enumerate(operators):
        try:
            v = _follow_dotted(op.meta, key)
        except ValueError as err:
            raise ValueError(f"operator #{idx}: {err}") from None
        if v is None:
            raise ValueError(f"operator #{idx}: metric {key!r} is None")
        values.append(float(v))
    return np.asarray(values, dtype=np.float64)


def normalize_weights(weights: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Clamp to ``>= 0``, ensure positive total, normalize to sum 1."""
    w = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain NaN or inf")
    w = np.maximum(w, 0.0)
    total = float(w.sum())
    if total <= eps:
        # fall back to uniform rather than emit zeros
        return np.full_like(w, 1.0 / w.size)
    return w / total


def compute_run_weights(
    operators: Sequence[ImplicitTransportOperator],
    policy: str = "uniform",
    *,
    key: str | None = None,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Produce a per-run weight vector according to ``policy``.

    Parameters
    ----------
    operators : sequence of run operators (from ``run_ensemble_sinkhorn`` /
        ``run_ensemble_gw``).
    policy : one of ``"uniform"``, ``"inverse"``, ``"softmax_negative"``,
        ``"softmax_positive"``, ``"rank_inverse"``.
    key : dotted path into ``op.meta`` (e.g. ``"metrics.marginal_error_row"``).
        Required for every non-uniform policy.
    temperature : positive scale for softmax policies.
    eps : numerical floor for inverse / normalization.
    """
    if len(operators) == 0:
        raise ValueError("operators must be non-empty")
    if policy not in _ALL_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; supported: {sorted(_ALL_POLICIES)}")

    n = len(operators)

    if policy == "uniform":
        return np.full(n, 1.0 / n, dtype=np.float64)

    if key is None:
        raise ValueError(f"policy {policy!r} requires `key`")

    metric = extract_metric(operators, key)
    if not np.all(np.isfinite(metric)):
        raise ValueError(f"metric {key!r} contains NaN or inf")

    if policy == "inverse":
        raw = 1.0 / (metric + eps)
    elif policy in ("softmax_negative", "softmax_positive"):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        sign = -1.0 if policy == "softmax_negative" else 1.0
        logits = sign * metric / float(temperature)
        logits = logits - logits.max()  # numerical stability
        raw = np.exp(logits)
    elif policy == "rank_inverse":
        # rank 1 = smallest metric. Ties share averaged rank.
        order = np.argsort(metric, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        # ties: replace each tied group with the mean rank
        uniq, inv = np.unique(metric, return_inverse=True)
        if uniq.size < n:
            means = np.zeros(uniq.size, dtype=np.float64)
            counts = np.zeros(uniq.size, dtype=np.int64)
            for idx in range(n):
                means[inv[idx]] += ranks[idx]
                counts[inv[idx]] += 1
            means /= counts
            ranks = means[inv]
        raw = 1.0 / ranks
    else:
        raise AssertionError("unreachable")

    return normalize_weights(raw, eps=eps)
