"""Thin convenience wrappers: run-metrics → weights → weighted aggregation.

These helpers just glue together the primitives introduced in earlier stages
(`compute_run_weights`, `make_weighted_mean_operator`, `weighted_consensus_edges`).
No new aggregation logic lives here.
"""

from __future__ import annotations

from typing import Sequence

from .aggregate import (
    ConsensusEdge,
    WeightedMeanTransportOperator,
    make_weighted_mean_operator,
    weighted_consensus_edges,
)
from .operator import ImplicitTransportOperator
from .weights import compute_run_weights


def make_metric_weighted_mean_operator(
    operators: Sequence[ImplicitTransportOperator],
    policy: str = "uniform",
    *,
    key: str | None = None,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> WeightedMeanTransportOperator:
    """One-shot: derive run weights from ``op.meta["metrics"]`` and build a
    :class:`WeightedMeanTransportOperator`."""
    weights = compute_run_weights(
        operators, policy=policy, key=key, temperature=temperature, eps=eps
    )
    return make_weighted_mean_operator(operators, weights)


def metric_weighted_consensus_edges(
    operators: Sequence[ImplicitTransportOperator],
    threshold: float,
    *,
    policy: str = "uniform",
    key: str | None = None,
    temperature: float = 1.0,
    eps: float = 1e-12,
    min_frequency: float = 1.0,
    topk_per_source: int | None = None,
) -> list[ConsensusEdge]:
    """One-shot: derive run weights from metrics and extract consensus edges."""
    weights = compute_run_weights(
        operators, policy=policy, key=key, temperature=temperature, eps=eps
    )
    return weighted_consensus_edges(
        operators,
        weights,
        threshold=threshold,
        min_frequency=min_frequency,
        topk_per_source=topk_per_source,
    )
