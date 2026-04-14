"""Aggregation across multiple run-level ImplicitTransportOperators.

Stage 5 provides two aggregation primitives:

1. ``MeanTransportOperator`` — an *operator-level* mean. ``apply_to_features``
   and ``apply_transpose_to_features`` never form a dense n_x × n_y matrix:
   they delegate to each underlying operator's implicit apply and average the
   resulting (n_x, F) / (n_y, F) arrays.

2. ``consensus_edges`` — extracts sample-level edges that are consistently
   large across runs. This helper materializes per-run dense transports for
   now (small-scale use); a larger-scale implementation would stream through
   cluster-level blocks. The scan-once design is kept isolated here so it can
   be replaced later without touching the Mean operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .operator import ImplicitTransportOperator


@dataclass(frozen=True)
class ConsensusEdge:
    i: int
    j: int
    mean_weight: float
    frequency: float


@dataclass
class MeanTransportOperator:
    """Implicit average of several ImplicitTransportOperators.

    The underlying operators must share the same ``(n_x, n_y)`` shape.
    No dense transport is ever constructed by the apply routines.
    """

    operators: list[ImplicitTransportOperator]

    def __post_init__(self) -> None:
        if len(self.operators) == 0:
            raise ValueError("MeanTransportOperator requires at least one operator")
        shape0 = self.operators[0].shape
        for op in self.operators[1:]:
            if op.shape != shape0:
                raise ValueError(
                    f"operator shape mismatch: {op.shape} vs {shape0}"
                )

    @property
    def n_runs(self) -> int:
        return len(self.operators)

    @property
    def shape(self) -> tuple[int, int]:
        return self.operators[0].shape

    def apply_to_features(self, Y: np.ndarray) -> np.ndarray:
        acc = self.operators[0].apply_to_features(Y)
        for op in self.operators[1:]:
            acc = acc + op.apply_to_features(Y)
        return acc / float(self.n_runs)

    def apply_transpose_to_features(self, X: np.ndarray) -> np.ndarray:
        acc = self.operators[0].apply_transpose_to_features(X)
        for op in self.operators[1:]:
            acc = acc + op.apply_transpose_to_features(X)
        return acc / float(self.n_runs)

    def materialize_dense(self) -> np.ndarray:
        """Debug/testing only. Do not call on large problems."""
        acc = self.operators[0].materialize_dense()
        for op in self.operators[1:]:
            acc = acc + op.materialize_dense()
        return acc / float(self.n_runs)


def make_mean_operator(
    operators: Sequence[ImplicitTransportOperator],
) -> MeanTransportOperator:
    return MeanTransportOperator(list(operators))


def _validate_shapes(operators: Sequence[ImplicitTransportOperator]) -> None:
    if len(operators) == 0:
        raise ValueError("operators must be non-empty")
    shape0 = operators[0].shape
    for op in operators[1:]:
        if op.shape != shape0:
            raise ValueError(f"operator shape mismatch: {op.shape} vs {shape0}")


def _normalize_weights(
    weights: np.ndarray | Sequence[float],
    n_operators: int,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("weights must be 1-D")
    if w.shape[0] != n_operators:
        raise ValueError(
            f"weights length {w.shape[0]} does not match n_operators {n_operators}"
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights must have a strictly positive total")
    return w / total


@dataclass
class WeightedMeanTransportOperator:
    """Implicit *weighted* average of several ImplicitTransportOperators.

    ``weights`` are validated and normalized (sum = 1) in ``__post_init__``.
    ``apply_to_features`` / ``apply_transpose_to_features`` never build an
    n_x × n_y dense matrix; they delegate to each run's implicit apply and
    combine the resulting (n_x, F) / (n_y, F) arrays with the stored weights.
    """

    operators: list[ImplicitTransportOperator]
    weights: np.ndarray

    def __post_init__(self) -> None:
        _validate_shapes(self.operators)
        self.weights = _normalize_weights(self.weights, len(self.operators))

    @property
    def n_runs(self) -> int:
        return len(self.operators)

    @property
    def shape(self) -> tuple[int, int]:
        return self.operators[0].shape

    def apply_to_features(self, Y: np.ndarray) -> np.ndarray:
        w = self.weights
        acc = w[0] * self.operators[0].apply_to_features(Y)
        for k in range(1, len(self.operators)):
            acc = acc + w[k] * self.operators[k].apply_to_features(Y)
        return acc

    def apply_transpose_to_features(self, X: np.ndarray) -> np.ndarray:
        w = self.weights
        acc = w[0] * self.operators[0].apply_transpose_to_features(X)
        for k in range(1, len(self.operators)):
            acc = acc + w[k] * self.operators[k].apply_transpose_to_features(X)
        return acc

    def materialize_dense(self) -> np.ndarray:
        """Debug/testing only. Do not call on large problems."""
        w = self.weights
        acc = w[0] * self.operators[0].materialize_dense()
        for k in range(1, len(self.operators)):
            acc = acc + w[k] * self.operators[k].materialize_dense()
        return acc


def make_weighted_mean_operator(
    operators: Sequence[ImplicitTransportOperator],
    weights: np.ndarray | Sequence[float],
) -> WeightedMeanTransportOperator:
    return WeightedMeanTransportOperator(list(operators), np.asarray(weights, dtype=np.float64))


def _stack_dense(
    operators: Sequence[ImplicitTransportOperator],
) -> np.ndarray:
    """Materialize per-run dense transports into a (R, n_x, n_y) array.

    Small-scale helper used by ``consensus_edges``. Not intended for large
    problems — a future streamed variant can replace this function alone.
    """
    if len(operators) == 0:
        raise ValueError("operators must be non-empty")
    shape0 = operators[0].shape
    for op in operators[1:]:
        if op.shape != shape0:
            raise ValueError(f"operator shape mismatch: {op.shape} vs {shape0}")
    return np.stack([op.materialize_dense() for op in operators], axis=0)


def consensus_edges(
    operators: Sequence[ImplicitTransportOperator],
    threshold: float,
    min_frequency: float = 1.0,
    topk_per_source: int | None = None,
) -> list[ConsensusEdge]:
    """Extract (i, j) edges that are consistently large across runs.

    For each candidate (i, j):

        mean_weight = mean over runs of T^{(r)}[i, j]
        frequency   = fraction of runs with T^{(r)}[i, j] > threshold

    Keep only edges with ``frequency >= min_frequency``. If
    ``topk_per_source`` is set, keep at most k edges per source i
    (ranked by ``mean_weight`` desc).
    """
    stack = _stack_dense(operators)                     # (R, n_x, n_y)
    R, n_x, n_y = stack.shape

    mean = stack.mean(axis=0)                            # (n_x, n_y)
    freq = (stack > threshold).mean(axis=0)              # (n_x, n_y)

    mask = freq >= min_frequency

    edges: list[ConsensusEdge] = []
    for i in range(n_x):
        js = np.flatnonzero(mask[i])
        if js.size == 0:
            continue
        if topk_per_source is not None and js.size > topk_per_source:
            weights_i = mean[i, js]
            order = np.argsort(-weights_i)[:topk_per_source]
            js = js[order]
        for j in js:
            edges.append(
                ConsensusEdge(
                    i=int(i),
                    j=int(j),
                    mean_weight=float(mean[i, j]),
                    frequency=float(freq[i, j]),
                )
            )
    return edges


def weighted_consensus_edges(
    operators: Sequence[ImplicitTransportOperator],
    weights: np.ndarray | Sequence[float],
    threshold: float,
    min_frequency: float = 1.0,
    topk_per_source: int | None = None,
) -> list[ConsensusEdge]:
    """Weighted counterpart of :func:`consensus_edges`.

    For each (i, j):

        mean_weight = Σ_r w_r · T^{(r)}[i, j]
        frequency   = Σ_r w_r · 1[T^{(r)}[i, j] > threshold]

    where ``w`` is the input ``weights`` vector, normalized to sum to 1.
    """
    _validate_shapes(operators)
    w = _normalize_weights(weights, len(operators))
    stack = _stack_dense(operators)                      # (R, n_x, n_y)
    R, n_x, n_y = stack.shape

    mean = np.tensordot(w, stack, axes=1)                 # (n_x, n_y)
    freq = np.tensordot(w, (stack > threshold).astype(np.float64), axes=1)

    mask = freq >= min_frequency

    edges: list[ConsensusEdge] = []
    for i in range(n_x):
        js = np.flatnonzero(mask[i])
        if js.size == 0:
            continue
        if topk_per_source is not None and js.size > topk_per_source:
            weights_i = mean[i, js]
            order = np.argsort(-weights_i)[:topk_per_source]
            js = js[order]
        for j in js:
            edges.append(
                ConsensusEdge(
                    i=int(i),
                    j=int(j),
                    mean_weight=float(mean[i, j]),
                    frequency=float(freq[i, j]),
                )
            )
    return edges
