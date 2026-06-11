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

    def apply_to_features(self, Y: np.ndarray, normalize: bool = True) -> np.ndarray:
        acc = self.operators[0].apply_to_features(Y, normalize=False)
        for op in self.operators[1:]:
            acc = acc + op.apply_to_features(Y, normalize=False)
        if not normalize:
            return acc / float(self.n_runs)
        ones = np.ones(self.shape[1], dtype=float)
        denom = self.operators[0].apply_to_features(ones, normalize=False)
        for op in self.operators[1:]:
            denom = denom + op.apply_to_features(ones, normalize=False)
        denom = np.where(np.abs(denom) > 1e-30, denom, 1.0)
        return acc / (denom[:, None] if acc.ndim == 2 else denom)

    def apply_transpose_to_features(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        acc = self.operators[0].apply_transpose_to_features(X, normalize=False)
        for op in self.operators[1:]:
            acc = acc + op.apply_transpose_to_features(X, normalize=False)
        if not normalize:
            return acc / float(self.n_runs)
        ones = np.ones(self.shape[0], dtype=float)
        denom = self.operators[0].apply_transpose_to_features(ones, normalize=False)
        for op in self.operators[1:]:
            denom = denom + op.apply_transpose_to_features(ones, normalize=False)
        denom = np.where(np.abs(denom) > 1e-30, denom, 1.0)
        return acc / (denom[:, None] if acc.ndim == 2 else denom)

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

    def apply_to_features(self, Y: np.ndarray, normalize: bool = True) -> np.ndarray:
        w = self.weights
        acc = w[0] * self.operators[0].apply_to_features(Y, normalize=False)
        for k in range(1, len(self.operators)):
            acc = acc + w[k] * self.operators[k].apply_to_features(Y, normalize=False)
        if not normalize:
            return acc
        ones = np.ones(self.shape[1], dtype=float)
        denom = w[0] * self.operators[0].apply_to_features(ones, normalize=False)
        for k in range(1, len(self.operators)):
            denom = denom + w[k] * self.operators[k].apply_to_features(ones, normalize=False)
        denom = np.where(np.abs(denom) > 1e-30, denom, 1.0)
        return acc / (denom[:, None] if acc.ndim == 2 else denom)

    def apply_transpose_to_features(self, X: np.ndarray, normalize: bool = True) -> np.ndarray:
        w = self.weights
        acc = w[0] * self.operators[0].apply_transpose_to_features(X, normalize=False)
        for k in range(1, len(self.operators)):
            acc = acc + w[k] * self.operators[k].apply_transpose_to_features(X, normalize=False)
        if not normalize:
            return acc
        ones = np.ones(self.shape[0], dtype=float)
        denom = w[0] * self.operators[0].apply_transpose_to_features(ones, normalize=False)
        for k in range(1, len(self.operators)):
            denom = denom + w[k] * self.operators[k].apply_transpose_to_features(ones, normalize=False)
        denom = np.where(np.abs(denom) > 1e-30, denom, 1.0)
        return acc / (denom[:, None] if acc.ndim == 2 else denom)

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
    """Reference dense stack (R, n_x, n_y). Used only for tests / debugging.

    The default consensus-edge path no longer calls this; it is kept as a
    correctness reference for the streaming implementation below.
    """
    _validate_shapes(operators)
    return np.stack([op.materialize_dense() for op in operators], axis=0)


def _block_dense_submatrix(
    op: ImplicitTransportOperator,
    i_start: int,
    i_end: int,
) -> np.ndarray:
    """Materialize T[i_start:i_end, :] for a single operator without
    building the full (n_x, n_y) transport matrix.

    Only a ``(block, n_y)`` array is allocated here.
    """
    labels_block = op.labels_x[i_start:i_end]
    T = np.asarray(op.T_cluster, dtype=np.float64)
    mx = np.asarray(op.cluster_mass_x, dtype=np.float64)
    my = np.asarray(op.cluster_mass_y, dtype=np.float64)
    mx_safe = np.where(mx > 0, mx, 1.0)
    my_safe = np.where(my > 0, my, 1.0)
    rows = T[labels_block] / mx_safe[labels_block][:, None]          # (block, K_y)
    block = rows[:, op.labels_y] / my_safe[op.labels_y][None, :]      # (block, n_y)
    return block


def _streaming_consensus_core(
    operators: Sequence[ImplicitTransportOperator],
    weights: np.ndarray,
    threshold: float,
    min_frequency: float,
    topk_per_source: int | None,
    block_size: int | None,
) -> list[ConsensusEdge]:
    """Streaming consensus-edge extraction.

    Works in source-index blocks. Per block we hold only two ``(block, n_y)``
    accumulators (``mean_block``, ``freq_block``). No ``(R, n_x, n_y)`` stack
    and no full ``(n_x, n_y)`` mean matrix is ever allocated.
    """
    _validate_shapes(operators)
    n_x, n_y = operators[0].shape
    if block_size is None:
        block_size = max(1, min(n_x, 256))
    if block_size <= 0:
        raise ValueError("block_size must be >= 1")
    w = np.asarray(weights, dtype=np.float64)

    edges: list[ConsensusEdge] = []
    for start in range(0, n_x, block_size):
        end = min(start + block_size, n_x)
        bsz = end - start
        mean_block = np.zeros((bsz, n_y), dtype=np.float64)
        freq_block = np.zeros((bsz, n_y), dtype=np.float64)
        for r, op in enumerate(operators):
            block = _block_dense_submatrix(op, start, end)           # (bsz, n_y)
            mean_block += w[r] * block
            freq_block += w[r] * (block > threshold).astype(np.float64)

        mask = freq_block >= min_frequency
        for li in range(bsz):
            js = np.flatnonzero(mask[li])
            if js.size == 0:
                continue
            if topk_per_source is not None and js.size > topk_per_source:
                order = np.argsort(-mean_block[li, js])[:topk_per_source]
                js = js[order]
            gi = start + li
            for j in js:
                edges.append(
                    ConsensusEdge(
                        i=int(gi),
                        j=int(j),
                        mean_weight=float(mean_block[li, j]),
                        frequency=float(freq_block[li, j]),
                    )
                )
    return edges


def consensus_edges(
    operators: Sequence[ImplicitTransportOperator],
    threshold: float,
    min_frequency: float = 1.0,
    topk_per_source: int | None = None,
    block_size: int | None = None,
) -> list[ConsensusEdge]:
    """Extract (i, j) edges consistently large across runs (uniform weights).

    Streaming implementation — no ``(R, n_x, n_y)`` or ``(n_x, n_y)`` array
    is ever built. See :func:`weighted_consensus_edges` for the weighted
    version.
    """
    _validate_shapes(operators)
    R = len(operators)
    w = np.full(R, 1.0 / R, dtype=np.float64)
    return _streaming_consensus_core(
        operators, w, threshold, min_frequency, topk_per_source, block_size
    )


def weighted_consensus_edges(
    operators: Sequence[ImplicitTransportOperator],
    weights: np.ndarray | Sequence[float],
    threshold: float,
    min_frequency: float = 1.0,
    topk_per_source: int | None = None,
    block_size: int | None = None,
) -> list[ConsensusEdge]:
    """Streaming weighted counterpart of :func:`consensus_edges`.

    For each (i, j):

        mean_weight = Σ_r w_r · T^{(r)}[i, j]
        frequency   = Σ_r w_r · 1[T^{(r)}[i, j] > threshold]

    ``weights`` is validated/normalized (sum = 1), and the scan is done
    block-by-block over source indices — no ``(R, n_x, n_y)`` stack is
    materialized.
    """
    _validate_shapes(operators)
    w = _normalize_weights(weights, len(operators))
    return _streaming_consensus_core(
        operators, w, threshold, min_frequency, topk_per_source, block_size
    )
