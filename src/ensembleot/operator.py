"""Implicit transport operator with uniform lifting.

A sample-level transport plan is represented as

    T = A_x @ T_cluster @ A_y.T

where A_x ∈ R^{n_x × K_x}, A_y ∈ R^{n_y × K_y} are the (implicit) cluster
assignment matrices and T_cluster ∈ R^{K_x × K_y} is the cluster-level
transport. A_x, A_y are *never* materialized — only the label vectors
`labels_x`, `labels_y` are stored.

Uniform lifting. For a sample i in cluster a and j in cluster b:

    T[i, j] = T_cluster[a, b] / (m_x[a] * m_y[b])

`cluster_mass_x`, `cluster_mass_y` are **per-cluster normalization factors**
used in the lifting — not specifically cluster cardinalities. The caller
chooses the semantics:

  * cardinality  (m_a = |C_a|)        — matches the Stage-1 uniform-lifting spec
                                        and is what the current Sinkhorn
                                        pipeline supplies.
  * sample mass  (m_a = |C_a| / n_x)  — also supported; the operator is
                                        agnostic and simply divides by
                                        whatever factors the caller stores.

What the operator requires is only that the same normalizer appears in
`T[i,j] = T_cluster[a,b] / (m_x[a] m_y[b])` consistently across
`materialize_entry`, `materialize_dense`, and the apply routines — which
it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lifting import aggregate_by_label, broadcast_to_samples


@dataclass
class ImplicitTransportOperator:
    labels_x: np.ndarray          # (n_x,) int
    labels_y: np.ndarray          # (n_y,) int
    T_cluster: np.ndarray         # (K_x, K_y)
    cluster_mass_x: np.ndarray    # (K_x,)
    cluster_mass_y: np.ndarray    # (K_y,)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        K_x, K_y = self.T_cluster.shape
        if self.cluster_mass_x.shape != (K_x,):
            raise ValueError("cluster_mass_x shape mismatch")
        if self.cluster_mass_y.shape != (K_y,):
            raise ValueError("cluster_mass_y shape mismatch")
        if self.labels_x.ndim != 1 or self.labels_y.ndim != 1:
            raise ValueError("labels must be 1-D")
        if self.labels_x.size and (self.labels_x.max() >= K_x or self.labels_x.min() < 0):
            raise ValueError("labels_x out of range")
        if self.labels_y.size and (self.labels_y.max() >= K_y or self.labels_y.min() < 0):
            raise ValueError("labels_y out of range")

    @property
    def n_x(self) -> int:
        return int(self.labels_x.shape[0])

    @property
    def n_y(self) -> int:
        return int(self.labels_y.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_x, self.n_y)

    def apply_to_features(self, Y: np.ndarray) -> np.ndarray:
        """Compute T @ Y without forming the full n_x × n_y matrix.

        Derivation. With T[i,j] = T_cluster[a,b] / (m_x[a] m_y[b]),

            (T Y)[i, :] = Σ_j T[i,j] Y[j,:]
                        = (1/m_x[a]) Σ_b T_cluster[a,b] (1/m_y[b]) Σ_{j∈D_b} Y[j,:]
                        = (1/m_x[a]) [T_cluster @ Ȳ]_{a,:}

        where Ȳ[b,:] = (1/m_y[b]) Σ_{j∈D_b} Y[j,:]. So:

          1. aggregate Y by target-cluster (sum → divide by m_y)
          2. multiply by T_cluster
          3. broadcast to samples via labels_x and divide by m_x
        """
        Y = np.asarray(Y)
        squeeze = False
        if Y.ndim == 1:
            Y = Y[:, None]
            squeeze = True
        K_x, K_y = self.T_cluster.shape
        Y_sum = aggregate_by_label(Y, self.labels_y, K_y, reduce="sum")      # (K_y, F)
        my = self.cluster_mass_y.astype(Y_sum.dtype)
        my_safe = np.where(my > 0, my, 1.0)
        Y_bar = Y_sum / my_safe[:, None]
        cluster_out = self.T_cluster @ Y_bar                                  # (K_x, F)
        sample_out = broadcast_to_samples(cluster_out, self.labels_x)         # (n_x, F)
        mx = self.cluster_mass_x.astype(sample_out.dtype)
        mx_safe = np.where(mx > 0, mx, 1.0)
        out = sample_out / mx_safe[self.labels_x][:, None]
        return out[:, 0] if squeeze else out

    def apply_transpose_to_features(self, X: np.ndarray) -> np.ndarray:
        """Compute T.T @ X without forming the full matrix. Symmetric to apply_to_features."""
        X = np.asarray(X)
        squeeze = False
        if X.ndim == 1:
            X = X[:, None]
            squeeze = True
        K_x, K_y = self.T_cluster.shape
        X_sum = aggregate_by_label(X, self.labels_x, K_x, reduce="sum")       # (K_x, F)
        mx = self.cluster_mass_x.astype(X_sum.dtype)
        mx_safe = np.where(mx > 0, mx, 1.0)
        X_bar = X_sum / mx_safe[:, None]
        cluster_out = self.T_cluster.T @ X_bar                                # (K_y, F)
        sample_out = broadcast_to_samples(cluster_out, self.labels_y)         # (n_y, F)
        my = self.cluster_mass_y.astype(sample_out.dtype)
        my_safe = np.where(my > 0, my, 1.0)
        out = sample_out / my_safe[self.labels_y][:, None]
        return out[:, 0] if squeeze else out

    def materialize_entry(self, i: int, j: int) -> float:
        a = int(self.labels_x[i])
        b = int(self.labels_y[j])
        denom = float(self.cluster_mass_x[a]) * float(self.cluster_mass_y[b])
        if denom == 0.0:
            return 0.0
        return float(self.T_cluster[a, b]) / denom

    def materialize_dense(self) -> np.ndarray:
        """Debug/testing only. Do not call on large problems."""
        mx = self.cluster_mass_x.astype(self.T_cluster.dtype)
        my = self.cluster_mass_y.astype(self.T_cluster.dtype)
        mx_safe = np.where(mx > 0, mx, 1.0)
        my_safe = np.where(my > 0, my, 1.0)
        scaled = self.T_cluster / (mx_safe[:, None] * my_safe[None, :])
        return scaled[self.labels_x][:, self.labels_y]
