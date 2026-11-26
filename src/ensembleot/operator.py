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

# A sample counts as "not transported" when its transport mass is zero up to
# round-off: below ``_MASS_RTOL`` times the largest per-sample mass (exact
# solvers leave residues around 1e-17 where the true mass is 0), or below the
# absolute floor ``_MASS_EPS``.
_MASS_EPS = 1e-30
_MASS_RTOL = 1e-10


def _has_mass(mass: np.ndarray) -> np.ndarray:
    """Boolean mask of entries of ``mass`` that carry real transport mass."""
    mass = np.asarray(mass, dtype=float)
    if mass.size == 0:
        return np.zeros(mass.shape, dtype=bool)
    return mass > max(_MASS_EPS, _MASS_RTOL * float(np.max(mass)))


def _divide_or_nan(num: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """``num / mass`` row-wise, with NaN rows where ``mass`` is (numerically) zero.

    ``mass`` is the per-sample transport mass, shape ``(n,)`` or ``(n, 1)``;
    ``num`` is ``(n,)`` or ``(n, F)``. A sample that receives no transport
    mass has no barycentric image — this happens routinely with partial and
    unbalanced couplings — so its row is NaN rather than a made-up value.
    """
    num = np.asarray(num, dtype=float)
    mass = np.asarray(mass, dtype=float).reshape(-1)
    ok = _has_mass(mass)
    out = np.full(num.shape, np.nan, dtype=float)
    if num.ndim == 1:
        out[ok] = num[ok] / mass[ok]
    else:
        out[ok] = num[ok] / mass[ok][:, None]
    return out


def _stochastic_sample(
    T_plan: np.ndarray,
    labels_out: np.ndarray,
    labels_in: np.ndarray,
    F_in: np.ndarray,
    rng: np.random.Generator,
    out_idx: np.ndarray,
) -> np.ndarray:
    """One stochastic-lifting draw, oriented out <- in. No dense matrix.

    For every requested output sample (``out_idx``) belonging to out-cluster
    ``a``: draw an in-cluster ``b`` with probability ∝ ``T_plan[a, b]``, then
    pick a uniform random member of in-cluster ``b`` and copy its ``F_in``
    row. Unlike the barycentric mean this preserves the within-cluster
    spread / tails of ``F_in``. Output samples whose cluster carries no
    transport mass (possible with partial / unbalanced couplings) get NaN.

    ``T_plan`` is (K_out, K_in); ``F_in`` is (n_in, F) and the result is
    (len(out_idx), F). Only (K_out, K_in) and (m, K_in) temporaries are
    allocated — never (n_out, n_in).
    """
    K_out, K_in = T_plan.shape
    rowsum = T_plan.sum(axis=1, keepdims=True)
    P = np.divide(T_plan, rowsum, out=np.zeros_like(T_plan, dtype=float), where=rowsum > 0)
    cumP = np.cumsum(P, axis=1)
    if K_in:
        cumP[:, -1] = np.where(rowsum[:, 0] > 0, 1.0, cumP[:, -1])

    a = labels_out[out_idx]
    m = a.shape[0]
    u = rng.random(m)
    chosen = (u[:, None] <= cumP[a]).argmax(axis=1)          # (m,) in-cluster per output

    sizes_in = np.bincount(labels_in, minlength=K_in)
    order = np.argsort(labels_in, kind="stable")             # group members by cluster
    starts = np.zeros(K_in, dtype=int)
    if K_in:
        starts[1:] = np.cumsum(sizes_in)[:-1]

    cnt = sizes_in[chosen]
    sizes_out = np.bincount(labels_out, minlength=K_out).astype(float)
    density = np.divide(rowsum[:, 0], sizes_out, out=np.zeros(K_out), where=sizes_out > 0)
    valid = _has_mass(density)[a] & (cnt > 0)                # per-sample mass, as in barycentric
    offsets = np.zeros(m, dtype=int)
    nz = cnt > 0
    offsets[nz] = np.floor(rng.random(int(nz.sum())) * cnt[nz]).astype(int)
    idx = np.clip(starts[chosen] + offsets, 0, max(labels_in.shape[0] - 1, 0))
    targets = order[idx]

    out = F_in[targets].astype(float)                        # (m, F)
    if not valid.all():
        out[~valid] = np.nan                                 # no transport mass: no image
    return out


def _quantile_match(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Monotone (rank-preserving) remap of ``values`` onto the empirical
    distribution of ``reference``, column by column.

    Each output value is replaced by the ``reference`` value at the same
    rank position, so the result has ``reference``'s marginal distribution
    while keeping the rank order of ``values`` (ties broken by position).
    Both inputs are (n, F); the remap is independent per feature column.
    Used by ``mode="matched"`` to restore the target feature distribution
    that the barycentric mean collapses, without changing spatial ranking.
    Rows of ``values`` that are NaN (samples with no transport mass) are left
    NaN and do not take part in the ranking.
    """
    n = values.shape[0]
    if n == 0:
        return values.astype(float)
    out = np.full(values.shape, np.nan, dtype=float)
    for k in range(values.shape[1]):
        col = np.asarray(values[:, k], dtype=float)
        ok = np.isfinite(col)                     # untransported rows stay NaN
        n_ok = int(ok.sum())
        ref = np.asarray(reference[:, k], dtype=float)
        ref_sorted = np.sort(ref[np.isfinite(ref)])
        m = ref_sorted.shape[0]
        if n_ok == 0 or m == 0:
            continue
        span = max(n_ok - 1, 1)
        ranks = np.argsort(np.argsort(col[ok], kind="stable"), kind="stable")
        idx = np.floor(ranks / span * (m - 1)).astype(int) if m > 1 else np.zeros(n_ok, dtype=int)
        out[ok, k] = ref_sorted[idx]
    return out


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

    def _apply_raw(self, Y: np.ndarray) -> np.ndarray:
        """Raw operator action ``T @ Y`` (2-D in, 2-D out). No normalization."""
        K_x, K_y = self.T_cluster.shape
        Y_sum = aggregate_by_label(Y, self.labels_y, K_y, reduce="sum")      # (K_y, F)
        my = self.cluster_mass_y.astype(Y_sum.dtype)
        my_safe = np.where(my > 0, my, 1.0)
        Y_bar = Y_sum / my_safe[:, None]
        cluster_out = self.T_cluster @ Y_bar                                  # (K_x, F)
        sample_out = broadcast_to_samples(cluster_out, self.labels_x)         # (n_x, F)
        mx = self.cluster_mass_x.astype(sample_out.dtype)
        mx_safe = np.where(mx > 0, mx, 1.0)
        return sample_out / mx_safe[self.labels_x][:, None]

    def _apply_transpose_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw transpose action ``T.T @ X`` (2-D in, 2-D out). No normalization."""
        K_x, K_y = self.T_cluster.shape
        X_sum = aggregate_by_label(X, self.labels_x, K_x, reduce="sum")       # (K_x, F)
        mx = self.cluster_mass_x.astype(X_sum.dtype)
        mx_safe = np.where(mx > 0, mx, 1.0)
        X_bar = X_sum / mx_safe[:, None]
        cluster_out = self.T_cluster.T @ X_bar                                # (K_y, F)
        sample_out = broadcast_to_samples(cluster_out, self.labels_y)         # (n_y, F)
        my = self.cluster_mass_y.astype(sample_out.dtype)
        my_safe = np.where(my > 0, my, 1.0)
        return sample_out / my_safe[self.labels_y][:, None]

    def _stochastic_subset(
        self, Y: np.ndarray, rng: np.random.Generator, src_idx: np.ndarray
    ) -> np.ndarray:
        """Stochastic transport of ``Y`` for the source samples ``src_idx`` (2-D)."""
        return _stochastic_sample(
            np.asarray(self.T_cluster, dtype=float),
            self.labels_x, self.labels_y, Y, rng, src_idx,
        )

    def _stochastic_transpose_subset(
        self, X: np.ndarray, rng: np.random.Generator, dst_idx: np.ndarray
    ) -> np.ndarray:
        """Stochastic transpose-transport of ``X`` for target samples ``dst_idx`` (2-D)."""
        return _stochastic_sample(
            np.asarray(self.T_cluster, dtype=float).T,
            self.labels_y, self.labels_x, X, rng, dst_idx,
        )

    def apply_to_features(
        self,
        Y: np.ndarray,
        normalize: bool = True,
        mode: str = "barycentric",
        random_state: int | None = None,
    ) -> np.ndarray:
        """Transport target features to the source side.

        Two ``mode`` choices:

        * ``"barycentric"`` (default) — deterministic. Returns, per source
          sample ``i`` in cluster ``a``,

              (1/m_x[a]) [T_cluster @ Ȳ]_{a,:},   Ȳ[b,:]=(1/m_y[b]) Σ_{j∈D_b} Y[j,:]

          When ``normalize=True`` (default) each row is divided by its
          transport mass Σ_j T[i,j], giving a proper weighted average of
          the target features (the barycentric projection / conditional
          mean E[Y | x_i]) on the same scale as ``Y``. Source samples with
          no transport mass (see :meth:`transported_mass_x`; possible with
          partial / unbalanced couplings) have no image and come back as
          NaN. ``normalize=False`` returns the raw linear operator action
          ``T @ Y`` (rows summing to ``1/n_x`` for a balanced coupling).

        * ``"stochastic"`` — for each source sample draw a target cluster
          ∝ ``T_cluster[a,·]`` and copy a random member's feature row. This
          preserves the within-cluster spread / tails of ``Y`` that the
          barycentric mean averages away. ``random_state`` seeds the draw;
          ``normalize`` is ignored.

        * ``"matched"`` — deterministic. Compute the barycentric output,
          then remap each feature column onto the empirical distribution of
          the corresponding ``Y`` column (rank-preserving quantile match).
          This keeps the barycentric spatial ranking (so any Spearman-type
          score is unchanged) while restoring the feature distribution the
          mean collapses. ``normalize`` / ``random_state`` are ignored.
        """
        Y = np.asarray(Y)
        squeeze = False
        if Y.ndim == 1:
            Y = Y[:, None]
            squeeze = True
        if mode == "barycentric":
            out = self._apply_raw(Y)
            if normalize:
                denom = self._apply_raw(np.ones((self.n_y, 1), dtype=float))       # (n_x, 1)
                out = _divide_or_nan(out, denom)
        elif mode == "stochastic":
            rng = np.random.default_rng(random_state)
            out = self._stochastic_subset(Y, rng, np.arange(self.n_x))
        elif mode == "matched":
            out = self.apply_to_features(Y, normalize=True, mode="barycentric")
            out = out[:, None] if out.ndim == 1 else out
            out = _quantile_match(out, Y)
        else:
            raise ValueError(
                f"unknown mode {mode!r} (use 'barycentric', 'stochastic' or 'matched')"
            )
        return out[:, 0] if squeeze else out

    def apply_transpose_to_features(
        self,
        X: np.ndarray,
        normalize: bool = True,
        mode: str = "barycentric",
        random_state: int | None = None,
    ) -> np.ndarray:
        """Transport source features to the target side.

        Symmetric to :meth:`apply_to_features`. ``mode="barycentric"``
        (default) returns the (optionally row-normalized) ``T.T @ X``;
        ``mode="stochastic"`` draws, per target sample, a source cluster
        ∝ ``T_cluster[·,b]`` and copies a random member's feature row,
        preserving the within-cluster spread of ``X``; ``mode="matched"``
        remaps the barycentric output onto each ``X`` column's empirical
        distribution (rank-preserving), restoring the feature distribution
        while keeping the spatial ranking.
        """
        X = np.asarray(X)
        squeeze = False
        if X.ndim == 1:
            X = X[:, None]
            squeeze = True
        if mode == "barycentric":
            out = self._apply_transpose_raw(X)
            if normalize:
                denom = self._apply_transpose_raw(np.ones((self.n_x, 1), dtype=float))      # (n_y, 1)
                out = _divide_or_nan(out, denom)
        elif mode == "stochastic":
            rng = np.random.default_rng(random_state)
            out = self._stochastic_transpose_subset(X, rng, np.arange(self.n_y))
        elif mode == "matched":
            out = self.apply_transpose_to_features(X, normalize=True, mode="barycentric")
            out = out[:, None] if out.ndim == 1 else out
            out = _quantile_match(out, X)
        else:
            raise ValueError(
                f"unknown mode {mode!r} (use 'barycentric', 'stochastic' or 'matched')"
            )
        return out[:, 0] if squeeze else out

    def transported_mass_x(self, relative: bool = False) -> np.ndarray:
        """Per-source-sample transport mass ``Σ_j T[i, j]``, shape ``(n_x,)``.

        For a balanced coupling every entry equals ``1/n_x``. Partial and
        unbalanced couplings move less (or, for unbalanced, possibly more)
        mass per sample; a value of 0 means the sample is left unmatched.
        ``relative=True`` rescales by ``n_x`` so that 1.0 means "all of this
        sample's original mass is transported". Computed from the cluster
        row sums — no dense matrix.
        """
        mx = np.asarray(self.cluster_mass_x, dtype=float)
        row = np.asarray(self.T_cluster, dtype=float).sum(axis=1)
        per_cluster = np.divide(row, mx, out=np.zeros_like(row), where=mx > 0)
        out = per_cluster[self.labels_x]
        return out * self.n_x if relative else out

    def transported_mass_y(self, relative: bool = False) -> np.ndarray:
        """Per-target-sample transport mass ``Σ_i T[i, j]``, shape ``(n_y,)``.

        Symmetric to :meth:`transported_mass_x`.
        """
        my = np.asarray(self.cluster_mass_y, dtype=float)
        col = np.asarray(self.T_cluster, dtype=float).sum(axis=0)
        per_cluster = np.divide(col, my, out=np.zeros_like(col), where=my > 0)
        out = per_cluster[self.labels_y]
        return out * self.n_y if relative else out

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
