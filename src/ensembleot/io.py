"""Compact save/load for ImplicitTransportOperator collections.

Design notes
------------
* Only the *compact* operator representation is persisted: labels,
  cluster masses, cluster-level T, and a small metadata dict. The
  sample × sample dense transport is **never** written to disk.
* The file format is a single NumPy ``.npz`` archive holding multiple
  operators. Keys use the pattern ``op{idx}_<field>`` so a future
  migration to zarr/hdf5 groups (``operators/{idx}/<field>``) is a
  straightforward rename.
* A version string (`_version`) is stored so incompatible formats can
  be rejected at load time.

Format v1 layout (``FORMAT_VERSION = "ensembleot-operators-v1"``):

    _version        : ()   U-string
    _kind           : ()   U-string   ("list" or "mean")
    _n_operators    : ()   int64
    op{i}_labels_x       : (n_x,) int64
    op{i}_labels_y       : (n_y,) int64
    op{i}_cluster_mass_x : (K_x,) float64
    op{i}_cluster_mass_y : (K_y,) float64
    op{i}_T_cluster      : (K_x, K_y) float64
    op{i}_shape          : (2,)  int64   [n_x, n_y]  (cross-check on load)
    op{i}_meta_json      : ()    U-string (JSON of operator.meta)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .aggregate import MeanTransportOperator
from .barycenter import BarycenterResult
from .operator import ImplicitTransportOperator

FORMAT_VERSION = "ensembleot-operators-v1"
_SUPPORTED_FORMATS = {"npz"}


class StorageFormatError(ValueError):
    """Raised when a saved file is corrupt or version-incompatible."""


def _key(idx: int, field: str) -> str:
    return f"op{idx}_{field}"


def _serialize_operators(
    operators: Sequence[ImplicitTransportOperator],
    kind: str,
) -> dict[str, np.ndarray]:
    if len(operators) == 0:
        raise ValueError("operators must be non-empty")

    arrays: dict[str, np.ndarray] = {
        "_version": np.array(FORMAT_VERSION),
        "_kind": np.array(kind),
        "_n_operators": np.array(len(operators), dtype=np.int64),
    }
    for idx, op in enumerate(operators):
        arrays[_key(idx, "labels_x")] = np.asarray(op.labels_x, dtype=np.int64)
        arrays[_key(idx, "labels_y")] = np.asarray(op.labels_y, dtype=np.int64)
        arrays[_key(idx, "cluster_mass_x")] = np.asarray(op.cluster_mass_x, dtype=np.float64)
        arrays[_key(idx, "cluster_mass_y")] = np.asarray(op.cluster_mass_y, dtype=np.float64)
        arrays[_key(idx, "T_cluster")] = np.asarray(op.T_cluster, dtype=np.float64)
        arrays[_key(idx, "shape")] = np.array([op.n_x, op.n_y], dtype=np.int64)
        arrays[_key(idx, "meta_json")] = np.array(json.dumps(op.meta))
    return arrays


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _read_npz(path: Path) -> np.lib.npyio.NpzFile:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as err:
        raise StorageFormatError(f"cannot open {path!s}: {err}") from err


def _deserialize_operators(npz: np.lib.npyio.NpzFile) -> tuple[list[ImplicitTransportOperator], str]:
    if "_version" not in npz.files:
        raise StorageFormatError("missing _version; not an EnsembleOT operator file")
    version = str(npz["_version"])
    if version != FORMAT_VERSION:
        raise StorageFormatError(
            f"unsupported storage version {version!r}; expected {FORMAT_VERSION!r}"
        )
    if "_n_operators" not in npz.files:
        raise StorageFormatError("missing _n_operators")
    n = int(npz["_n_operators"])
    kind = str(npz["_kind"]) if "_kind" in npz.files else "list"

    required = ("labels_x", "labels_y", "cluster_mass_x", "cluster_mass_y", "T_cluster", "shape")
    ops: list[ImplicitTransportOperator] = []
    shape0: tuple[int, int] | None = None
    for idx in range(n):
        for field_name in required:
            if _key(idx, field_name) not in npz.files:
                raise StorageFormatError(
                    f"missing field {_key(idx, field_name)!r} in operator #{idx}"
                )
        labels_x = npz[_key(idx, "labels_x")]
        labels_y = npz[_key(idx, "labels_y")]
        mx = npz[_key(idx, "cluster_mass_x")]
        my = npz[_key(idx, "cluster_mass_y")]
        T = npz[_key(idx, "T_cluster")]
        shape = tuple(int(v) for v in npz[_key(idx, "shape")])
        if labels_x.shape != (shape[0],) or labels_y.shape != (shape[1],):
            raise StorageFormatError(
                f"operator #{idx}: labels shape inconsistent with stored shape {shape}"
            )
        if shape0 is None:
            shape0 = shape  # type: ignore[assignment]
        elif kind == "mean" and shape != shape0:
            raise StorageFormatError(
                f"mean-operator members disagree on shape: {shape} vs {shape0}"
            )
        meta_key = _key(idx, "meta_json")
        meta: dict = {}
        if meta_key in npz.files:
            try:
                meta = json.loads(str(npz[meta_key]))
            except json.JSONDecodeError as err:
                raise StorageFormatError(f"operator #{idx}: corrupt meta JSON: {err}") from err
        ops.append(
            ImplicitTransportOperator(
                labels_x=labels_x,
                labels_y=labels_y,
                T_cluster=T,
                cluster_mass_x=mx,
                cluster_mass_y=my,
                meta=meta,
            )
        )
    return ops, kind


# ----------------------------- public API -----------------------------

def save_operators(
    path: str | Path,
    operators: Sequence[ImplicitTransportOperator],
    *,
    format: str = "npz",
) -> None:
    """Save a list of operators to disk. Empty list is rejected."""
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format {format!r}; supported: {_SUPPORTED_FORMATS}")
    arrays = _serialize_operators(operators, kind="list")
    _write_npz(Path(path), arrays)


def load_operators(path: str | Path) -> list[ImplicitTransportOperator]:
    """Load a list of operators previously written by :func:`save_operators`."""
    with _read_npz(Path(path)) as npz:
        ops, _kind = _deserialize_operators(npz)
    return ops


def save_mean_operator(
    path: str | Path,
    mean_operator: MeanTransportOperator,
    *,
    format: str = "npz",
) -> None:
    """Save the per-run operators backing a MeanTransportOperator."""
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format {format!r}; supported: {_SUPPORTED_FORMATS}")
    arrays = _serialize_operators(mean_operator.operators, kind="mean")
    _write_npz(Path(path), arrays)


def load_mean_operator(path: str | Path) -> MeanTransportOperator:
    """Load operators and reconstruct a MeanTransportOperator."""
    with _read_npz(Path(path)) as npz:
        ops, kind = _deserialize_operators(npz)
    if kind not in ("mean", "list"):
        raise StorageFormatError(f"unexpected _kind {kind!r}")
    return MeanTransportOperator(ops)


# ----------------------------- barycenters -----------------------------
#
# Barycenter files share FORMAT_VERSION with operator files and are told
# apart by ``_kind == "barycenters"``. They add (no existing key changes):
#
#     _n_barycenters      : ()   int64
#     bc{i}_support       : (n_atoms, d) float64
#     bc{i}_weights       : (n_atoms,)   float64
#     bc{i}_meta_json     : ()   U-string (JSON of result.meta)
#     bc{i}_run_index     : (n_atoms,)   int64   (pooled results only)
#     bc{i}_n_operators   : ()   int64
#     bc{i}_<operator-file key>  — when bc{i}_n_operators > 0, the complete
#         v1 operator layout of result.operators under the ``bc{i}_`` prefix
#         (``bc{i}__n_operators``, ``bc{i}_op{k}_labels_x``, ...).
#
# Operator files have no ``_n_barycenters`` and barycenter files have no
# top-level ``_n_operators``, so each loader rejects the other kind with a
# StorageFormatError.

_BARYCENTER_KIND = "barycenters"


def _bc_key(idx: int, field: str) -> str:
    return f"bc{idx}_{field}"


class _PrefixedNpz:
    """Read-only view of the npz entries under one key prefix, so nested
    operators can be parsed by :func:`_deserialize_operators` unchanged."""

    def __init__(self, npz: np.lib.npyio.NpzFile, prefix: str) -> None:
        self._npz = npz
        self._prefix = prefix
        self.files = [k[len(prefix):] for k in npz.files if k.startswith(prefix)]

    def __getitem__(self, key: str) -> np.ndarray:
        return self._npz[self._prefix + key]


def save_barycenters(
    path: str | Path,
    barycenters: Sequence[BarycenterResult],
    *,
    format: str = "npz",
) -> None:
    """Save a list of :class:`BarycenterResult` (per-run and/or pooled) to a
    single ``.npz``, including each result's barycenter <-> dataset operators
    in their compact form. Empty list is rejected."""
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format {format!r}; supported: {_SUPPORTED_FORMATS}")
    if len(barycenters) == 0:
        raise ValueError("barycenters must be non-empty")

    arrays: dict[str, np.ndarray] = {
        "_version": np.array(FORMAT_VERSION),
        "_kind": np.array(_BARYCENTER_KIND),
        "_n_barycenters": np.array(len(barycenters), dtype=np.int64),
    }
    for idx, res in enumerate(barycenters):
        arrays[_bc_key(idx, "support")] = np.asarray(res.support, dtype=np.float64)
        arrays[_bc_key(idx, "weights")] = np.asarray(res.weights, dtype=np.float64)
        arrays[_bc_key(idx, "meta_json")] = np.array(json.dumps(res.meta))
        if res.run_index is not None:
            arrays[_bc_key(idx, "run_index")] = np.asarray(res.run_index, dtype=np.int64)
        arrays[_bc_key(idx, "n_operators")] = np.array(len(res.operators), dtype=np.int64)
        if res.operators:
            nested = _serialize_operators(res.operators, kind="list")
            arrays.update({_bc_key(idx, key): value for key, value in nested.items()})
    _write_npz(Path(path), arrays)


def load_barycenters(path: str | Path) -> list[BarycenterResult]:
    """Load barycenters previously written by :func:`save_barycenters`."""
    with _read_npz(Path(path)) as npz:
        if "_version" not in npz.files:
            raise StorageFormatError("missing _version; not an EnsembleOT file")
        version = str(npz["_version"])
        if version != FORMAT_VERSION:
            raise StorageFormatError(
                f"unsupported storage version {version!r}; expected {FORMAT_VERSION!r}"
            )
        kind = str(npz["_kind"]) if "_kind" in npz.files else "list"
        if kind != _BARYCENTER_KIND or "_n_barycenters" not in npz.files:
            raise StorageFormatError(
                f"not a barycenter file (_kind={kind!r}); use load_operators / "
                "load_mean_operator for operator files"
            )

        results: list[BarycenterResult] = []
        for idx in range(int(npz["_n_barycenters"])):
            for field_name in ("support", "weights", "n_operators"):
                if _bc_key(idx, field_name) not in npz.files:
                    raise StorageFormatError(
                        f"missing field {_bc_key(idx, field_name)!r} in barycenter #{idx}"
                    )
            meta: dict = {}
            if _bc_key(idx, "meta_json") in npz.files:
                try:
                    meta = json.loads(str(npz[_bc_key(idx, "meta_json")]))
                except json.JSONDecodeError as err:
                    raise StorageFormatError(
                        f"barycenter #{idx}: corrupt meta JSON: {err}"
                    ) from err
            run_index = (
                npz[_bc_key(idx, "run_index")]
                if _bc_key(idx, "run_index") in npz.files
                else None
            )
            n_ops = int(npz[_bc_key(idx, "n_operators")])
            operators: list[ImplicitTransportOperator] = []
            if n_ops > 0:
                operators, _ = _deserialize_operators(_PrefixedNpz(npz, _bc_key(idx, "")))
                if len(operators) != n_ops:
                    raise StorageFormatError(
                        f"barycenter #{idx}: expected {n_ops} operators, "
                        f"found {len(operators)}"
                    )
            try:
                results.append(
                    BarycenterResult(
                        support=npz[_bc_key(idx, "support")],
                        weights=npz[_bc_key(idx, "weights")],
                        meta=meta,
                        run_index=run_index,
                        operators=operators,
                    )
                )
            except ValueError as err:
                raise StorageFormatError(f"barycenter #{idx}: {err}") from err
    return results
