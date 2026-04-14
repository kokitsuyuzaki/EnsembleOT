"""Stage 6: compact operator save/load."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot import (
    FORMAT_VERSION,
    MeanTransportOperator,
    StorageFormatError,
    load_mean_operator,
    load_operators,
    make_mean_operator,
    save_mean_operator,
    save_operators,
)
from ensembleot.operator import ImplicitTransportOperator


def _random_operator(rng, n_x=10, n_y=8, K_x=3, K_y=2, meta=None) -> ImplicitTransportOperator:
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
        labels_x=labels_x,
        labels_y=labels_y,
        T_cluster=T,
        cluster_mass_x=sizes_x,
        cluster_mass_y=sizes_y,
        meta=meta or {},
    )


def test_save_load_single_operator_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    op = _random_operator(rng, meta={"solver": "sinkhorn", "reg": 0.1, "seed": 7})

    path = tmp_path / "ops.npz"
    save_operators(path, [op])
    loaded = load_operators(path)

    assert len(loaded) == 1
    lo = loaded[0]
    np.testing.assert_array_equal(lo.labels_x, op.labels_x)
    np.testing.assert_array_equal(lo.labels_y, op.labels_y)
    np.testing.assert_allclose(lo.cluster_mass_x, op.cluster_mass_x)
    np.testing.assert_allclose(lo.cluster_mass_y, op.cluster_mass_y)
    np.testing.assert_allclose(lo.T_cluster, op.T_cluster)
    assert lo.meta == {"solver": "sinkhorn", "reg": 0.1, "seed": 7}
    np.testing.assert_allclose(lo.materialize_dense(), op.materialize_dense())


def test_save_load_multiple_operators_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    ops = [_random_operator(rng) for _ in range(4)]
    path = tmp_path / "ensemble.npz"
    save_operators(path, ops)
    loaded = load_operators(path)
    assert len(loaded) == len(ops)

    rng2 = np.random.default_rng(123)
    F = rng2.standard_normal((ops[0].n_y, 5))
    for orig, lo in zip(ops, loaded):
        np.testing.assert_allclose(lo.apply_to_features(F), orig.apply_to_features(F),
                                   atol=1e-12)


def test_save_load_preserves_shape_and_entries(tmp_path):
    rng = np.random.default_rng(2)
    op = _random_operator(rng)
    path = tmp_path / "a.npz"
    save_operators(path, [op])
    lo = load_operators(path)[0]
    assert lo.shape == op.shape
    for i in range(op.n_x):
        for j in range(op.n_y):
            assert lo.materialize_entry(i, j) == pytest.approx(op.materialize_entry(i, j))


def test_save_rejects_empty_operator_list(tmp_path):
    with pytest.raises(ValueError):
        save_operators(tmp_path / "empty.npz", [])


def test_load_rejects_invalid_or_incomplete_file(tmp_path):
    # (a) completely unrelated file
    bogus = tmp_path / "bogus.npz"
    np.savez(bogus, some_array=np.arange(3))
    with pytest.raises(StorageFormatError):
        load_operators(bogus)

    # (b) wrong version string
    wrong_ver = tmp_path / "wrong_ver.npz"
    np.savez(wrong_ver, _version=np.array("ensembleot-operators-v999"),
             _n_operators=np.array(0, dtype=np.int64))
    with pytest.raises(StorageFormatError):
        load_operators(wrong_ver)

    # (c) version OK but a required field is missing
    rng = np.random.default_rng(0)
    op = _random_operator(rng)
    path = tmp_path / "partial.npz"
    save_operators(path, [op])
    with np.load(path) as data:
        payload = {k: data[k] for k in data.files if k != "op0_T_cluster"}
    np.savez(path, **payload)
    with pytest.raises(StorageFormatError):
        load_operators(path)

    # (d) non-existent path
    with pytest.raises(StorageFormatError):
        load_operators(tmp_path / "does_not_exist.npz")


def test_mean_operator_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    ops = [_random_operator(rng) for _ in range(3)]
    mean_op = make_mean_operator(ops)

    path = tmp_path / "mean.npz"
    save_mean_operator(path, mean_op)
    loaded = load_mean_operator(path)
    assert isinstance(loaded, MeanTransportOperator)
    assert loaded.n_runs == mean_op.n_runs
    assert loaded.shape == mean_op.shape

    rng2 = np.random.default_rng(11)
    F = rng2.standard_normal((mean_op.shape[1], 4))
    np.testing.assert_allclose(
        loaded.apply_to_features(F),
        mean_op.apply_to_features(F),
        atol=1e-12,
    )


def test_format_version_is_exported():
    assert isinstance(FORMAT_VERSION, str)
    assert FORMAT_VERSION.startswith("ensembleot-operators-")
