"""Stage 27: persistence of BarycenterResult collections."""

from __future__ import annotations

import numpy as np
import pytest

from ensembleot import (
    FORMAT_VERSION,
    BarycenterResult,
    StorageFormatError,
    load_barycenters,
    load_mean_operator,
    load_operators,
    pool_barycenters,
    run_ensemble_barycenter,
    run_ensemble_free_support_barycenter,
    run_ensemble_sinkhorn,
    save_barycenters,
    save_operators,
)


def _toy(K: int = 3, n: int = 60, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shifts = [(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)][:K]
    return [rng.standard_normal((n + 5 * k, 2)) + s for k, s in enumerate(shifts)]


def _assert_same(a: BarycenterResult, b: BarycenterResult) -> None:
    np.testing.assert_array_equal(a.support, b.support)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.meta == b.meta
    if a.run_index is None:
        assert b.run_index is None
    else:
        np.testing.assert_array_equal(a.run_index, b.run_index)
    assert len(a.operators) == len(b.operators)
    for oa, ob in zip(a.operators, b.operators):
        np.testing.assert_array_equal(oa.labels_x, ob.labels_x)
        np.testing.assert_array_equal(oa.labels_y, ob.labels_y)
        np.testing.assert_array_equal(oa.T_cluster, ob.T_cluster)
        np.testing.assert_array_equal(oa.cluster_mass_x, ob.cluster_mass_x)
        np.testing.assert_array_equal(oa.cluster_mass_y, ob.cluster_mass_y)
        assert oa.meta == ob.meta


def test_format_version_is_unchanged():
    assert FORMAT_VERSION == "ensembleot-operators-v1"


@pytest.mark.parametrize("variant", ["free", "fixed"])
def test_roundtrip_runs(tmp_path, variant):
    Xs = _toy()
    if variant == "free":
        runs = run_ensemble_free_support_barycenter(
            Xs, n_clusters=6, n_support=8, n_runs=3, random_state=0
        )
    else:
        runs = run_ensemble_barycenter(Xs, n_clusters=10, n_runs=3, random_state=0)
    path = tmp_path / "sub" / "bary.npz"
    save_barycenters(path, runs)
    loaded = load_barycenters(path)
    assert len(loaded) == 3
    for orig, lo in zip(runs, loaded):
        _assert_same(orig, lo)
        F = np.random.default_rng(0).standard_normal((Xs[2].shape[0], 3))
        np.testing.assert_array_equal(
            lo.operators[2].apply_to_features(F), orig.operators[2].apply_to_features(F)
        )


def test_roundtrip_pooled_and_mixed(tmp_path):
    runs = run_ensemble_free_support_barycenter(
        _toy(K=2), n_clusters=6, n_support=8, n_runs=11, random_state=0
    )
    pooled = pool_barycenters(runs)
    path = tmp_path / "mixed.npz"
    save_barycenters(path, [pooled, *runs])  # > 10 entries: bc1_ vs bc10_ prefixes
    loaded = load_barycenters(path)
    assert len(loaded) == 12
    _assert_same(pooled, loaded[0])
    assert loaded[0].operators == []
    np.testing.assert_array_equal(loaded[0].run_index, pooled.run_index)
    for orig, lo in zip(runs, loaded[1:]):
        _assert_same(orig, lo)


def test_no_dense_sample_matrix_is_written(tmp_path):
    Xs = _toy()
    runs = run_ensemble_free_support_barycenter(
        Xs, n_clusters=6, n_support=8, n_runs=1, random_state=0
    )
    path = tmp_path / "b.npz"
    save_barycenters(path, runs)
    with np.load(path) as npz:
        assert str(npz["_version"]) == FORMAT_VERSION
        assert str(npz["_kind"]) == "barycenters"
        n_max = max(X.shape[0] for X in Xs)
        for key in npz.files:
            assert npz[key].size <= max(n_max, 8 * 6, 8 * 2), key


def test_save_validation(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        save_barycenters(tmp_path / "x.npz", [])
    res = BarycenterResult(support=np.zeros((2, 2)), weights=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="unsupported format"):
        save_barycenters(tmp_path / "x.zarr", [res], format="zarr")


def test_kinds_are_not_interchangeable(tmp_path):
    """Operator loaders reject barycenter files and vice versa."""
    runs = run_ensemble_barycenter(_toy(), n_clusters=10, n_runs=1, random_state=0)
    bary_path = tmp_path / "bary.npz"
    save_barycenters(bary_path, runs)
    with pytest.raises(StorageFormatError):
        load_operators(bary_path)
    with pytest.raises(StorageFormatError):
        load_mean_operator(bary_path)

    X, Y = _toy(K=2)
    ops = run_ensemble_sinkhorn(X, Y, n_clusters_x=4, n_clusters_y=4, n_runs=2, random_state=0)
    ops_path = tmp_path / "ops.npz"
    save_operators(ops_path, ops)
    assert len(load_operators(ops_path)) == 2  # existing files still load
    with pytest.raises(StorageFormatError, match="not a barycenter file"):
        load_barycenters(ops_path)


def _resave(src, dst, mutate) -> None:
    with np.load(src) as npz:
        arrays = {k: npz[k] for k in npz.files}
    mutate(arrays)
    np.savez(dst, **arrays)


def test_corrupt_files_raise_storage_format_error(tmp_path):
    runs = run_ensemble_barycenter(_toy(), n_clusters=10, n_runs=1, random_state=0)
    good = tmp_path / "good.npz"
    save_barycenters(good, runs)

    def bad(name, mutate, match):
        path = tmp_path / f"{name}.npz"
        _resave(good, path, mutate)
        with pytest.raises(StorageFormatError, match=match):
            load_barycenters(path)

    bad("version", lambda a: a.update(_version=np.array("ensembleot-operators-v0")),
        "unsupported storage version")
    bad("noversion", lambda a: a.pop("_version"), "missing _version")
    bad("nosupport", lambda a: a.pop("bc0_support"), "missing field 'bc0_support'")
    bad("noop", lambda a: a.pop("bc0_op1_T_cluster"), "missing field 'op1_T_cluster'")
    bad("meta", lambda a: a.update(bc0_meta_json=np.array("{not json")), "corrupt meta JSON")
    bad("weights", lambda a: a.update(bc0_weights=np.ones(3) / 3), "weights shape mismatch")
    bad("count", lambda a: a.update(bc0_n_operators=np.array(5)), "expected 5 operators")

    not_npz = tmp_path / "garbage.npz"
    not_npz.write_bytes(b"this is not an npz archive")
    with pytest.raises(StorageFormatError):
        load_barycenters(not_npz)
