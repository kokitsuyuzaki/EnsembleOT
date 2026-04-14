"""Smoke tests: public API surface + placeholder entry points."""

from __future__ import annotations

import numpy as np
import pytest

import ensembleot as eot


def test_public_api_surface():
    assert callable(eot.run_ensemble_sinkhorn)
    assert callable(eot.run_ensemble_gw)
    for name in (
        "ImplicitTransportOperator",
        "ClusteringConfig",
        "SinkhornConfig",
        "GWConfig",
        "EnsembleConfig",
        "TrialResult",
        "EnsembleResult",
    ):
        assert hasattr(eot, name)


def test_stage1_entry_points_are_placeholders():
    with pytest.raises(NotImplementedError):
        eot.run_ensemble_sinkhorn(
            X=np.zeros((2, 2)),
            Y=np.zeros((2, 2)),
            clustering=eot.ClusteringConfig(n_clusters_x=1, n_clusters_y=1),
            sinkhorn=eot.SinkhornConfig(),
            ensemble=eot.EnsembleConfig(n_trials=1),
        )
    with pytest.raises(NotImplementedError):
        eot.run_ensemble_gw(
            X=np.zeros((2, 2)),
            Y=np.zeros((2, 2)),
            clustering=eot.ClusteringConfig(n_clusters_x=1, n_clusters_y=1),
            gw=eot.GWConfig(),
            ensemble=eot.EnsembleConfig(n_trials=1),
        )
