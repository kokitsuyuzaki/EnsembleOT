"""Smoke tests: public API surface + placeholder entry points."""

from __future__ import annotations

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


