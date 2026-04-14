"""Gromov–Wasserstein ensemble OT entry point.

Stage 1: skeleton only. POT-backed GW solver lands in Stage 3.
"""

from __future__ import annotations

import numpy as np

from .config import ClusteringConfig, EnsembleConfig, GWConfig
from .storage import EnsembleResult


def run_ensemble_gw(
    X: np.ndarray,
    Y: np.ndarray,
    clustering: ClusteringConfig,
    gw: GWConfig,
    ensemble: EnsembleConfig,
    p: np.ndarray | None = None,
    q: np.ndarray | None = None,
) -> EnsembleResult:
    """Run an ensemble of cluster-level (entropic) Gromov–Wasserstein trials.

    X and Y may live in different feature spaces; intra-space cost matrices
    are built at the *cluster* level inside each trial.
    """
    raise NotImplementedError("run_ensemble_gw: implemented in Stage 3")
