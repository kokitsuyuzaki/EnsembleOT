"""Entropic (Sinkhorn) ensemble OT entry point.

Stage 1: skeleton only. The actual POT-backed solver is implemented in Stage 3.
"""

from __future__ import annotations

import numpy as np

from .config import ClusteringConfig, EnsembleConfig, SinkhornConfig
from .storage import EnsembleResult


def run_ensemble_sinkhorn(
    X: np.ndarray,
    Y: np.ndarray,
    clustering: ClusteringConfig,
    sinkhorn: SinkhornConfig,
    ensemble: EnsembleConfig,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
) -> EnsembleResult:
    """Run an ensemble of cluster-level Sinkhorn OT trials.

    Parameters
    ----------
    X, Y : (n_x, d), (n_y, d) feature matrices.
    a, b : optional sample-level marginals (default: uniform).

    Returns
    -------
    EnsembleResult holding one TrialResult per trial, each with an
    ImplicitTransportOperator.
    """
    raise NotImplementedError("run_ensemble_sinkhorn: implemented in Stage 3")
