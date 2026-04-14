"""Shared ensemble trial-loop scaffolding.

This module will host the common per-trial orchestration used by both
the Sinkhorn and GW entry points (clustering -> solver -> operator
-> storage). Stage 1 only declares the placeholder surface.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import ClusteringConfig, EnsembleConfig
from .storage import TrialResult


TrialSolver = Callable[..., TrialResult]


def iter_trial_seeds(ensemble: EnsembleConfig) -> list[int]:
    rng = np.random.default_rng(ensemble.base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=ensemble.n_trials)]


def run_trials(
    solver: TrialSolver,
    ensemble: EnsembleConfig,
    clustering: ClusteringConfig,
    **solver_kwargs,
) -> list[TrialResult]:
    raise NotImplementedError("run_trials: implemented in Stage 2/3")
