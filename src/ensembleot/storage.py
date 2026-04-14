"""Containers for per-trial and ensemble results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .operator import ImplicitTransportOperator


@dataclass
class TrialResult:
    trial_id: int
    seed: int
    labels_x: np.ndarray
    labels_y: np.ndarray
    cluster_mass_x: np.ndarray
    cluster_mass_y: np.ndarray
    T_cluster: np.ndarray
    clustering_method: str
    solver_name: str
    solver_params: dict[str, Any]
    operator: ImplicitTransportOperator
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    trials: list[TrialResult]
    solver_name: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.trials)

    def __iter__(self):
        return iter(self.trials)
