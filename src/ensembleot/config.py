"""Configuration dataclasses for EnsembleOT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClusteringConfig:
    method: str = "kmeans"
    n_clusters_x: int = 10
    n_clusters_y: int = 10
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SinkhornConfig:
    reg: float = 0.1
    numItermax: int = 1000
    stopThr: float = 1e-6
    metric: str = "sqeuclidean"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GWConfig:
    loss_fun: str = "square_loss"
    epsilon: float = 0.05
    max_iter: int = 1000
    tol: float = 1e-6
    metric: str = "sqeuclidean"
    entropic: bool = True
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleConfig:
    n_trials: int = 10
    base_seed: int = 0
    n_jobs: int = 1
