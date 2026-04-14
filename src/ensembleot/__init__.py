"""EnsembleOT: Ensemble Optimal Transport with randomized clustering."""

from .config import (
    ClusteringConfig,
    SinkhornConfig,
    GWConfig,
    EnsembleConfig,
)
from .operator import ImplicitTransportOperator
from .storage import TrialResult, EnsembleResult
from .sinkhorn import run_ensemble_sinkhorn
from .gw import run_ensemble_gw

__all__ = [
    "ClusteringConfig",
    "SinkhornConfig",
    "GWConfig",
    "EnsembleConfig",
    "ImplicitTransportOperator",
    "TrialResult",
    "EnsembleResult",
    "run_ensemble_sinkhorn",
    "run_ensemble_gw",
]

__version__ = "0.0.1"
