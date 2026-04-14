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
from .aggregate import (
    MeanTransportOperator,
    ConsensusEdge,
    make_mean_operator,
    consensus_edges,
)
from .io import (
    save_operators,
    load_operators,
    save_mean_operator,
    load_mean_operator,
    StorageFormatError,
    FORMAT_VERSION,
)

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
    "MeanTransportOperator",
    "ConsensusEdge",
    "make_mean_operator",
    "consensus_edges",
    "save_operators",
    "load_operators",
    "save_mean_operator",
    "load_mean_operator",
    "StorageFormatError",
    "FORMAT_VERSION",
]

__version__ = "0.0.1"
