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
from .fgw import run_ensemble_fgw
from .aggregate import (
    MeanTransportOperator,
    WeightedMeanTransportOperator,
    ConsensusEdge,
    make_mean_operator,
    make_weighted_mean_operator,
    consensus_edges,
    weighted_consensus_edges,
)
from .weights import compute_run_weights, extract_metric, normalize_weights
from .convenience import (
    make_metric_weighted_mean_operator,
    metric_weighted_consensus_edges,
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
    "run_ensemble_fgw",
    "MeanTransportOperator",
    "WeightedMeanTransportOperator",
    "ConsensusEdge",
    "make_mean_operator",
    "make_weighted_mean_operator",
    "consensus_edges",
    "weighted_consensus_edges",
    "save_operators",
    "load_operators",
    "save_mean_operator",
    "load_mean_operator",
    "StorageFormatError",
    "FORMAT_VERSION",
    "compute_run_weights",
    "extract_metric",
    "normalize_weights",
    "make_metric_weighted_mean_operator",
    "metric_weighted_consensus_edges",
]

__version__ = "0.0.1"
