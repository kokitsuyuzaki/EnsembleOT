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
from .unbalanced import run_ensemble_unbalanced
from .partial import run_ensemble_partial, scan_partial_mass
from .partial_gw import run_ensemble_partial_gw
from .partial_fgw import run_ensemble_partial_fgw
from .unbalanced_gw import run_ensemble_fused_unbalanced_gw
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
from .barycenter import (
    BarycenterResult,
    run_ensemble_free_support_barycenter,
    pool_barycenters,
    weighted_pool_barycenters,
    run_ensemble_barycenter,
    transport_features_to_barycenter,
)
from .io import save_barycenters, load_barycenters

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
    "run_ensemble_unbalanced",
    "run_ensemble_partial",
    "scan_partial_mass",
    "run_ensemble_partial_gw",
    "run_ensemble_partial_fgw",
    "run_ensemble_fused_unbalanced_gw",
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
    "BarycenterResult",
    "run_ensemble_free_support_barycenter",
    "pool_barycenters",
    "weighted_pool_barycenters",
    "run_ensemble_barycenter",
    "transport_features_to_barycenter",
    "save_barycenters",
    "load_barycenters",
]

__version__ = "0.4.0"
