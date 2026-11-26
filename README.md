# EnsembleOT

[![DOI](https://zenodo.org/badge/1210430822.svg)](https://doi.org/10.5281/zenodo.24490690)
[![DockerHub/GHCR](https://github.com/kokitsuyuzaki/EnsembleOT/actions/workflows/build_test_push.yml/badge.svg)](https://github.com/kokitsuyuzaki/EnsembleOT/actions/workflows/build_test_push.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ensemble Optimal Transport via randomized clustering, with implicit
sample-level lifting. Backend: [POT](https://pythonot.github.io/).

## Installation

```bash
pip install git+https://github.com/kokitsuyuzaki/EnsembleOT.git
```

For development (Python >= 3.11):

```bash
git clone https://github.com/kokitsuyuzaki/EnsembleOT.git
cd EnsembleOT
pip install -e ".[dev]"
pytest
```

### Docker

Pre-built images are published to Docker Hub and the GitHub Container
Registry on every push to `main`:

```bash
docker pull koki/ensembleot:latest
# or
docker pull ghcr.io/kokitsuyuzaki/ensembleot:latest

docker run --rm -it koki/ensembleot:latest python -c "import ensembleot; print(ensembleot.__all__)"
```

## Public API policy

- **The Sinkhorn family and the Gromov–Wasserstein family are exposed as
  separate entry points.** They are never merged into a single function.
  - `run_ensemble_sinkhorn(...)` — entropic / EMD family (feature cost only)
  - `run_ensemble_gw(...)` — Gromov–Wasserstein family (structure cost only)
  - `run_ensemble_fgw(...)` — Fused Gromov–Wasserstein family (uses feature
    and structure costs jointly)
    - Stage 12a (default behavior): **only when X and Y share the same
      feature dimension**. `M = ot.dist(centers_x, centers_y, metric=...)`
      is generated internally.
    - Stage 12b extension: by passing `cross_feature_cost_fn=...`, the
      cluster-level cross-domain cost `M` (shape `(K_x, K_y)`) of each run
      can be supplied externally. This enables **cross-modal FGW** with
      `X.shape[1] != Y.shape[1]`. The callable receives `centers_x`,
      `centers_y`, `labels_x`, `labels_y`, `seed`, `metric`, etc.; extra
      arguments are passed through `cross_feature_cost_kwargs`.
    - The meaning of `alpha` follows POT's FGW implementation
      (`ot.gromov.fused_gromov_wasserstein` /
      `ot.gromov.entropic_fused_gromov_wasserstein`) as is. EnsembleOT does
      not redefine it.
- **Unbalanced and partial OT** (since v0.4.0) have their own entry points,
  one per POT solver family — see
  [Unbalanced and partial OT](#unbalanced-and-partial-ot-since-v040):
  `run_ensemble_unbalanced`, `run_ensemble_partial`,
  `run_ensemble_partial_gw`, `run_ensemble_partial_fgw`,
  `run_ensemble_fused_unbalanced_gw`.
- **Wasserstein barycenters** of two or more datasets (since v0.4.0) —
  see [Wasserstein barycenters](#wasserstein-barycenters-since-v040):
  `run_ensemble_free_support_barycenter`, `run_ensemble_barycenter`.
- All entry points, aggregation over multiple runs, metrics, weighted
  aggregation, and result persistence (storage) are implemented.

## End-to-end example (Sinkhorn + metric-weighted aggregation)

```python
import numpy as np
from ensembleot import (
    run_ensemble_sinkhorn,
    make_metric_weighted_mean_operator,
    metric_weighted_consensus_edges,
)

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 10))
Y = rng.standard_normal((180, 10)) + 0.3

# 1. Solve cluster-level OT for multiple runs
runs = run_ensemble_sinkhorn(
    X, Y,
    n_clusters_x=15, n_clusters_y=12,
    n_runs=8,
    solver_method="sinkhorn",
    reg=0.05,
    random_state=42,
)

# 2. Weighted mean operator that favors runs with smaller marginal error
mean_op = make_metric_weighted_mean_operator(
    runs,
    policy="inverse",
    key="metrics.marginal_error_row",
)

# 3. Transport Y-side features to the X side (without building the full T)
F_y = rng.standard_normal((Y.shape[0], 4))
F_x = mean_op.apply_to_features(F_y)      # shape (200, 4); row-normalized (barycentric) by default

# 4. Extract only the edges that are consistently strong across all runs
edges = metric_weighted_consensus_edges(
    runs,
    threshold=1e-4,
    policy="softmax_negative",
    key="metrics.transport_entropy",
    temperature=0.5,
    min_frequency=0.75,
    topk_per_source=3,
)
```

To use the GW variant, simply replace `run_ensemble_sinkhorn` with
`run_ensemble_gw`; the downstream aggregation pipeline works unchanged.

## Clustering backends

- `"kmeans"` (default) — scikit-learn's k-means, with iterative optimization.
- `"random_voronoi"` — a Voronoi-style spatial partition that picks seed
  points at random from X and assigns each sample to its nearest seed. It
  performs no iterative updates, so it is fast and well suited to producing a
  different coarse spatial partition in every run. It is available via
  `clustering_method="random_voronoi"` in any of `run_ensemble_sinkhorn` /
  `run_ensemble_gw` / `run_ensemble_fgw`.

## Run weights from metrics

- From each run's `op.meta["metrics"]`,
  `compute_run_weights(operators, policy=..., key=...)` produces a run-weight
  vector (non-negative, summing to 1) for weighted aggregation.
- The resulting weights can be passed directly to
  `make_weighted_mean_operator` / `weighted_consensus_edges`.
- Supported policies: `"uniform"`, `"inverse"`, `"softmax_negative"`,
  `"softmax_positive"`, `"rank_inverse"`.
- `key` is a dot-separated path such as `"metrics.marginal_error_row"` that
  is resolved by traversing `op.meta`.

## Memory characteristics of consensus edge extraction

`consensus_edges` / `weighted_consensus_edges` stream over source indices in
chunks of `block_size`. Each block allocates only two `(block_size, n_y)`
accumulation buffers (mean / frequency) plus one temporary submatrix of the
same size per run; the `(R, n_x, n_y)` stack and the `(n_x, n_y)` full mean
matrix are never materialized. When `block_size=None`, `min(n_x, 256)` is
used.

## Persistence (storage)

- EnsembleOT **never writes a dense sample × sample transport matrix to
  disk**.
- Only the compact operator representation of each run (`labels_x`,
  `labels_y`, `cluster_mass_x`, `cluster_mass_y`, `T_cluster`, meta) is
  saved, bundled into a single `.npz` (`save_operators` / `load_operators`).
- `MeanTransportOperator` likewise saves its constituent operators and is
  rebuilt at load time (`save_mean_operator` / `load_mean_operator`).

## Design invariants

- The huge sample × sample transport matrix `T` is not built during normal
  processing.
- The transport of each run is held implicitly in an
  `ImplicitTransportOperator`, and features are transported through the
  clusters with `apply_to_features` / `apply_transpose_to_features`.

## Normalization of feature transport (`normalize`, since v0.1.0)

`apply_to_features(F_y)` / `apply_transpose_to_features(F_x)` perform **row
normalization by default (`normalize=True`)**: for each destination sample
they return the weighted average divided by the coupling mass Σ_j T[i,j],
i.e. the barycentric projection (the conditional mean E[F_y | x_i]). The
output is on the same scale as `F_y`.

> **Breaking change from v0.0.x:** the old implementation returned the raw
> linear operator `T @ F_y`. Because the OT constraints make each row of `T`
> sum to `1/n_x`, the output shrank to roughly `1/n_x` of the original
> features and the contrast (tails) of the distribution appeared flattened.
> v0.1.0 changed the default to normalization.

- Specify `normalize=False` only when you need the raw operator `T @ F` (and
  `T.T @ X`). `MeanTransportOperator` / `WeightedMeanTransportOperator`
  accept the same `normalize` argument and normalize the aggregated
  transport in the same way.
- Note that `normalize` fixes the scale (i.e. the apparent contrast). The
  variance within a destination cluster is lost by cluster averaging, so
  when sharper tails are needed, combine it with tuning such as lowering
  `reg`, using `solver_method="emd"`, or increasing the number of clusters.

### Transport modes (`mode`, since v0.2.0 / matched since v0.3.0)

`apply_to_features` / `apply_transpose_to_features` switch between three
kinds of transport through the `mode` argument.

- `mode="barycentric"` (default) — deterministic. Returns the (normalized)
  weighted average described above, i.e. the conditional mean
  E[F_y | x_i]. It is smooth, but the variance within a destination cluster
  vanishes in the average, so **the tails are flattened**.
- `mode="stochastic"` — stochastic. For each source, one destination cluster
  is drawn with probability proportional to the coupling weights
  `T_cluster[a,·]`, and the **features of an actual sample are copied as
  is**. Since nothing is averaged, **the shape and tails of the F_y
  distribution are preserved**. Reproducible via `random_state`.
  `normalize` is ignored.
- `mode="matched"` — deterministic. The barycentric output is **remapped,
  per feature column and preserving ranks, onto the empirical distribution
  of the input F_y (quantile matching)**. Because the ranks do not change,
  **spatial correlation (Spearman, etc.) stays identical to barycentric**
  while only the flattened distribution is matched to F_y. `normalize` /
  `random_state` are ignored.

```python
F_smooth = mean_op.apply_to_features(F_y)                              # barycentric (default)
F_keep_tails = mean_op.apply_to_features(                              # preserves the distribution shape (stochastic)
    F_y, mode="stochastic", random_state=0
)
F_matched = mean_op.apply_to_features(F_y, mode="matched")            # restores the distribution while keeping correlation
```

barycentric and stochastic have mirror-image weaknesses: barycentric gives
"good spatial structure but a flattened distribution", whereas stochastic
"preserves the distribution but breaks spatial correlation because each
coordinate is drawn independently". `matched` is a deterministic option that
fills the gap between the two, restoring the distribution while keeping the
barycentric ranking (on real ST→SL data it was confirmed to drive the KS
distance of the distribution to 0 while keeping the same spatial Spearman
correlation as the smooth output).

`mode` is a feature of `ImplicitTransportOperator`, so it works not only
with `run_ensemble_sinkhorn` but **equally with operators obtained from
`run_ensemble_gw` / `run_ensemble_fgw`**. For `MeanTransportOperator` /
`WeightedMeanTransportOperator`, stochastic performs mixture sampling — for
each source, a run is drawn with probability proportional to its weight and
that run's stochastic transport is used — and matched applies quantile
matching to the aggregated barycentric output (neither the `(R, n_x, n_y)`
stack nor a dense T is built).

## Unbalanced and partial OT (since v0.4.0)

Balanced OT forces every bit of mass to be transported. These entry points
relax that, which matters when one dataset contains a population the other
lacks, or when the same populations occur in different proportions.

| Entry point | Cost | POT solvers (`solver_method`) |
|---|---|---|
| `run_ensemble_unbalanced` | feature | `sinkhorn_unbalanced` (default), `mm_unbalanced`, `lbfgsb_unbalanced` |
| `run_ensemble_partial` | feature | `partial_wasserstein` (default), `entropic_partial_wasserstein`, `partial_wasserstein_lagrange` |
| `run_ensemble_partial_gw` | structure | `partial_gromov_wasserstein` (default), `entropic_partial_gromov_wasserstein` |
| `run_ensemble_partial_fgw` | feature + structure | `partial_fused_gromov_wasserstein` (default), `entropic_partial_fused_gromov_wasserstein` |
| `run_ensemble_fused_unbalanced_gw` | structure (+ feature) | `fused_unbalanced_gromov_wasserstein` |

The two relaxations differ in kind:

- **Partial** transports exactly a mass `m` (a hard constraint). Mass can be
  dropped but never created. Cluster marginals sum to 1, so `m` is the
  *fraction* of mass to transport, exactly as in POT.
- **Unbalanced** penalizes deviation from the marginals with strength
  `reg_m` (a soft constraint). Mass can be dropped *or created*, and the
  total transported mass is an outcome rather than an input.

### Arguments follow POT

- Solver arguments keep POT's names, meanings and defaults and are forwarded
  verbatim. They are keyword-only. An argument left at `None` is not passed,
  so POT's own default for the selected solver applies.
- Passing an argument the selected solver does not accept raises
  `ValueError` instead of being silently ignored.
- `solver_method` is the name of the POT function that solves the
  cluster-level problem.
- The GW-family unbalanced solver uses the argument names of its POT
  function (`reg_marginals`, `epsilon`, `divergence`, `unbalanced_solver`,
  `max_iter`, `tol`), not the `reg` / `reg_m` names of `ot.unbalanced`.
  Its `alpha` is also POT's: a free non-negative weight on the feature term
  (`alpha=0`, the default, is plain unbalanced GW) — **not** the convex
  weight of balanced FGW. `run_ensemble_partial_fgw` keeps the balanced FGW
  meaning of `alpha`.
- `normalize_cost=True` (default) divides each cluster-level cost matrix by
  its maximum, as the balanced entry points do. **`reg_m` / `reg_marginals`
  act relative to the cost scale**, so set `normalize_cost=False` to give
  POT the raw `ot.dist` cost that a direct POT call would see.
- `float("inf")` in `reg_m` (keep one marginal exact) is supported only by
  `sinkhorn_unbalanced`; the other unbalanced solvers need finite values.

```python
import numpy as np
from ensembleot import (
    run_ensemble_partial, run_ensemble_unbalanced, make_mean_operator,
)

rng = np.random.default_rng(0)
shared = rng.standard_normal((300, 5))
only_in_x = rng.standard_normal((300, 5)) + 8.0     # no counterpart in Y
X = np.vstack([shared, only_in_x])
Y = rng.standard_normal((400, 5))

# Partial: transport half of the mass
runs = run_ensemble_partial(X, Y, 20, 15, n_runs=8, m=0.5, random_state=0)
mean_op = make_mean_operator(runs)

kept = mean_op.transported_mass_x(relative=True)    # (n_x,), 1.0 = fully matched
F_x = mean_op.apply_to_features(Y)                  # NaN where nothing was matched

# Unbalanced: let the solver decide how much mass to move
runs_u = run_ensemble_unbalanced(
    X, Y, 20, 15, n_runs=8, reg=0.05, reg_m=0.5, random_state=0
)
```

### Unmatched samples

Row and column sums of these couplings no longer equal the sample masses.

- `transported_mass_x()` / `transported_mass_y()` (on single-run and mean
  operators) return the transport mass of every sample;
  `relative=True` rescales so that 1.0 means "all of the sample's mass is
  transported". Averaged over runs this is a soft matching score.
- A sample with no transport mass has no image: `apply_to_features` returns
  **NaN** for it in every `mode`. Samples with small but non-zero mass get a
  finite value computed from little mass — threshold on the transported mass
  if you need a clean mask.
- `op.meta["metrics"]` gains `transported_mass`, `transport_cost` (where a
  feature cost exists) and, for partial solvers, `mass_error`. For these
  solvers `marginal_error_row` / `marginal_error_col` measure the intended
  relaxation, not solver error, so do not use them to weight runs.
- **Granularity.** Mass is dropped per cluster and spread uniformly inside
  it, so a single run says "every sample of this cluster keeps 60 % of its
  mass". Sample-level resolution comes from averaging runs with different
  clusterings.

### Choosing `m`

There is no canonical rule. Prior knowledge of the shared fraction is best;
otherwise sweep `m` and look for the bend where forcing unmatched mass
across makes the cost jump. The cluster-level problems are small, so the
sweep is cheap:

```python
from ensembleot import scan_partial_mass

res = scan_partial_mass(
    X, Y, 20, 15, n_runs=8, m_values=np.linspace(0.1, 1.0, 10), random_state=0
)
res["m_mid"], res["marginal_cost"]    # cost of the last unit of mass vs. m
```

### Behavior changes in v0.4.0

- Samples with zero transport mass now yield NaN instead of 0
  (`mode="barycentric"` / `"matched"`) or the global feature mean
  (`mode="stochastic"`). Balanced couplings are unaffected.
- `mode="stochastic"` on mean operators draws the run for each sample with
  probability ∝ run weight × that sample's transport mass. The distribution
  is unchanged for balanced runs, but a fixed `random_state` gives a
  different draw than v0.3.0.
- POT >= 0.9.6 is required.

## Wasserstein barycenters (since v0.4.0)

The barycenter of two or more datasets living in the same feature space,
again computed on cluster-level measures only.

| Entry point | Clustering | POT solvers (`solver_method`) |
|---|---|---|
| `run_ensemble_free_support_barycenter` | each dataset independently | `free_support_barycenter` (default), `free_support_sinkhorn_barycenter` |
| `run_ensemble_barycenter` | pooled data, one shared set of centers | `lp_barycenter` (default), `bregman_barycenter` |

- **Free support** optimizes the *positions* of `n_support` atoms (weights
  `b` fixed, uniform by default).
- **Fixed support** optimizes the *weights* on the shared cluster centers.

```python
from ensembleot import (
    run_ensemble_free_support_barycenter, pool_barycenters,
    transport_features_to_barycenter,
)

rng = np.random.default_rng(0)
Xs = [rng.standard_normal((1000, 2)) + c for c in ([0, 0], [6, 0], [3, 5])]

runs = run_ensemble_free_support_barycenter(
    Xs, n_clusters=20, n_support=15, n_runs=8, random_state=0
)
bary = pool_barycenters(runs)          # all runs' atoms, each run weighted 1/R
bary.support, bary.weights             # (8 * 15, 2), (8 * 15,)
bary.mean()                            # ~ (3, 5/3)

# carry features of dataset 0 onto the pooled atoms
F_atoms = transport_features_to_barycenter(runs, 0, Xs[0])
```

Things to know:

- Atoms of different runs have no correspondence, so runs are combined by
  **pooling atoms**, not by averaging coordinates. The pooled measure is a
  mixture of barycenter estimates (a bagging-style estimator); it is not
  itself the minimizer of the barycenter objective.
- **Free support shrinks, fixed support spreads.** Cluster means lose the
  within-cluster spread, so the free-support barycenter is somewhat too
  tight (use more clusters to reduce this). Fixed-support centers only exist
  where data exists, while a barycenter can sit *between* datasets, so its
  mass is smeared onto distant centers. Prefer free support when the
  datasets are far apart; fixed support is fine when they overlap.
- Each run's result carries one `ImplicitTransportOperator` per dataset
  (barycenter atoms ↔ samples), so the usual `apply_to_features` /
  `apply_transpose_to_features` work between the barycenter and every
  dataset. Atoms of zero weight return NaN.
- `save_barycenters` / `load_barycenters` persist results in a single
  `.npz`.
- Gromov–Wasserstein barycenters are not provided: their output is a
  structure matrix whose nodes have no correspondence across runs, and no
  sound way to aggregate them is implemented.

## Authors

- Koki Tsuyuzaki

## License

MIT — see `LICENSE`.
