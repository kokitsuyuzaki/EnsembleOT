# EnsembleOT

Ensemble Optimal Transport via randomized clustering, with implicit
sample-level lifting. Backend: [POT](https://pythonot.github.io/).

## 公開 API の方針

- **Sinkhorn 系と Gromov–Wasserstein 系は別のエントリポイント**として提供します。
  1 つの関数に両者を統合しません。
  - `run_ensemble_sinkhorn(...)` — entropic / EMD 系 (feature cost のみ)
  - `run_ensemble_gw(...)` — Gromov–Wasserstein 系 (structure cost のみ)
  - `run_ensemble_fgw(...)` — Fused Gromov–Wasserstein 系 (feature + structure を同時に使う)
    - Stage 12a (デフォルト挙動): **X, Y が同じ feature 次元を持つ場合のみ**。
      `M = ot.dist(centers_x, centers_y, metric=...)` を内部で自動生成します。
    - Stage 12b 拡張: `cross_feature_cost_fn=...` を渡せば、各 run の
      cluster-level cross-domain cost `M` (shape `(K_x, K_y)`) を外部から
      供給できます。これにより `X.shape[1] != Y.shape[1]` の **cross-modal
      FGW** が可能になります。callable は `centers_x`, `centers_y`,
      `labels_x`, `labels_y`, `seed`, `metric` 等を受け取り、追加引数は
      `cross_feature_cost_kwargs` で渡します。
    - `alpha` の意味は引き続き POT の FGW 実装 (`ot.gromov.fused_gromov_wasserstein`
      / `ot.gromov.entropic_fused_gromov_wasserstein`) にそのまま準拠します。
      EnsembleOT 側で再定義はしません。
- Sinkhorn / GW / FGW の 3 エントリポイント、複数試行の集約 (aggregation)、
  metrics、weighted aggregation、結果の永続化 (storage) はいずれも実装済みです。

## End-to-end 使用例 (Sinkhorn + metric-weighted aggregation)

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

# 1. 複数 run の cluster-level OT を解く
runs = run_ensemble_sinkhorn(
    X, Y,
    n_clusters_x=15, n_clusters_y=12,
    n_runs=8,
    solver_method="sinkhorn",
    reg=0.05,
    random_state=42,
)

# 2. marginal 誤差が小さい run ほど重視した weighted mean operator
mean_op = make_metric_weighted_mean_operator(
    runs,
    policy="inverse",
    key="metrics.marginal_error_row",
)

# 3. Y 側の特徴量を X 側へ搬送 (full T を作らない)
F_y = rng.standard_normal((Y.shape[0], 4))
F_x = mean_op.apply_to_features(F_y)      # shape (200, 4)

# 4. 全 run で一貫して強い edge のみ抽出
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

GW 版を使う場合は `run_ensemble_sinkhorn` を `run_ensemble_gw` に差し替えるだけで、
以降の aggregation パイプラインはそのまま使えます。

## Run weights from metrics

- 各 run の `op.meta["metrics"]` から、`compute_run_weights(operators, policy=..., key=...)`
  で weighted aggregation 用の run-weight ベクトル (非負・総和 1) を生成できます。
- 得られた weights はそのまま `make_weighted_mean_operator` /
  `weighted_consensus_edges` に渡せます。
- 対応 policy: `"uniform"`, `"inverse"`, `"softmax_negative"`,
  `"softmax_positive"`, `"rank_inverse"`。
- `key` は `"metrics.marginal_error_row"` のようなドット区切りで
  `op.meta` を辿って値を取ります。

## Consensus edge 抽出のメモリ特性

`consensus_edges` / `weighted_consensus_edges` は source index を
`block_size` 単位でストリーム処理します。各ブロックでは `(block_size, n_y)`
の累積バッファ 2 本 (mean / frequency) と、各 run あたり同サイズの一時
submatrix だけを確保し、`(R, n_x, n_y)` の stack や `(n_x, n_y)` の全平均
行列は一切作りません。`block_size=None` のときは `min(n_x, 256)` を使います。

## 永続化 (storage)

- EnsembleOT は **sample × sample の dense な transport 行列をディスクに保存しません**。
- 保存対象は各 run の compact operator 表現 (`labels_x`, `labels_y`,
  `cluster_mass_x`, `cluster_mass_y`, `T_cluster`, meta) のみで、これを
  単一の `.npz` にまとめます (`save_operators` / `load_operators`)。
- `MeanTransportOperator` も構成要素の operators を保存し、ロード時に
  再構築する方式です (`save_mean_operator` / `load_mean_operator`)。

## 設計上の不変条件

- サンプル × サンプルの巨大な輸送行列 `T` は通常処理で作らない。
- 各 run の輸送は `ImplicitTransportOperator` で implicit に保持し、
  `apply_to_features` / `apply_transpose_to_features` で `T @ Y`, `T.T @ X`
  をクラスタ経由で計算する。
