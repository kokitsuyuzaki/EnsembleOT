# EnsembleOT

Ensemble Optimal Transport via randomized clustering, with implicit
sample-level lifting. Backend: [POT](https://pythonot.github.io/).

## 公開 API の方針

- **Sinkhorn 系と Gromov–Wasserstein 系は別のエントリポイント**として提供します。
  1 つの関数に両者を統合しません。
  - `run_ensemble_sinkhorn(...)` — entropic / EMD 系
  - `run_ensemble_gw(...)` — Gromov–Wasserstein 系
- **現段階では `run_ensemble_sinkhorn` のみ実装済み**です。
  `run_ensemble_gw` は後続 Stage で実装され、現状は `NotImplementedError` を返します。
- **複数試行の集約 (aggregation) と結果の永続化 (storage) は後続 Stage** で追加します。
  現状 `run_ensemble_sinkhorn` は各 run の `ImplicitTransportOperator` を
  そのままリストで返すだけのミニマル実装です。

## Run weights from metrics

- 各 run の `op.meta["metrics"]` から、`compute_run_weights(operators, policy=..., key=...)`
  で weighted aggregation 用の run-weight ベクトル (非負・総和 1) を生成できます。
- 得られた weights はそのまま `make_weighted_mean_operator` /
  `weighted_consensus_edges` に渡せます。
- 対応 policy: `"uniform"`, `"inverse"`, `"softmax_negative"`,
  `"softmax_positive"`, `"rank_inverse"`。
- `key` は `"metrics.marginal_error_row"` のようなドット区切りで
  `op.meta` を辿って値を取ります。

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
