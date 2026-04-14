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

## 設計上の不変条件

- サンプル × サンプルの巨大な輸送行列 `T` は通常処理で作らない。
- 各 run の輸送は `ImplicitTransportOperator` で implicit に保持し、
  `apply_to_features` / `apply_transpose_to_features` で `T @ Y`, `T.T @ X`
  をクラスタ経由で計算する。
