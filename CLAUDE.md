# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## コマンド

```bash
pip install -e ".[dev]"      # 開発インストール (Python >= 3.11)
pytest                        # 全テスト (tests/ 配下)
pytest tests/test_sinkhorn.py::test_name -x   # 単一テスト
```

Lint/format 設定は未整備。POT (`pot`), numpy, scipy, scikit-learn が実行時依存。

## アーキテクチャ

EnsembleOT は「ランダム化クラスタリング + cluster-level OT + implicit な sample-level lifting」を複数 run 束ねる Python パッケージ。**dense な `(n_x, n_y)` 輸送行列 `T` は原則として作らない**ことが設計上の不変条件。

### データフロー (1 run)

`run_ensemble_sinkhorn` / `run_ensemble_gw` → `clustering.py` で X, Y を独立にクラスタリング (k-means 等、random_state を run 毎にずらす) → cluster-level のコスト行列から POT で `T_cluster` を解く (`sinkhorn.py` / `gw.py`) → `ImplicitTransportOperator` (`operator.py`) に `labels_x, labels_y, cluster_mass_x, cluster_mass_y, T_cluster, meta` を格納して返す。

`ImplicitTransportOperator.apply_to_features(F_y)` は `T_cluster` とクラスタ所属だけで `T @ F_y` を計算する。`lifting.py` に implicit lifting の本体がある。`T` を直接展開する API は意図的に提供しない。

### 集約 (`aggregate.py`)

複数 run の operators を `MeanTransportOperator` / `WeightedMeanTransportOperator` で束ねる。`consensus_edges` / `weighted_consensus_edges` は source index を `block_size` 単位でストリーム処理し、`(R, n_x, n_y)` スタックや `(n_x, n_y)` 全平均行列を決して作らない (README 参照)。新しい集約系を足すときもこのメモリ特性を壊さないこと。

### Run weights (`weights.py`, `convenience.py`)

各 `op.meta["metrics"]` (`metrics.py` が生成) を元に `compute_run_weights(policy=..., key="metrics.marginal_error_row")` で非負・総和 1 の run weight を作る。`key` はドット区切りで `op.meta` を辿る。`convenience.py` の `make_metric_weighted_mean_operator` / `metric_weighted_consensus_edges` は「weights 計算 → weighted aggregation」を 1 つにまとめたショートカット。

### 永続化 (`io.py`, `storage.py`)

`save_operators` / `load_operators` は operators の compact 表現だけを 1 つの `.npz` にまとめる。`MeanTransportOperator` も構成 operators を保存しロード時に再構築する (`save_mean_operator` / `load_mean_operator`)。`FORMAT_VERSION` を上げるときは `StorageFormatError` 経路を維持すること。`storage.py` の `TrialResult` / `EnsembleResult` は in-memory の軽量コンテナ。

### 公開 API の切り分け

- Sinkhorn 系 (`run_ensemble_sinkhorn`) と GW 系 (`run_ensemble_gw`) は**別エントリポイント**。1 つの関数に統合しない。
- `run_ensemble_gw` 側の実装は段階的に埋めている (README では未実装扱いの記述があるが `src/ensembleot/gw.py` は存在する — 実体を確認してから触ること)。
- 公開シンボルは `ensembleot/__init__.py` の `__all__` が source of truth。

## リポジトリの進め方

開発は `Stage N.0` という粒度のコミットで段階実装されている (`git log` 参照)。新機能追加時もこの粒度に合わせると履歴が追いやすい。
