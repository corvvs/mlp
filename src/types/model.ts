import type { EarlyStopping } from "./es.js";
import type { InitializationMethod } from "./initialization.js";
import type { LayerInfo } from "./layer.js";
import type { EpochMetrics, LossFunction } from "./loss.js";
import type { OptimizationMethod } from "./optimization.js";
import type { RegularizationMethod } from "./regularization.js";

export type LayerParameter = {
  weights: number[][]; // サイズは層i+1のサイズ x 層iのサイズに等しい
  biases: number[]; // サイズは層i+1のサイズに等しい
};

// モデルデータ本体
export type ModelData = {
  // [学習に使う情報]
  version: string; // 適当

  // 広義のハイパーパラメータ
  seed: number; // 乱数シード
  splitRatio: number; // データ分割比率 (train: splitRatio, validation: 1 - splitRatio)
  maxEpochs: number; // 最大エポック数; 正の整数
  batchSize: number; // バッチサイズ; 非負整数; 0の場合は全データ
  layerNumber: number; // 入力層・出力層を含む層の数; 2未満の場合は入力エラー
  layers: LayerInfo[]; // 要素数は layerNumber; 最初の要素は InputLayer, 最初以外の要素は GeneralLayer でなければ入力エラー
  initialization: InitializationMethod; // 初期化手法
  lossFunction: LossFunction; // 損失関数; BCE(Binary Cross Entropy) 固定
  regularization?: RegularizationMethod; // 正則化; しない場合は省略する
  optimization: OptimizationMethod; // 最適化方法
  earlyStopping?: EarlyStopping;

  // [学習の結果の情報]
  bestEpoch: number; // 最良モデルが得られたエポック数; 1始まり

  // 学習されたパラメータ; 要素数は layerNumber - 1
  parameters: LayerParameter[];

  trainMetrics: EpochMetrics[];
  valMetrics: EpochMetrics[];
};
