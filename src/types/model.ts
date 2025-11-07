import type { InitializationMethod } from "./initialization.js";
import type { LayerInfo } from "./layer.js";
import type { LossFunction } from "./loss.js";
import type { OptimizationMethod } from "./optimization.js";
import type { RegularizationMethod } from "./regularization.js";

export type LayerParameter = {
  weights: number[][]; // サイズは層i+1のサイズ x 層iのサイズに等しい
  biases: number[]; // サイズは層i+1のサイズに等しい
};

// モデルデータ本体
export type ModelData = {
  version: string; // 適当

  // 広義のハイパーパラメータ
  batchSize: number; // バッチサイズ; 非負整数; 0の場合は全データ
  layerNumber: number; // 入力層・出力層を含む層の数; 2未満の場合は入力エラー
  layers: LayerInfo[]; // 要素数は layerNumber; 最初の要素は InputLayer, 最初以外の要素は GeneralLayer でなければ入力エラー
  initialization: InitializationMethod; // 初期化手法
  lossFunction: LossFunction; // 損失関数; BCE(Binary Cross Entropy) 固定
  regularization?: RegularizationMethod; // 正則化; しない場合は省略する
  optimization: OptimizationMethod; // 最適化方法

  // 学習されたパラメータ; 要素数は layerNumber - 1
  parameters: LayerParameter[];
};
