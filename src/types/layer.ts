// 層の情報

import type {
  ActivationFunction,
  ActivationFunctionSingleArgument,
  AFSoftmax,
} from "./af.js";

export type ScaleFactor = {
  mean: number; // 平均
  stddev: number; // 標準偏差; stddev == 0 の列は前処理で除かれていることを期待する
};

// 入力層
export type InputLayerInfo = {
  layerType: "input";
  size: number; // 層のサイズ

  // 標準化を行った列についての標準化前の統計情報; 標準化を行わなかった列については null
  scaleFactors: (ScaleFactor | null)[];
};

// 隠れ層
export type HiddenLayerInfo = {
  layerType: "hidden";
  size: number; // 層のサイズ
  activationFunction: ActivationFunctionSingleArgument; // 活性化関数; 本課題では, 出力層以外に softmax を指定すると入力エラー
};

export type OutputLayerInfo = {
  layerType: "output";
  size: 2;
  activationFunction: AFSoftmax; // 活性化関数
};

export type LayerInfo = InputLayerInfo | HiddenLayerInfo | OutputLayerInfo;
