// 層の情報

import type { ActivationFunction } from "./af.js";

// 入力層
export type InputLayerInfo = {
  size: number; // 層のサイズ

  scaleFactors: ({
    // 標準化を行った列についての標準化前の統計情報; 標準化を行わなかった列については null
    mean: number; // 平均
    stddev: number; // 標準偏差; stddev == 0 の列は前処理で除かれていることを期待する
  } | null)[];
};

// 隠れ層および出力層
export type GeneralLayerInfo = {
  size: number; // 層のサイズ; 本課題では, 出力層において 2 でなければ入力エラー
  activationFunction: ActivationFunction; // 活性化関数; 本課題では, 出力層以外に softmax を指定すると入力エラー
};

export type LayerInfo = InputLayerInfo | GeneralLayerInfo;
