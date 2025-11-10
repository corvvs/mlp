// 損失関数の情報
export type LossCCE = {
  method: "CCE";
  eps: number; // log-sum-exp のパラメータ
};

export type LossWeightedBCE = {
  method: "WeightedBCE";
  posWeight: number; // 正例の重み
  negWeight: number; // 負例の重み
  eps: number; // log-sum-exp のパラメータ
};

export type LossFunction = LossCCE | LossWeightedBCE;

export type LossFunctionActual = (yAnswer: number[], yPred: number[]) => number;

export type DerivativeLossFunctionActual = (
  aVec: number[],
  answers: number[]
) => number[];

export type EpochMetrics = {
  loss: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1Score: number;
};

// Improvement: 減少するほど良い, という含意がある
export type EpochMetricsImprovement = {
  loss: number;
  accuracy: number;
  precision: number;
  recall: number;
  specificity: number;
  f1Score: number;
};
