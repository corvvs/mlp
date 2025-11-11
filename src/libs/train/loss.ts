import type {
  LossFunction,
  LossFunctionActual,
  EpochMetrics as EpochMetrics,
  EpochMetricsImprovement,
  DerivativeLossFunctionActual,
} from "../../types/loss.js";
import type { RegularizationFunction } from "../../types/regularization.js";
import { subVector } from "../arithmetics.js";

export function parseLossFunction(str: string): LossFunction {
  const parts = str.split(",");
  const method = parts[0];
  switch (method.toLowerCase()) {
    case "cce": {
      const f: LossFunction = {
        method: "CCE",
        eps: 1e-9,
      };
      if (parts.length >= 2) {
        const eps = parseFloat(parts[1]);
        if (isNaN(eps) || eps <= 0 || eps >= 0.5) {
          throw new Error(`不正なCCE損失関数パラメータ: ${parts[1]}`);
        }
        f.eps = eps;
      }
      return f;
    }
    case "weightedbce": {
      const f: LossFunction = {
        method: "WeightedBCE",
        posWeight: 1.0,
        negWeight: 1.0,
        eps: 1e-9,
      };
      if (parts.length >= 2) {
        const posWeight = parseFloat(parts[1]);
        if (isNaN(posWeight) || posWeight <= 0) {
          throw new Error(`不正なWeightedBCE損失関数パラメータ: ${parts[1]}`);
        }
        f.posWeight = posWeight;
      }
      if (parts.length >= 3) {
        const negWeight = parseFloat(parts[2]);
        if (isNaN(negWeight) || negWeight <= 0) {
          throw new Error(`不正なWeightedBCE損失関数パラメータ: ${parts[2]}`);
        }
        f.negWeight = negWeight;
      }
      if (parts.length >= 4) {
        const eps = parseFloat(parts[3]);
        if (isNaN(eps) || eps <= 0 || eps >= 0.5) {
          throw new Error(`不正なWeightedBCE損失関数パラメータ: ${parts[3]}`);
        }
        f.eps = eps;
      }
      return f;
    }
    default:
      throw new Error(`未知の損失関数: ${method}`);
  }
}

export function getLossFunctionActual(
  lossFunction: LossFunction
): LossFunctionActual {
  switch (lossFunction.method) {
    case "CCE":
      const eps = lossFunction.eps;
      return (yAnswer, yPred) => {
        // yAnswerは0 or 1
        // yPredは2次元ベクトル [p_M, p_B]
        const yPredC0 = Math.min(Math.max(yPred[0], eps), 1 - eps);
        const yPredC1 = Math.min(Math.max(1 - yPred[0], eps), 1 - eps);
        // Cross-Entropy = -Σ(y_true * log(y_pred))
        return -(
          yAnswer[0] * Math.log(yPredC0) +
          (1 - yAnswer[0]) * Math.log(yPredC1)
        );
      };
    case "WeightedBCE":
      const posWeight = lossFunction.posWeight;
      const negWeight = lossFunction.negWeight;
      const epsW = lossFunction.eps;
      return (yAnswer, yPred) => {
        const yPredC0 = Math.min(Math.max(yPred[0], epsW), 1 - epsW);
        return -(
          posWeight * yAnswer[0] * Math.log(yPredC0) +
          negWeight * (1 - yAnswer[0]) * Math.log(1 - yPredC0)
        );
      };
    default:
      throw new Error(`未知の損失関数: ${lossFunction}`);
  }
}

export function getDerivativeLossFunctionActual(
  lossFunction: LossFunction
): DerivativeLossFunctionActual {
  switch (lossFunction.method) {
    case "CCE":
      return subVector;
    case "WeightedBCE":
      const posWeight = lossFunction.posWeight;
      const negWeight = lossFunction.negWeight;
      return (aVec, answers) => {
        // answers[0]が正例の確率(0 or 1)
        // 正例の場合はposWeight、負例の場合はnegWeightを適用
        const weight = answers[0] * posWeight + (1 - answers[0]) * negWeight;
        return subVector(aVec, answers).map((x) => x * weight);
      };
    default:
      throw new Error(`未知の損失関数: ${lossFunction}`);
  }
}

export function getLoss(props: {
  inputVectors: number[][];
  outputMats: number[][][];
  wMats: number[][][];
  lossFunction: LossFunctionActual;
  regularizationFunction: RegularizationFunction | null;
}) {
  let meanLoss = 0;
  const {
    inputVectors,
    outputMats,
    lossFunction,
    wMats,
    regularizationFunction,
  } = props;
  const outputMat = outputMats[outputMats.length - 1];
  const regularizationContribution = regularizationFunction
    ? regularizationFunction(wMats)
    : 0;
  let corrects = 0;
  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  inputVectors.forEach((inputVector, k) => {
    const outVector = outputMat[k];
    const yAnswer = inputVector[0];
    const yPred = outVector[0]; // 正解ラベル1に対応する出力
    const answerIsPositive = yAnswer === 1;
    const predIsPositive = yPred >= 0.5;
    const isCorrect = answerIsPositive === predIsPositive;
    if (answerIsPositive && predIsPositive) {
      tp++;
    } else if (!answerIsPositive && !predIsPositive) {
      tn++;
    } else if (!answerIsPositive && predIsPositive) {
      fp++;
    } else if (answerIsPositive && !predIsPositive) {
      fn++;
    }
    // 損失の計算
    const answer0 = yAnswer === 1 ? 1 : 0;
    const loss = lossFunction([answer0, 1 - answer0], outVector);
    meanLoss += loss;
    corrects += isCorrect ? 1 : 0;
  });
  meanLoss /= inputVectors.length;
  meanLoss += regularizationContribution;
  return { meanLoss, corrects, tp, tn, fp, fn };
}

export function getMetrics(props: {
  loss: number;
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}): EpochMetrics {
  const { tp, tn, fp, fn } = props;
  // 精度 = 陽性陰性関係ない正解率
  const accuracy = (tp + tn) / (tp + tn + fp + fn);
  // 適合率 = 陽性に対する真陽性率
  const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
  // 再現率 = 感度 = 真陽性に対する陽性検出率
  const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
  // 特異度 = 真陰性に対する陰性検出率
  const specificity = tn + fp === 0 ? 0 : tn / (tn + fp);
  // F1スコア = 適合率と再現率の調和平均
  const f1Score =
    precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);
  return {
    loss: props.loss,
    accuracy,
    precision,
    recall,
    specificity,
    f1Score,
  };
}

export function getMetricsImprovement(
  metricsPrev: EpochMetrics,
  metricsCurr: EpochMetrics
): EpochMetricsImprovement {
  // 減少するほど良い, という量を返す
  return {
    loss: metricsCurr.loss - metricsPrev.loss,
    accuracy: metricsPrev.accuracy - metricsCurr.accuracy,
    precision: metricsPrev.precision - metricsCurr.precision,
    recall: metricsPrev.recall - metricsCurr.recall,
    specificity: metricsPrev.specificity - metricsCurr.specificity,
    f1Score: metricsPrev.f1Score - metricsCurr.f1Score,
  };
}
