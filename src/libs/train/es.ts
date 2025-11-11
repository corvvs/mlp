import type {
  EarlyStopping,
  EarlyStoppingFunction,
  LatestGoodModel,
} from "../../types/es.js";
import type { EpochMetrics } from "../../types/loss.js";
import type { ModelData } from "../../types/model.js";

export function parseEarlyStopping(args: {
  metric: string;
  patience: number;
}): EarlyStopping | null {
  if (!args.metric) {
    return null;
  }
  if (args.patience === 0) {
    return null;
  } else if (args.patience < 0) {
    throw new Error(
      `EarlyStopping の patience は非負整数でなければなりません: ${args.patience}`
    );
  }

  switch (args.metric.toLowerCase()) {
    case "accuracy":
      return {
        metric: "accuracy",
        patience: args.patience,
      };
    case "loss":
      return {
        metric: "loss",
        patience: args.patience,
      };
    case "precision":
      return {
        metric: "precision",
        patience: args.patience,
      };
    case "recall":
      return {
        metric: "recall",
        patience: args.patience,
      };
    case "f1score":
      return {
        metric: "f1Score",
        patience: args.patience,
      };
    default:
      throw new Error(`未知の EarlyStopping metric: ${args.metric}`);
  }
}

export function getEarlyStoppingActual(
  earlyStopping: EarlyStopping | null
): EarlyStoppingFunction | null {
  if (!earlyStopping) {
    return null;
  }
  const getScore = (() => {
    switch (earlyStopping.metric) {
      case "accuracy":
        return (metrics: EpochMetrics) => 1 - metrics.accuracy;
      case "loss":
        return (metrics: EpochMetrics) => metrics.loss;
      case "precision":
        return (metrics: EpochMetrics) => 1 - metrics.precision;
      case "recall":
        return (metrics: EpochMetrics) => 1 - metrics.recall;
      case "f1Score":
        return (metrics: EpochMetrics) => 1 - metrics.f1Score;
      default:
        throw new Error(
          `Unsupported EarlyStopping metric: ${earlyStopping.metric}`
        );
    }
  })();
  let scoreDeteriorationCount = 0;
  const deteriorationEps = 0.00001;
  return (
    currentModel: ModelData,
    latestGoodModel: LatestGoodModel,
    epoch: number,
    currentMetrics: EpochMetrics
  ) => {
    const currentScore = getScore(currentMetrics);

    if (latestGoodModel.score - currentScore < deteriorationEps) {
      scoreDeteriorationCount++;
      if (scoreDeteriorationCount >= earlyStopping.patience) {
        return `Early stopping: ${earlyStopping.metric} did not improve in ${
          earlyStopping.patience
        } epochs; best score: ${latestGoodModel.score.toFixed(6)} at epoch ${
          latestGoodModel.epoch
        }.`;
      }
    } else {
      scoreDeteriorationCount = 0;
    }

    if (currentScore < latestGoodModel.score) {
      latestGoodModel.score = currentScore;
      latestGoodModel.model = JSON.parse(JSON.stringify(currentModel));
      latestGoodModel.epoch = epoch;
    }

    if (currentScore * 0.5 > latestGoodModel.score) {
      return `Early stopping: ${earlyStopping.metric} deteriorated significantly.`;
    }
    if (latestGoodModel.epoch < epoch - 100) {
      return "Early stopping: too many epochs since last improvement.";
    }
    return null;
  };
}
