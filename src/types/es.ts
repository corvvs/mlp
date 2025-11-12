import type { EpochMetrics } from "./loss.js";
import type { ModelData } from "./model.js";

export type EarlyStoppingMetric = keyof EpochMetrics;

export type EarlyStopping = {
  metric: EarlyStoppingMetric;
  patience: number;
};

export type LatestGoodModel = {
  epoch: number;
  score: number;
  model: ModelData | null;
};

export type EarlyStoppingFunction = (
  currentModel: ModelData,
  latestGoodModel: LatestGoodModel,
  epoch: number,
  currentMetrics: EpochMetrics
) => string | null;
