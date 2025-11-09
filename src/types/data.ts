import type { ScaleFactor } from "./layer.js";

export type StandardizeResult = {
  rows: number[][];
  scaleFactors: (ScaleFactor | null)[];
};

export type TrainingProgress = {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAccuracy: number;
  valAccuracy: number;
};
