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
  trainPrecision: number;
  valPrecision: number;
  trainRecall: number;
  valRecall: number;
  trainSpecificity: number;
  valSpecificity: number;
  trainF1Score: number;
  valF1Score: number;
};
