import type { ScaleFactor } from "./layer.js";

export type StandardizeResult = {
  rows: number[][];
  scaleFactors: (ScaleFactor | null)[];
};
